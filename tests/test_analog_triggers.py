"""Run in Blender: blender --background --factory-startup --python tests/test_analog_triggers.py.

Requires the bundled SDL wheels to be installed or on PYTHONPATH.
Uses actual Blender RNA and RegionView3D; no physical controller is simulated as hardware evidence.
"""
import importlib
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
# Optional path for an isolated local test environment, outside the add-on.
if (ROOT.parent / 'deps').exists():
    sys.path.insert(0, str(ROOT.parent / 'deps'))
addon = importlib.import_module(ROOT.name)
prefs = importlib.import_module(ROOT.name + '.preferences')
actions = importlib.import_module(ROOT.name + '.controller_actions')
io = importlib.import_module(ROOT.name + '.io_operations')
sdl = importlib.import_module(ROOT.name + '.sdl_handler')
import sdl2


class AnalogTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for c in prefs.PREFERENCE_CLASSES:
            bpy.utils.register_class(c)
        bpy.types.WindowManager.test_gamepad_modes = bpy.props.CollectionProperty(type=prefs.GamepadModeSettings)

    @classmethod
    def tearDownClass(cls):
        del bpy.types.WindowManager.test_gamepad_modes
        for c in reversed(prefs.PREFERENCE_CLASSES):
            bpy.utils.unregister_class(c)

    def setUp(self):
        modes = bpy.context.window_manager.test_gamepad_modes
        modes.clear()
        self.mode = modes.add()
        self.mode.name = 'Test'
        self.side = self.mode.left_side
        self.side.trigger_mode = 'ANALOG'
        self.side.trigger_deadzone = 0.05
        self.executor = actions.ControllerActionExecutor()
        self.area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
        self.rv = self.area.spaces.active.region_3d
        self.rv.view_perspective = 'PERSP'
        self.rv.view_distance = 10.0
        self.snapshot = {o.name: tuple(tuple(row) for row in o.matrix_world) for o in bpy.data.objects}

    def tearDown(self):
        self.assertEqual(self.snapshot, {o.name: tuple(tuple(row) for row in o.matrix_world) for o in bpy.data.objects})

    def press(self, pressure, side=None, key=None):
        side = side or self.side
        key = key or self.executor.LEFT_TRIGGER_KEY
        self.executor._reader = {key: pressure}
        self.executor._process_trigger(key, self.mode, side.trigger_action, side)

    def test_deadzone_release_and_clamping(self):
        for p in (-1.0, 0.0, 0.04, 0.05, float('nan')):
            self.press(p)
            self.assertEqual(self.rv.view_distance, 10.0)
        self.press(2.0)
        self.assertAlmostEqual(self.rv.view_distance, 10 * math.exp(-0.05), places=5)
        stopped = self.rv.view_distance
        self.press(0.0)
        self.assertEqual(self.rv.view_distance, stopped)

    def test_pressure_is_continuous_and_hold_repeats(self):
        self.press(0.525)  # Half of the usable travel after deadzone.
        self.assertAlmostEqual(self.rv.view_distance, 10 * math.exp(-0.025), places=5)
        self.press(0.525)
        self.assertAlmostEqual(self.rv.view_distance, 10 * math.exp(-0.05), places=5)
        self.press(1.0)
        self.assertAlmostEqual(self.rv.view_distance, 10 * math.exp(-0.10), places=5)

    def test_sensitivity_invert_and_simultaneous_triggers(self):
        self.side.trigger_sensitivity = 2.0
        self.press(1.0)
        self.assertAlmostEqual(self.rv.view_distance, 10 * math.exp(-0.10), places=5)
        self.side.trigger_invert = True
        self.press(1.0)
        self.assertAlmostEqual(self.rv.view_distance, 10.0, places=5)
        right = self.mode.right_side
        right.trigger_mode = 'ANALOG'
        self.side.trigger_sensitivity = 1.0
        self.press(0.525)
        self.press(0.525, right, self.executor.RIGHT_TRIGGER_KEY)
        self.assertAlmostEqual(self.rv.view_distance, 10.0, places=5)

    def test_poll_rate_independence(self):
        results = []
        for rate in (30, 60, 120):
            self.rv.view_distance = 10.0
            self.executor._analog_frame_scale = 60.0 / rate
            for _ in range(rate):
                self.press(0.525)
            results.append(self.rv.view_distance)
        for result in results:
            self.assertAlmostEqual(result, 10 * math.exp(-1.5), places=5)

    def test_all_navigation_actions_and_camera_lock(self):
        self.area.spaces.active.lock_camera = True
        for action, *_ in prefs.SEPARATE_AXIS_ACTION_ITEMS:
            self.side.trigger_axis_action = action
            self.rv.view_perspective = 'CAMERA'
            with patch.object(self.executor, '_event_simulate', side_effect=AssertionError('OS event')):
                self.press(1.0)
            self.assertEqual(self.rv.view_perspective, 'CAMERA' if action == 'NONE' else 'PERSP')
        self.area.spaces.active.lock_camera = False

    def test_each_supported_action_dispatches_proportional_value(self):
        for action, *_ in prefs.SEPARATE_AXIS_ACTION_ITEMS:
            if action in {'NONE', 'ZOOM_VIEW'}:
                continue
            self.side.trigger_axis_action = action
            with patch.object(self.executor, '_execute_separate_axis_action') as call:
                self.press(0.525)
                call.assert_called_once()
                self.assertEqual(call.call_args.args[1], action)
                self.assertAlmostEqual(call.call_args.args[2], 0.5, places=6)

    def test_missing_viewport_and_zero_sensitivity(self):
        with patch.object(self.executor, '_find_view3d_area', return_value=(None, None)):
            self.press(1.0)
        self.side.trigger_sensitivity = 0.0
        self.press(1.0)
        self.assertEqual(self.rv.view_distance, 10.0)

    def test_legacy_button_edges_and_switch_to_analog(self):
        self.side.trigger_mode = 'BUTTON'
        self.side.trigger_action = 'MOUSE_LEFT'
        with patch.object(self.executor, '_execute_button_press') as down, patch.object(self.executor, '_execute_button_release') as up:
            for p in (0.2, 0.5, 1.0, 0.0):
                self.press(p)
            down.assert_called_once_with('MOUSE_LEFT')
            up.assert_called_once_with('MOUSE_LEFT')
            self.press(1.0)
            self.side.trigger_mode = 'ANALOG'
            self.press(1.0)
            self.assertEqual(up.call_count, 2)

    def test_serialization_roundtrip_and_old_settings(self):
        self.side.trigger_axis_action = 'PAN_LR'
        self.side.trigger_sensitivity = 2.5
        self.side.trigger_deadzone = 0.2
        self.side.trigger_invert = True
        data = io.side_settings_to_dict(self.side)
        io.dict_to_side_settings(data, self.mode.right_side)
        self.assertEqual(data, io.side_settings_to_dict(self.mode.right_side))
        legacy = bpy.context.window_manager.test_gamepad_modes.add().left_side
        io.dict_to_side_settings({'trigger_action': 'MOUSE_WHEEL_UP'}, legacy)
        self.assertEqual(legacy.trigger_mode, 'BUTTON')
        self.assertEqual(legacy.trigger_action, 'MOUSE_WHEEL_UP')

    def test_elapsed_time_clamped_and_reset(self):
        with patch.object(actions, 'get_addon_preferences', return_value=None), patch.object(actions.time, 'monotonic', side_effect=[1.0, 1.02, 5.0]):
            self.executor.apply(bpy.context, {})
            self.assertAlmostEqual(self.executor._analog_frame_scale, 1.0)
            self.executor.apply(bpy.context, {})
            self.assertAlmostEqual(self.executor._analog_frame_scale, 1.2)
            self.executor.apply(bpy.context, {})
            self.assertAlmostEqual(self.executor._analog_frame_scale, 6.0)
        self.executor.reset()
        self.assertIsNone(self.executor._analog_last_time)

    def test_sdl_poll_preserves_trigger_pressure(self):
        class Reader(dict):
            location = (0, 0, 0)
        reader = Reader(keep=True)
        handler = object.__new__(sdl.SDL2_Controller_Handler)
        handler.controller, handler.joystick = object(), None
        for raw in (0, 8192, 16384, 32767):
            with patch.object(handler, '_discover_devices'), patch.object(sdl, 'get_reader', return_value=reader), patch.object(sdl2, 'SDL_PumpEvents'), patch.object(sdl2, 'SDL_GameControllerHasAxis', return_value=True), patch.object(sdl2, 'SDL_GameControllerGetAxis', return_value=raw), patch.object(sdl2, 'SDL_GameControllerHasButton', return_value=False):
                handler.poll(None)
            self.assertAlmostEqual(reader[self.executor.LEFT_TRIGGER_KEY], raw / 32767.0)
            self.assertAlmostEqual(reader[self.executor.RIGHT_TRIGGER_KEY], raw / 32767.0)


if __name__ == '__main__':
    print('Runtime:', bpy.app.version_string, sys.platform)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalogTriggerTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
