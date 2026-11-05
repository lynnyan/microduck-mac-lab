"""Regression tests for driving keys, focus loss and stop semantics."""
import unittest
from unittest.mock import patch

import glfw
import numpy as np

from driving_viewer import DrivingViewer
from keyboard_control import KeyboardCommands


class DrivingControlsTests(unittest.TestCase):
    def setUp(self):
        self.commands = KeyboardCommands()
        # Exercise the actual window event handler without requiring a display.
        self.viewer = DrivingViewer.__new__(DrivingViewer)
        self.viewer.key_callback = self.commands.on_key
        self.viewer.stop_callback = self.commands.stop

    def press(self, key, action=glfw.PRESS):
        self.viewer._on_key(None, key, 0, action, 0)

    def test_wasd_are_robot_commands(self):
        self.press(glfw.KEY_W)
        np.testing.assert_allclose(self.commands.snapshot(), [.3, 0, 0])
        self.press(glfw.KEY_A)
        np.testing.assert_allclose(self.commands.snapshot(), [.3, 0, .5])
        self.press(glfw.KEY_S)
        self.press(glfw.KEY_D)
        np.testing.assert_allclose(self.commands.snapshot(), [-.1, 0, -.5])

    def test_space_clears_combined_axes(self):
        for key in (glfw.KEY_W, glfw.KEY_Q, glfw.KEY_A):
            self.press(key)
        self.press(glfw.KEY_SPACE)
        np.testing.assert_array_equal(self.commands.snapshot(), np.zeros(3))

    def test_release_and_repeat_do_not_reinterpret_command(self):
        self.press(glfw.KEY_W)
        self.press(glfw.KEY_S, glfw.RELEASE)
        self.press(glfw.KEY_S, glfw.REPEAT)
        np.testing.assert_allclose(self.commands.snapshot(), [.3, 0, 0])

    def test_losing_focus_stops(self):
        self.press(glfw.KEY_W)
        self.viewer._on_focus(None, False)
        np.testing.assert_array_equal(self.commands.snapshot(), np.zeros(3))

    def test_escape_stops_and_closes(self):
        self.press(glfw.KEY_W)
        with patch('driving_viewer.glfw.set_window_should_close') as close:
            self.press(glfw.KEY_ESCAPE)
            close.assert_called_once_with(None, True)
        np.testing.assert_array_equal(self.commands.snapshot(), np.zeros(3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
