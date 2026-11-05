"""Single-threaded GLFW driving window with exclusive robot keyboard bindings.

Run with normal Python on macOS: unlike launch_passive, GLFW is created on the
Python main thread. There is no MuJoCo simulator UI to intercept W/A/S/D.
"""
from __future__ import annotations

import glfw
import mujoco


class DrivingViewer:
    def __init__(self, model, data, key_callback, command_provider, stop_callback):
        self.model, self.data = model, data
        self.key_callback = key_callback
        self.command_provider, self.stop_callback = command_provider, stop_callback
        self.window = None
        self.context = None
        self.scene = None
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        try:
            glfw.window_hint(glfw.SAMPLES, 4)
            self.window = glfw.create_window(1200, 800, "MicroDuck Drive | WASD controls robot", None, None)
            if not self.window:
                raise RuntimeError("Could not create driving window; run with .venv/bin/python, not mjpython")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            self.pert = mujoco.MjvPerturb()
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._last_cursor = None
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_window_focus_callback(self.window, self._on_focus)
            glfw.set_scroll_callback(self.window, self._on_scroll)
            glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        except Exception:
            self.close()
            raise

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            self.stop_callback()
            glfw.set_window_should_close(window, True)
        else:
            # Directly deliver physical GLFW codes; never forward to simulator shortcuts.
            self.key_callback(key)

    def _on_focus(self, window, focused):
        if not focused:
            self.stop_callback()

    def _on_scroll(self, window, dx, dy):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05*dy, self.scene, self.cam)

    def _on_cursor(self, window, x, y):
        previous, self._last_cursor = self._last_cursor, (x, y)
        if previous is None:
            return
        dx, dy = x-previous[0], y-previous[1]
        height = max(glfw.get_window_size(window)[1], 1)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)

    def is_running(self):
        return self.window is not None and not glfw.window_should_close(self.window)

    def sync(self):
        glfw.poll_events()
        if not self.is_running():
            return
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width == 0 or height == 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        command = self.command_provider()
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                           "MICRODUCK DRIVE\nW/S forward/back | A/D turn | Q/E sidestep\nSPACE stop | ESC exit | mouse drag/scroll: camera\nCommands persist; leaving window stops commands",
                           f"\nvx {command[0]:+.2f} m/s | vy {command[1]:+.2f} m/s | yaw {command[2]:+.2f} rad/s", self.context)
        glfw.swap_buffers(self.window)

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
            if self.context is not None:
                self.context.free()
                self.context = None
            if self.scene is not None:
                # MjvScene is released by the Python binding's destructor.
                self.scene = None
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
