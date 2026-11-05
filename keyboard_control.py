"""Thread-safe velocity commands for the dedicated driving window."""
from threading import Lock

import numpy as np


class KeyboardCommands:
    def __init__(self):
        self._lock = Lock()
        self._command = np.zeros(3, dtype=np.float32)

    def stop(self):
        with self._lock:
            self._command[:] = 0

    def snapshot(self):
        with self._lock:
            return self._command.copy()

    def on_key(self, key):
        # GLFW arrow codes: up/down/left/right. Commands persist after key release.
        changes = {ord('W'): (0, .30), 265: (0, .30),
                   ord('S'): (0, -.10), 264: (0, -.10),
                   ord('A'): (2, .5), 263: (2, .5),
                   ord('D'): (2, -.5), 262: (2, -.5),
                   ord('Q'): (1, .08), ord('E'): (1, -.08)}
        if key != 32 and key not in changes:
            return
        with self._lock:
            if key == 32:
                self._command[:] = 0
            else:
                axis, value = changes[key]
                self._command[axis] = value
            command = self._command.copy()
        print(f"Keyboard vx={command[0]:+.2f} m/s vy={command[1]:+.2f} m/s yaw={command[2]:+.2f} rad/s", flush=True)
