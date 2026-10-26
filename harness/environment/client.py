"""Adapter over the unchanged HTTP transport (or injected LocalSimulator).

The episode owner calls reset exactly once before constructing an executor.
observe() returns the last confirmed response: the legacy server has no read-only
observation endpoint. External mutation is unsupported by this R1 adapter.
"""

import numpy as np

from harness.environment.base import CAMERAS, copy_images, vector14
from harness.perception.rgbd import camera_frames_from_payload
from .base import Observation


class EnvironmentClient:
    def __init__(self, transport=None, *, host="127.0.0.1", port=8110):
        if transport is None:
            from harness.environment.robotwin import call_sim
            transport = call_sim
        self._transport, self.host, self.port = transport, host, port
        self._observation = None
        self._native_success = False
        self._uncertain = False

    def _accept(self, raw, sequence):
        success = np.asarray(raw["success"])
        if success.shape != () or success.dtype.kind != "b":
            raise ValueError("invalid evaluator terminal signal")
        images = copy_images({k: raw[k] for k in CAMERAS})
        camera_frames = camera_frames_from_payload(raw, images)
        observation = Observation(sequence, images, vector14(raw["state"]), camera_frames or None)
        self._observation = observation
        self._native_success = bool(success)
        return observation.copy()

    def reset(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._uncertain = True
        raw, _ = self._transport(self.host, self.port, "/reset", {"seed": seed})
        observation = self._accept(raw, 0)
        self._uncertain = False
        return observation

    def observe(self):
        if self._observation is None or self._uncertain:
            raise RuntimeError("environment has no confirmed current observation")
        return self._observation.copy()

    @property
    def native_success(self):
        self.observe()  # Do not report stale status after an ambiguous request.
        return self._native_success

    def step(self, action):
        return self._step(action, allow_after_success=False)

    def step_after_success(self, action):
        """Explicitly permit one caller-bounded follow-through target."""
        return self._step(action, allow_after_success=True)

    def _step(self, action, *, allow_after_success):
        previous = self.observe()
        if self._native_success and not allow_after_success:
            raise RuntimeError("environment is terminal")
        action = vector14(action)
        self._uncertain = True  # A failed response may still have moved the robot.
        raw, _ = self._transport(self.host, self.port, "/step", {"action": list(action)})
        observation = self._accept(raw, previous.sequence + 1)
        self._uncertain = False
        return observation
