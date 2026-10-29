"""Process-local RoboTwin adapter for offline experiments, never supervisor input.

Owns one simulator. Use in a dedicated process: RoboTwin requires its checkout as
the working directory. No additional HTTP service or policy model is started.
"""

from pathlib import Path
import time

import numpy as np

from harness.environment.base import vector14


class LocalSimulator:
    def __init__(self, robotwin_root, *, rgbd_enabled=False, diagnostics=None, perception_cameras=()):
        """``perception_cameras``: ``FixedCamera`` specs added to the scene after every reset."""
        if type(rgbd_enabled) is not bool:
            raise ValueError("rgbd_setting_must_be_boolean")
        from harness.environment.perception_cameras import FixedCamera
        if (not isinstance(perception_cameras, (tuple, list))
                or any(not isinstance(spec, FixedCamera) for spec in perception_cameras)
                or len({spec.name for spec in perception_cameras}) != len(perception_cameras)):
            raise ValueError("invalid_perception_cameras")
        self.perception_camera_specs = tuple(perception_cameras)
        self.perception_cameras = {}
        from harness.environment.robotwin import build_task
        self.task, self.setup = build_task(str(Path(robotwin_root).resolve()), "hanging_mug")
        if rgbd_enabled:
            # Enable depth only for this local skill-run process. The unchanged
            # baseline runner retains demo_clean's original depth=false setting.
            data_type = self.setup.get("data_type")
            if not isinstance(data_type, dict):
                raise ValueError("simulator_data_type_unavailable")
            data_type["depth"] = True
        self.blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self.initialized = False
        self.diagnostics = diagnostics
        self._step_count = 0

    def _diagnostic(self, event_type, **fields):
        if self.diagnostics is None:
            return
        try:
            self.diagnostics({"event_type": event_type, **fields})
        except Exception:
            # Diagnostics must never change whether a simulator action runs.
            pass

    def __call__(self, host, port, path, payload):
        import torch
        start = time.perf_counter()
        step_id = None
        phase = "request"
        self._diagnostic("simulator_request_started", path=path)
        with torch.enable_grad():
            if path == "/reset":
                phase = "reset"
                self._diagnostic("simulator_phase_started", phase=phase)
                phase_start = time.perf_counter()
                try:
                    self.task.setup_demo(**dict(self.setup, seed=int(payload["seed"]), is_test=True))
                except Exception as exc:
                    self._diagnostic("simulator_phase_failed", phase=phase,
                                     error_type=type(exc).__name__)
                    raise
                # Added after setup_demo has placed the objects, so no seeded
                # random numbers are drawn; setup_demo builds a new scene each reset.
                from harness.environment.perception_cameras import add_fixed_camera
                self.perception_cameras = {spec.name: add_fixed_camera(self.task.scene, spec)
                                           for spec in self.perception_camera_specs}
                # This flag is scoped to the local agentic runner. The frozen
                # baseline runner and RoboTwin's ordinary evaluation path keep
                # their native terminal behavior.
                self.task.harness_continue_after_success = True
                self.task.harness_native_success_latched = False
                self.initialized = True
                success = False
                self._diagnostic("simulator_phase_completed", phase=phase,
                                 elapsed_ms=(time.perf_counter() - phase_start) * 1000)
            elif path == "/step" and self.initialized:
                step_id = self._step_count + 1
                success = self.act(payload["action"])
            else:
                raise ValueError("local simulator requires reset followed by steps")
        after_action = time.perf_counter()
        phase = "observation"
        self._diagnostic("simulator_phase_started", phase=phase, step_id=step_id)
        phase_start = time.perf_counter()
        try:
            observation = self.observation(success)
        except Exception as exc:
            self._diagnostic("simulator_phase_failed", phase=phase, step_id=step_id,
                             error_type=type(exc).__name__)
            raise
        observation_finished = time.perf_counter()
        self._diagnostic("simulator_phase_completed", phase=phase, step_id=step_id,
                         elapsed_ms=(observation_finished - phase_start) * 1000)
        self._diagnostic("simulator_request_completed", path=path, step_id=step_id,
                         elapsed_ms=(observation_finished - start) * 1000)
        return observation, {
            "sim_http_ms": (time.perf_counter() - start) * 1000,
            "sim_action_ms": (after_action - start) * 1000,
            "sim_observation_ms": (observation_finished - after_action) * 1000,
            "sim_encode_ms": 0.0,
        }

    def act(self, action):
        """Apply one action without rendering and return the native success flag.

        Rendering every camera with depth costs most of a step (about 80% on a
        Modal A10, more on a laptop GPU), so a caller that needs only some frames
        acts here and calls ``observation`` for those.
        """
        import torch
        if not self.initialized:
            raise ValueError("local simulator requires reset followed by steps")
        self._step_count += 1
        step_id = self._step_count
        phase = "task_action"
        self._diagnostic("simulator_phase_started", phase=phase, step_id=step_id)
        phase_start = time.perf_counter()
        try:
            with torch.enable_grad():
                self.task.take_action(np.asarray(vector14(action), dtype=np.float64))
                success = (bool(self.task.eval_success)
                           or bool(getattr(self.task, "harness_native_success_latched", False))
                           or bool(self.task.check_success()))
        except Exception as exc:
            self._diagnostic("simulator_phase_failed", phase=phase, step_id=step_id,
                             error_type=type(exc).__name__)
            raise
        self._diagnostic("simulator_phase_completed", phase=phase, step_id=step_id,
                         elapsed_ms=(time.perf_counter() - phase_start) * 1000)
        return success

    def observation(self, success):
        """Render the current observation payload, including any perception cameras."""
        from harness.environment.robotwin import observation_of
        return observation_of(self.task, self.blank, success, self.perception_cameras)

    def close(self):
        if self.initialized:
            self.task.close_env()
            self.perception_cameras = {}
            self.initialized = False
