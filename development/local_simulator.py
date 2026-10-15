"""Process-local RoboTwin adapter for offline experiments, never supervisor input.

Owns one simulator. Use in a dedicated process: RoboTwin requires its checkout as
the working directory. No additional HTTP service or policy model is started.
"""

from pathlib import Path
import time

import numpy as np

from harness.trace import vector14


class LocalSimulator:
    def __init__(self, robotwin_root):
        from run_inference_chunked import build_task
        self.task, self.setup = build_task(str(Path(robotwin_root).resolve()), "hanging_mug")
        self.blank = np.zeros((240, 320, 3), dtype=np.uint8)
        self.initialized = False

    def __call__(self, host, port, path, payload):
        import torch
        from run_inference_chunked import observation_of
        start = time.perf_counter()
        with torch.enable_grad():
            if path == "/reset":
                self.task.setup_demo(**dict(self.setup, seed=int(payload["seed"]), is_test=True))
                self.initialized = True
                success = False
            elif path == "/step" and self.initialized:
                self.task.take_action(np.asarray(vector14(payload["action"]), dtype=np.float64))
                success = bool(self.task.eval_success) or bool(self.task.check_success())
            else:
                raise ValueError("local simulator requires reset followed by steps")
        after_action = time.perf_counter()
        observation = observation_of(self.task, self.blank, success)
        return observation, {
            "sim_http_ms": (time.perf_counter() - start) * 1000,
            "sim_action_ms": (after_action - start) * 1000,
            "sim_observation_ms": (time.perf_counter() - after_action) * 1000,
            "sim_encode_ms": 0.0,
        }

    def close(self):
        if self.initialized:
            self.task.close_env()
            self.initialized = False
