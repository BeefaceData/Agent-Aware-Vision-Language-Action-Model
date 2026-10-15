"""Reuse the deployed Neotix client without changing weights or serving code."""

from harness.environment.base import Observation


class ModalVLAClient:
    def __init__(self, worker=None, *, app_name="robotwin-policy", class_name="PolicyWorker"):
        if worker is None:
            from run_inference_chunked import ModalPolicyClient
            worker = ModalPolicyClient(app_name, class_name)
        self.worker = worker

    def reset_history(self):
        info, _ = self.worker.reset()
        if not isinstance(info, dict) or info.get("ok") is not True:
            raise RuntimeError("incompatible Modal reset response")

    def infer(self, observation: Observation, requested_actions: int):
        observation = observation.copy()
        actions, _ = self.worker.act_chunk(observation.images, observation.command_state,
                                          requested_actions)
        return actions
