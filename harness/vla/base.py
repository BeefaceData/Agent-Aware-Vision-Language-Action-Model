from typing import Protocol

from harness.environment.base import Observation


class VLAClient(Protocol):
    def reset_history(self) -> None: ...

    def infer(self, observation: Observation, requested_actions: int): ...
