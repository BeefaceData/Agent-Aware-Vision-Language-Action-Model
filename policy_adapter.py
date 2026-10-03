"""Explicit opt-in adapter for policies whose reset invalidates action queues."""

from typing import Any, Callable

from episode_harness import ObservationPacket


class ResetOnResumePolicyAdapter:
    """Use only when clearing all policy state is the declared resume behavior.

    ``infer`` owns native preprocessing/inference/postprocessing. ``reset`` must
    invalidate every cached action. Resume itself performs no inference: the
    next act uses the harness's accepted post-override observation. Policies
    needing a different state transition should implement resume themselves.
    """

    def __init__(self, reset: Callable[[], None],
                 infer: Callable[[ObservationPacket], Any]):
        self._reset = reset
        self._infer = infer

    def reset(self) -> None:
        self._reset()

    def resume(self, observation: ObservationPacket) -> None:
        self._reset()

    def act(self, observation: ObservationPacket) -> Any:
        return self._infer(observation)
