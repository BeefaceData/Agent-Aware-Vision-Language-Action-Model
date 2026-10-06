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
        self._resume_identity = None

    def reset(self) -> None:
        self._resume_identity = None
        self._reset()

    def resume(self, observation: ObservationPacket) -> None:
        self._resume_identity = (observation.episode_id, observation.sequence)
        self._reset()

    def act(self, observation: ObservationPacket) -> Any:
        if self._resume_identity is not None:
            episode, sequence = self._resume_identity
            if observation.episode_id != episode or observation.sequence < sequence:
                raise ValueError('policy observation predates recovery resumption')
        return self._infer(observation)
