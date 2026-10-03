"""Bounded, detached deployable history for one episode's supervisor."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from episode_harness import ObservationPacket, supervisor_observation


@dataclass(frozen=True)
class WindowSettings:
    """Retain at most N paired camera/state packets and M executed actions.

    Thus camera payload count is at most 2*N and state sample count at most N.
    These are sample-count limits, not image-byte or model-token limits.
    """

    max_observations: int = 8
    max_actions: int = 8

    def __post_init__(self):
        if type(self.max_observations) is not int or self.max_observations < 1:
            raise ValueError('max_observations must be a positive integer')
        if type(self.max_actions) is not int or self.max_actions < 0:
            raise ValueError('max_actions must be a nonnegative integer')


@dataclass(frozen=True)
class SequenceInterval:
    """Inclusive sequence interval with no supplied observation evidence."""

    first: int
    last: int


@dataclass(frozen=True)
class WindowAction:
    """Acknowledged action at a source observation; no evaluator outcomes."""

    source_sequence: int
    proposal_id: str
    proposed_action: Any
    executed_action: Any
    executed_at: float


@dataclass(frozen=True)
class ObservationWindow:
    episode_id: str
    task: str
    settings: WindowSettings
    observations: tuple[ObservationPacket, ...]
    actions: tuple[WindowAction, ...]
    omitted_prefix: SequenceInterval | None
    missing_intervals: tuple[SequenceInterval, ...]
    omitted_action_count: int
    ordering: str = 'observation_sequence_ascending'


class ObservationWindowBuilder:
    """Append ordered packets; gaps remain visible and are never interpolated.

    Duplicate, reversed, cross-episode and time-reversed inputs are rejected
    without changing the history. Missing task fields retain the declared task;
    conflicting instructions are rejected. Each snapshot is detached from both
    the producer and prior snapshots. Actions must precede the current decision.
    """

    def __init__(self, episode_id: str, task: str,
                 settings: WindowSettings = WindowSettings()):
        if type(episode_id) is not str or not episode_id:
            raise ValueError('nonempty episode identity required')
        if type(task) is not str or not task:
            raise ValueError('nonempty task instruction required')
        if not isinstance(settings, WindowSettings):
            raise ValueError('WindowSettings required')
        self.episode_id, self.task, self.settings = episode_id, task, settings
        self._observations = deque(maxlen=settings.max_observations)
        self._actions = deque(maxlen=settings.max_actions)
        self._action_count = 0
        self._last_action_sequence = -1

    def append(self, packet: ObservationPacket) -> None:
        if (not isinstance(packet, ObservationPacket) or
            packet.episode_id != self.episode_id or
            type(packet.sequence) is not int or packet.sequence < 0 or
            type(packet.captured_at) is not datetime or
            packet.captured_at.utcoffset() is None or
            type(packet.captured_monotonic) not in (int, float) or
            not isfinite(packet.captured_monotonic)):
            raise ValueError('valid episode packet identity and timestamps required')
        if self._observations:
            previous = self._observations[-1]
            if (packet.sequence <= previous.sequence or
                packet.captured_at < previous.captured_at or
                packet.captured_monotonic < previous.captured_monotonic):
                raise ValueError('observations must arrive in chronological sequence order')
        clean = supervisor_observation(packet)
        if clean.observation.get('task', self.task) != self.task:
            raise ValueError('task instruction changed within episode')
        clean.observation['task'] = self.task
        self._observations.append(clean)

    def record_action(self, action: WindowAction) -> None:
        if (not isinstance(action, WindowAction) or not self._observations or
            type(action.source_sequence) is not int or
            action.source_sequence != self._observations[-1].sequence or
            action.source_sequence <= self._last_action_sequence or
            type(action.proposal_id) is not str or not action.proposal_id or
            type(action.executed_at) not in (int, float) or
            not isfinite(action.executed_at) or
            action.executed_at < self._observations[-1].captured_monotonic):
            raise ValueError('ordered acknowledged action for the latest observation required')
        self._actions.append(deepcopy(action))
        self._action_count += 1
        self._last_action_sequence = action.source_sequence

    def snapshot(self) -> ObservationWindow:
        if not self._observations:
            raise ValueError('at least one observation required')
        observations = tuple(self._observations)
        first = observations[0].sequence
        missing = tuple(SequenceInterval(left.sequence + 1, right.sequence - 1)
                        for left, right in zip(observations, observations[1:])
                        if right.sequence > left.sequence + 1)
        return deepcopy(ObservationWindow(
            self.episode_id, self.task, self.settings, observations,
            tuple(self._actions), SequenceInterval(0, first - 1) if first else None,
            missing, self._action_count - len(self._actions)))
