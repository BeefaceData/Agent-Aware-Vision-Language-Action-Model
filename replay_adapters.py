"""Small, deterministic episode adapters for replay without model or simulator.

Each ``successful_replay`` call returns fresh adapters. The scripted policy
checks observations, and the environment checks executed actions, so a changed
episode cannot silently receive the fixture's successful evaluator outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

from episode_harness import (EpisodeConfig, IngestionOutcome, ObservationPacket,
                             StepFailure, StepResult, ViewCapture,
                             frame_references_for_observation)


_CAPTURE_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ReplayStep:
    """Scripted environment return before episode packet metadata is attached."""

    observation: Any
    reward: float
    success: bool
    terminated: bool
    truncated: bool
    camera_captures: Mapping[str, ViewCapture] | None = None


class ReplayPolicy:
    def __init__(self, turns: Sequence[tuple[Any, Any]]):
        self._turns = tuple(turns)
        self.observations: list[Any] = []

    def reset(self) -> None:
        self.observations.clear()

    def act(self, observation: ObservationPacket) -> Any:
        index = len(self.observations)
        if index >= len(self._turns):
            raise AssertionError(f'Unexpected policy turn {index + 1}')
        expected_observation, action = self._turns[index]
        if observation.observation != expected_observation:
            raise AssertionError(f'Unexpected observation at turn {index + 1}')
        self.observations.append(observation)
        return action


class ReplayEnvironment:
    def __init__(self, seed: int, initial_observation: Any,
                 turns: Sequence[tuple[Any, ReplayStep]],
                 clock: Callable[[], float] = monotonic,
                 initial_camera_captures: Mapping[str, ViewCapture] | None = None,
                 max_camera_skew_seconds: float | None = None):
        self._seed = seed
        self._initial_observation = initial_observation
        self._turns = tuple(turns)
        self.actions: list[Any] = []
        self._episode_id = ''
        self._clock = clock
        self._initial_camera_captures = initial_camera_captures
        self._max_camera_skew_seconds = max_camera_skew_seconds

    def _packet(self, episode_id: str, sequence: int, captured_at: datetime,
                observation: Any, captured_monotonic: float,
                camera_captures: Mapping[str, ViewCapture] | None) -> ObservationPacket:
        references = (frame_references_for_observation(
            observation, sequence, captured_at, captured_monotonic,
            camera_captures, self._max_camera_skew_seconds)
            if isinstance(observation, Mapping) else ())
        return ObservationPacket(episode_id, sequence, captured_at,
                                 observation, captured_monotonic, references)

    def reset(self, seed: int, episode_id: str) -> ObservationPacket:
        if seed != self._seed:
            raise ValueError(f'Replay requires seed {self._seed}')
        self.actions.clear()
        self._episode_id = episode_id
        return self._packet(episode_id, 0, _CAPTURE_START,
                            self._initial_observation, self._clock(),
                            self._initial_camera_captures)

    def step(self, action: Any) -> StepResult:
        index = len(self.actions)
        if index >= len(self._turns):
            raise AssertionError(f'Unexpected environment step {index + 1}')
        expected_action, result = self._turns[index]
        self.actions.append(action)
        if action != expected_action:
            raise AssertionError(f'Unexpected action at step {index + 1}')
        return StepResult(
            observation=self._packet(
                self._episode_id, index + 1,
                _CAPTURE_START + timedelta(seconds=index + 1),
                result.observation, self._clock(), result.camera_captures),
            reward=result.reward,
            success=result.success,
            terminated=result.terminated,
            truncated=result.truncated,
        )


class ReplayRecorder:
    def __init__(self) -> None:
        self.observations: list[ObservationPacket] = []
        self.steps: list[tuple[int, ObservationPacket, Any, StepResult,
                               IngestionOutcome]] = []
        self.failures: list[tuple[int, ObservationPacket, Any, StepFailure]] = []
        self.finalized = False

    def begin(self, observation: ObservationPacket) -> None:
        self.observations = [observation]
        self.steps = []
        self.failures = []
        self.finalized = False

    def record_step(self, step: int, source: ObservationPacket,
                    action: Any, result: StepResult,
                    ingestion: IngestionOutcome) -> None:
        self.steps.append((step, source, action, result, ingestion))
        if ingestion.accepted:
            self.observations.append(result.observation)

    def record_failure(self, step: int, source: ObservationPacket,
                       action: Any, failure: StepFailure) -> None:
        self.failures.append((step, source, action, failure))

    def finish(self) -> dict[str, str]:
        self.finalized = True
        return {}  # An in-memory replay creates no file artifacts.


@dataclass(frozen=True)
class ReplayFixture:
    config: EpisodeConfig
    policy: ReplayPolicy
    environment: ReplayEnvironment
    recorder: ReplayRecorder


def successful_replay(clock: Callable[[], float] = monotonic) -> ReplayFixture:
    """Return a fresh two-action synthetic episode ending in task success."""
    initial = 'item visible'
    after_reach = 'item held'
    terminal = 'item placed'
    reach = ('reach', 0.25)
    place = ('place', 0.75)
    return ReplayFixture(
        config=EpisodeConfig(seed=17, max_steps=5),
        policy=ReplayPolicy(((initial, reach), (after_reach, place))),
        environment=ReplayEnvironment(17, initial, (
            (reach, ReplayStep(after_reach, 0.0, False, False, False)),
            (place, ReplayStep(terminal, 1.0, True, True, False)),
        ), clock),
        recorder=ReplayRecorder(),
    )
