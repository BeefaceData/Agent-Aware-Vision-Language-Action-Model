"""Small, deterministic episode adapters for replay without model or simulator.

Each ``successful_replay`` call returns fresh adapters. The scripted policy
checks observations, and the environment checks executed actions, so a changed
episode cannot silently receive the fixture's successful evaluator outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from episode_harness import EpisodeConfig, StepResult


class ReplayPolicy:
    def __init__(self, turns: Sequence[tuple[Any, Any]]):
        self._turns = tuple(turns)
        self.observations: list[Any] = []

    def reset(self) -> None:
        self.observations.clear()

    def act(self, observation: Any) -> Any:
        index = len(self.observations)
        if index >= len(self._turns):
            raise AssertionError(f'Unexpected policy turn {index + 1}')
        expected_observation, action = self._turns[index]
        if observation != expected_observation:
            raise AssertionError(f'Unexpected observation at turn {index + 1}')
        self.observations.append(observation)
        return action


class ReplayEnvironment:
    def __init__(self, seed: int, initial_observation: Any,
                 turns: Sequence[tuple[Any, StepResult]]):
        self._seed = seed
        self._initial_observation = initial_observation
        self._turns = tuple(turns)
        self.actions: list[Any] = []

    def reset(self, seed: int) -> Any:
        if seed != self._seed:
            raise ValueError(f'Replay requires seed {self._seed}')
        self.actions.clear()
        return self._initial_observation

    def step(self, action: Any) -> StepResult:
        index = len(self.actions)
        if index >= len(self._turns):
            raise AssertionError(f'Unexpected environment step {index + 1}')
        expected_action, result = self._turns[index]
        self.actions.append(action)
        if action != expected_action:
            raise AssertionError(f'Unexpected action at step {index + 1}')
        return result


class ReplayRecorder:
    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.steps: list[tuple[int, Any, StepResult]] = []
        self.finalized = False

    def begin(self, observation: Any) -> None:
        self.observations = [observation]
        self.steps = []
        self.finalized = False

    def record_step(self, step: int, action: Any, result: StepResult) -> None:
        self.steps.append((step, action, result))
        self.observations.append(result.observation)

    def finish(self) -> dict[str, str]:
        self.finalized = True
        return {}  # An in-memory replay creates no file artifacts.


@dataclass(frozen=True)
class ReplayFixture:
    config: EpisodeConfig
    policy: ReplayPolicy
    environment: ReplayEnvironment
    recorder: ReplayRecorder


def successful_replay() -> ReplayFixture:
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
            (reach, StepResult(after_reach, 0.0, False, False, False)),
            (place, StepResult(terminal, 1.0, True, True, False)),
        )),
        recorder=ReplayRecorder(),
    )
