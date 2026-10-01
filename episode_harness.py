"""Baseline episode orchestration with injectable policy, environment and recorder.

The caller owns the adapters and releases any resources they hold. The recorder
finalizes its artifacts on success; the caller handles cleanup on failure.
Environment step results contain evaluator outcomes, which are never sent to the
policy through this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class EpisodeConfig:
    seed: int
    max_steps: int


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    success: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    steps: int
    stop_reason: str
    sum_rewards: float
    rollout_seconds: float
    artifacts: Mapping[str, str]


class PolicyAdapter(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: Any) -> Any: ...


class EnvironmentAdapter(Protocol):
    def reset(self, seed: int) -> Any: ...
    def step(self, action: Any) -> StepResult: ...


class EpisodeRecorder(Protocol):
    def begin(self, observation: Any) -> None: ...
    def record_step(self, step: int, action: Any, result: StepResult) -> None: ...
    def finish(self) -> Mapping[str, str]: ...


def run_episode(
    config: EpisodeConfig,
    policy: PolicyAdapter,
    environment: EnvironmentAdapter,
    recorder: EpisodeRecorder,
    on_step: Callable[[int, StepResult], None] | None = None,
) -> EpisodeOutcome:
    """Run one attempt and return its outcome after artifacts are finalized.

    ``policy`` provides reset/act(observation); ``environment`` provides
    reset(seed)/step(action); ``recorder`` provides begin(observation),
    record_step(step, action, result), and finish() -> artifact references.
    The caller releases adapter resources on any exception.
    """
    if config.max_steps <= 0:
        raise ValueError('max_steps must be positive')

    policy.reset()
    observation = environment.reset(config.seed)
    recorder.begin(observation)
    rollout_start = perf_counter()
    reward_sum = 0.0
    stop_reason = 'step_limit'
    success = False
    steps = 0

    for step in range(1, config.max_steps + 1):
        action = policy.act(observation)
        result = environment.step(action)
        recorder.record_step(step, action, result)
        reward_sum += result.reward
        steps = step
        success = result.success
        if on_step is not None:
            on_step(step, result)
        if success or result.terminated or result.truncated:
            stop_reason = ('success' if success else
                           'terminated' if result.terminated else 'truncated')
            break
        observation = result.observation

    rollout_seconds = perf_counter() - rollout_start
    artifacts = recorder.finish()
    return EpisodeOutcome(success, steps, stop_reason, reward_sum,
                          rollout_seconds, artifacts)
