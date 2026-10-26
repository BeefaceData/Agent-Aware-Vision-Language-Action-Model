"""Bounded public observation history and durable policy execution traces.

No method accepts a raw simulator observation or oracle outcome. Context snapshots
copy images and decisions; neither caller mutation nor context mutation can change
the runner's observations or the retained trace.
"""

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from harness.environment.base import copy_images, vector14
from harness.jsonl import append_jsonl
from harness.legacy.config import TraceOptions
from harness.legacy.schemas import SupervisorDecision


@dataclass(frozen=True)
class StateSample:
    step: int
    command_state: tuple[float, ...]


@dataclass(frozen=True)
class ActionSample:
    step: int
    policy_action: tuple[float, ...]
    executed_action: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Keyframe:
    step: int
    images: dict[str, np.ndarray]


@dataclass(frozen=True)
class TraceContext:
    episode_id: str
    states: tuple[StateSample, ...]
    actions: tuple[ActionSample, ...]
    keyframes: tuple[Keyframe, ...]
    decisions: tuple[SupervisorDecision, ...]


class TraceManager:
    def __init__(self, options: TraceOptions):
        self.options = options
        self.reset("")

    def reset(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._states = deque(maxlen=self.options.state_history)
        self._actions = deque(maxlen=self.options.action_history)
        self._keyframes = deque(maxlen=self.options.keyframes)
        self._decisions = []

    def observe_state(self, step: int, state) -> tuple[float, ...]:
        if step < 0 or (self._states and step < self._states[-1].step):
            raise ValueError("observation steps must be nonnegative and ordered")
        value = vector14(state)
        if self._states and self._states[-1].step == step:
            self._states.pop()
        self._states.append(StateSample(step, value))
        return value

    def propose(self, step, images, state, action, decision) -> Keyframe | None:
        if self._actions and step <= self._actions[-1].step:
            raise ValueError("action steps must increase")
        action = vector14(action)
        # Validate all images, including unsampled steps, without retaining them.
        copied = copy_images(images)
        self.observe_state(step, state)
        self._actions.append(ActionSample(step, action))
        self._decisions.append(decision.model_copy(deep=True))
        if step % self.options.keyframe_interval == 0:
            frame = Keyframe(step, copied)
            self._keyframes.append(frame)
            return Keyframe(step, copy_images(copied))
        return None

    def executed(self, step, action, post_state) -> None:
        if not self._actions or self._actions[-1].step != step or self._actions[-1].executed_action is not None:
            raise ValueError("execution must match the latest unexecuted proposal")
        action = vector14(action)
        self.observe_state(step + 1, post_state)
        self._actions[-1] = replace(self._actions[-1], executed_action=action)

    def context(self) -> TraceContext:
        return TraceContext(
            self.episode_id, tuple(self._states), tuple(self._actions),
            tuple(Keyframe(f.step, copy_images(f.images)) for f in self._keyframes),
            tuple(d.model_copy(deep=True) for d in self._decisions),
        )


class TraceWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "trace.jsonl"

    def append(self, event: dict) -> None:
        append_jsonl(self.path, event)

    def save_keyframe(self, frame: Keyframe) -> str:
        relative = f"keyframes/step_{frame.step:06d}.npz"
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **frame.images)
        return relative
