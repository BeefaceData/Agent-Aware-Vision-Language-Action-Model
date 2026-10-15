"""One validated skill per turn; no environment handles in planner input."""

from dataclasses import dataclass, field
from typing import Protocol

from harness.environment.base import Observation
from harness.predicates.base import PredicateResult
from harness.skills.schemas import SkillCommand, SkillResult
from harness.memory.snapshot import MemorySelection


@dataclass(frozen=True)
class PlannerContext:
    turn: int
    observation: Observation
    predicates: tuple[PredicateResult, ...]
    previous: SkillResult | None
    remaining_budget: dict[str, int] = field(default_factory=dict)
    memory: MemorySelection | None = None

    def value(self, name):
        return next((p.value for p in self.predicates if p.name == name), None)


@dataclass(frozen=True)
class PlannerSelection:
    command: SkillCommand
    phase: str
    retry_count: int = 0
    metadata: dict = field(default_factory=dict)


class Planner(Protocol):
    def select(self, context: PlannerContext) -> PlannerSelection: ...
