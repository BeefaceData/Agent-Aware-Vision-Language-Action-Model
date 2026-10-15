"""Parameterized task sequences; never serialized simulator poses or joint paths."""

from typing import Annotated
from pydantic import Field, model_validator
from harness.schemas import Model
from harness.skills.schemas import SkillCommand
from .base import MemoryRecord, Metric, Name, Text


class RecipeStep(Model):
    phase: Name
    command: SkillCommand


class TaskMemory(MemoryRecord):
    objective: Metric
    description: Text
    trace: Annotated[tuple[RecipeStep, ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def success_claim(self):
        if self.evidence_level != "hypothesis":
            successes = [e for e in self.evidence if e.execution == "live"
                         and e.metric == self.objective and e.outcome == "success"]
            count = 2 if self.evidence_level == "repeated" else 1
            if len({e.episode for e in successes}) < count or len({e.seed for e in successes}) < count:
                raise ValueError("sequence_success_requires_matching_objective_evidence")
        return self

    def planner_payload(self):
        return self.model_dump(mode="json", exclude={"evidence", "source", "planner_reviewed"}) | {"kind": "task"}
