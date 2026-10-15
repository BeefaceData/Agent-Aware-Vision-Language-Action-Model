"""Scoped failure, success, staging and retry rules with operating conditions."""

from typing import Annotated, Literal
from pydantic import AliasChoices, Field
from harness.skills.schemas import MoveParameters, SkillType
from .base import MemoryRecord, Name, Text


class GlobalMemory(MemoryRecord):
    # 'all' is an explicit cross-task or cross-phase scope, not a wildcard path.
    task: Name = "all"
    category: Literal["success", "failure", "staging", "retry"]
    skill: SkillType | None = Field(default=None, validation_alias=AliasChoices("skill", "primitive"))
    statement: Text
    operating_conditions: Annotated[tuple[Text, ...], Field(max_length=6)] = ()
    recommended_staging: MoveParameters | None = None

    def planner_payload(self):
        return self.model_dump(mode="json", exclude={"evidence", "source", "planner_reviewed"}) | {"kind": "global"}
