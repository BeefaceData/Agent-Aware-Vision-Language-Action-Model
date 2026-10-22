"""Evidence labels describe claims; they do not certify physical validity."""

from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator
from harness.schemas import Model

Name = Annotated[StrictStr, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")]
Text = Annotated[StrictStr, Field(min_length=1, max_length=1000)]
Metric = Literal["mug_lifted", "handoff_confirmed", "task_success", "native_task_success", "strict_rack_success"]


class Evidence(Model):
    episode: Name
    seed: Annotated[StrictInt, Field(ge=0)]
    artifact: Annotated[StrictStr, Field(min_length=1, max_length=500)]
    metric: Metric
    outcome: Literal["success", "failure", "unknown"]
    source: Literal["public", "development_oracle"]
    execution: Literal["live", "synthetic"]


class MemoryRecord(Model):
    id: Name
    revision: Annotated[StrictInt, Field(ge=1)] = 1
    task: Name = "hanging_mug"
    phases: Annotated[tuple[Name, ...], Field(min_length=1, max_length=20)]
    source: Literal["public", "development_oracle"]
    evidence_level: Literal["hypothesis", "single_case", "repeated"] = "hypothesis"
    planner_reviewed: StrictBool = False
    evidence: Annotated[tuple[Evidence, ...], Field(max_length=100)] = ()

    @model_validator(mode="after")
    def provenance(self):
        if self.source == "public" and any(e.source != "public" for e in self.evidence):
            raise ValueError("private_evidence_cannot_be_declared_public")
        live = [e for e in self.evidence if e.execution == "live" and e.outcome != "unknown"]
        if self.evidence_level != "hypothesis" and not live:
            raise ValueError("observed_claim_requires_live_evidence")
        if self.evidence_level == "repeated" and (len({e.episode for e in live}) < 2 or len({e.seed for e in live}) < 2):
            raise ValueError("repeated_claim_requires_distinct_live_episodes_and_seeds")
        return self

    @property
    def eligible(self):
        return self.source == "public" and self.planner_reviewed
