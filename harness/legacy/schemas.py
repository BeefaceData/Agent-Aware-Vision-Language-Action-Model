"""Decision and assessment records of the legacy pass-through supervisor.

Decision parsing is structural validation, not permission to execute a skill.
The planner-driven runtime uses ``harness.skills.schemas`` instead.
"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictStr, model_validator

from harness.schemas import Confidence, Identifier, Model, NonnegativeInt, Parameter, StateSummary, Vector14


class TaskPhase(str, Enum):
    APPROACH_MUG = "approach_mug"
    GRASP_MUG = "grasp_mug"
    LIFT_MUG = "lift_mug"
    TRANSFER = "transfer"
    HANDOFF = "handoff"
    APPROACH_RACK = "approach_rack"
    HANG_MUG = "hang_mug"
    COMPLETE = "complete"
    UNKNOWN = "unknown"


class FailureType(str, Enum):
    NONE = "none"
    MISSED_GRASP = "missed_grasp"
    UNSTABLE_GRASP = "unstable_grasp"
    DROPPED_MUG = "dropped_mug"
    FAILED_HANDOFF = "failed_handoff"
    BAD_RACK_APPROACH = "bad_rack_approach"
    STALLED = "stalled"
    LOST_PROGRESS = "lost_progress"
    UNRECOVERABLE = "unrecoverable"
    UNKNOWN = "unknown"


class DecisionType(str, Enum):
    PASS = "pass"
    EXECUTE_SKILL = "execute_skill"
    ABORT = "abort"


class SupervisorDecision(Model):
    decision: DecisionType
    phase: TaskPhase
    failure_type: FailureType
    confidence: Confidence
    skill_id: Identifier | None = None
    parameters: dict[StrictStr, Parameter] = Field(default_factory=dict)
    evidence: tuple[StrictStr, ...] = ()
    reason: Annotated[StrictStr, Field(min_length=1)]

    @model_validator(mode="after")
    def check_skill_fields(self):
        if self.decision == DecisionType.EXECUTE_SKILL:
            if self.skill_id is None:
                raise ValueError("execute_skill requires a skill_id")
        elif self.skill_id is not None or self.parameters:
            raise ValueError("PASS/ABORT cannot carry skill IDs or parameters")
        return self


class SkillResult(Model):
    skill_id: Identifier
    executed_steps: NonnegativeInt
    success_hint: StrictBool | None = None
    termination_reason: Annotated[StrictStr, Field(min_length=1)]
    final_state: Vector14 | None = None
    metadata: dict[StrictStr, Parameter] = Field(default_factory=dict)


class AssessmentRecord(Model):
    schema_version: Literal[1] = 1
    run_id: Identifier
    episode_id: Identifier
    event_id: Identifier
    event_type: Literal["decision", "action_result"]
    step: NonnegativeInt
    chunk_id: NonnegativeInt
    chunk_pos: NonnegativeInt
    timestamp: StrictFloat
    phase: TaskPhase
    failure_type: FailureType
    supervisor_confidence: Confidence
    evidence: tuple[StrictStr, ...] = ()
    policy_action: Vector14
    decision: DecisionType
    skill_id: Identifier | None = None
    skill_parameters: dict[StrictStr, Parameter] = Field(default_factory=dict)
    pre_state_summary: StateSummary
    post_state_summary: StateSummary | None = None
    executed_action: Vector14 | None = None
    outcome: Literal["helped", "hurt", "neutral", "unknown"] | None = None
    notes: StrictStr | None = None

    @model_validator(mode="after")
    def check_event(self):
        SupervisorDecision(
            decision=self.decision, phase=self.phase, failure_type=self.failure_type,
            confidence=self.supervisor_confidence, skill_id=self.skill_id,
            parameters=self.skill_parameters, evidence=self.evidence,
            reason="assessment validation",
        )
        if self.event_type == "decision":
            if any(x is not None for x in (
                self.post_state_summary, self.executed_action, self.outcome,
            )):
                raise ValueError("decision cannot claim execution or an outcome")
        elif self.post_state_summary is None or self.executed_action is None:
            raise ValueError("action_result requires post-state and executed action")
        return self
