"""Additive R1 contracts; legacy decisions and serialized records are unchanged."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from harness.schemas import Model, NonnegativeInt, StateSummary
from harness.predicates.base import PredicateResult
from harness.predicates.registry import PREDICATE_NAMES


class SkillType(str, Enum):
    MOVE_TO = "move_to"
    ROTATE_WRIST = "rotate_wrist"
    SET_GRIPPER = "set_gripper"
    RELEASE = "release"
    VLA_ACT = "vla_act"
    ABORT = "abort"


class VLAParameters(Model):
    max_chunks: Annotated[StrictInt, Field(ge=1, le=32)] = 1
    chunk_size: Annotated[StrictInt, Field(ge=1, le=24)] = 8
    max_policy_actions: Annotated[StrictInt, Field(ge=1, le=768)] = 768


class EmptyParameters(Model):
    pass


class AnalyticParameters(Model):
    max_physics_ticks: Annotated[StrictInt, Field(ge=1, le=3000)] = 1000


class MoveParameters(AnalyticParameters):
    target: Annotated[StrictStr, Field(min_length=1, max_length=80)] = "current_ee"
    offset_xyz: tuple[StrictFloat, StrictFloat, StrictFloat] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def bounded_offset(self):
        if sum(x*x for x in self.offset_xyz) > 0.20**2:
            raise ValueError("offset exceeds 20cm")
        return self


class RotateParameters(AnalyticParameters):
    delta_yaw_deg: Annotated[StrictFloat, Field(ge=-30, le=30)]


class GripperParameters(AnalyticParameters):
    value: Annotated[StrictFloat, Field(ge=0, le=1)]
    ramp_ticks: Annotated[StrictInt, Field(ge=50, le=500)] = 200


class ReleaseParameters(AnalyticParameters):
    ramp_ticks: Annotated[StrictInt, Field(ge=50, le=500)] = 200


PARAMETERS = {
    SkillType.VLA_ACT: VLAParameters, SkillType.ABORT: EmptyParameters,
    SkillType.MOVE_TO: MoveParameters, SkillType.ROTATE_WRIST: RotateParameters,
    SkillType.SET_GRIPPER: GripperParameters, SkillType.RELEASE: ReleaseParameters,
}


class SkillCommand(Model):
    action: SkillType
    arm: Literal["left", "right", "both", "auto"] = "auto"
    parameters: VLAParameters | MoveParameters | RotateParameters | GripperParameters | ReleaseParameters | EmptyParameters = Field(default_factory=EmptyParameters)
    stop: StrictStr | None = None
    reason: Annotated[StrictStr, Field(max_length=1000)] | None = None

    @model_validator(mode="before")
    @classmethod
    def typed_parameters(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            parameters = value.get("parameters", {})
            if isinstance(parameters, Model):
                parameters = parameters.model_dump()
            parameter_type = PARAMETERS.get(value.get("action"), EmptyParameters)
            value["parameters"] = parameter_type.model_validate(parameters)
        return value

    @model_validator(mode="after")
    def supported_contract(self):
        if self.stop is not None and self.stop not in PREDICATE_NAMES:
            raise ValueError("unknown_stop_predicate")
        if self.action == SkillType.VLA_ACT:
            if self.arm not in ("auto", "both"):
                raise ValueError("Neotix controls both arms; exclusive arm control is unsupported")
        elif self.action == SkillType.ABORT:
            if self.stop is not None:
                raise ValueError("ABORT does not accept a stop predicate")
            if self.action == SkillType.ABORT and self.arm != "auto":
                raise ValueError("ABORT has no arm binding")
        elif self.arm not in ("left", "right") or self.stop is not None:
            raise ValueError("analytic skills require one explicit arm and their own completion checks")
        return self


class EpisodeBudget(Model):
    max_analytic_ticks: Annotated[StrictInt, Field(ge=1, le=30000)] = 12000
    max_policy_actions: Annotated[StrictInt, Field(ge=1, le=1200)] = 1200
    max_vla_chunks: Annotated[StrictInt, Field(ge=1, le=1200)] = 150
    max_skill_calls: Annotated[StrictInt, Field(ge=1, le=1000)] = Field(
        default=100, validation_alias=AliasChoices("max_skill_calls", "max_primitive_calls"))


class SkillResult(Model):
    schema_version: Literal[1, 2] = 2
    call_id: NonnegativeInt
    skill: SkillType = Field(validation_alias=AliasChoices("skill", "primitive"))
    completed: StrictBool
    stop_reason: StrictStr
    executed_policy_actions: NonnegativeInt
    vla_chunks_requested: NonnegativeInt
    discarded_actions: NonnegativeInt
    executed_physics_ticks: NonnegativeInt = 0
    state_before: StateSummary
    state_after: StateSummary | None
    observation_before: NonnegativeInt
    observation_after: NonnegativeInt | None
    episode_terminal: StrictBool
    task_success: StrictBool | None
    objective_satisfied: StrictBool | None = None
    controlled_arms: Literal["left", "right", "both", "none"]
    information_source: Literal["public", "development_oracle"] = "public"
    predicate: PredicateResult | None = None
    position_error_m: StrictFloat | None = None
    orientation_error_rad: StrictFloat | None = None
    policy_history_reset: StrictBool = False
    execution_uncertain: StrictBool = False
    elapsed_seconds: Annotated[StrictFloat, Field(ge=0)]
    error_type: StrictStr | None = None
