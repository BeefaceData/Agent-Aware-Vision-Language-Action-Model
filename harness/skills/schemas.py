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
    # Freeze one arm at its most recent commanded target while the other arm
    # performs a bounded VLA phase. Action layout is left[0:7], right[7:14].
    lock_arm: Literal["left", "right"] | None = None
    # Optional evidence-based early stop for a bounded VLA attempt. This is
    # configured by the runtime, not an unbounded model-selected action.
    failure_stop: Literal["grasp_attempt_failed"] | None = None
    failure_arm: Literal["left", "right"] = "left"


class EmptyParameters(Model):
    pass


class AnalyticParameters(Model):
    max_physics_ticks: Annotated[StrictInt, Field(ge=1, le=3000)] = 1000


class GroundingPixel(Model):
    """Planner-selected RGB pixel normalized to its full source camera view."""

    camera: Literal["head_camera", "left_camera", "right_camera"]
    uv: tuple[Annotated[StrictFloat, Field(ge=0.0, le=1.0)],
              Annotated[StrictFloat, Field(ge=0.0, le=1.0)]]


class MoveParameters(AnalyticParameters):
    target: Annotated[StrictStr, Field(min_length=1, max_length=80)] = "current_ee"
    offset_xyz: tuple[StrictFloat, StrictFloat, StrictFloat] = (0.0, 0.0, 0.0)
    grounding: GroundingPixel | None = None

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
    authorization: Literal["observed_support", "rack_seat_alignment"] = "observed_support"


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
        if (self.action == SkillType.MOVE_TO and self.parameters.grounding is not None
                and self.parameters.target not in ("mug_pregrasp", "rack_approach")):
            raise ValueError("grounding_pixel_requires_perception_target")
        return self


class EpisodeBudget(Model):
    max_analytic_ticks: Annotated[StrictInt, Field(ge=1, le=30000)] = 12000
    max_policy_actions: Annotated[StrictInt, Field(ge=1, le=1200)] = 1200
    max_vla_chunks: Annotated[StrictInt, Field(ge=1, le=1200)] = 150
    max_skill_calls: Annotated[StrictInt, Field(ge=1, le=1000)] = Field(
        default=100, validation_alias=AliasChoices("max_skill_calls", "max_primitive_calls"))


class TrackingError(Model):
    """Confirmed joint tracking residual that caused an analytic stop."""

    arm: Literal["left", "right"]
    comparison: Literal["controlled_arm", "inactive_arm"]
    joint_index: Annotated[StrictInt, Field(ge=0, le=5)]
    measured_position_rad: StrictFloat
    commanded_position_rad: StrictFloat
    residual_rad: Annotated[StrictFloat, Field(gt=0)]
    threshold_rad: Annotated[StrictFloat, Field(gt=0)]


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
    # Public contact feedback; None when the active environment cannot measure it.
    arm_overlap_detected: StrictBool | None = None
    task_success: StrictBool | None
    native_success_seen: StrictBool = False
    post_success_actions: NonnegativeInt = 0
    objective_satisfied: StrictBool | None = None
    controlled_arms: Literal["left", "right", "both", "none"]
    information_source: Literal["public", "development_oracle"] = "public"
    predicate: PredicateResult | None = None
    tracking_error: TrackingError | None = None
    position_error_m: StrictFloat | None = None
    orientation_error_rad: StrictFloat | None = None
    policy_history_reset: StrictBool = False
    execution_uncertain: StrictBool = False
    elapsed_seconds: Annotated[StrictFloat, Field(ge=0)]
    error_type: StrictStr | None = None


def is_retryable_pre_motion_planning_failure(result: SkillResult | None) -> bool:
    """Whether a failed MOVE_TO is known to have rejected before any motion.

    This narrow case is safe to hand back to a planner for a different bounded
    action. It excludes lost responses, partial execution, changed command state,
    and failures from every other skill.
    """
    return bool(
        result is not None
        and result.skill == SkillType.MOVE_TO
        and not result.completed
        and result.stop_reason == "planning_failure"
        and not result.episode_terminal
        and not result.execution_uncertain
        and result.executed_physics_ticks == 0
        and result.executed_policy_actions == 0
        and result.vla_chunks_requested == 0
        and result.state_after is not None
        and result.state_before.command_state == result.state_after.command_state
    )


def is_recoverable_executed_motion_failure(result: SkillResult | None) -> bool:
    """Whether a known analytic MOVE_TO miss has enough state for replanning."""
    return bool(
        result is not None
        and result.skill == SkillType.MOVE_TO
        and not result.completed
        and result.stop_reason == "target_not_reached"
        and result.information_source == "public"
        and not result.episode_terminal
        and not result.execution_uncertain
        and result.executed_physics_ticks > 0
        and result.executed_policy_actions == 0
        and result.vla_chunks_requested == 0
        and result.state_after is not None
        and result.observation_after is not None
        and result.observation_after > result.observation_before
        and result.position_error_m is not None
        and result.orientation_error_rad is not None
    )


def is_recoverable_move_feedback(result: SkillResult | None) -> bool:
    """Known MOVE_TO failures with sufficient state for a fresh planner turn."""
    return (is_retryable_pre_motion_planning_failure(result)
            or is_recoverable_executed_motion_failure(result))


def is_recoverable_arm_overlap(result: SkillResult | None) -> bool:
    """A confirmed contact stop with fresh state may be handed to the planner."""
    return bool(
        result is not None and result.skill == SkillType.VLA_ACT
        and not result.completed and result.stop_reason == "arm_overlap_detected"
        and result.information_source == "public" and not result.episode_terminal
        and not result.execution_uncertain and result.executed_policy_actions > 0
        and result.state_after is not None
        and result.observation_after is not None
        and result.observation_after > result.observation_before
        and result.arm_overlap_detected is True
    )


def is_recoverable_predicate_unavailable(result: SkillResult | None) -> bool:
    """Return a certain, fresh perception dropout to the planner for reassessment.

    This does not certify task progress. The planner receives the new observation
    and decides whether to use another bounded VLA chunk, re-ground from another
    view, choose a safe analytic skill, or abort.
    """
    predicate = result.predicate if result is not None else None
    safe_observation = bool(
        result is not None
        and result.state_after is not None
        and result.observation_after is not None
        and (
            (result.executed_policy_actions > 0
             and result.observation_after > result.observation_before)
            or (result.executed_policy_actions == 0
                and result.vla_chunks_requested == 0
                and result.observation_after == result.observation_before
                and result.state_after.command_state == result.state_before.command_state)
        )
    )
    return bool(
        result is not None
        and result.skill == SkillType.VLA_ACT
        and not result.completed
        and result.stop_reason == "predicate_unavailable"
        and result.information_source == "public"
        and not result.episode_terminal
        and not result.execution_uncertain
        and safe_observation
        and predicate is not None
        and predicate.source == "perception"
        and predicate.value is None
        and predicate.observation_sequence == result.observation_after
    )


def is_recoverable_release_support_gap(result: SkillResult | None) -> bool:
    """Return known mid-release evidence loss/alignment drift for reassessment.

    Release must have started with authorization and execution must be known.
    Rack-seat alignment is measured pose feedback, not physical contact proof.
    """
    predicate = result.predicate if result is not None else None
    return bool(
        result is not None
        and result.skill == SkillType.RELEASE
        and not result.completed
        and result.stop_reason in {"release_support_lost", "rack_seat_alignment_lost"}
        and not result.episode_terminal
        and result.information_source == "public"
        and not result.execution_uncertain
        and result.executed_physics_ticks > 0
        and result.executed_policy_actions == 0
        and result.vla_chunks_requested == 0
        and result.state_after is not None
        and result.observation_after is not None
        and result.observation_after > result.observation_before
        and predicate is not None
        and ((result.stop_reason == "release_support_lost"
              and predicate.name == "release_ready"
              and predicate.source == "perception" and predicate.value is None)
             or (result.stop_reason == "rack_seat_alignment_lost"
                 and predicate.name == "rack_seat_aligned"
                 and predicate.source == "runtime" and predicate.value is False))
        and predicate.observation_sequence == result.observation_after
    )
