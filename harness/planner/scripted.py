"""R4/R5 acquisition hypothesis; not an admitted full-task controller."""

from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictInt, model_validator

from harness.schemas import Model
from harness.skills.schemas import MoveParameters, SkillCommand, SkillType, VLAParameters
from .base import PlannerSelection

MAX_RETRY_STAGE_POSITION_ERROR_M = 0.03
MAX_RETRY_STAGE_ORIENTATION_ERROR_RAD = 0.10


def monitored_grasp_parameters(parameters, arm):
    """Attach the runtime-owned local failed-grasp monitor to a VLA call."""
    values = parameters.model_dump() if hasattr(parameters, "model_dump") else dict(parameters)
    values.update(failure_stop="grasp_attempt_failed", failure_arm=arm)
    return values


def verified_recovery_clearance(result):
    """Require measured completion of the clearance move before VLA resumes."""
    return bool(
        result is not None
        and result.skill == SkillType.MOVE_TO
        and result.completed
        and result.stop_reason == "completed"
        and not result.execution_uncertain
        and result.executed_physics_ticks > 0
        and result.position_error_m is not None
        and result.position_error_m <= 0.015
        and result.orientation_error_rad is not None
        and result.orientation_error_rad <= 0.10
    )


def verified_recovery_gripper_open(result):
    """Require measured gripper motion before starting a fresh VLA attempt."""
    return bool(
        result is not None
        and result.skill == SkillType.SET_GRIPPER
        and result.completed
        and result.stop_reason == "completed"
        and not result.execution_uncertain
        and result.executed_physics_ticks > 0
    )


def retry_open_authorized(context):
    """Open only with current proof of support and an upright mug."""
    return (context.value("release_ready") is True
            and context.value("mug_toppled") is False)


def near_grounded_retry_stage(context, result):
    """Whether the frozen VLA can safely finish a nearly reached pre-grasp."""
    visual = getattr(context, "visual_context", None) or {}
    return (result is not None and result.stop_reason == "target_not_reached"
            and result.position_error_m is not None
            and result.position_error_m <= MAX_RETRY_STAGE_POSITION_ERROR_M
            and result.orientation_error_rad is not None
            and result.orientation_error_rad <= MAX_RETRY_STAGE_ORIENTATION_ERROR_RAD
            and visual.get("status", {}).get("mug") == "grounded")


class GraspRetrySettings(Model):
    stage_before_first_vla: StrictBool = True
    arm: Literal["left", "right"] = "left"
    max_retries: Annotated[StrictInt, Field(ge=0, le=5)] = 2
    stage: MoveParameters = Field(default_factory=lambda: MoveParameters(target="mug_pregrasp"))
    retreat: MoveParameters = Field(default_factory=lambda: MoveParameters(
        target="mug_recovery_clearance", orientation_mode="preserve_current"))
    vla: VLAParameters = Field(default_factory=lambda: VLAParameters(max_chunks=3, max_policy_actions=24))

    @model_validator(mode="after")
    def validate_retreat(self):
        retreat = self.retreat
        if (retreat.target != "mug_recovery_clearance"
                or retreat.offset_xyz != (0.0, 0.0, 0.0)
                or retreat.path_mode != "target"
                or retreat.orientation_mode != "preserve_current"):
            raise ValueError("grasp_retry_requires_fixed_mug_clearance")
        return self


class GraspRetryPlanner:
    def __init__(self, settings=None):
        self.settings = GraspRetrySettings.model_validate(
            (settings or GraspRetrySettings()).model_dump())
        self.phase = "APPROACH_MUG" if self.settings.stage_before_first_vla else "GRASP_MUG"
        self.retries = 0

    def resume_development_retry(self, *, retries=1):
        """Start at the first recovery action after a replayed failed grasp."""
        if self.phase != "APPROACH_MUG" and self.phase != "GRASP_MUG":
            raise ValueError("planner_already_started")
        if type(retries) is not int or not 1 <= retries <= self.settings.max_retries:
            raise ValueError("invalid_development_retry_count")
        self.phase = "RETRY_RETREAT"
        self.retries = retries

    def select(self, context):
        previous = context.previous
        if previous is not None:
            if previous.episode_terminal:
                return self._abort("previous_skill_failed")
            if not previous.completed:
                if (self.phase == "APPROACH_MUG"
                        and self.retries > 0
                        and previous.skill.value == "move_to"
                        and near_grounded_retry_stage(context, previous)):
                    self.phase = "GRASP_MUG"
                    return self._grasp_selection(
                        "Continue the contact approach with the frozen VLA from a near pre-grasp")
                return self._abort("previous_skill_failed")
            if self.phase == "APPROACH_MUG":
                self.phase = "GRASP_MUG"
            elif self.phase == "GRASP_MUG":
                # A chunk budget is an attempt boundary, never evidence of success.
                if context.value("mug_grasp_confirmed") is None:
                    return self._abort("grasp_evidence_unavailable")
                if (context.value("mug_grasp_confirmed")
                        and context.value("release_ready") is not True):
                    return self._abort("grasp_without_verified_lift")
                if self.retries >= self.settings.max_retries:
                    return self._abort("grasp_retry_budget_exhausted")
                self.retries += 1
                self.phase = "RETRY_RETREAT"
            elif self.phase == "RETRY_RETREAT":
                if not verified_recovery_clearance(previous):
                    return self._abort("grasp_retry_clearance_unverified")
                if not retry_open_authorized(context):
                    return self._abort("grasp_retry_open_not_authorized")
                self.phase = "RETRY_OPEN"
                return self.retry_open_selection()
            elif self.phase == "RETRY_OPEN":
                if not verified_recovery_gripper_open(previous):
                    return self._abort("grasp_retry_open_unverified")
                self.phase = "GRASP_MUG"
                return self._grasp_selection(
                    "Resume the frozen VLA from verified clearance with an open gripper")

        if context.value("mug_lifted") is None:
            return self._abort("lift_evidence_unavailable")
        if self.phase == "APPROACH_MUG":
            parameters = self.settings.stage.model_dump()
            command = dict(action="move_to", arm=self.settings.arm, parameters=parameters,
                           reason="Stage at a freshly grounded mug pre-approach")
        elif self.phase == "GRASP_MUG":
            return self._grasp_selection("Bounded frozen-policy acquisition")
        else:
            return self.retry_retreat_selection()
        return PlannerSelection(SkillCommand.model_validate(command), self.phase, self.retries)

    def _grasp_selection(self, reason):
        command = SkillCommand(action="vla_act", arm="both", stop="mug_lifted",
            parameters=monitored_grasp_parameters(self.settings.vla, self.settings.arm),
            reason=reason)
        return PlannerSelection(command, "GRASP_MUG", self.retries)

    def retry_retreat_selection(self, *, arm=None, phase="RETRY_RETREAT", retries=None):
        """Move once to the grounded fixed clearance above the mug."""
        command = SkillCommand(action="move_to", arm=arm or self.settings.arm,
            parameters=self.settings.retreat.model_dump(),
            reason="Move to the fixed grounded clearance above the mug before fresh VLA inference")
        return PlannerSelection(command, phase,
                                self.retries if retries is None else retries)

    def retry_open_selection(self, *, arm=None, phase="RETRY_OPEN", retries=None):
        """Open only after clearance; the skill enforces verified object support."""
        command = SkillCommand(action="set_gripper", arm=arm or self.settings.arm,
            parameters={"value": 1.0},
            reason="Normalize the cleared gripper before fresh VLA inference")
        return PlannerSelection(command, phase,
                                self.retries if retries is None else retries)

    def _abort(self, reason):
        return PlannerSelection(SkillCommand(action="abort", reason=reason), "FAILED", self.retries)
