"""R4/R5 acquisition hypothesis; not an admitted full-task controller."""

from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from harness.schemas import Model
from harness.skills.schemas import MoveParameters, SkillCommand, VLAParameters
from .base import PlannerSelection

MAX_RETRY_STAGE_POSITION_ERROR_M = 0.03
MAX_RETRY_STAGE_ORIENTATION_ERROR_RAD = 0.10


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
        target="current_ee", path_mode="reverse_approach", retreat_distance_m=0.05,
        orientation_mode="preserve_current"))
    # Optional finite pregrasp offsets, indexed by retry number. The trusted
    # anchor preparer refreshes mug_pregrasp from the current observation first.
    retry_stage_offsets: tuple[tuple[StrictFloat, StrictFloat, StrictFloat], ...] = Field(
        default=(), max_length=5)
    vla: VLAParameters = Field(default_factory=lambda: VLAParameters(max_chunks=3, max_policy_actions=24))

    @model_validator(mode="after")
    def validate_retry_offsets(self):
        if any(sum(value*value for value in offset) > 0.20**2 for offset in self.retry_stage_offsets):
            raise ValueError("retry stage offset exceeds 20cm")
        if len(self.retry_stage_offsets) > self.max_retries:
            raise ValueError("retry stage offset exceeds retry budget")
        retreat = self.retreat
        if (retreat.target != "current_ee" or retreat.offset_xyz != (0.0, 0.0, 0.0)
                or retreat.path_mode != "reverse_approach"
                or retreat.orientation_mode != "preserve_current"):
            raise ValueError("grasp_retry_requires_measured_reverse_retreat")
        return self


class GraspRetryPlanner:
    def __init__(self, settings=None):
        self.settings = GraspRetrySettings.model_validate(
            (settings or GraspRetrySettings()).model_dump())
        self.phase = "APPROACH_MUG" if self.settings.stage_before_first_vla else "GRASP_MUG"
        self.retries = 0
        self.stage_offset = self.settings.stage.offset_xyz
        self.retry_stage_orientation_mode = None
        self.reorientation_attempted = False

    def resume_development_retry(self, *, retries=1):
        """Start at the first recovery action after a replayed failed grasp."""
        if self.phase != "APPROACH_MUG" and self.phase != "GRASP_MUG":
            raise ValueError("planner_already_started")
        if type(retries) is not int or not 1 <= retries <= self.settings.max_retries:
            raise ValueError("invalid_development_retry_count")
        self.phase = "RETRY_OPEN"
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
                if (self.phase == "APPROACH_MUG" and self.reorientation_attempted
                        and previous.skill.value == "move_to"
                        and previous.stop_reason in {"trajectory_limits", "planning_failure"}):
                    self.reorientation_attempted = False
                    return self._fallback_restage()
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
                if not _retry_open_authorized(context, arm=self.settings.arm,
                        failed_predicate=context.value("mug_grasp_confirmed")):
                    return self._abort("retry_open_not_authorized")
                self.retries += 1
                if self.retries <= len(self.settings.retry_stage_offsets):
                    self.stage_offset = self.settings.retry_stage_offsets[self.retries-1]
                self.phase = "RETRY_OPEN"
            elif self.phase == "RETRY_OPEN":
                self.phase = "RETRY_RETREAT"
            elif self.phase == "RETRY_RETREAT":
                self.phase = "APPROACH_MUG"
                self.retry_stage_orientation_mode = (
                    "preserve_current" if self.retries == 1 else "target")
                self.reorientation_attempted = self.retries > 1

        if context.value("mug_lifted") is None:
            return self._abort("lift_evidence_unavailable")
        if self.phase == "APPROACH_MUG":
            parameters = self.settings.stage.model_dump()
            parameters["offset_xyz"] = self.stage_offset
            if self.retry_stage_orientation_mode is not None:
                parameters["orientation_mode"] = self.retry_stage_orientation_mode
                self.retry_stage_orientation_mode = None
            command = dict(action="move_to", arm=self.settings.arm, parameters=parameters,
                           reason=("Re-stage without changing wrist orientation" if
                                   parameters["orientation_mode"] == "preserve_current" else
                                   "Stage at a freshly grounded mug pregrasp"))
        elif self.phase == "GRASP_MUG":
            return self._grasp_selection("Bounded frozen-policy acquisition")
        elif self.phase == "RETRY_OPEN":
            command = dict(action="set_gripper", arm=self.settings.arm,
                           parameters={"value": 1.0}, reason="Open supported or empty hand before retry")
        else:
            retreat = self.settings.retreat.model_dump()
            if self.retries > 1:
                retreat["retreat_distance_m"] = 0.10
            command = dict(action="move_to", arm=self.settings.arm,
                           parameters=retreat, reason="Retreat before re-grounding pregrasp")
        return PlannerSelection(SkillCommand.model_validate(command), self.phase, self.retries)

    def _grasp_selection(self, reason):
        command = SkillCommand(action="vla_act", arm="both", stop="mug_lifted",
            parameters=self.settings.vla.model_dump(), reason=reason)
        return PlannerSelection(command, "GRASP_MUG", self.retries)

    def _fallback_restage(self):
        """Keep the measured wrist orientation if target reorientation is infeasible."""
        parameters = self.settings.stage.model_dump()
        parameters["offset_xyz"] = self.stage_offset
        parameters["orientation_mode"] = "preserve_current"
        command = SkillCommand(action="move_to", arm=self.settings.arm,
            parameters=parameters,
            reason="Re-stage at the mug while preserving the measured wrist orientation")
        self.phase = "APPROACH_MUG"
        return PlannerSelection(command, "RETRY_RESTAGE_FALLBACK", self.retries)

    def _abort(self, reason):
        return PlannerSelection(SkillCommand(action="abort", reason=reason), "FAILED", self.retries)


def _retry_open_authorized(context, *, arm, failed_predicate):
    """Allow a retry open after a verified miss on a publicly grounded mug."""
    if context.value("release_ready") is True:
        return True
    arm_lift = "right_mug_lifted" if arm == "right" else "mug_lifted"
    visual = getattr(context, "visual_context", None) or {}
    return (failed_predicate is False and context.value(arm_lift) is False
            and visual.get("status", {}).get("mug") == "grounded")
