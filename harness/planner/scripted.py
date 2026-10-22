"""R4/R5 acquisition hypothesis; not an admitted full-task controller."""

from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from harness.schemas import Model
from harness.skills.schemas import MoveParameters, SkillCommand, VLAParameters
from .base import PlannerSelection


class GraspRetrySettings(Model):
    stage_before_first_vla: StrictBool = True
    arm: Literal["left", "right"] = "left"
    max_retries: Annotated[StrictInt, Field(ge=0, le=5)] = 2
    stage: MoveParameters = Field(default_factory=lambda: MoveParameters(target="mug_pregrasp"))
    retreat: MoveParameters = Field(default_factory=lambda: MoveParameters(offset_xyz=(0.0, 0.0, 0.05)))
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
        return self


class GraspRetryPlanner:
    def __init__(self, settings=None):
        self.settings = GraspRetrySettings.model_validate(
            (settings or GraspRetrySettings()).model_dump())
        self.phase = "APPROACH_MUG" if self.settings.stage_before_first_vla else "GRASP_MUG"
        self.retries = 0
        self.stage_offset = self.settings.stage.offset_xyz

    def select(self, context):
        previous = context.previous
        if previous is not None:
            if previous.episode_terminal or not previous.completed:
                return self._abort("previous_skill_failed")
            if self.phase == "APPROACH_MUG":
                self.phase = "GRASP_MUG"
            elif self.phase == "GRASP_MUG":
                # A chunk budget is an attempt boundary, never evidence of success.
                if context.value("mug_grasp_confirmed") is None:
                    return self._abort("grasp_evidence_unavailable")
                if context.value("mug_grasp_confirmed"):
                    return self._abort("grasp_without_verified_lift")
                if self.retries >= self.settings.max_retries:
                    return self._abort("grasp_retry_budget_exhausted")
                if context.value("release_ready") is not True:
                    return self._abort("retry_open_not_authorized")
                self.retries += 1
                if self.retries <= len(self.settings.retry_stage_offsets):
                    self.stage_offset = self.settings.retry_stage_offsets[self.retries-1]
                self.phase = "RETRY_OPEN"
            elif self.phase == "RETRY_OPEN":
                self.phase = "RETRY_RETREAT"
            elif self.phase == "RETRY_RETREAT":
                self.phase = "APPROACH_MUG"

        if context.value("mug_lifted") is None:
            return self._abort("lift_evidence_unavailable")
        if self.phase == "APPROACH_MUG":
            parameters = self.settings.stage.model_dump()
            parameters["offset_xyz"] = self.stage_offset
            command = dict(action="move_to", arm=self.settings.arm, parameters=parameters,
                           reason="Stage at a freshly grounded mug pregrasp")
        elif self.phase == "GRASP_MUG":
            command = dict(action="vla_act", arm="both", stop="mug_lifted",
                           parameters=self.settings.vla.model_dump(), reason="Bounded frozen-policy acquisition")
        elif self.phase == "RETRY_OPEN":
            command = dict(action="set_gripper", arm=self.settings.arm,
                           parameters={"value": 1.0}, reason="Open supported or empty hand before retry")
        else:
            command = dict(action="move_to", arm=self.settings.arm,
                           parameters=self.settings.retreat.model_dump(), reason="Retreat before re-grounding pregrasp")
        return PlannerSelection(SkillCommand.model_validate(command), self.phase, self.retries)

    def _abort(self, reason):
        return PlannerSelection(SkillCommand(action="abort", reason=reason), "FAILED", self.retries)
