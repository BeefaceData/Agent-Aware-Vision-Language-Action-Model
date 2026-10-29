"""Shared single-episode settings for the runner and exploration lifecycle."""

from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictInt, model_validator
from harness.schemas import Model
from harness.planner.scripted import GraspRetrySettings
from harness.planner.anthropic import AnthropicSettings
from harness.skills.schemas import EpisodeBudget, VLAParameters


class PerceptionSettings(Model):
    # Sensor payload for local public perception only; default preserves the
    # legacy simulator observation cost and baseline behavior.
    rgbd_enabled: StrictBool = False
    # One RGB-only Opus call captures a fixed-rack reference at episode start.
    opus_rack_grounding: StrictBool = False
    # Experimental: cache a reset table plane, then ground Opus mug support
    # pixels against fresh head-camera RGB-D for temporal lift evidence.
    opus_mug_lift_grounding: StrictBool = False
    # Defer the first expensive public lift assessment until this many VLA
    # actions have run; later checks stay aligned to planner feedback turns.
    mug_lift_first_check_after_actions: Annotated[StrictInt, Field(ge=0, le=10000)] = 0
    rack_grounding_camera: Literal["head_camera", "left_camera", "right_camera"] = "head_camera"
    # Experimental rack grounding uses a separately calibrated SAM 3 prompt
    # and confidence threshold. Default/evaluation behavior remains unchanged.
    rack_prompt_profile: Literal["default", "explore_mug_rack"] = "default"


class RunSettings(Model):
    record_video: StrictBool = False
    # Local simulator only: render each policy step once instead of twice (the
    # base step also renders the pre-action state and discards it). Same
    # observations; off by default so existing runs keep their measured timing.
    render_once_per_step: StrictBool = False
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    max_turns: Annotated[StrictInt, Field(ge=1, le=1000)] = 20
    recipe: GraspRetrySettings = Field(default_factory=GraspRetrySettings)
    budget: EpisodeBudget = Field(default_factory=EpisodeBudget)
    anthropic: AnthropicSettings | None = None
    # Optional bounded scripted transfer recipe; Anthropic full-task planning
    # has its own task_scope and does not replace the planner with this recipe.
    handoff: VLAParameters | None = None
    # Allows validating left placement/retreat without also launching right-arm hanging.
    stop_after_left_transfer: StrictBool = False

    @model_validator(mode="after")
    def validate_transfer_stop(self):
        if self.perception.opus_rack_grounding:
            if not self.perception.rgbd_enabled:
                raise ValueError("opus_rack_grounding_requires_rgbd")
            if self.anthropic is None or self.anthropic.task_scope != "full_task":
                raise ValueError("opus_rack_grounding_requires_full_task_anthropic")
        if self.perception.opus_mug_lift_grounding:
            if not self.perception.rgbd_enabled:
                raise ValueError("opus_mug_lift_grounding_requires_rgbd")
            if self.anthropic is None:
                raise ValueError("opus_mug_lift_grounding_requires_anthropic")
        if (self.stop_after_left_transfer and self.handoff is None
                and not (self.anthropic is not None and self.anthropic.task_scope == "full_task")):
            raise ValueError("left_transfer_stop_requires_handoff_recipe")
        if (self.handoff is not None and self.anthropic is not None
                and self.anthropic.task_scope != "full_task"):
            raise ValueError("handoff_recipe_requires_full_task_anthropic_scope")
        return self

    @property
    def completion_predicate(self):
        if self.stop_after_left_transfer:
            return "left_transfer_complete"
        full_task = self.handoff is not None or (self.anthropic is not None
                                                  and self.anthropic.task_scope == "full_task")
        return "task_success" if full_task else "mug_lifted"
