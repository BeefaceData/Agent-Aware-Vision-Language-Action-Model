"""Shared, simulator-independent base types used across the harness."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, validate_default=True)


NonnegativeInt = Annotated[StrictInt, Field(ge=0)]
Confidence = Annotated[StrictFloat, Field(ge=0, le=1)]
Identifier = Annotated[StrictStr, Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")]
Vector14 = Annotated[tuple[StrictFloat, ...], Field(min_length=14, max_length=14)]
Parameter = StrictFloat | StrictInt | StrictStr | StrictBool


class StateSummary(Model):
    # Preserve the policy's existing input; these are not measured joint angles.
    command_state: Vector14
    semantics: Literal["arm_drive_targets_and_gripper_commands"] = (
        "arm_drive_targets_and_gripper_commands"
    )
