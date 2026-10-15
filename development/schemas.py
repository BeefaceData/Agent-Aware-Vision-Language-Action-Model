"""Versioned checkpoint/replay contracts, isolated from deployed observations."""

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictStr, field_validator, model_validator

from harness.schemas import FailureType, Model, NonnegativeInt, TaskPhase, Vector14


SafeID = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Digest = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
CAMERAS = ("head_camera", "left_camera", "right_camera")


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value or value == ".":
        raise ValueError("artifact paths must be normalized relative POSIX paths")
    return value


class FailureCheckpoint(Model):
    schema_version: Literal[1] = 1
    checkpoint_id: SafeID
    run_id: SafeID
    episode_id: SafeID
    step: NonnegativeInt
    boundary: Literal["after_executed_actions"] = "after_executed_actions"
    seed: NonnegativeInt
    task: Literal["hanging_mug"] = "hanging_mug"
    phase: TaskPhase = TaskPhase.UNKNOWN
    failure_type: FailureType = FailureType.UNKNOWN
    images: dict[StrictStr, StrictStr]
    state: Vector14
    recent_actions: tuple[Vector14, ...] = ()
    recent_states: tuple[Vector14, ...]
    simulator_snapshot_path: None = None  # Full physics snapshots are not supported yet.
    trace_path: StrictStr
    evaluation_path: StrictStr
    provenance_path: StrictStr
    artifact_hashes: dict[StrictStr, Digest]
    notes: StrictStr | None = None
    label_revision: NonnegativeInt = 0

    @field_validator("trace_path", "evaluation_path", "provenance_path")
    @classmethod
    def check_path(cls, value):
        return relative_path(value)

    @model_validator(mode="after")
    def check_assets_and_history(self):
        if set(self.images) != set(CAMERAS):
            raise ValueError("checkpoint requires all three camera references")
        for path in (*self.images.values(), *self.artifact_hashes):
            relative_path(path)
        expected = set(self.images.values()) | {self.trace_path, self.evaluation_path, self.provenance_path}
        if set(self.artifact_hashes) != expected:
            raise ValueError("hashes must cover exactly the checkpoint artifacts")
        if not self.recent_states or self.recent_states[-1] != self.state:
            raise ValueError("recent state history must end at the checkpoint state")
        if len(self.recent_actions) > self.step or len(self.recent_states) > self.step + 1:
            raise ValueError("history extends before episode reset")
        return self

    def restore(self, env):
        """Delegate to a restoration adapter; metadata alone cannot restore physics."""
        return env.restore_checkpoint(self)


class RecordedFrame(Model):
    schema_version: Literal[1] = 1
    step: NonnegativeInt
    action: Vector14 | None
    command_state: Vector14

    @model_validator(mode="after")
    def action_boundary(self):
        if (self.step == 0) != (self.action is None):
            raise ValueError("only reset frame zero has no executed action")
        return self


class ReplayOptions(Model):
    state_atol: Annotated[StrictFloat, Field(ge=0)] = 1e-5
    image_mae_tolerance: Annotated[StrictFloat, Field(ge=0, le=255)] = 5.0


class ReplayReport(Model):
    checkpoint_id: SafeID
    status: Literal["matched", "drifted", "unavailable", "error"]
    expected_steps: NonnegativeInt
    executed_steps: NonnegativeInt
    state_max_abs_error: StrictFloat | None = None
    image_mean_abs_errors: dict[StrictStr, StrictFloat] = Field(default_factory=dict)
    local_provenance: Literal["matched", "mismatch", "unchecked"] = "unchecked"
    full_state_restored: Literal[False] = False
    reason: StrictStr
    error_type: StrictStr | None = None
