"""Small environment contract; no simulator objects in public observations."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from harness.perception.rgbd import RGBDCamera
from harness.schemas import StateSummary
from .layout import CAMERAS


def vector14(value) -> tuple[float, ...]:
    array = np.asarray(value)
    if array.shape != (14,) or array.dtype.kind not in "fiu":
        raise ValueError("expected a numeric 14-element vector")
    # Validate before conversion (including overflow in unusually large integers).
    return StateSummary(command_state=tuple(array.tolist())).command_state


def copy_images(images: dict) -> dict[str, np.ndarray]:
    if set(images) != set(CAMERAS):
        raise ValueError("images must contain exactly the three public RGB cameras")
    copied = {}
    for camera in CAMERAS:
        image = np.asarray(images[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape) == 0:
            raise ValueError(f"{camera} must be nonempty HxWx3 uint8 RGB")
        copied[camera] = image.copy()
    return copied


@dataclass(frozen=True)
class Observation:
    sequence: int
    images: dict[str, np.ndarray]
    command_state: tuple[float, ...]
    camera_frames: dict[str, RGBDCamera] | None = None
    # Optional measured robot FK poses; used locally for arm attribution and
    # deliberately omitted from planner-facing prompt payloads.
    measured_tcp_poses: dict[str, tuple[float, ...]] | None = None

    def __post_init__(self):
        if self.camera_frames is not None and (not isinstance(self.camera_frames, dict)
                or any(name not in self.images or not isinstance(frame, RGBDCamera)
                       or frame.depth_m.shape != np.asarray(self.images[name]).shape[:2]
                       for name, frame in self.camera_frames.items())):
            raise ValueError("invalid_observation_camera_frames")
        if self.measured_tcp_poses is not None:
            if (not isinstance(self.measured_tcp_poses, dict)
                    or set(self.measured_tcp_poses) != {"left", "right"}):
                raise ValueError("invalid_measured_tcp_poses")
            poses = {}
            for arm, value in self.measured_tcp_poses.items():
                pose = np.asarray(value, dtype=float)
                if (pose.shape != (7,) or not np.isfinite(pose).all()
                        or abs(np.linalg.norm(pose[3:]) - 1.0) > 0.01):
                    raise ValueError("invalid_measured_tcp_poses")
                poses[arm] = tuple(float(x) for x in pose)
            object.__setattr__(self, "measured_tcp_poses", poses)

    def copy(self):
        frames = None if self.camera_frames is None else {
            name: frame.copy() for name, frame in self.camera_frames.items()}
        poses = None if self.measured_tcp_poses is None else {
            arm: tuple(pose) for arm, pose in self.measured_tcp_poses.items()}
        return Observation(self.sequence, copy_images(self.images),
                           vector14(self.command_state), frames, poses)


class Environment(Protocol):
    @property
    def native_success(self) -> bool: ...

    def observe(self) -> Observation: ...

    def step(self, action) -> Observation: ...

    def step_after_success(self, action) -> Observation: ...
