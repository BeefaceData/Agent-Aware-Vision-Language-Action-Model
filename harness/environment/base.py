"""Small environment contract; no simulator objects in public observations."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from harness.trace import copy_images, vector14
from harness.perception.rgbd import RGBDCamera


@dataclass(frozen=True)
class Observation:
    sequence: int
    images: dict[str, np.ndarray]
    command_state: tuple[float, ...]
    camera_frames: dict[str, RGBDCamera] | None = None

    def __post_init__(self):
        if self.camera_frames is None:
            return
        if (not isinstance(self.camera_frames, dict)
                or any(name not in self.images or not isinstance(frame, RGBDCamera)
                       or frame.depth_m.shape != np.asarray(self.images[name]).shape[:2]
                       for name, frame in self.camera_frames.items())):
            raise ValueError("invalid_observation_camera_frames")

    def copy(self):
        frames = None if self.camera_frames is None else {
            name: frame.copy() for name, frame in self.camera_frames.items()}
        return Observation(self.sequence, copy_images(self.images), vector14(self.command_state), frames)


class Environment(Protocol):
    @property
    def native_success(self) -> bool: ...

    def observe(self) -> Observation: ...

    def step(self, action) -> Observation: ...

    def step_after_success(self, action) -> Observation: ...
