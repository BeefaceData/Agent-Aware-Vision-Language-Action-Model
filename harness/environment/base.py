"""Small environment contract; no simulator objects in public observations."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from harness.trace import copy_images, vector14


@dataclass(frozen=True)
class Observation:
    sequence: int
    images: dict[str, np.ndarray]
    command_state: tuple[float, ...]

    def copy(self):
        return Observation(self.sequence, copy_images(self.images), vector14(self.command_state))


class Environment(Protocol):
    @property
    def native_success(self) -> bool: ...

    def observe(self) -> Observation: ...

    def step(self, action) -> Observation: ...
