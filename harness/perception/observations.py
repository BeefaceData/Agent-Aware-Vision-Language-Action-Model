"""Public camera/state artifacts referenced by skill-turn assessments."""

from pathlib import Path
import numpy as np
from .rgbd import camera_archive_values


class ObservationStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._saved = {}

    def save(self, observation):
        observation = observation.copy()  # Validate/copy only the public contract.
        sequence = observation.sequence
        if sequence not in self._saved:
            name = f"observation_{sequence:08d}.npz"
            with (self.directory / name).open("xb") as stream:
                np.savez_compressed(stream, **observation.images,
                    **camera_archive_values(observation.camera_frames),
                    command_state=np.asarray(observation.command_state), sequence=sequence)
            self._saved[sequence] = {"sequence": sequence, "path": f"{self.directory.name}/{name}"}
        return dict(self._saved[sequence])
