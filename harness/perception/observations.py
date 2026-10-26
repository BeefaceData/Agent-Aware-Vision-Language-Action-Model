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
            robot_state = {}
            if observation.measured_tcp_poses is not None:
                robot_state = {f"{arm}_tcp_pose": np.asarray(pose, dtype=float)
                               for arm, pose in observation.measured_tcp_poses.items()}
            with (self.directory / name).open("xb") as stream:
                np.savez_compressed(stream, **observation.images,
                    **camera_archive_values(observation.camera_frames),
                    command_state=np.asarray(observation.command_state), sequence=sequence,
                    **robot_state)
            self._saved[sequence] = {"sequence": sequence, "path": f"{self.directory.name}/{name}"}
        return dict(self._saved[sequence])
