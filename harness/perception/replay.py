"""Replay recorded semantic masks against public RGB-D observations offline."""

import json
from pathlib import Path

import numpy as np

from .grounding import Detection, DetectionBatch


def decode_mask_rle(shape, runs):
    """Decode sorted flat ``[start, length]`` runs into a bounded bool mask."""
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(item) is not int or item <= 0 for item in shape)):
        raise ValueError("invalid_mask_shape")
    height, width = shape
    total = height * width
    if total > 8_000_000 or not isinstance(runs, (list, tuple)):
        raise ValueError("invalid_mask_rle")
    flat = np.zeros(total, dtype=bool)
    previous_end = 0
    for run in runs:
        if (not isinstance(run, (list, tuple)) or len(run) != 2
                or any(type(item) is not int for item in run)):
            raise ValueError("invalid_mask_rle")
        start, length = run
        if length <= 0 or start < previous_end or start + length > total:
            raise ValueError("invalid_mask_rle")
        flat[start:start+length] = True
        previous_end = start + length
    return flat.reshape(height, width)


class RecordedDetectionAdapter:
    """A deterministic ``SemanticDetector`` backed by reviewed mask annotations.

    Input JSON schema: ``{"schema_version": 1, "frames": [{"sequence": 0,
    "camera": "head_camera", "detections": [{"label": "mug",
    "confidence": 0.95, "shape": [480, 640], "runs": [[10, 3]]}]}]}``.
    Missing annotations raise instead of being silently treated as no objects.
    """

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            data = json.loads(Path(source).read_text())
        else:
            data = source
        if (not isinstance(data, dict) or data.get("schema_version") != 1
                or not isinstance(data.get("frames"), list)):
            raise ValueError("unsupported_detection_replay")
        self._frames = {}
        for row in data["frames"]:
            if not isinstance(row, dict):
                raise ValueError("invalid_detection_replay_frame")
            sequence, camera = row.get("sequence"), row.get("camera")
            if (type(sequence) is not int or sequence < 0 or not isinstance(camera, str)
                    or not camera or sequence in self._frames
                    or not isinstance(row.get("detections"), list)):
                raise ValueError("invalid_detection_replay_frame")
            detections = []
            for item in row["detections"]:
                if not isinstance(item, dict):
                    raise ValueError("invalid_detection_replay_item")
                detections.append(Detection(item.get("label"), item.get("confidence"),
                    decode_mask_rle(item.get("shape"), item.get("runs"))))
            self._frames[sequence] = DetectionBatch(sequence, camera, tuple(detections))

    def detect(self, images, *, observation_sequence):
        batch = self._frames.get(observation_sequence)
        if batch is None:
            raise ValueError("detection_annotation_missing")
        if batch.camera not in images:
            raise ValueError("detection_annotation_camera_missing")
        if any(detection.mask.shape != np.asarray(images[batch.camera]).shape[:2]
               for detection in batch.detections):
            raise ValueError("detection_annotation_shape_mismatch")
        return batch


def load_public_observation(path):
    """Load only the versioned public observation fields from one NPZ archive."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        from harness.environment.layout import CAMERAS, PERCEPTION_CAMERAS
        required = {*CAMERAS, "command_state", "sequence"}
        if not required <= set(archive.files):
            raise ValueError("incomplete_public_observation_archive")
        # Perception-only cameras are optional: older captures do not have them.
        images = {name: np.asarray(archive[name]).copy()
                  for name in (*CAMERAS, *PERCEPTION_CAMERAS) if name in archive.files}
        from .rgbd import camera_frames_from_archive
        frames = camera_frames_from_archive(archive, images)
        from harness.environment.base import Observation
        sequence = int(archive["sequence"])
        pose_keys = {"left_tcp_pose", "right_tcp_pose"}
        present_poses = pose_keys & set(archive.files)
        if present_poses and present_poses != pose_keys:
            raise ValueError("incomplete_measured_tcp_pose_archive")
        measured_tcp_poses = ({arm: tuple(archive[f"{arm}_tcp_pose"].tolist())
                               for arm in ("left", "right")} if present_poses else None)
        expected = path.stem.removeprefix("observation_")
        if path.stem.startswith("observation_") and expected.isdigit() and int(expected) != sequence:
            raise ValueError("observation_archive_sequence_mismatch")
        return Observation(sequence, images, tuple(archive["command_state"].tolist()),
                          frames or None, measured_tcp_poses).copy()
