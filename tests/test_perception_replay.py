"""Offline detector replay contracts over public observation archives."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from development.replays.perception import replay
from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from harness.perception.replay import (
    RecordedDetectionAdapter, decode_mask_rle, load_public_observation,
)
from harness.perception.rgbd import RGBDCamera
from tests.helpers import images


class PerceptionReplayTests(unittest.TestCase):
    def test_rle_decoder_validates_bounds_order_and_dimensions(self):
        np.testing.assert_array_equal(decode_mask_rle((2, 3), [[1, 2], [5, 1]]),
            np.array([[False, True, True], [False, False, True]]))
        for shape, runs in (((0, 3), []), ((2, 3), [[0, 2], [1, 1]]),
                            ((2, 3), [[5, 2]]), ((2, 3), [[0, 0]])):
            with self.subTest(shape=shape, runs=runs), self.assertRaises(ValueError):
                decode_mask_rle(shape, runs)

    def test_recorded_detection_adapter_requires_exact_sequence_and_shape(self):
        data = {"schema_version": 1, "frames": [{"sequence": 7, "camera": "head_camera",
            "detections": [{"label": "mug", "confidence": .95,
                            "shape": [2, 3], "runs": [[1, 1]]}]}]}
        adapter = RecordedDetectionAdapter(data)
        result = adapter.detect(images(), observation_sequence=7)
        self.assertEqual(result.detections[0].label, "mug")
        with self.assertRaisesRegex(ValueError, "detection_annotation_missing"):
            adapter.detect(images(), observation_sequence=8)
        with self.assertRaisesRegex(ValueError, "detection_annotation_shape_mismatch"):
            adapter.detect({"head_camera": np.zeros((3, 3, 3), dtype=np.uint8)},
                           observation_sequence=7)

    def test_public_archive_loader_reads_rgbd_but_no_other_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observation_00000007.npz"
            frame = RGBDCamera(np.ones((2, 3), dtype=np.float32),
                np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]), np.eye(4))
            with path.open("xb") as stream:
                from harness.perception.rgbd import camera_archive_values
                np.savez_compressed(stream, **images(), **camera_archive_values({"head_camera": frame}),
                    command_state=np.zeros(14), sequence=7, privileged_marker=np.ones(1))
            observation = load_public_observation(path)
            self.assertEqual(observation.sequence, 7)
            self.assertEqual(set(observation.camera_frames), {"head_camera"})
            self.assertFalse(hasattr(observation, "privileged_marker"))

    def test_offline_replay_grounds_annotated_rgbd_and_keeps_rgb_only_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgbd_dir = root / "rgbd"
            store = ObservationStore(rgbd_dir)
            frame = RGBDCamera(np.full((2, 3), 2.0, dtype=np.float32),
                np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]), np.eye(4))
            store.save(Observation(3, images(), (0.,) * 14, {"head_camera": frame}))
            annotation = root / "masks.json"
            annotation.write_text(json.dumps({"schema_version": 1, "frames": [{
                "sequence": 3, "camera": "head_camera", "detections": [{
                    "label": "mug", "confidence": .95, "shape": [2, 3], "runs": [[1, 1]]}]}]}))
            report = replay(rgbd_dir, annotation)
            self.assertEqual(report["summary"]["rgbd_frame_count"], 1)
            self.assertEqual(report["summary"]["grounded_mug_frames"], 1)
            self.assertEqual(report["frames"][0]["status"]["mug"], "grounded")
            release = next(item for item in report["frames"][0]["evidence"]
                           if item["name"] == "release_ready")
            self.assertIsNone(release["value"])
            self.assertEqual(report["summary"]["release_ready_frames"], 0)
            self.assertFalse(report["robot_motion"])
            self.assertFalse(report["detector_model_executed"])

            rgb_dir = root / "rgb"
            rgb_dir.mkdir()
            rgb_path = rgb_dir / "observation_00000003.npz"
            with rgb_path.open("xb") as stream:
                np.savez_compressed(stream, **images(), command_state=np.zeros(14), sequence=3)
            rgb_only = replay(rgb_dir, annotation)
            self.assertEqual(rgb_only["summary"]["rgbd_frame_count"], 0)
            self.assertEqual(rgb_only["frames"][0]["status"]["mug"], "rgbd_unavailable")
            self.assertEqual(rgb_only["frames"][0]["evidence"][0]["value"], None)


if __name__ == "__main__":
    unittest.main()
