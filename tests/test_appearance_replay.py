"""CPU-only contracts for reviewed RGB-D appearance descriptor replay."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from development.appearance_replay import replay
from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from harness.perception.rgbd import RGBDCamera
from tests.helpers import images


class AppearanceReplayTests(unittest.TestCase):
    def fixture(self, root):
        observation_dir = root / "observations"
        store = ObservationStore(observation_dir)
        rgb = images()
        rgb["head_camera"][:] = (120, 55, 32)
        camera = RGBDCamera(np.full((2, 3), 1.0, np.float32),
            np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]), np.eye(4))
        path = store.save(Observation(0, rgb, (0.,)*14, {"head_camera": camera}))
        source = observation_dir / Path(path["path"]).name
        annotation = {"schema_version": 1, "frames": [{"sequence": 0,
            "camera": "head_camera", "detections": [{"label": "mug",
            "confidence": 1.0, "shape": [2, 3], "runs": [[0, 6]]}]}],
            "source_sha256": {source.name: sha256(source.read_bytes()).hexdigest()}}
        detections = root / "detections.json"
        detections.write_text(json.dumps(annotation))
        return observation_dir, detections, source

    def test_reviewed_mask_replay_reports_public_features_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            observation_dir, detections, _ = self.fixture(Path(tmp))
            result = replay(observation_dir, detections)
        self.assertFalse(result["robot_motion"])
        self.assertFalse(result["simulator_started"])
        self.assertFalse(result["detector_model_executed"])
        visual = result["frames"][0]["visual_context"]
        mug = visual["objects"]["mug"]
        self.assertEqual(mug["median_rgb_0_255"], (120.0, 55.0, 32.0))
        self.assertEqual(mug["normalized_bbox"], (0.0, 0.0, 1.0, 1.0))
        self.assertNotIn("position", mug)
        self.assertNotIn("world_to_camera", json.dumps(result))

    def test_source_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            observation_dir, detections, _ = self.fixture(Path(tmp))
            data = json.loads(detections.read_text())
            data["source_sha256"]["observation_00000000.npz"] = "0"*64
            detections.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "appearance_observation_hash_mismatch"):
                replay(observation_dir, detections)


if __name__ == "__main__":
    unittest.main()
