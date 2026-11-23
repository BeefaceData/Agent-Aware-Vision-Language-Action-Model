import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from development.hanging.check_grounding660 import check_grounding
from harness.environment.layout import CAMERAS
from tests.test_opus_frame_grounding import FakeClient


class Grounding660CheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.archive = root / "observation_00000660.npz"
        self.output = root / "review"
        images = {name: np.zeros((80, 80, 3), dtype=np.uint8)
                  for name in (*CAMERAS, "perception_camera")}
        np.savez(self.archive, **images, sequence=660, command_state=[0.]*14,
                 perception_camera__depth_m=np.ones((80, 80), dtype=np.float32),
                 perception_camera__intrinsic=np.array([[100., 0., 40.], [0., 100., 40.], [0., 0., 1.]]),
                 perception_camera__world_to_camera=np.eye(4))

    def client(self, *, weak=False):
        return FakeClient(dict(visible=True, keypoints_px=[[40, 40], [50, 40], [40, 50]],
                               description="synthetic points", confidence=.1 if weak else .9))

    def test_two_fake_requests_save_review_artifacts_without_motion(self):
        client = self.client()
        report = check_grounding(self.archive, self.output, model="test-model", client=client)
        self.assertEqual(len(client.messages.requests), 2)
        for target in ("rack_peg", "mug_handle"):
            self.assertEqual(report["targets"][target]["status"], "geometry_passed_needs_review")
            self.assertTrue((self.output / f"{target}_proposal.png").exists())
        self.assertTrue(report["human_review_required"])
        self.assertFalse(report["support_verified"])
        self.assertFalse(report["motion"])
        self.assertEqual(json.loads((self.output / "results.json").read_text()), report)

    def test_rejected_geometry_preserves_proposals_and_never_overwrites(self):
        client = self.client(weak=True)
        report = check_grounding(self.archive, self.output, model="test-model", client=client)
        for row in report["targets"].values():
            self.assertEqual(row["validation_error"], "grounding_confidence_too_low")
            self.assertIn("proposal", row)
            self.assertNotIn("frame_pose_camera_derived", row)
        with self.assertRaises(FileExistsError):
            check_grounding(self.archive, self.output, model="test-model", client=client)
        self.assertEqual(len(client.messages.requests), 2)
