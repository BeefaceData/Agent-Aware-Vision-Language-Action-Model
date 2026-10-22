"""CPU contracts for Opus image-only object-frame keypoint reports."""

from types import SimpleNamespace
import unittest

import numpy as np

from harness.environment.base import Observation
from harness.perception.opus_frame import (
    OpusObjectFrameGrounder, ground_public_object_frame,
)
from harness.perception.rgbd import RGBDCamera
from tests.helpers import images


class FakeMessages:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=[SimpleNamespace(
            type="tool_use", name="report_object_frame_keypoints", input=self.result)])


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)


def observation():
    return Observation(8, images(), (0.0,) * 14)


class OpusFrameGroundingTests(unittest.TestCase):
    def test_reports_normalized_context_and_three_ordered_pixels(self):
        client = FakeClient({"visible": True,
            "keypoints_px": [[0, 0], [1, 0], [2, 1]],
            "description": "visible peg frame", "confidence": .8})
        grounder = OpusObjectFrameGrounder(client, "claude-test-model",
            target="rack_peg", camera="head_camera")

        result = grounder.locate(observation())

        self.assertEqual(result["status"], "grounded")
        self.assertEqual(result["keypoints_px"], [[0., 0.], [1., 0.], [2., 1.]])
        request = client.messages.requests[0]
        self.assertEqual(request["tool_choice"], {
            "type": "tool", "name": "report_object_frame_keypoints"})
        self.assertIn("rack_frame_grounding.md", grounder.PROMPTS["rack_peg"])

    def test_unavailable_and_ambiguous_landmarks_fail_closed(self):
        missed = FakeClient({"visible": False, "keypoints_px": None,
            "description": "occluded", "confidence": .1})
        result = OpusObjectFrameGrounder(missed, "claude-test-model",
            target="mug_handle", camera="right_camera").locate(observation())
        self.assertEqual(result["status"], "not_detected")
        self.assertNotIn("keypoints_px", result)

        ambiguous = FakeClient({"visible": True, "keypoints_px": None,
            "description": "handle hidden", "confidence": .2})
        with self.assertRaisesRegex(ValueError, "requires_three_keypoints"):
            OpusObjectFrameGrounder(ambiguous, "claude-test-model",
                target="mug_handle", camera="right_camera").locate(observation())

    def test_invalid_or_out_of_view_keypoints_and_bad_configuration_are_rejected(self):
        outside = FakeClient({"visible": True,
            "keypoints_px": [[0, 0], [3, 0], [2, 1]],
            "description": "out of view", "confidence": .8})
        with self.assertRaisesRegex(ValueError, "keypoint_out_of_bounds"):
            OpusObjectFrameGrounder(outside, "claude-test-model",
                target="rack_peg", camera="head_camera").locate(observation())
        bad = FakeClient({"visible": True,
            "keypoints_px": [[0, 0], [1, 0], [float("nan"), 1]],
            "description": "bad coordinates", "confidence": .8})
        with self.assertRaisesRegex(ValueError, "keypoints_invalid"):
            OpusObjectFrameGrounder(bad, "claude-test-model",
                target="rack_peg", camera="head_camera").locate(observation())
        with self.assertRaisesRegex(ValueError, "unsupported_frame_grounding_target"):
            OpusObjectFrameGrounder(FakeClient({}), "claude-test-model",
                target="rack", camera="head_camera")

    def test_world_frame_requires_same_observation_rgbd_and_mask_support(self):
        shape = (80, 80)
        images_by_camera = {name: np.zeros((*shape, 3), dtype=np.uint8)
                            for name in ("head_camera", "left_camera", "right_camera")}
        rgbd = RGBDCamera(np.ones(shape, dtype=np.float32),
            np.array([[100., 0., 40.], [0., 100., 40.], [0., 0., 1.]]), np.eye(4))
        observation = Observation(8, images_by_camera, (0.,) * 14,
                                  {"head_camera": rgbd})
        report = {"status": "grounded", "camera": "head_camera",
            "observation_sequence": 8, "image_shape_px": list(shape), "confidence": .8,
            "keypoints_px": [[40., 40.], [42., 40.], [40., 42.]]}
        mask = np.zeros(shape, dtype=bool)
        mask[39:43, 39:43] = True

        pose = ground_public_object_frame(observation, report, mask)

        np.testing.assert_allclose(pose[:3], [0., 0., 1.], atol=1e-7)
        with self.assertRaisesRegex(ValueError, "outside_object_mask"):
            ground_public_object_frame(observation, report,
                                       np.pad(np.ones((2, 2), dtype=bool),
                                              ((0, 78), (0, 78))))
        with self.assertRaisesRegex(ValueError, "stale_or_wrong_camera"):
            ground_public_object_frame(observation, {**report,
                "observation_sequence": 7}, mask)

    def test_world_frame_snaps_thin_structure_keypoint_to_nearby_mask_depth(self):
        shape = (80, 80)
        images_by_camera = {name: np.zeros((*shape, 3), dtype=np.uint8)
                            for name in ("head_camera", "left_camera", "right_camera")}
        depth = np.ones(shape, dtype=np.float32)
        # The semantic pixel looks through an opening onto a farther surface.
        depth[29:32, 29:32] = 1.4
        rgbd = RGBDCamera(depth,
            np.array([[100., 0., 40.], [0., 100., 40.], [0., 0., 1.]]), np.eye(4))
        observation = Observation(8, images_by_camera, (0.,) * 14,
                                  {"right_camera": rgbd})
        report = {"status": "grounded", "camera": "right_camera",
            "observation_sequence": 8, "image_shape_px": list(shape), "confidence": .8,
            "keypoints_px": [[30., 30.], [40., 30.], [30., 40.]]}
        mask = np.zeros(shape, dtype=bool)
        mask[29:32, 33:36] = True
        mask[29:32, 40:43] = True
        mask[40:43, 29:32] = True

        pose = ground_public_object_frame(observation, report, mask)

        self.assertTrue(np.isfinite(pose).all())
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0)

    def test_world_frame_rejects_low_confidence_and_missing_rgbd(self):
        shape = (80, 80)
        images_by_camera = {name: np.zeros((*shape, 3), dtype=np.uint8)
                            for name in ("head_camera", "left_camera", "right_camera")}
        observation = Observation(8, images_by_camera, (0.,) * 14)
        report = {"status": "grounded", "camera": "head_camera",
            "observation_sequence": 8, "image_shape_px": list(shape), "confidence": .2,
            "keypoints_px": [[40., 40.], [42., 40.], [40., 42.]]}
        mask = np.ones(shape, dtype=bool)
        with self.assertRaisesRegex(ValueError, "confidence_too_low"):
            ground_public_object_frame(observation, report, mask)
        with self.assertRaisesRegex(ValueError, "rgbd_camera_unavailable"):
            ground_public_object_frame(observation, {**report, "confidence": .9}, mask)


if __name__ == "__main__":
    unittest.main()
