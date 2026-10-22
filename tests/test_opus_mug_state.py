"""CPU contract tests for the structured Opus mug-state assessor."""

from types import SimpleNamespace
import unittest

from harness.environment.base import Observation
from harness.perception.opus_mug_state import OpusMugStateAssessor
from tests.helpers import images


def valid_result(phase, cameras):
    return {"phase": phase, "visual_state": "mug_overlapping_gripper",
        "confidence": .8, "evidence": "Mug appears to overlap a gripper.",
        "views": [{"camera": camera, "mug_visibility": "visible",
            "mug_bbox_px": [0, 0, 2, 2],
            "gripper_visibility": "visible", "gripper_bbox_px": [1, 0, 3, 2],
            "evidence": "The boxes overlap in this view."} for camera in cameras]}


class FakeMessages:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
            name="report_mug_visual_state", input=self.result)])


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)


class OpusMugStateTests(unittest.TestCase):
    def test_grasp_assessment_uses_head_and_left_views_and_computes_overlap(self):
        result = valid_result("grasp", ("head_camera", "left_camera"))
        client = FakeClient(result)
        assessor = OpusMugStateAssessor(client, "claude-test-model")
        observation = Observation(7, images(), (0.0,) * 14)

        assessed = assessor.assess(observation, phase="grasp")

        self.assertEqual([view["camera"] for view in assessed["views"]],
                         ["head_camera", "left_camera"])
        self.assertAlmostEqual(assessed["views"][0]["gripper_overlap_fraction_of_mug"], .5)
        self.assertEqual(assessed["visual_state"], "mug_overlapping_gripper")
        self.assertNotIn("left_gripper_bbox_px", assessed["views"][0])
        request = client.messages.requests[0]
        self.assertEqual(request["tool_choice"], {
            "type": "tool", "name": "report_mug_visual_state"})
        self.assertEqual(sum(part["type"] == "image"
                             for part in request["messages"][0]["content"]), 2)

    def test_phase_camera_sets_include_both_wrist_views_during_handoff(self):
        cameras = ("head_camera", "left_camera", "right_camera")
        assessor = OpusMugStateAssessor(FakeClient(valid_result("handoff", cameras)),
                                        "claude-test-model")
        assessed = assessor.assess(Observation(3, images(), (0.0,) * 14), phase="handoff")
        self.assertEqual([view["camera"] for view in assessed["views"]], list(cameras))

    def test_assessment_can_use_head_camera_only(self):
        result = valid_result("grasp", ("head_camera",))
        assessor = OpusMugStateAssessor(FakeClient(result), "claude-test-model")
        assessed = assessor.assess(Observation(3, images(), (0.,) * 14),
                                   phase="grasp", cameras=("head_camera",))
        self.assertEqual([view["camera"] for view in assessed["views"]], ["head_camera"])

    def test_bad_camera_set_phase_and_bounding_boxes_are_rejected(self):
        cameras = ("head_camera", "right_camera")
        result = valid_result("hang", cameras)
        result["views"][1]["mug_bbox_px"] = [0, 0, 4, 2]
        assessor = OpusMugStateAssessor(FakeClient(result), "claude-test-model")
        observation = Observation(5, images(), (0.0,) * 14)
        with self.assertRaisesRegex(ValueError, "assessment_bbox"):
            assessor.assess(observation, phase="hang")
        with self.assertRaisesRegex(ValueError, "unsupported_mug_assessment_phase"):
            assessor.assess(observation, phase="unknown")

    def test_single_frame_overlap_never_returns_a_grasp_confirmation_field(self):
        cameras = ("head_camera", "left_camera")
        assessor = OpusMugStateAssessor(FakeClient(valid_result("grasp", cameras)),
                                        "claude-test-model")
        assessed = assessor.assess(Observation(6, images(), (0.0,) * 14), phase="grasp")
        self.assertNotIn("grasp_confirmed", assessed)
        self.assertNotIn("mug_lifted", assessed)


if __name__ == "__main__":
    unittest.main()
