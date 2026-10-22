"""Planner-selected image pixels are bounded and resolve through fresh depth."""

from types import SimpleNamespace
import unittest

import numpy as np

from harness.environment.base import Observation
from harness.perception.anchors import AnchorRegistry
from harness.perception.preparation import PlannerPixelAnchorPreparer
from harness.perception.rgbd import RGBDCamera
from harness.skills.schemas import GroundingPixel, SkillCommand
from tests.helpers import images


def observation(sequence=7, *, include_depth=True, depth_m=.50):
    frames = None
    if include_depth:
        frames = {name: RGBDCamera(
            np.full((2, 3), depth_m, dtype=np.float32),
            np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]),
            np.eye(4)) for name in images()}
    return Observation(sequence, images(), (0.,) * 14, frames)


class PlannerPixelGroundingTests(unittest.TestCase):
    def test_hybrid_preparer_uses_explicit_pixels_and_keeps_semantic_fallback(self):
        from harness.perception.preparation import HybridAnchorPreparer
        calls = []
        hybrid = HybridAnchorPreparer(
            lambda command, _obs: calls.append(("semantic", command.parameters.target)),
            lambda command, _obs: calls.append(("pixel", command.parameters.target)))
        obs = observation()
        semantic_command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_approach"}})
        pixel_command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_approach", "grounding": {
                "camera": "right_camera", "uv": [.5, .5]}}})
        hybrid(semantic_command, obs)
        hybrid(pixel_command, obs)
        self.assertEqual(calls, [("semantic", "rack_approach"), ("pixel", "rack_approach")])

    def test_normalized_pixel_binds_same_frame_bounded_anchor_from_measured_tcp(self):
        anchors = AnchorRegistry()
        measured = []
        environment = SimpleNamespace(arm_state=lambda arm: measured.append(arm) or {
            "pose": (0., 0., 0., 1., 0., 0., 0.)})
        prepare = PlannerPixelAnchorPreparer(environment, anchors)
        obs = observation()
        command = SkillCommand.model_validate({"action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp", "grounding": {
                "camera": "head_camera", "uv": [.5, .5]}}})
        anchor = prepare(command, obs)
        self.assertEqual(measured, ["left"])
        self.assertEqual(anchor.observation_sequence, obs.sequence)
        self.assertEqual(anchor.source, "perception")
        np.testing.assert_allclose(anchor.pose[:3], (0., 0., .38))
        self.assertEqual(anchors.resolve("mug_pregrasp", "left", obs), anchor)

    def test_rack_pixel_binds_fresh_rack_approach_from_selected_camera(self):
        anchors = AnchorRegistry()
        environment = SimpleNamespace(arm_state=lambda _arm: {
            "pose": (0., 0., 0., 1., 0., 0., 0.)})
        prepare = PlannerPixelAnchorPreparer(environment, anchors)
        obs = observation()
        command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_approach", "grounding": {
                "camera": "right_camera", "uv": [.5, .5]}}})
        anchor = prepare(command, obs)
        self.assertEqual(anchor.observation_sequence, obs.sequence)
        self.assertEqual(anchor.source, "perception")
        np.testing.assert_allclose(anchor.pose[:3], (0., 0., .38))
        self.assertEqual(anchors.resolve("rack_approach", "right", obs), anchor)

    def test_requires_rgbd_valid_target_depth_and_reachable_target(self):
        anchors = AnchorRegistry()
        environment = SimpleNamespace(arm_state=lambda _arm: {
            "pose": (0., 0., 0., 1., 0., 0., 0.)})
        prepare = PlannerPixelAnchorPreparer(environment, anchors)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp", "grounding": {
                "camera": "head_camera", "uv": [.5, .5]}}})
        with self.assertRaisesRegex(ValueError, "camera_frame_unavailable"):
            prepare(command, observation(include_depth=False))
        with self.assertRaisesRegex(ValueError, "depth_unavailable"):
            prepare(command, observation(depth_m=0.))
        with self.assertRaisesRegex(ValueError, "target_out_of_reach"):
            prepare(command, observation(depth_m=1.))

    def test_rejects_depth_visible_through_opening_before_binding_anchor(self):
        depth = np.full((9, 9), .74, dtype=np.float32)
        depth[4, 0] = .60  # Nearby mug wall is much closer than the selected opening.
        obs_images = {name: np.zeros((9, 9, 3), dtype=np.uint8)
                      for name in images()}
        obs = Observation(7, obs_images, (0.,) * 14, {"head_camera": RGBDCamera(
            depth, np.array([[4., 0., 4.], [0., 4., 4.], [0., 0., 1.]]), np.eye(4))})
        anchors = AnchorRegistry()
        environment = SimpleNamespace(arm_state=lambda _arm: {
            "pose": (0., 0., 0., 1., 0., 0., 0.)})
        prepare = PlannerPixelAnchorPreparer(environment, anchors)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "left",
            "parameters": {"target": "mug_pregrasp", "grounding": {
                "camera": "head_camera", "uv": [.5, .5]}}})

        with self.assertRaisesRegex(ValueError, "grounding_pixel_behind_local_surface"):
            prepare(command, obs)
        with self.assertRaisesRegex(ValueError, "anchor_missing_or_stale"):
            anchors.resolve("mug_pregrasp", "left", obs)

    def test_pixel_schema_is_normalized_and_only_valid_for_symbolic_pregrasp(self):
        for uv in ((-0.01, .5), (.5, 1.01), (float("nan"), .5)):
            with self.subTest(uv=uv), self.assertRaises(ValueError):
                GroundingPixel(camera="head_camera", uv=uv)
        with self.assertRaises(ValueError):
            GroundingPixel(camera="wrist_camera", uv=(.5, .5))
        command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_approach", "grounding": {
                "camera": "head_camera", "uv": [.5, .5]}}})
        self.assertIsNotNone(command.parameters.grounding)
        with self.assertRaisesRegex(ValueError, "grounding_pixel_requires_perception_target"):
            SkillCommand.model_validate({"action": "move_to", "arm": "left",
                "parameters": {"target": "current_ee", "grounding": {
                    "camera": "head_camera", "uv": [.5, .5]}}})


if __name__ == "__main__":
    unittest.main()
