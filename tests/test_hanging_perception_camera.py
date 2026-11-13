"""Synthetic camera-contract coverage, not real peg/handle perception evidence."""

from types import SimpleNamespace
import unittest

from harness.environment.base import Observation
from harness.environment.layout import CAMERAS, PERCEPTION_CAMERAS
from harness.perception.anchors import AnchorRegistry
from harness.perception.grounding import RGBDPerceptionProvider
from harness.perception.hang_targets import StaticRackFrameCache
from harness.perception.opus_frame import OpusObjectFrameGrounder
from harness.perception.preparation import PerceptionAnchorPreparer
from harness.skills.schemas import SkillCommand
from tests.test_opus_frame_grounding import FakeClient
from tests.test_hang_targets import make_observation, report, RACK_POINTS, MUG_POINTS


class HangingPerceptionCameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = PERCEPTION_CAMERAS[0]
        base = make_observation()
        image = base.images["right_camera"]
        self.observation = Observation(0, {name: image.copy() for name in (*CAMERAS, self.camera)},
            base.command_state, {self.camera: base.camera_frames["right_camera"]})
        self.rack = {**report("rack_peg", 0, RACK_POINTS), "camera": self.camera}
        self.mug = {**report("mug_handle", 0, MUG_POINTS), "camera": self.camera}
        self.cache = StaticRackFrameCache()
        self.cache.cache(self.observation, self.rack)
        self.anchors = AnchorRegistry()
        self.command = SkillCommand.model_validate(dict(action="move_to", arm="right",
                                                        parameters=dict(target="rack_seat")))

    def preparer(self, camera=None):
        return PerceptionAnchorPreparer(
            RGBDPerceptionProvider(lambda *_a, **_k: None, self.anchors),
            SimpleNamespace(arm_state=lambda arm: {"pose": (0., 0., 1., 1., 0., 0., 0.)}),
            self.anchors, hang_target_cache=self.cache,
            mug_frame_grounder=SimpleNamespace(locate=lambda obs: self.mug),
            mug_frame_camera=self.camera if camera is None else camera)

    def test_extra_camera_reports_and_prepares_current_public_anchor(self):
        for target, points in (("rack_peg", RACK_POINTS), ("mug_handle", MUG_POINTS)):
            client = FakeClient(dict(visible=True, keypoints_px=points,
                                     description="synthetic material", confidence=.9))
            result = OpusObjectFrameGrounder(client, "test-model", target=target,
                                             camera=self.camera).locate(self.observation)
            self.assertEqual(result["camera"], self.camera)
            self.assertEqual(result["observation_sequence"], 0)
        anchor = self.preparer()(self.command, self.observation)
        self.assertEqual(anchor.source, "perception")
        self.assertEqual(anchor.observation_sequence, 0)

    def test_missing_rgbd_is_rejected_without_binding_targets(self):
        no_depth = Observation(0, self.observation.images, self.observation.command_state)
        with self.assertRaisesRegex(ValueError, "grounding_rgbd_camera_unavailable"):
            StaticRackFrameCache().cache(no_depth, self.rack)
        with self.assertRaisesRegex(ValueError, "rack_frame_camera_unavailable"):
            self.preparer()(self.command, no_depth)
        with self.assertRaisesRegex(ValueError, "anchor_missing_or_stale"):
            self.anchors.resolve("rack_seat", "right", self.observation)

    def test_wrong_camera_and_unknown_camera_remain_rejected(self):
        self.mug = {**self.mug, "camera": "right_camera"}
        with self.assertRaisesRegex(ValueError, "mug_frame_camera_mismatch"):
            self.preparer()(self.command, self.observation)
        with self.assertRaisesRegex(ValueError, "invalid_hang_frame_camera"):
            self.preparer("invented_camera")
        with self.assertRaisesRegex(ValueError, "unsupported_frame_grounding_target"):
            OpusObjectFrameGrounder(FakeClient({}), "test-model",
                                    target="mug_handle", camera="invented_camera")
