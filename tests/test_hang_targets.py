"""CPU tests for episode rack-frame caching and grasp-aware hang targets."""

import unittest
from types import SimpleNamespace

import numpy as np

from harness.environment.base import Observation
from harness.perception.anchors import AnchorRegistry
from harness.perception.hang_targets import StaticRackFrameCache, bind_hang_targets
from harness.perception.grounding import (Detection, DetectionBatch,
                                          RGBDPerceptionProvider)
from harness.perception.preparation import PerceptionAnchorPreparer
from harness.perception.rgbd import RGBDCamera
from harness.skills.schemas import SkillCommand


CAMERA = "right_camera"


def make_observation(sequence=0, *, calibration_shift=0.0, camera_motion=0.0,
                     surface_depth=1.0):
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    depth = np.full((80, 80), surface_depth, dtype=np.float32)
    intrinsic = np.array([[100., 0., 40.], [0., 100., 40.], [0., 0., 1.]])
    intrinsic[0, 2] += calibration_shift
    extrinsic = np.eye(4)
    extrinsic[0, 3] = camera_motion
    camera = RGBDCamera(depth, intrinsic, extrinsic)
    return Observation(sequence, {CAMERA: image}, (0.0,) * 14,
                       {CAMERA: camera})


def mask_for(points):
    mask = np.zeros((80, 80), dtype=bool)
    for u, v in points:
        mask[max(0, v-1):v+2, max(0, u-1):u+2] = True
    return mask


def report(target, sequence, points):
    return {"status": "grounded", "target": target, "camera": CAMERA,
            "observation_sequence": sequence, "image_shape_px": [80, 80],
            "confidence": .9, "keypoints_px": [list(p) for p in points]}


class Detector:
    def __init__(self):
        self.calls = 0

    def detect(self, images, *, observation_sequence):
        self.calls += 1
        return DetectionBatch(observation_sequence, CAMERA, (
            Detection("rack", .95, mask_for(RACK_POINTS)),
            Detection("mug", .95, mask_for(MUG_POINTS))))


class MugGrounder:
    def locate(self, observation):
        return report("mug_handle", observation.sequence, MUG_POINTS)


RACK_POINTS = ((40, 40), (50, 40), (40, 50))
MUG_POINTS = ((20, 20), (30, 20), (20, 30))


class HangTargetTests(unittest.TestCase):
    def setUp(self):
        self.obs = make_observation()
        self.cache = StaticRackFrameCache()
        self.cache.cache(self.obs, report("rack_peg", 0, RACK_POINTS))

    def test_rack_frame_resolves_when_later_view_is_occluded(self):
        later = make_observation(1)
        np.testing.assert_allclose(self.cache.resolve(later),
                                   self.cache.resolve(self.obs))
        self.assertEqual(self.cache.metadata()["observation_sequence"], 0)

    def test_camera_calibration_change_invalidates_episode_cache(self):
        with self.assertRaisesRegex(ValueError, "rack_frame_camera_changed"):
            self.cache.resolve(make_observation(1, calibration_shift=1.0))

    def test_camera_motion_preserves_cached_world_rack_frame(self):
        # Grounding already produced a world pose at episode start; a moving
        # wrist camera must not invalidate the static rack reference.
        moved = make_observation(1, camera_motion=0.25)
        np.testing.assert_allclose(self.cache.resolve(moved),
                                   self.cache.resolve(self.obs))

    def test_cached_rack_world_position_comes_from_reset_frame_depth(self):
        deeper_observation = make_observation(surface_depth=1.12)
        deeper_cache = StaticRackFrameCache()
        deeper_cache.cache(deeper_observation, report("rack_peg", 0, RACK_POINTS))

        reset_pose = self.cache.resolve(self.obs)
        deeper_pose = deeper_cache.resolve(deeper_observation)
        np.testing.assert_allclose(deeper_pose[:3] - reset_pose[:3],
                                   [0., 0., .12], atol=1e-6)

    def test_cache_requires_current_rack_keypoints_and_rgbd(self):
        with self.assertRaisesRegex(ValueError, "rack_frame_report_required"):
            StaticRackFrameCache().cache(self.obs, report("mug_handle", 0, RACK_POINTS))
        stale = report("rack_peg", 3, RACK_POINTS)
        with self.assertRaisesRegex(ValueError, "stale_or_wrong_camera"):
            StaticRackFrameCache().cache(self.obs, stale)

    def test_cache_rejects_keypoints_on_a_discontinuous_depth_surface(self):
        obs = make_observation()
        camera = obs.camera_frames[CAMERA]
        depth = camera.depth_m.copy()
        depth[39:42, 39:42] = 1.5
        obs.camera_frames[CAMERA] = RGBDCamera(depth, camera.intrinsic,
                                                camera.world_to_camera)
        with self.assertRaisesRegex(ValueError, "keypoint_depth_discontinuity"):
            StaticRackFrameCache().cache(obs, report("rack_peg", 0, RACK_POINTS))

    def test_binds_targets_from_cached_rack_current_mug_and_measured_tcp(self):
        anchors = AnchorRegistry()
        tcp = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        result = bind_hang_targets(
            self.cache, self.obs, report("mug_handle", 0, MUG_POINTS),
            tcp_pose=tcp, anchors=anchors, arm="right")
        self.assertEqual(set(result), {"rack_approach", "rack_seat"})
        self.assertGreater(result["rack_approach"].pose[0], result["rack_seat"].pose[0])
        for anchor in result.values():
            self.assertEqual(anchor.observation_sequence, 0)
            self.assertEqual(anchor.source, "perception")

    def test_rejects_wrong_or_stale_mug_frame(self):
        args = (self.cache, self.obs, report("rack_peg", 0, MUG_POINTS))
        with self.assertRaisesRegex(ValueError, "mug_handle_report_required"):
            bind_hang_targets(*args, tcp_pose=(0., 0., 1., 1., 0., 0., 0.),
                              anchors=AnchorRegistry(), arm="right")
        stale = report("mug_handle", 4, MUG_POINTS)
        with self.assertRaisesRegex(ValueError, "stale_or_wrong_camera"):
            bind_hang_targets(self.cache, self.obs, stale,
                              tcp_pose=(0., 0., 1., 1., 0., 0., 0.),
                              anchors=AnchorRegistry(), arm="right")

    def test_move_to_preparer_preserves_same_observation_hang_anchor(self):
        anchors = AnchorRegistry()
        tcp = (0., 0., 1., 1., 0., 0., 0.)
        bind_hang_targets(self.cache, self.obs, report("mug_handle", 0, MUG_POINTS),
                          tcp_pose=tcp, anchors=anchors,
                          arm="right")
        provider = RGBDPerceptionProvider(lambda *args, **kwargs: None, anchors)
        environment = SimpleNamespace(arm_state=lambda arm: {"pose": tcp})
        prepare = PerceptionAnchorPreparer(provider, environment, anchors)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_seat"}})

        resolved = prepare(command, self.obs)

        self.assertEqual(resolved, anchors.resolve("rack_seat", "right", self.obs))

    def test_move_to_rejects_unprepared_rack_seat(self):
        anchors = AnchorRegistry()
        provider = RGBDPerceptionProvider(lambda *args, **kwargs: None, anchors)
        prepare = PerceptionAnchorPreparer(provider,
            SimpleNamespace(arm_state=lambda arm: {"pose": (0., 0., 1., 1., 0., 0., 0.)}),
            anchors)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_seat"}})
        with self.assertRaisesRegex(ValueError, "hang_target_not_prepared"):
            prepare(command, self.obs)

    def test_runtime_preparer_regrounds_mug_and_binds_cached_rack_composition(self):
        anchors = AnchorRegistry()
        detector = Detector()
        provider = RGBDPerceptionProvider(detector, anchors)
        provider.set_static_rack_frame_cache(self.cache)
        environment = SimpleNamespace(arm_state=lambda arm: {
            "pose": (0., 0., 1., 1., 0., 0., 0.)})
        prepare = PerceptionAnchorPreparer(provider, environment, anchors,
            hang_target_cache=self.cache, mug_frame_grounder=MugGrounder(),
            mug_frame_camera=CAMERA)
        command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_seat"}})

        anchor = prepare(command, self.obs)

        self.assertEqual(detector.calls, 0)
        self.assertEqual(anchor, anchors.resolve("rack_seat", "right", self.obs))
        summary = provider.planner_summary(self.obs)
        self.assertTrue(summary["rack_static_reference"]["world_anchor_available"])
        self.assertNotIn("world_position_xyz", summary["rack_static_reference"])


if __name__ == "__main__":
    unittest.main()
