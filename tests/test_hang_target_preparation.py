"""Real hang preparation and RGB-D geometry through runtime rejection handling."""

from types import SimpleNamespace
import unittest

import numpy as np

from harness.perception.anchors import AnchorRegistry
from harness.perception.grounding import RGBDPerceptionProvider
from harness.perception.hang_targets import StaticRackFrameCache
from harness.perception.preparation import PerceptionAnchorPreparer
from harness.perception.rgbd import RGBDCamera
from harness.runtime.orchestrator import grounding_rejection_code
from tests.hanging_scenario import HangingScenario
from tests.test_hang_targets import CAMERA, RACK_POINTS, MUG_POINTS, make_observation, report


class HangPreparationTests(unittest.TestCase):
    def scenario(self, mode):
        scenario = HangingScenario()
        cache = StaticRackFrameCache()
        cache.cache(make_observation(), report("rack_peg", 0, RACK_POINTS))
        if mode == "cache_from_future":
            cache.cache(make_observation(4), report("rack_peg", 4, RACK_POINTS))
        if mode == "unreachable":
            cache.cache(make_observation(camera_motion=1.0), report("rack_peg", 0, RACK_POINTS))
        if mode == "missing_rack":
            cache.clear()

        def locate(observation):
            points = MUG_POINTS
            if mode == "degenerate_x":
                points = ((20, 20), (20, 20), (20, 30))
            if mode == "degenerate_plane":
                points = ((20, 20), (30, 20), (40, 20))
            result = report("mug_handle", observation.sequence, points)
            if mode == "wrong_camera":
                result["camera"] = "head_camera"
            if mode == "stale_mug":
                result["observation_sequence"] += 1
            if mode == "weak_confidence":
                result["confidence"] = .1
            return result

        provider = RGBDPerceptionProvider(lambda *_a, **_k: None, scenario.anchors)
        preparer = PerceptionAnchorPreparer(provider, scenario.environment, scenario.anchors,
            hang_target_cache=cache, mug_frame_grounder=SimpleNamespace(locate=locate),
            mug_frame_camera=CAMERA)

        def prepare(command, observation):
            current = make_observation(observation.sequence)
            if mode == "changed_calibration":
                current = make_observation(observation.sequence, calibration_shift=1.)
            if mode == "depth_discontinuity":
                camera = current.camera_frames[CAMERA]
                depth = camera.depth_m.copy()
                depth[19:22, 19:22] = 1.5
                current.camera_frames[CAMERA] = RGBDCamera(
                    depth, camera.intrinsic, camera.world_to_camera)
            if mode == "missing_depth":
                camera = current.camera_frames[CAMERA]
                current.camera_frames[CAMERA] = RGBDCamera(
                    np.zeros_like(camera.depth_m), camera.intrinsic, camera.world_to_camera)
            return preparer(command, current)

        scenario.prepare = prepare
        return scenario

    def test_actual_preparation_failures_return_feedback_before_motion(self):
        for mode in ("degenerate_x", "degenerate_plane", "missing_depth", "wrong_camera",
                     "stale_mug", "missing_rack", "cache_from_future", "unreachable"):
            with self.subTest(mode=mode):
                scenario = self.scenario(mode)
                summary = scenario.run()
                self.assertIsNone(summary["error_type"])
                self.assertEqual(summary["outcome"], "scenario_grounding_rejected")
                self.assertEqual(scenario.environment.ticks, 0)
                self.assertIsNotNone(scenario.contexts[1].grounding_feedback)
                rejected = next(e for e in scenario.assessments if e.get("grounding_rejected"))
                self.assertFalse(rejected["skill_executed"])

    def test_weak_or_inconsistent_synthetic_rgbd_returns_feedback_before_motion(self):
        # Public input contract only: these arrays are not captured scene evidence.
        for mode in ("weak_confidence", "depth_discontinuity", "changed_calibration"):
            with self.subTest(mode=mode):
                scenario = self.scenario(mode)
                summary = scenario.run()
                self.assertIsNone(summary["error_type"])
                self.assertEqual(summary["outcome"], "scenario_grounding_rejected")
                self.assertEqual(scenario.environment.ticks, 0)
                self.assertIsNotNone(scenario.contexts[1].grounding_feedback)
                self.assertFalse(any(e.get("skill_executed") for e in scenario.assessments))

    def test_unrecognized_error_is_not_silently_treated_as_grounding_failure(self):
        command = HangingScenario().commands[0]
        self.assertIsNone(grounding_rejection_code(command, "invalid_measured_robot_state"))

    def test_preparer_reads_current_measured_tcp_for_each_new_observation(self):
        cache = StaticRackFrameCache()
        cache.cache(make_observation(), report("rack_peg", 0, RACK_POINTS))
        anchors = AnchorRegistry()
        provider = RGBDPerceptionProvider(lambda *_a, **_k: None, anchors)
        measured = np.array([0., 0., 1., 1., 0., 0., 0.])
        environment = SimpleNamespace(arm_state=lambda arm: {"pose": measured.copy()})
        prepare = PerceptionAnchorPreparer(provider, environment, anchors,
            hang_target_cache=cache, mug_frame_grounder=SimpleNamespace(
                locate=lambda obs: report("mug_handle", obs.sequence, MUG_POINTS)),
            mug_frame_camera=CAMERA)
        command = self.scenario("valid").commands[1]
        first = prepare(command, make_observation())
        measured[0] += .04
        second = prepare(command, make_observation(1))
        np.testing.assert_allclose(np.array(second.pose[:3])-first.pose[:3], [.04, 0., 0.], atol=1e-12)
        self.assertEqual(second.observation_sequence, 1)
        with self.assertRaisesRegex(ValueError, "anchor_missing_or_stale"):
            anchors.resolve("rack_seat", "right", make_observation())
