"""CPU contracts for reviewed planner-pixel projection replay."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from development.replays.pixel_grounding import replay, replay_reselection
from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from harness.perception.rgbd import RGBDCamera
from tests.helpers import images


class PixelGroundingReplayTests(unittest.TestCase):
    def test_replay_unprojects_reviewed_pixel_with_saved_same_frame_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObservationStore(root / "observations")
            frame = RGBDCamera(np.full((2, 3), .8, dtype=np.float32),
                np.array([[2., 0., 1.], [0., 2., .5], [0., 0., 1.]]), np.eye(4))
            store.save(Observation(7, images(), (0.,) * 14, {"head_camera": frame}))
            annotations = root / "pixels.json"
            annotations.write_text(json.dumps({"schema_version": 1, "points": [{
                "sequence": 7, "camera": "head_camera", "uv": [.5, .5]}]}))

            report = replay(root / "observations", annotations)

            self.assertEqual(report["summary"], {"annotated_points": 1, "grounded_points": 1,
                                                  "foreground_supported_points": 1})
            self.assertAlmostEqual(report["points"][0]["sampled_depth_m"], .8)
            np.testing.assert_allclose(report["points"][0]["world_xyz_m"], [0., 0., .8])
            self.assertFalse(report["robot_motion"])
            self.assertFalse(report["simulator_started"])
            self.assertFalse(report["planner_or_detector_executed"])

    def test_rgb_only_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ObservationStore(root / "observations")
            store.save(Observation(3, images(), (0.,) * 14))
            annotations = root / "pixels.json"
            annotations.write_text(json.dumps({"schema_version": 1, "points": [{
                "sequence": 3, "camera": "head_camera", "uv": [.4, .6]}]}))

            with self.assertRaisesRegex(ValueError, "rgbd_camera_frame_unavailable"):
                replay(root / "observations", annotations)

    def test_replay_reports_background_depth_as_not_foreground_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations"
            store = ObservationStore(observations)
            depth = np.full((9, 9), .74, dtype=np.float32)
            depth[4, 0] = .60
            rgb = {name: np.zeros((9, 9, 3), dtype=np.uint8) for name in images()}
            frame = RGBDCamera(depth,
                np.array([[4., 0., 4.], [0., 4., 4.], [0., 0., 1.]]), np.eye(4))
            store.save(Observation(8, rgb, (0.,) * 14, {"head_camera": frame}))
            annotations = root / "pixels.json"
            annotations.write_text(json.dumps({"schema_version": 1, "points": [{
                "sequence": 8, "camera": "head_camera", "uv": [.5, .5]}]}))

            report = replay(observations, annotations)

            self.assertEqual(report["points"][0]["foreground_supported"], False)
            self.assertGreater(report["points"][0]["local_depth_gap_m"], .06)
            self.assertEqual(report["summary"]["foreground_supported_points"], 0)

    def test_recorded_frame_reselection_accepts_second_pixel_or_exhausts_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations"
            store = ObservationStore(observations)
            depth = np.full((9, 9), .74, dtype=np.float32)
            depth[0:2, 3:6] = .60
            rgb = {name: np.zeros((9, 9, 3), dtype=np.uint8) for name in images()}
            frame = RGBDCamera(depth,
                np.array([[4., 0., 4.], [0., 4., 4.], [0., 0., 1.]]), np.eye(4))
            store.save(Observation(8, rgb, (0.,) * 14, {"head_camera": frame}))
            scenarios = root / "scenarios.json"
            tcp = [0., -.5, .4, 1., 0., 0., 0.]
            bad = {"camera": "head_camera", "uv": [.5, .5]}
            good = {"camera": "head_camera", "uv": [.5, 0.]}
            scenarios.write_text(json.dumps({"schema_version": 1, "scenarios": [
                {"name": "reselect_to_supported_wall", "sequence": 8, "arm": "left",
                 "measured_tcp_pose": tcp, "pixels": [bad, good]},
                {"name": "three_bad_pixels_stop", "sequence": 8, "arm": "left",
                 "measured_tcp_pose": tcp, "pixels": [bad, bad, bad]}]}))

            report = replay_reselection(observations, scenarios)

            first = report["scenarios"][0]
            self.assertEqual([item["outcome"] for item in first["selections"]],
                             ["rejected_depth_inconsistent", "accepted"])
            self.assertEqual(first["selections"][1]["anchor_source"], "perception")
            second = report["scenarios"][1]
            self.assertEqual([item["outcome"] for item in second["selections"]],
                             ["rejected_depth_inconsistent"] * 3)
            self.assertTrue(second["reselection_budget_exhausted"])
            self.assertEqual(report["summary"], {"scenarios": 2,
                "accepted_after_reselection": 1, "budget_exhausted": 1})
            self.assertFalse(report["robot_motion"])
            self.assertFalse(report["skill_executor_executed"])


if __name__ == "__main__":
    unittest.main()
