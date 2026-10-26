from types import SimpleNamespace
import unittest

import numpy as np

from harness.environment.base import Observation
from harness.perception.rgbd import RGBDCamera
from harness.perception.table_plane import (
    MugLiftEvidenceTracker, OpusMugSupportGrounder, StaticTablePlaneCache,
)


def observation(sequence, *, lifted=False):
    shape = (48, 48)
    images = {name: np.zeros((*shape, 3), dtype=np.uint8)
              for name in ("head_camera", "left_camera", "right_camera")}
    depth = np.ones(shape, dtype=np.float32)
    if lifted:
        depth[19:28, 19:28] = 0.95
    intrinsic = np.array([[60., 0., 24.], [0., 60., 24.], [0., 0., 1.]])
    # Camera looks downward: a closer pixel maps to a higher world Z.
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.diag([1., -1., -1.])
    extrinsic[2, 3] = 1.
    frame = RGBDCamera(depth, intrinsic, extrinsic)
    return Observation(sequence, images, (0.,) * 14, {"head_camera": frame})


class FakeGrounder:
    def __init__(self, reports):
        self.reports = iter(reports)
        self.calls = 0

    def locate(self, obs):
        self.calls += 1
        return next(self.reports)


class MugSupportTests(unittest.TestCase):
    def test_plane_cache_measures_table_and_lifted_support_points(self):
        cache = StaticTablePlaneCache()
        reset = observation(0)
        metadata = cache.cache(reset, np.ones((48, 48), dtype=bool))
        self.assertTrue(metadata["available"])
        self.assertLess(metadata["p95_residual_m"], 1e-6)
        table_distance = cache.signed_distance(reset, [23, 23])
        raised_distance = cache.signed_distance(observation(1, lifted=True), [23, 23])
        self.assertAlmostEqual(table_distance, 0., places=5)
        self.assertAlmostEqual(raised_distance, .05, places=4)

    def test_lift_requires_two_fresh_support_observations_and_latches(self):
        cache = StaticTablePlaneCache()
        reset = observation(0)
        cache.cache(reset, np.ones((48, 48), dtype=bool))
        reports = [{"status": "grounded", "camera": "head_camera",
                    "observation_sequence": seq, "confidence": .8,
                    "support_pixel": [23., 23.]} for seq in (1, 2, 3)]
        grounder = FakeGrounder(reports)
        tracker = MugLiftEvidenceTracker(cache, grounder)
        self.assertIsNone(tracker.evaluate(observation(1, lifted=True))["value"])
        self.assertTrue(tracker.evaluate(observation(2, lifted=True))["value"])
        self.assertTrue(tracker.evaluate(observation(3))["value"])
        self.assertEqual(grounder.calls, 3)

    def test_table_support_is_false_and_unavailable_support_stays_unknown(self):
        cache = StaticTablePlaneCache()
        reset = observation(0)
        cache.cache(reset, np.ones((48, 48), dtype=bool))
        on_table = FakeGrounder([{"status": "grounded", "camera": "head_camera",
            "observation_sequence": 1, "confidence": .8, "support_pixel": [23., 23.]}])
        self.assertFalse(MugLiftEvidenceTracker(cache, on_table).evaluate(
            observation(1))["value"])
        invisible = FakeGrounder([{"status": "not_detected",
            "observation_sequence": 1}])
        result = MugLiftEvidenceTracker(cache, invisible).evaluate(observation(1))
        self.assertIsNone(result["value"])

    def test_opus_support_checks_wait_until_configured_acquisition_window(self):
        cache = StaticTablePlaneCache()
        cache.cache(observation(0), np.ones((48, 48), dtype=bool))
        grounder = FakeGrounder([{
            "status": "grounded", "camera": "head_camera",
            "observation_sequence": seq, "confidence": .8,
            "support_pixel": [23., 23.],
        } for seq in (74, 82)])
        tracker = MugLiftEvidenceTracker(cache, grounder,
            minimum_policy_actions=64)

        self.assertEqual(tracker.evaluate(observation(0))["reason"],
                         "mug_lift_check_deferred_until_feedback_window")
        tracker.set_policy_action_count(63)
        self.assertIsNone(tracker.evaluate(observation(73, lifted=True))["value"])
        self.assertEqual(grounder.calls, 0)
        tracker.set_policy_action_count(64)
        self.assertIsNone(tracker.evaluate(observation(74, lifted=True))["value"])
        tracker.set_policy_action_count(72)
        self.assertTrue(tracker.evaluate(observation(82, lifted=True))["value"])
        self.assertEqual(grounder.calls, 2)

    def test_opus_schema_requests_pixel_only_and_rejects_bad_reports(self):
        class Client:
            class messages:
                requests = []

                @classmethod
                def create(cls, **request):
                    cls.requests.append(request)
                    return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
                        name="report_mug_support_pixel", input={"visible": True,
                        "support_pixel": [23, 24], "description": "base edge",
                        "confidence": .8})])

        report = OpusMugSupportGrounder(Client(), "claude-test").locate(observation(7))
        self.assertEqual(report["support_pixel"], [23., 24.])
        tool = Client.messages.requests[0]["tools"][0]
        self.assertEqual(tool["name"], "report_mug_support_pixel")
        self.assertNotIn("depth", str(tool["input_schema"]).lower())

        class BadClient:
            class messages:
                @staticmethod
                def create(**request):
                    return SimpleNamespace(content=[SimpleNamespace(type="tool_use",
                        name="report_mug_support_pixel", input={"visible": True,
                        "support_pixel": [99, 24], "description": "bad",
                        "confidence": .8})])

        with self.assertRaisesRegex(ValueError, "out_of_bounds"):
            OpusMugSupportGrounder(BadClient(), "claude-test").locate(observation(7))

    def test_cache_rejects_shape_and_calibration_changes(self):
        cache = StaticTablePlaneCache()
        reset = observation(0)
        cache.cache(reset, np.ones((48, 48), dtype=bool))
        changed = observation(1)
        frame = changed.camera_frames["head_camera"]
        moved = RGBDCamera(frame.depth_m, frame.intrinsic,
            np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                      [0., 0., -1., 1.01], [0., 0., 0., 1.]]))
        changed.camera_frames["head_camera"] = moved
        with self.assertRaisesRegex(ValueError, "calibration_changed"):
            cache.signed_distance(changed, [23, 23])


if __name__ == "__main__":
    unittest.main()
