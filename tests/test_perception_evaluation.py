"""CPU evaluation metrics for candidate instance-mask detectors."""

import unittest

import numpy as np

from harness.perception.evaluation import evaluate_detector, evaluate_mask_batches
from harness.perception.grounding import Detection, DetectionBatch
from tests.test_perception_grounding import FakeDetector, mask, observation


class PerceptionEvaluationTests(unittest.TestCase):
    def test_instance_metrics_count_matches_false_positives_and_misses(self):
        truth = DetectionBatch(1, "head_camera", (
            Detection("mug", 1.0, mask()), Detection("rack", 1.0, mask((1, 1)))))
        predicted = DetectionBatch(1, "head_camera", (
            Detection("mug", .95, mask()), Detection("table", .9, mask((1, 2)))))
        result = evaluate_mask_batches(predicted, truth)
        self.assertEqual(result["labels"]["mug"]["true_positive"], 1)
        self.assertEqual(result["labels"]["mug"]["mean_matched_iou"], 1.0)
        self.assertEqual(result["labels"]["rack"]["false_negative"], 1)
        self.assertEqual(result["labels"]["table"]["false_positive"], 1)

    def test_detector_replay_aggregates_frames_and_reports_rgbd_availability(self):
        detector = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", (
            Detection("mug", .95, mask()),)))
        truth = DetectionBatch(4, "head_camera", (Detection("mug", 1., mask()),))
        report = evaluate_detector(detector, [(observation(4), truth)])
        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["rgbd_frame_count"], 1)
        self.assertEqual(report["summary"]["mug"]["precision"], 1.0)
        self.assertEqual(report["summary"]["mug"]["recall"], 1.0)
        self.assertFalse(report["robot_motion"])
        self.assertFalse(report["simulator_started"])

    def test_confidence_iou_and_camera_shape_gates_fail_closed(self):
        truth = DetectionBatch(2, "head_camera", (Detection("mug", 1., mask()),))
        low = DetectionBatch(2, "head_camera", (Detection("mug", .4, mask()),))
        result = evaluate_mask_batches(low, truth, confidence_threshold=.6)
        self.assertEqual(result["labels"]["mug"]["false_negative"], 1)
        partial = np.zeros((2, 3), dtype=bool)
        partial[0, 2] = True
        low_iou = DetectionBatch(2, "head_camera", (Detection("mug", .9, partial),))
        result = evaluate_mask_batches(low_iou, truth, iou_threshold=.5)
        self.assertEqual(result["labels"]["mug"]["false_positive"], 1)
        self.assertEqual(result["labels"]["mug"]["false_negative"], 1)
        bad = DetectionBatch(2, "left_camera", ())
        with self.assertRaisesRegex(ValueError, "evaluation_camera_mismatch"):
            evaluate_detector(FakeDetector(lambda seq: bad), [(observation(2), truth)])

    def test_evaluator_rejects_stale_batches_empty_inputs_and_nonincreasing_frames(self):
        truth = DetectionBatch(3, "head_camera", ())
        stale = FakeDetector(lambda seq: DetectionBatch(seq-1, "head_camera", ()))
        with self.assertRaisesRegex(ValueError, "detector_returned_stale_or_invalid_batch"):
            evaluate_detector(stale, [(observation(3), truth)])
        good = FakeDetector(lambda seq: DetectionBatch(seq, "head_camera", ()))
        with self.assertRaisesRegex(ValueError, "empty_detector_evaluation"):
            evaluate_detector(good, [])
        with self.assertRaisesRegex(ValueError, "evaluation_observations_not_increasing"):
            evaluate_detector(good, [(observation(3), truth), (observation(3), truth)])


if __name__ == "__main__":
    unittest.main()
