import unittest

from modal_sam3 import MUG_SCORE_THRESHOLD, RACK_PROFILES, confidence_meets_threshold


class ModalSam3ServiceTests(unittest.TestCase):
    def test_exploration_rack_cutoff_is_applied_after_postprocessing(self):
        _, threshold = RACK_PROFILES["explore_mug_rack"]
        self.assertEqual(threshold, 0.18)
        self.assertTrue(confidence_meets_threshold(0.419, threshold))
        self.assertTrue(confidence_meets_threshold(0.191, threshold))
        self.assertFalse(confidence_meets_threshold(0.179, threshold))

    def test_mug_cutoff_uses_lowest_observed_test_score(self):
        self.assertEqual(MUG_SCORE_THRESHOLD, 0.40)
        self.assertTrue(confidence_meets_threshold(0.40, MUG_SCORE_THRESHOLD))
        self.assertFalse(confidence_meets_threshold(0.399, MUG_SCORE_THRESHOLD))

    def test_default_rack_cutoff_stays_at_point_six(self):
        _, threshold = RACK_PROFILES["default"]
        self.assertTrue(confidence_meets_threshold(0.60, threshold))
        self.assertFalse(confidence_meets_threshold(0.577, threshold))

    def test_invalid_scores_fail_closed(self):
        for score in (float("nan"), float("inf"), "bad", None):
            self.assertFalse(confidence_meets_threshold(score, 0.18))


if __name__ == "__main__":
    unittest.main()
