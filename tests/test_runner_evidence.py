"""Runner evidence modes stay explicit and gate-only oracle use is supported."""

import unittest

from run_skills import validate_evidence_mode


class RunnerEvidenceTests(unittest.TestCase):
    def test_pixel_grounding_can_use_private_development_gates(self):
        validate_evidence_mode(development_oracle=True, perception_factory=None,
                               planner_pixel_grounding=True)

    def test_public_perception_can_be_combined_with_pixel_anchor_fallback(self):
        validate_evidence_mode(development_oracle=False, perception_factory="pkg:factory",
                               planner_pixel_grounding=True)

    def test_oracle_and_public_detector_remain_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually_exclusive"):
            validate_evidence_mode(development_oracle=True, perception_factory="pkg:factory",
                                   planner_pixel_grounding=False)

    def test_evidence_source_remains_required(self):
        with self.assertRaisesRegex(ValueError, "evidence_provider_required"):
            validate_evidence_mode(development_oracle=False, perception_factory=None,
                                   planner_pixel_grounding=False)


if __name__ == "__main__":
    unittest.main()
