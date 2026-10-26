import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from harness.legacy.assessment_log import AssessmentLogger
from tests.helpers import assessment


class AssessmentLogTests(unittest.TestCase):
    def test_append_across_instances_and_linked_results(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "assessments.jsonl"
            AssessmentLogger(path).append(assessment())
            prefix = path.read_bytes()
            AssessmentLogger(path).append(assessment(
                event_type="action_result", executed_action=[0.0] * 14,
                post_state_summary={"command_state": [1.0] * 14}, outcome="unknown",
            ))
            self.assertTrue(path.read_bytes().startswith(prefix))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["event_id"], rows[1]["event_id"])
            self.assertIsNone(rows[0]["executed_action"])
            self.assertEqual(rows[1]["outcome"], "unknown")

    def test_invalid_mutated_record_does_not_touch_log(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "assessments.jsonl"
            logger = AssessmentLogger(path)
            logger.append(assessment())
            previous = path.read_bytes()
            invalid = assessment()
            invalid.skill_parameters["joint_targets"] = [0.0] * 14
            with self.assertRaises(ValidationError):
                logger.append(invalid)
            self.assertEqual(path.read_bytes(), previous)
