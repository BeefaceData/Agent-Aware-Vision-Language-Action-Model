import math
import unittest

from pydantic import ValidationError

from harness.schemas import DecisionType, StateSummary, SupervisorDecision
from tests.helpers import assessment, decision


class SchemaTests(unittest.TestCase):
    def test_pass_abort_and_structural_skill_roundtrip(self):
        for value in (
            decision(), decision(decision="abort"),
            decision(decision="execute_skill", skill_id="candidate_v1", parameters={"scale": 0.05}),
        ):
            self.assertEqual(SupervisorDecision.model_validate_json(value.model_dump_json()), value)
        self.assertEqual(decision().decision, DecisionType.PASS)

    def test_confidence_strict_finite_bounds(self):
        for invalid in (-0.1, 1.1, math.nan, math.inf, -math.inf, "0.5", True):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                decision(confidence=invalid)

    def test_unknown_fields_enums_missing_fields(self):
        for update in ({"joint_actions": [0] * 14}, {"decision": "move"},
                       {"phase": "invented"}, {"failure_type": "invented"}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                decision(**update)
        with self.assertRaises(ValidationError):
            SupervisorDecision.model_validate({"decision": "pass"})

    def test_skill_field_consistency(self):
        for update in ({"decision": "execute_skill"}, {"decision": "execute_skill", "skill_id": " "},
                       {"skill_id": "retry"}, {"parameters": {"offset": 1}},
                       {"decision": "abort", "skill_id": "retry"}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                decision(**update)

    def test_parameters_are_finite_scalars_not_action_arrays(self):
        for value in ([0.0] * 14, {"joint": 1}, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                decision(decision="execute_skill", skill_id="candidate", parameters={"x": value})
        self.assertEqual(decision().parameters, {})
        first = decision()
        first.parameters["x"] = 1
        self.assertEqual(decision().parameters, {})

    def test_state_requires_finite_14_values_and_excludes_oracle(self):
        for vector in ([0.0] * 13, [0.0] * 15, [math.nan] * 14, [True] * 14, ["1"] * 14):
            with self.subTest(vector=vector), self.assertRaises(ValidationError):
                StateSummary(command_state=vector)
        with self.assertRaises(ValidationError):
            StateSummary(command_state=[0.0] * 14, mug_pose=[0, 0, 0])

    def test_assessment_event_validation(self):
        for update in ({"step": -1}, {"chunk_pos": True}, {"success": True},
                       {"event_type": "action_result"}, {"outcome": "helped"},
                       {"pre_state_summary": {"command_state": [0.0] * 14, "success": True}}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                assessment(**update)
        self.assertEqual(assessment().post_state_summary, None)
