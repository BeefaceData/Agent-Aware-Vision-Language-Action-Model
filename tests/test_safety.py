import math
import unittest

from pydantic import ValidationError

from harness.config import SafetyOptions
from harness.safety import SafetyValidator
from harness.schemas import DecisionType, SkillResult, TaskPhase
from harness.legacy.recovery_skills.retry_grasp import RetryGrasp
from tests.helpers import decision
from tests.skill_fixtures import registry_with_verified_skill, skill_decision


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry_with_verified_skill()
        self.safety = SafetyValidator(self.registry)
        self.phase = TaskPhase.GRASP_MUG

    def evaluate(self, request, **kwargs):
        return self.safety.evaluate(request, current_phase=self.phase, **kwargs)

    def reserve(self):
        return self.safety.reserve(skill_decision(), current_phase=self.phase)

    def finish(self, recovered):
        self.safety.finish_intervention(SkillResult(
            skill_id="test_only", executed_steps=5, success_hint=True,
            termination_reason="test_complete",
        ), recovered=recovered)

    def test_verified_request_defaults_and_pass_abort(self):
        result = self.evaluate(skill_decision(parameters={}))
        self.assertTrue(result.accepted)
        self.assertEqual(result.decision.parameters, {"distance": 0.05})
        self.assertEqual(result.max_steps, 10)
        self.assertEqual(self.safety.interventions, 0)
        self.assertEqual(self.evaluate(decision()).decision.decision, DecisionType.PASS)
        self.assertEqual(self.evaluate(decision(decision="abort")).decision.decision, DecisionType.ABORT)

    def test_unknown_unverified_phase_failure_and_confidence(self):
        self.registry.register(RetryGrasp())
        cases = [
            ({"skill_id": "invented"}, "unknown_skill"),
            ({"skill_id": "retry_grasp"}, "unverified_skill"),
            ({"phase": "hang_mug"}, "unsupported_phase"),
            ({"failure_type": "dropped_mug"}, "unsupported_failure_type"),
            ({"confidence": 0.2}, "low_confidence"),
        ]
        for update, reason in cases:
            with self.subTest(update=update):
                result = self.evaluate(skill_decision(**update))
                self.assertFalse(result.accepted)
                self.assertEqual(result.decision.decision, DecisionType.PASS)
                self.assertEqual(result.reason, reason)
        self.assertEqual(self.safety.interventions, 0)

    def test_unknown_out_of_range_nonfinite_or_coerced_parameters_rejected(self):
        for parameters in ({"invented": 1}, {"distance": -0.1}, {"distance": 0.11},
                           {"distance": "0.05"}, {"distance": True},
                           {"distance": math.nan}, {"distance": [0] * 14}):
            raw = skill_decision().model_dump()
            raw["parameters"] = parameters
            with self.subTest(parameters=parameters):
                result = self.evaluate(raw)
                self.assertFalse(result.accepted)
                self.assertEqual(result.decision.decision, DecisionType.PASS)
        self.assertEqual(self.evaluate({"action": [0] * 14}).reason, "invalid_decision_schema")

    def test_independent_abort_overrides_invalid_response_and_is_latched(self):
        self.assertEqual(self.evaluate({}, safety_abort=True).decision.decision, DecisionType.ABORT)
        self.safety.reserve({}, current_phase=self.phase, safety_abort=True)
        self.assertEqual(self.evaluate(decision()).decision.decision, DecisionType.ABORT)
        self.safety.reset_episode()
        self.assertEqual(self.evaluate(decision()).decision.decision, DecisionType.PASS)

    def test_intervention_count_reserved_once_and_episode_reset(self):
        self.safety = SafetyValidator(self.registry, SafetyOptions(max_interventions_per_episode=1))
        self.assertTrue(self.reserve().accepted)
        self.assertEqual(self.safety.interventions, 1)
        self.assertEqual(self.evaluate(skill_decision()).reason, "intervention_already_active")
        self.finish(True)
        result = self.evaluate(skill_decision())
        self.assertEqual(result.reason, "intervention_limit")
        self.assertEqual(result.decision.decision, DecisionType.ABORT)
        self.safety.reset_episode()
        self.assertEqual(self.safety.interventions, 0)
        self.assertTrue(self.reserve().accepted)

    def test_recovery_labels_not_success_hint_control_failure_budget(self):
        for _ in range(2):
            self.reserve()
            self.finish(False)
        self.assertEqual(self.safety.consecutive_failures, 2)
        result = self.evaluate(decision())
        self.assertEqual(result.reason, "consecutive_skill_failure_limit")
        self.assertEqual(result.decision.decision, DecisionType.ABORT)

    def test_success_resets_failures_but_not_attempts(self):
        self.reserve()
        self.finish(False)
        self.reserve()
        self.finish(True)
        self.assertEqual(self.safety.consecutive_failures, 0)
        self.assertEqual(self.safety.interventions, 2)

    def test_step_budget_and_configured_invalid_request_abort(self):
        self.safety = SafetyValidator(self.registry, SafetyOptions(max_skill_steps=9))
        self.assertEqual(self.evaluate(skill_decision()).reason, "skill_step_budget_exceeded")
        self.safety = SafetyValidator(self.registry, SafetyOptions(abort_on_invalid_skill_request=True))
        self.assertEqual(self.evaluate(skill_decision(skill_id="unknown")).decision.decision, DecisionType.ABORT)

    def test_result_mismatch_budget_overrun_and_execution_exception_abort(self):
        for skill_id, steps in (("wrong", 5), ("test_only", 11)):
            self.safety.reset_episode()
            self.reserve()
            with self.subTest(skill_id=skill_id, steps=steps), self.assertRaises(ValueError):
                self.safety.finish_intervention(SkillResult(
                    skill_id=skill_id, executed_steps=steps, termination_reason="test",
                ), recovered=True)
            self.assertEqual(self.evaluate(decision()).decision.decision, DecisionType.ABORT)
        self.safety.reset_episode()
        self.reserve()
        self.safety.execution_failed()
        self.assertEqual(self.safety.interventions, 1)
        self.assertEqual(self.safety.consecutive_failures, 1)
        self.assertEqual(self.evaluate(decision()).decision.decision, DecisionType.ABORT)

    def test_skill_result_strict_validation(self):
        for update in ({"executed_steps": -1}, {"executed_steps": True},
                       {"final_state": [0.0] * 13}, {"metadata": {"x": math.inf}},
                       {"success_hint": "true"}, {"unknown": 1}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                SkillResult.model_validate({"skill_id": "test_only", "executed_steps": 0,
                                            "termination_reason": "done", **update})
