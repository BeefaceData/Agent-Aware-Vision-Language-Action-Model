"""CPU tests for the one-shot measured rack-seat release authorization."""

from types import SimpleNamespace
import unittest

from harness.environment.base import Observation
from harness.perception.anchors import Anchor
from harness.predicates.registry import PredicateRegistry
from harness.predicates.rack_seat_gate import RackSeatReleaseGate
from harness.skills.schemas import SkillCommand, SkillResult
from harness.schemas import StateSummary


class RackSeatGateTests(unittest.TestCase):
    def setUp(self):
        self.pose = (0.1, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0)
        self.environment = SimpleNamespace(
            arm_state=lambda arm: {"pose": self.pose})
        self.gate = RackSeatReleaseGate(self.environment)
        self.registry = PredicateRegistry(rack_seat_gate=self.gate)
        self.command = SkillCommand.model_validate({"action": "move_to", "arm": "right",
            "parameters": {"target": "rack_seat"}})
        self.anchor = Anchor(self.pose, 4, "perception")
        state = StateSummary(command_state=(0.0,) * 14)
        self.result = SkillResult(call_id=1, skill="move_to", completed=True,
            stop_reason="completed", executed_policy_actions=0,
            vla_chunks_requested=0, discarded_actions=0, executed_physics_ticks=120,
            state_before=state, state_after=state, observation_before=4,
            observation_after=5, episode_terminal=False, task_success=None,
            controlled_arms="right", position_error_m=.005,
            orientation_error_rad=.02, elapsed_seconds=.2)

    def test_only_completed_grounded_seat_move_authorizes_measured_alignment(self):
        self.gate.record(self.command, self.result, self.anchor)
        observation = Observation(5, {}, (0.0,) * 14)

        predicate = self.registry.evaluate("rack_seat_aligned", observation, arm="right")

        self.assertTrue(predicate.value)
        self.assertEqual(predicate.source, "runtime")
        self.assertEqual(predicate.reason, "measured_tcp_at_grounded_rack_seat")
        self.assertFalse(self.registry.evaluate(
            "rack_seat_aligned", observation, arm="left").value)

    def test_gate_rejects_failed_or_uncertain_move_and_tcp_drift(self):
        failed = self.result.model_copy(update={"completed": False,
            "stop_reason": "target_not_reached"})
        self.gate.record(self.command, failed, self.anchor)
        self.assertFalse(self.gate.available)

        self.gate.record(self.command, self.result, self.anchor)
        self.pose = (0.13, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0)
        predicate = self.registry.evaluate("rack_seat_aligned",
            Observation(6, {}, (0.0,) * 14), arm="right")
        self.assertFalse(predicate.value)
        self.assertEqual(predicate.reason, "tcp_left_rack_seat")

    def test_alignment_authorization_is_explicit_and_defaults_to_observed_support(self):
        default = SkillCommand.model_validate({"action": "release", "arm": "right",
            "parameters": {}})
        aligned = SkillCommand.model_validate({"action": "release", "arm": "right",
            "parameters": {"authorization": "rack_seat_alignment"}})
        self.assertEqual(default.parameters.authorization, "observed_support")
        self.assertEqual(aligned.parameters.authorization, "rack_seat_alignment")
        with self.assertRaises(ValueError):
            SkillCommand.model_validate({"action": "release", "arm": "right",
                "parameters": {"authorization": "simulator_contact"}})

    def test_gate_requires_current_public_anchor_and_matching_motion_result(self):
        cases = (
            (Anchor(self.pose, 3, "perception"), self.result),
            (Anchor(self.pose, 5, "perception"), self.result),
            (Anchor(self.pose, 4, "development_oracle"), self.result),
            (self.anchor, self.result.model_copy(update={"controlled_arms": "left"})),
            (self.anchor, self.result.model_copy(update={"skill": "rotate_wrist"})),
        )
        for anchor, result in cases:
            with self.subTest(anchor=anchor, skill=result.skill, arms=result.controlled_arms):
                self.gate.record(self.command, self.result, self.anchor)
                self.gate.record(self.command, result, anchor)
                self.assertFalse(self.gate.available)

    def test_uncertain_or_terminal_seat_clears_prior_authorization(self):
        for update in ({"execution_uncertain": True}, {"episode_terminal": True},
                       {"completed": False, "stop_reason": "target_not_reached"}):
            with self.subTest(update=update):
                self.gate.record(self.command, self.result, self.anchor)
                self.gate.record(self.command, self.result.model_copy(update=update), self.anchor)
                self.assertFalse(self.gate.available)


if __name__ == "__main__":
    unittest.main()
