"""Withdrawal mechanics and separate support outcomes, without live physics."""

import unittest

import numpy as np

from development.hanging.evaluation import HangTrial, evaluate
from tests.hanging_scenario import HangingScenario
from tests.test_hang_evaluation import trial_data


class HangingRetreatTests(unittest.TestCase):
    def released_scenario(self):
        scenario = HangingScenario()
        results = []
        for command in scenario.commands[:3]:
            anchor = scenario.prepare(command, scenario.environment.observe())
            result = scenario.executor.execute(command)
            self.assertTrue(result.completed)
            scenario.gate.record(command, result, anchor)
            results.append(result)
        self.assertEqual(scenario.environment.arms["right"]["gripper"], 1.0)
        self.assertEqual(results[-1].skill, "release")
        return scenario

    def test_withdrawal_starts_from_post_release_measured_state_and_keeps_hand_open(self):
        scenario = self.released_scenario()
        env = scenario.environment
        # The actual measured state can differ from the earlier seat target.
        env.arms["right"]["pose"][0] += .01
        env.arms["right"]["joints"] += .02
        measured = env.arm_state("right")
        left = env.arm_state("left")
        before = len(scenario.events)
        result = scenario.executor.execute(scenario.commands[3])
        events = scenario.events[before:]
        start = next(e for e in events if e.get("event_type") == "analytic_start")
        self.assertEqual(result.stop_reason, "completed")
        np.testing.assert_allclose(start["measured_start_pose"], measured["pose"])
        np.testing.assert_allclose(start["measured_start_joints"], measured["joints"])
        expected = measured["pose"].copy()
        expected[2] += .05
        np.testing.assert_allclose(start["target_pose"], expected)
        self.assertTrue(all(e["right_gripper"] == 1.0 for e in events if e.get("event") == "tick_requested"))
        np.testing.assert_allclose(env.arm_state("left")["joints"], left["joints"])
        self.assertFalse(result.task_success)

    def test_withdrawal_outside_workspace_rejected_before_planning_or_writing(self):
        scenario = self.released_scenario()
        env = scenario.environment
        env.arms["right"]["pose"][2] = 1.38
        writes, plans = len(env.writes), len(env.goals)
        result = scenario.executor.execute(scenario.commands[3])
        self.assertEqual(result.stop_reason, "pose_limits")
        self.assertEqual(result.executed_physics_ticks, 0)
        self.assertEqual(len(env.writes), writes)
        self.assertEqual(len(env.goals), plans)

    def test_retreat_planning_failure_is_separate_from_completed_release(self):
        scenario = self.released_scenario()
        env = scenario.environment
        env.plan_status = "Fail"
        writes = len(env.writes)
        result = scenario.executor.execute(scenario.commands[3])
        self.assertFalse(result.completed)
        self.assertEqual(result.stop_reason, "planning_failure")
        self.assertEqual(len(env.writes), writes)
        self.assertEqual(env.arms["right"]["gripper"], 1.0)

    def test_retreat_completion_does_not_override_unknown_or_lost_support(self):
        for support, expected in ((None, "hang_unknown"), (False, "hang_failed")):
            with self.subTest(support=support):
                data = trial_data()
                data["support_samples"][-1]["supported"] = support
                data["native_success"] = True
                report = evaluate(HangTrial.model_validate(data))
                self.assertEqual(report.stages["release"].status, "passed")
                self.assertEqual(report.stages["retreat"].status, "passed")
                self.assertEqual(report.outcome, expected)
