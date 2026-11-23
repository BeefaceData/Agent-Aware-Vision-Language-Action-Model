"""Measured-start hanging motion edge cases with the real skill executor."""

import unittest

from tests.test_analytic_skills import AnalyticEnvironment, EndpointLagEnvironment, executor


MOVE = {"action": "move_to", "arm": "right",
        "parameters": {"target": "current_ee", "offset_xyz": (0.02, 0., 0.)}}


class HangMotionTests(unittest.TestCase):
    def test_one_final_in_tolerance_sample_does_not_prove_stable_convergence(self):
        for command in (MOVE, {"action": "rotate_wrist", "arm": "right",
                              "parameters": {"delta_yaw_deg": 15.0}}):
            with self.subTest(action=command["action"]):
                # One trajectory endpoint tick + 125 settling ticks: only the
                # last sample is in tolerance, short of three stable samples.
                env = EndpointLagEnvironment(off_target_commands=125)
                run, _, events = executor(env)
                result = run.execute(command)
                settling = next(e for e in events if e.get("event_type") == "analytic_settling")
                self.assertEqual(settling["stable_samples"], 1)
                self.assertFalse(result.completed)
                self.assertEqual(result.stop_reason, "target_not_reached")
                self.assertEqual(result.position_error_m, 0.0)
                self.assertFalse(settling["converged"])

    def test_cancellation_after_planning_prevents_drive_writes(self):
        class CancelAfterPlan(AnalyticEnvironment):
            cancelled = False

            def plan(self, arm, pose):
                result = super().plan(arm, pose)
                self.cancelled = True
                return result

        env = CancelAfterPlan()
        run, _, _ = executor(env)
        run.cancelled = lambda: env.cancelled
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(env.writes, [])
        self.assertEqual(result.executed_physics_ticks, 0)

    def test_mid_motion_cancellation_stops_before_next_tick_and_returns_state(self):
        env = AnalyticEnvironment()
        run, _, _ = executor(env)
        run.cancelled = lambda: env.ticks >= 2
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(result.executed_physics_ticks, 2)
        self.assertEqual(len(env.writes), 2)
        self.assertEqual(env.holds, 1)
        self.assertFalse(result.execution_uncertain)
        self.assertGreater(result.observation_after, result.observation_before)

    def test_remaining_episode_ticks_bound_motion_before_any_drive_writes(self):
        env = AnalyticEnvironment()
        run, _, _ = executor(env)
        run.budget = run.budget.model_copy(update={"max_analytic_ticks": 7})
        result = run.execute(MOVE)
        self.assertEqual(result.stop_reason, "step_budget")
        self.assertEqual(env.writes, [])
        self.assertEqual(result.executed_physics_ticks, 0)
