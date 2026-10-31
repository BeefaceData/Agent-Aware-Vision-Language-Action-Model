"""Composed hanging contracts; intentionally not physical support validation."""

import unittest

import numpy as np

from tests.hanging_scenario import HangingScenario


class HangingScenarioTests(unittest.TestCase):
    def test_failed_seat_returns_to_planner_without_opening_or_retreat(self):
        scenario = HangingScenario(seat_plan_fails=True)
        summary = scenario.run()
        results = [e["result"] for e in scenario.events if e.get("event_type") == "skill_result"]
        self.assertEqual(results[0]["stop_reason"], "completed")
        self.assertFalse(results[1]["completed"])
        self.assertEqual(results[1]["stop_reason"], "planning_failure")
        self.assertEqual([r["skill"] for r in results], ["move_to", "move_to", "abort"])
        self.assertEqual(scenario.contexts[2].previous.stop_reason, "planning_failure")
        self.assertFalse(scenario.gate.available)
        self.assertEqual(scenario.environment.arms["right"]["gripper"], 0.0)
        self.assertFalse(summary["task_success"])

    def test_each_move_records_measured_start_from_two_entry_poses(self):
        for x in (-.2, -.25):
            with self.subTest(x=x):
                scenario = HangingScenario(start_x=x)
                initial = scenario.environment.arm_state("right")
                scenario.run()
                starts = [e for e in scenario.events if e.get("event_type") == "analytic_start"]
                np.testing.assert_allclose(starts[0]["measured_start_pose"], initial["pose"])
                np.testing.assert_allclose(starts[0]["measured_start_joints"], initial["joints"])
                np.testing.assert_allclose(starts[1]["measured_start_pose"], starts[0]["target_pose"], atol=1e-12)

    def test_approach_seat_release_retreat_return_separate_results(self):
        scenario = HangingScenario()
        left = scenario.environment.arm_state("left")
        summary = scenario.run()
        results = [e["result"] for e in scenario.events if e.get("event_type") == "skill_result"]
        self.assertEqual([r["skill"] for r in results],
                         ["move_to", "move_to", "release", "move_to", "abort"])
        self.assertEqual([r["stop_reason"] for r in results[:4]], ["completed"]*4)
        self.assertEqual(scenario.environment.arms["right"]["gripper"], 1.0)
        self.assertFalse(summary["task_success"])
        self.assertEqual(summary["policy_actions"], 0)
        self.assertEqual(summary["outcome"], "scenario_complete")
        np.testing.assert_array_equal(scenario.environment.arm_state("left")["joints"], left["joints"])
        self.assertEqual(len(scenario.prepared), 2)
        self.assertGreater(scenario.prepared[1][1], scenario.prepared[0][1])
        self.assertTrue(all(r["information_source"] == "public" for r in results))

    def test_seating_authorizes_alignment_but_not_contact_or_hanging(self):
        scenario = HangingScenario()
        scenario.run()
        before_release = scenario.contexts[2]
        self.assertTrue(before_release.value("rack_seat_aligned"))
        self.assertIsNone(before_release.value("rack_contact_detected"))
        self.assertIsNone(before_release.value("mug_released"))
        self.assertFalse(before_release.value("task_success"))
        self.assertIsNone(scenario.contexts[3].value("rack_seat_aligned"))

    def test_lost_alignment_blocks_release_and_prevents_retreat(self):
        scenario = HangingScenario(drift_before_release=True)
        scenario.run()
        results = [e["result"] for e in scenario.events if e.get("event_type") == "skill_result"]
        release = next(r for r in results if r["skill"] == "release")
        self.assertEqual(release["stop_reason"], "release_not_authorized")
        self.assertEqual(release["executed_physics_ticks"], 0)
        self.assertEqual(scenario.environment.arms["right"]["gripper"], 0.0)
        self.assertEqual([r["skill"] for r in results].count("move_to"), 2)

    def test_grounding_failure_reaches_planner_before_any_motion(self):
        for error in ("rack_frame_not_cached", "grounding_report_stale_or_wrong_camera",
                      "keypoint_depth_discontinuity", "hang_target_out_of_reach"):
            with self.subTest(error=error):
                scenario = HangingScenario(grounding_error=error)
                summary = scenario.run()
                rejected = next(e for e in scenario.assessments if e.get("grounding_rejected"))
                self.assertFalse(rejected["skill_executed"])
                self.assertEqual(rejected["grounding_rejection_code"], "hang_target_grounding_unavailable")
                self.assertIsNotNone(scenario.contexts[1].grounding_feedback)
                self.assertEqual(scenario.environment.ticks, 0)
                self.assertEqual(summary["outcome"], "scenario_grounding_rejected")
