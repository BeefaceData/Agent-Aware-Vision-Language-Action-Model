"""Guarded rack release with measured drift during opening; no simulator."""

import unittest

from tests.hanging_scenario import HangingScenario


class HangingReleaseTests(unittest.TestCase):
    def ready(self):
        scenario = HangingScenario()
        for command in scenario.commands[:2]:
            observation = scenario.environment.observe()
            anchor = scenario.prepare(command, observation)
            result = scenario.executor.execute(command)
            self.assertTrue(result.completed)
            scenario.gate.record(command, result, anchor)
        self.assertTrue(scenario.gate.available)
        return scenario

    def test_wrong_arm_release_is_rejected_before_opening(self):
        scenario = self.ready()
        result = scenario.executor.execute({"action": "release", "arm": "left",
            "parameters": {"authorization": "rack_seat_alignment", "ramp_ticks": 50}})
        self.assertEqual(result.stop_reason, "release_not_authorized")
        self.assertEqual(result.executed_physics_ticks, 0)

    def test_alignment_loss_or_unavailable_pose_stops_opening_at_monitor_boundary(self):
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                scenario = self.ready()
                env = scenario.environment
                original_tick, original_state = env.tick, env.arm_state
                opening_ticks = 0

                def tick(*args):
                    nonlocal opening_ticks
                    original_tick(*args)
                    opening_ticks += 1
                    if opening_ticks == 25 and not unavailable:
                        env.arms["right"]["pose"][0] += .04

                def arm_state(arm):
                    state = original_state(arm)
                    if unavailable and arm == "right" and opening_ticks >= 25:
                        state["pose"][:] = float("nan")
                    return state

                env.tick, env.arm_state = tick, arm_state
                result = scenario.executor.execute(scenario.commands[2])
                self.assertFalse(result.completed)
                self.assertEqual(result.stop_reason, "rack_seat_alignment_lost")
                self.assertEqual(result.executed_physics_ticks, 25)
                self.assertEqual(opening_ticks, 25)
                self.assertLess(env.arms["right"]["gripper"], 1.0)
                self.assertFalse(result.execution_uncertain)
