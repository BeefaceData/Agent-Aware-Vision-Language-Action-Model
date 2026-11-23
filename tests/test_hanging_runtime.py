"""CPU integration of hanging assembly, retry caps and native checker outcomes."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.perception.anchors import AnchorRegistry
from harness.perception.grounding import RGBDPerceptionProvider
from harness.perception.hang_targets import StaticRackFrameCache
from harness.planner.base import PlannerSelection
from harness.runtime import assembly
from tests.hanging_scenario import HangingScenario
from tests.test_analytic_skills import AnalyticEnvironment
from tests.test_hang_targets import CAMERA, RACK_POINTS, MugGrounder, make_observation, report


class HangingRuntimeTests(unittest.TestCase):
    def test_repeated_hang_grounding_failure_is_bounded_before_motion(self):
        scenario = HangingScenario(grounding_error="rack_frame_not_cached")

        def select(context):
            scenario.contexts.append(context)
            return PlannerSelection(scenario.commands[0], "HANG")

        scenario.select = select
        summary = scenario.run()
        self.assertEqual(summary["outcome"], "grounding_reselection_budget_exhausted")
        self.assertEqual(summary["grounding_reselections"], 3)
        self.assertEqual(len(scenario.contexts), 3)
        self.assertEqual(scenario.executor.calls, 0)
        self.assertEqual(scenario.environment.ticks, 0)
        self.assertTrue(all(c.grounding_feedback for c in scenario.contexts[1:]))

    def test_assembly_shares_gate_and_prepares_current_observation_anchors(self):
        scenario = HangingScenario()
        anchors = AnchorRegistry()
        perception = RGBDPerceptionProvider(lambda *_a, **_k: None, anchors)
        cache = StaticRackFrameCache()
        cache.cache(make_observation(), report("rack_peg", 0, RACK_POINTS))
        planner = SimpleNamespace(client=None, settings=SimpleNamespace(model="synthetic"))
        with patch("harness.runtime.assembly.OpusObjectFrameGrounder", return_value=MugGrounder()) as grounder:
            predicates, prepare = assembly.public_perception_evidence(perception,
                scenario.environment, anchors, planner=planner, rack_frame_cache=cache,
                release_gate=scenario.gate, hang_camera=CAMERA)
        self.assertIs(predicates.rack_seat_gate, scenario.gate)
        self.assertIs(predicates.perception, perception)
        self.assertIsNone(predicates.hanging_mug)
        self.assertEqual(grounder.call_args.kwargs["target"], "mug_handle")
        for sequence in (0, 1):
            observation = make_observation(sequence)
            target = prepare(scenario.commands[1], observation)
            self.assertEqual(target.observation_sequence, sequence)
            self.assertEqual(target.source, "perception")
            self.assertEqual(anchors.resolve("rack_seat", "right", observation), target)

    def run_native_success_release(self, lose_alignment=False):
        scenario = HangingScenario()
        env = scenario.environment
        original = env.tick
        opening_ticks = 0

        def tick(*args):
            nonlocal opening_ticks
            original(*args)
            if env.arms["right"]["gripper"] > 0:
                opening_ticks += 1
                if lose_alignment and opening_ticks >= 45:
                    env.arms["right"]["pose"][0] += .04

        env.tick = tick
        with patch.object(AnalyticEnvironment, "native_success", property(
                lambda e: e.arms["right"]["gripper"] > .8)):
            summary = scenario.run()
        return scenario, summary

    def test_native_success_during_release_finishes_opening_but_does_not_prove_retreat(self):
        scenario, summary = self.run_native_success_release()
        results = [e["result"] for e in scenario.events if e.get("event_type") == "skill_result"]
        self.assertEqual(results[-1]["skill"], "release")
        self.assertTrue(results[-1]["completed"])
        self.assertEqual(scenario.environment.arms["right"]["gripper"], 1.0)
        self.assertEqual(summary["outcome"], "task_success")
        self.assertEqual(len(results), 3)  # Benchmark success terminates before retreat.

    def test_native_checker_does_not_hide_failed_guarded_hang_release(self):
        scenario, summary = self.run_native_success_release(lose_alignment=True)
        results = [e["result"] for e in scenario.events if e.get("event_type") == "skill_result"]
        self.assertFalse(results[-1]["completed"])
        self.assertEqual(results[-1]["stop_reason"], "rack_seat_alignment_lost")
        self.assertTrue(summary["task_success"])  # Preserve separate native checker evidence.
        self.assertEqual(summary["outcome"], "rack_seat_alignment_lost")
        self.assertEqual(summary["phase"], "FAILED")
