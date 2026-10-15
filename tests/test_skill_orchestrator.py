"""Small CPU composition checks using the real skill executor."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from harness.perception.anchors import Anchor
from harness.perception.observations import ObservationStore
from harness.planner.scripted import GraspRetryPlanner, GraspRetrySettings
from harness.skill_orchestrator import SkillOrchestrator
from harness.predicates.hanging_mug import HangingMugPredicates
from harness.predicates.registry import PredicateRegistry
from tests.test_analytic_skills import AnalyticEnvironment, evidence, executor


class AcquisitionEnvironment(AnalyticEnvironment):
    def __init__(self, succeeds_after=None):
        super().__init__()
        self.policy_steps = 0
        self.succeeds_after = succeeds_after

    def step(self, action):
        self.policy_steps += 1
        return super().step(action)

    def evidence(self, obs):
        held = self.succeeds_after is not None and self.policy_steps >= self.succeeds_after
        return evidence(obs.sequence, left_contact=held, left_bilateral=held,
                        table_contact=not held, mug_position=(-0.2, 0.0, 0.80 if held else 0.75))


class OrchestratorTests(unittest.TestCase):
    def run_recipe(self, env, *, retries=1, turns=20, predicates=True, prepare_failure=False):
        gates = PredicateRegistry(HangingMugPredicates(env.evidence, source="development_oracle")) if predicates else None
        run, worker, events = executor(env, predicates=gates, oracle=predicates)
        stages, assessments = [], []

        def prepare(command, obs):
            if command.action == "move_to" and command.parameters.target == "mug_pregrasp":
                if prepare_failure:
                    raise ValueError("sensitive provider details must not enter assessments")
                stages.append(obs.sequence)
                pose = env.arm_state(command.arm)["pose"]
                run.anchors.bind("mug_pregrasp", command.arm,
                    Anchor(tuple(pose), obs.sequence, source="development_oracle"))

        planner = GraspRetryPlanner(GraspRetrySettings(max_retries=retries,
            vla={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 4}))
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            controller = SkillOrchestrator(run, planner, log=assessments.append,
                save_observation=store.save, episode_id="test", seed=3, max_turns=turns, prepare=prepare)
            summary = controller.run()
            for artifact in store.directory.glob("*.npz"):
                with np.load(artifact, allow_pickle=False) as saved:
                    self.assertEqual(set(saved.files), {"head_camera", "left_camera", "right_camera",
                                                       "command_state", "sequence"})
            with self.assertRaises(RuntimeError):
                controller.run()
        return summary, run, worker, events, assessments, stages

    def test_failed_attempt_restages_then_fresh_vla_lifts(self):
        summary, run, worker, events, assessments, stages = self.run_recipe(AcquisitionEnvironment(5))
        commands = [e["command"]["action"] for e in events if e.get("event_type") == "skill_requested"]
        self.assertEqual(commands, ["move_to", "vla_act", "set_gripper", "move_to", "move_to", "vla_act"])
        self.assertEqual(summary["outcome"], "acquisition_confirmed")
        self.assertEqual(summary["phase"], "TRANSFER")
        self.assertFalse(summary["task_success"])
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(worker.policy_resets, 2)
        self.assertEqual(worker.chunks, 2)
        self.assertGreater(stages[1], stages[0])
        self.assertTrue(run.terminal)
        result = [e for e in events if e.get("event_type") == "skill_result"][-1]["result"]
        self.assertEqual(result["discarded_actions"], 2)
        self.assertEqual(len([a for a in assessments if a["event_type"] == "planner_turn_result"]), 6)

    def test_retry_exhaustion_aborts_without_more_inference(self):
        summary, _, worker, events, _, _ = self.run_recipe(AcquisitionEnvironment())
        self.assertEqual(summary["outcome"], "grasp_retry_budget_exhausted")
        self.assertFalse(summary["milestone_reached"])
        self.assertEqual(worker.chunks, 2)
        self.assertEqual(events[-1]["result"]["skill"], "abort")

    def test_unknown_evidence_never_moves_or_calls_policy(self):
        env = AcquisitionEnvironment()
        summary, _, worker, _, _, stages = self.run_recipe(env, predicates=False)
        self.assertEqual(summary["outcome"], "lift_evidence_unavailable")
        self.assertEqual((worker.chunks, env.ticks, stages), (0, 0, []))

    def test_turn_budget_and_failed_preparation_close_episode(self):
        summary, run, worker, _, _, _ = self.run_recipe(AcquisitionEnvironment(), turns=1)
        self.assertEqual(summary["outcome"], "planner_turn_budget_exhausted")
        self.assertTrue(run.terminal)
        self.assertEqual(worker.chunks, 0)
        summary, run, worker, _, assessments, _ = self.run_recipe(AcquisitionEnvironment(), prepare_failure=True)
        self.assertEqual(summary["outcome"], "controller_error")
        self.assertEqual(summary["error_type"], "ValueError")
        self.assertNotIn("sensitive", str(assessments))
        self.assertEqual(run.calls, 0)

    def test_held_but_unlifted_mug_is_not_opened_for_retry(self):
        env = AcquisitionEnvironment()
        env.evidence = lambda obs: evidence(obs.sequence, left_contact=env.policy_steps > 0,
            left_bilateral=env.policy_steps > 0, table_contact=True)
        summary, _, worker, events, _, _ = self.run_recipe(env)
        self.assertEqual(summary["outcome"], "grasp_without_verified_lift")
        self.assertEqual(worker.chunks, 1)
        commands = [e["command"]["action"] for e in events if e.get("event_type") == "skill_requested"]
        self.assertEqual(commands, ["move_to", "vla_act", "abort"])

    def test_uncertain_execution_does_not_retry_or_claim_success(self):
        env = AcquisitionEnvironment()
        def lost_response(action):
            raise ConnectionError("response lost")
        env.step = lost_response
        summary, run, worker, events, _, _ = self.run_recipe(env)
        self.assertEqual(summary["outcome"], "execution_error")
        self.assertIsNone(summary["task_success"])
        self.assertFalse(summary["milestone_reached"])
        self.assertTrue(run.terminal)
        self.assertEqual(worker.chunks, 1)
        self.assertTrue(events[-1]["result"]["execution_uncertain"])
