"""Small CPU composition checks using the real skill executor."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from harness.perception.anchors import Anchor
from harness.perception.observations import ObservationStore
from harness.planner.scripted import GraspRetryPlanner, GraspRetrySettings
from harness.runtime.orchestrator import SkillOrchestrator
from harness.predicates.hanging_mug import HangingMugPredicates
from harness.predicates.registry import PredicateRegistry
from tests.test_analytic_skills import (AnalyticEnvironment, EndpointLagEnvironment,
                                        evidence, executor)
from tests.scenario_replay import scenario


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
    def test_release_support_perception_dropout_returns_to_planner(self):
        from harness.planner.base import PlannerSelection
        from harness.predicates.base import PredicateResult
        from harness.skills.schemas import SkillCommand

        class SupportDropout:
            source = "perception"

            def evaluate(self, name, observation, *, arm="auto"):
                value = True if observation.sequence == 0 else None
                return PredicateResult(name=name, value=value, source="perception",
                    observation_sequence=observation.sequence,
                    reason="supported_before_open" if value else "temporary_mask_dropout")

            def planner_summary(self, observation):
                return {"source": "public_rgbd", "observation_sequence": observation.sequence,
                        "status": {"mug": "not_detected"}, "objects": {}}

        env = AcquisitionEnvironment()
        run, _, _ = executor(env, predicates=PredicateRegistry(perception=SupportDropout()))
        assessments, decisions = [], []

        class Planner:
            def select(self, context):
                decisions.append(context)
                if context.previous is None:
                    return PlannerSelection(SkillCommand(action="release", arm="left",
                        parameters={"ramp_ticks": 50}, reason="Open with verified table support"),
                        "HANDOFF")
                return PlannerSelection(SkillCommand(action="abort",
                    reason="Stop after planner receives the changed public evidence"), "REASSESS")

        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=assessments.append,
                save_observation=store.save, episode_id="release-support-gap", seed=7,
                max_turns=3, completion_predicate="task_success").run()

        gap = next(event for event in assessments if event.get("event_type") == "planner_turn_result"
                   and event.get("failure_type") == "release_support_lost")
        self.assertTrue(gap["recoverable_perception_gap"])
        self.assertFalse(gap["result"]["episode_terminal"])
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[1].previous.stop_reason, "release_support_lost")
        self.assertGreater(decisions[1].observation.sequence,
                           decisions[1].previous.observation_before)
        self.assertEqual(summary["outcome"], "Stop after planner receives the changed public evidence")

    def test_public_perception_gap_returns_to_planner_with_fresh_state(self):
        from harness.planner.base import PlannerSelection
        from harness.predicates.base import PredicateResult
        from harness.skills.schemas import (SkillCommand,
                                            is_recoverable_predicate_unavailable)

        class IntermittentPerception:
            source = "perception"

            def __init__(self):
                self.phases = []

            def set_task_phase(self, phase):
                self.phases.append(phase)

            def evaluate(self, name, observation, *, arm="auto", native_success=False,
                         budget_exhausted=False, allow_oracle=False):
                if name == "chunk_budget_exhausted":
                    return PredicateResult(name=name, value=budget_exhausted, source="runtime",
                        observation_sequence=observation.sequence, reason="runtime_status")
                value = False if name == "mug_lifted" and observation.sequence == 0 else None
                reason = "visible_not_lifted" if value is False else "temporary_mask_dropout"
                return PredicateResult(name=name, value=value, source="perception",
                    observation_sequence=observation.sequence, reason=reason)

            def planner_summary(self, observation):
                return {"source": "public_rgbd", "observation_sequence": observation.sequence,
                        "status": {"mug": "not_detected"}, "objects": {}}

        env = AcquisitionEnvironment()
        gates = PredicateRegistry(perception=IntermittentPerception())
        run, worker, events = executor(env, predicates=gates)
        assessments, decisions = [], []

        class Planner:
            def select(self, context):
                decisions.append(context)
                if context.previous is None:
                    return PlannerSelection(SkillCommand(action="vla_act", arm="both",
                        stop="mug_lifted", parameters={"max_chunks": 1, "chunk_size": 4,
                            "max_policy_actions": 4}, reason="Approach and assess the mug"), "GRASP")
                if context.previous.stop_reason == "predicate_unavailable":
                    return PlannerSelection(SkillCommand(action="vla_act", arm="both",
                        stop="chunk_budget_exhausted", parameters={"max_chunks": 1,
                            "chunk_size": 4, "max_policy_actions": 4},
                        reason="Use one feedback chunk while visual evidence recovers"), "GRASP", 1)
                return PlannerSelection(SkillCommand(action="abort", reason="end CPU scenario"), "FAILED")

        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=assessments.append,
                save_observation=store.save, episode_id="perception-gap-retry", seed=12,
                max_turns=3, completion_predicate="task_success").run()

        gap_events = [event for event in assessments if event.get("event_type") == "planner_turn_result"
                      and event.get("failure_type") == "predicate_unavailable"]
        self.assertTrue(gap_events, repr(assessments))
        gap = gap_events[0]
        self.assertTrue(gap["recoverable_perception_gap"])
        self.assertFalse(gap["result"]["episode_terminal"])
        self.assertEqual(decisions[1].previous.predicate.value, None)
        self.assertGreater(decisions[1].observation.sequence,
                           decisions[1].previous.observation_before)
        actions = [event["command"]["action"] for event in events
                   if event.get("event_type") == "skill_requested"]
        self.assertEqual(actions, ["vla_act", "vla_act", "abort"])
        self.assertEqual(worker.chunks, 2)
        self.assertEqual(gates.perception.phases[:2], ["APPROACH_MUG", "GRASP"])
        first = next(event["result"] for event in assessments
                     if event.get("event_type") == "planner_turn_result"
                     and event.get("failure_type") == "predicate_unavailable")
        self.assertEqual(first["discarded_actions"], 3)
        result = decisions[1].previous
        self.assertTrue(is_recoverable_predicate_unavailable(result))
        self.assertFalse(is_recoverable_predicate_unavailable(
            result.model_copy(update={"execution_uncertain": True})))
        self.assertFalse(is_recoverable_predicate_unavailable(
            result.model_copy(update={"information_source": "development_oracle"})))
        self.assertFalse(is_recoverable_predicate_unavailable(
            result.model_copy(update={"observation_after": result.observation_before})))
        no_motion_gap = result.model_copy(update={
            "executed_policy_actions": 0, "vla_chunks_requested": 0,
            "observation_after": result.observation_before,
            "state_after": result.state_before,
            "predicate": result.predicate.model_copy(update={
                "observation_sequence": result.observation_before}),
        })
        self.assertTrue(is_recoverable_predicate_unavailable(no_motion_gap))
        self.assertEqual(summary["outcome"], "end CPU scenario")

    def test_inter_arm_contact_is_returned_to_planner_for_retreat(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand

        class ContactEnvironment(AcquisitionEnvironment):
            def __init__(self):
                super().__init__()
                self.overlap = False

            def arm_overlap_detected(self):
                return self.overlap

            def step(self, action):
                self.overlap = True
                return super().step(action)

            def tick(self, arm, position, velocity, gripper, other_hold):
                super().tick(arm, position, velocity, gripper, other_hold)
                if arm == "right":
                    self.overlap = False

        env = ContactEnvironment()
        run, _, events = executor(env)
        assessments, decisions = [], []

        class Planner:
            def select(self, context):
                decisions.append(context)
                if context.previous is None:
                    command = SkillCommand(action="vla_act", arm="both", stop="chunk_budget_exhausted",
                        parameters={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 4},
                        reason="Run one bounded segment")
                    return PlannerSelection(command, "HANDOFF")
                if context.previous.stop_reason == "arm_overlap_detected":
                    command = SkillCommand(action="move_to", arm="right",
                        parameters={"target": "current_ee", "offset_xyz": (0.05, 0.0, 0.0)},
                        reason="Retreat from the measured contact")
                    return PlannerSelection(command, "HANDOFF")
                return PlannerSelection(SkillCommand(action="abort", reason="end CPU scenario"), "FAILED")

        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=assessments.append,
                save_observation=store.save, episode_id="inter-arm-retry", seed=5,
                max_turns=3, completion_predicate="task_success").run()

        turn = next(event for event in assessments if event.get("event_type") == "planner_turn_result"
                    and event.get("failure_type") == "arm_overlap_detected")
        self.assertTrue(turn["recoverable_arm_overlap"])
        self.assertFalse(turn["result"]["episode_terminal"])
        self.assertTrue(decisions[1].previous.arm_overlap_detected)
        self.assertEqual([event["command"]["action"] for event in events
                          if event.get("event_type") == "skill_requested"],
                         ["vla_act", "move_to", "abort"])
        self.assertEqual(summary["outcome"], "end CPU scenario")

    def test_chunk_budget_stop_is_a_feedback_boundary_not_a_task_failure(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand

        env = AcquisitionEnvironment()
        gates = PredicateRegistry(HangingMugPredicates(env.evidence, source="development_oracle"))
        run, _, _ = executor(env, predicates=gates, oracle=True)
        assessments = []

        class Planner:
            calls = 0
            def select(self, context):
                self.calls += 1
                if self.calls == 1:
                    command = SkillCommand(action="vla_act", arm="both", stop="chunk_budget_exhausted",
                        parameters={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 8},
                        reason="Run one bounded segment and inspect the new image")
                    return PlannerSelection(command, "HANDOFF")
                return PlannerSelection(SkillCommand(action="abort", reason="stop test"), "FAILED")

        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=assessments.append,
                save_observation=store.save, episode_id="chunk-feedback", seed=4,
                max_turns=2, completion_predicate="task_success").run()

        first_result = next(event for event in assessments
            if event.get("event_type") == "planner_turn_result")
        self.assertEqual(first_result["result"]["stop_reason"], "chunk_budget_exhausted")
        self.assertIsNone(first_result["failure_type"])
        self.assertFalse(first_result["milestone_reached"])
        self.assertFalse(summary["task_success"])

    def test_pre_motion_retreat_plan_rejection_allows_one_planner_fallback(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand, is_retryable_pre_motion_planning_failure

        env = AcquisitionEnvironment(succeeds_after=3)
        env.plan_status = "Fail"
        gates = PredicateRegistry(HangingMugPredicates(env.evidence, source="development_oracle"))
        run, worker, events = executor(env, predicates=gates, oracle=True)
        assessments, decisions = [], []

        class Planner:
            def select(self, context):
                decisions.append(context)
                if context.previous is None:
                    return PlannerSelection(SkillCommand(action="move_to", arm="left",
                        parameters={"target": "mug_pregrasp"}, reason="test staging"), "APPROACH_MUG")
                self_result = context.previous
                if is_retryable_pre_motion_planning_failure(self_result):
                    return PlannerSelection(SkillCommand(action="vla_act", arm="both",
                        stop="mug_lifted", parameters={"chunk_size": 4,
                        "max_chunks": 1, "max_policy_actions": 4},
                        reason="bounded contact-rich retry after rejected pre-motion plan"), "GRASP_MUG", 1)
                return PlannerSelection(SkillCommand(action="abort", reason="unsafe retry"), "FAILED")

        def prepare(command, observation):
            if command.action.value == "move_to":
                pose = env.arm_state(command.arm)["pose"]
                run.anchors.bind("mug_pregrasp", command.arm,
                    Anchor(tuple(pose), observation.sequence, source="development_oracle"))

        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=assessments.append,
                save_observation=store.save, episode_id="plan-retry", seed=4,
                max_turns=3, prepare=prepare).run()

        self.assertTrue(is_retryable_pre_motion_planning_failure(decisions[1].previous))
        self.assertEqual([e["command"]["action"] for e in events
                          if e.get("event_type") == "skill_requested"], ["move_to", "vla_act"])
        self.assertEqual(summary["outcome"], "acquisition_confirmed")
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(worker.chunks, 1)
        failed_turn = next(e for e in assessments if e.get("event_type") == "planner_turn_result"
                           and e.get("failure_type") == "planning_failure")
        self.assertTrue(failed_turn["retryable_pre_motion_planning_failure"])

    def test_executed_target_miss_returns_fresh_state_for_release_replan(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand

        env = EndpointLagEnvironment(fixed_error=(0.034, 0.16))
        gates = PredicateRegistry(HangingMugPredicates(
            lambda obs: evidence(obs.sequence), source="development_oracle"))
        run, _, events = executor(env, predicates=gates, oracle=True)
        assessments, contexts = [], []

        class Planner:
            def select(self, context):
                contexts.append(context)
                if context.previous is None:
                    command = SkillCommand(action="move_to", arm="left",
                        parameters={"target": "current_ee", "offset_xyz": (0.02, 0., 0.)},
                        reason="Move to the bounded transfer position")
                    return PlannerSelection(command, "HANDOFF")
                if context.previous.stop_reason == "target_not_reached":
                    command = SkillCommand(action="release", arm="left",
                        parameters={"ramp_ticks": 50},
                        reason="The new image and support check allow releasing onto the table")
                    return PlannerSelection(command, "HANDOFF")
                return PlannerSelection(SkillCommand(action="abort", reason="end CPU scenario"), "FAILED")

        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=assessments.append,
                save_observation=store.save, episode_id="target-miss-replan", seed=9,
                max_turns=3, completion_predicate="task_success").run()

        failed_turn = next(e for e in assessments if e.get("event_type") == "planner_turn_result"
            and e.get("failure_type") == "target_not_reached")
        self.assertTrue(failed_turn["recoverable_executed_motion_failure"])
        self.assertFalse(failed_turn["result"]["episode_terminal"])
        self.assertEqual(contexts[1].previous.stop_reason, "target_not_reached")
        self.assertGreater(contexts[1].observation.sequence, contexts[1].previous.observation_before)
        commands = [e["command"]["action"] for e in events if e.get("event_type") == "skill_requested"]
        self.assertEqual(commands, ["move_to", "release", "abort"])
        self.assertEqual(summary["outcome"], "end CPU scenario")

    def test_target_miss_with_uncertain_or_terminal_state_is_not_recoverable(self):
        from harness.skills.schemas import SkillResult, is_recoverable_executed_motion_failure

        state = {"command_state": [0.] * 14}
        result = SkillResult.model_validate({
            "call_id": 1, "skill": "move_to", "completed": False,
            "stop_reason": "target_not_reached", "executed_policy_actions": 0,
            "vla_chunks_requested": 0, "discarded_actions": 0,
            "executed_physics_ticks": 10, "state_before": state, "state_after": state,
            "observation_before": 1, "observation_after": 2, "episode_terminal": False,
            "task_success": False, "controlled_arms": "left", "elapsed_seconds": 0.1,
            "position_error_m": 0.03, "orientation_error_rad": 0.12,
        })
        self.assertTrue(is_recoverable_executed_motion_failure(result))
        self.assertFalse(is_recoverable_executed_motion_failure(
            result.model_copy(update={"episode_terminal": True})))
        self.assertFalse(is_recoverable_executed_motion_failure(
            result.model_copy(update={"execution_uncertain": True})))
        self.assertFalse(is_recoverable_executed_motion_failure(
            result.model_copy(update={"observation_after": 1})))
        self.assertFalse(is_recoverable_executed_motion_failure(
            result.model_copy(update={"information_source": "development_oracle"})))

    def run_recipe(self, env, *, retries=1, retry_offsets=(), turns=20, predicates=True,
                   prepare_failure=False, stage=True, vla_actions=4, failure_stop=None):
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
            retry_stage_offsets=retry_offsets, stage_before_first_vla=stage,
            vla={"max_chunks": (vla_actions + 3) // 4, "chunk_size": 4,
                 "max_policy_actions": vla_actions, "failure_stop": failure_stop}))
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

    def test_retry_uses_bounded_configured_offset_after_fresh_anchor_binding(self):
        summary, _, _, events, _, stages = self.run_recipe(AcquisitionEnvironment(5),
            retry_offsets=((0.01, 0.0, 0.0),))
        moves = [e["command"] for e in events if e.get("event_type") == "skill_requested"
                 and e["command"]["action"] == "move_to"]
        self.assertEqual(summary["outcome"], "acquisition_confirmed")
        self.assertEqual(len(stages), 2)
        self.assertEqual(moves[0]["parameters"]["offset_xyz"], [0.0, 0.0, 0.0])
        self.assertEqual(moves[-1]["parameters"]["offset_xyz"], [0.01, 0.0, 0.0])
        with self.assertRaises(ValueError):
            GraspRetrySettings(max_retries=1, retry_stage_offsets=((0.3, 0.0, 0.0),))

    def test_retry_exhaustion_aborts_without_more_inference(self):
        summary, _, worker, events, _, _ = self.run_recipe(AcquisitionEnvironment())
        self.assertEqual(summary["outcome"], "grasp_retry_budget_exhausted")
        self.assertFalse(summary["milestone_reached"])
        self.assertEqual(worker.chunks, 2)
        self.assertEqual(events[-1]["result"]["skill"], "abort")

    def test_no_lift_failure_monitor_stops_vla_early_then_uses_existing_retry(self):
        summary, _, worker, events, _, _ = self.run_recipe(AcquisitionEnvironment(),
            retries=1, stage=False, vla_actions=64, failure_stop="grasp_attempt_failed")
        results = [e["result"] for e in events if e.get("event_type") == "skill_result"]
        self.assertEqual([r["stop_reason"] for r in results if r["skill"] == "vla_act"],
                         ["failure_predicate_satisfied", "failure_predicate_satisfied"])
        self.assertEqual([r["executed_policy_actions"] for r in results if r["skill"] == "vla_act"],
                         [40, 40])
        self.assertEqual(summary["outcome"], "grasp_retry_budget_exhausted")
        self.assertEqual(worker.chunks, 20)

    def test_recorded_empty_grasp_retries_and_successful_attempt_lifts_within_window(self):
        # Replay the observed miss -> backoff/restage -> successful retry shape
        # without RoboTwin, Modal, or GPU.
        recorded = scenario("empty_grasp_then_successful_retry")
        failure_window = recorded["failure_window_closed_actions"]
        retry_lift_action = failure_window + recorded["retry_lift_after_policy_action"] - 1
        env = AcquisitionEnvironment()
        def replayed_evidence(obs):
            lifted = env.policy_steps >= retry_lift_action
            return evidence(obs.sequence, left_contact=lifted, left_bilateral=lifted,
                table_contact=not lifted,
                mug_position=(-0.2, 0.0, 0.80 if lifted else 0.75))
        env.evidence = replayed_evidence
        summary, _, worker, events, _, _ = self.run_recipe(env, retries=1,
            stage=False, vla_actions=64, failure_stop="grasp_attempt_failed")
        attempts = [event["result"] for event in events
                    if event.get("event_type") == "skill_result" and event["result"]["skill"] == "vla_act"]
        self.assertEqual([attempt["stop_reason"] for attempt in attempts], recorded["expected_vla_stops"])
        self.assertEqual([attempt["executed_policy_actions"] for attempt in attempts], recorded["expected_vla_actions"])
        self.assertEqual(summary["outcome"], recorded["expected_outcome"])
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(worker.policy_resets, 2)
        self.assertEqual([event["command"]["action"] for event in events
                          if event.get("event_type") == "skill_requested"], recorded["expected_skills"])

    def test_natural_entry_stops_on_lift_and_discards_tail_without_staging(self):
        summary, _, worker, events, _, stages = self.run_recipe(AcquisitionEnvironment(1), stage=False)
        self.assertEqual(summary["outcome"], "acquisition_confirmed")
        self.assertEqual(stages, [])
        self.assertEqual(summary["analytic_ticks"], 0)
        self.assertEqual(worker.policy_resets, 1)
        self.assertEqual([e['command']['action'] for e in events if e.get('event_type')=='skill_requested'], ['vla_act'])
        self.assertGreater(events[-1]['result']['discarded_actions'], 0)

    def test_natural_entry_exhaustion_aborts_without_retry(self):
        summary, _, worker, events, _, stages = self.run_recipe(AcquisitionEnvironment(), stage=False, retries=0)
        self.assertEqual(summary['outcome'], 'grasp_retry_budget_exhausted')
        self.assertEqual(stages, [])
        self.assertEqual(worker.chunks, 1)
        self.assertEqual([e['command']['action'] for e in events if e.get('event_type')=='skill_requested'], ['vla_act', 'abort'])

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

    def test_rejected_grounding_gets_one_feedback_turn_without_motion(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand
        env = AcquisitionEnvironment()
        gates = PredicateRegistry(HangingMugPredicates(env.evidence, source="development_oracle"))
        run, worker, events = executor(env, predicates=gates, oracle=True)
        feedback_seen = []

        class Planner:
            def select(self, context):
                feedback_seen.append(context.grounding_feedback)
                if context.grounding_feedback:
                    return PlannerSelection(SkillCommand(action="abort", reason="pixel_rejected"), "FAILED")
                return PlannerSelection(SkillCommand.model_validate({"action": "move_to", "arm": "left",
                    "parameters": {"target": "mug_pregrasp", "grounding": {
                        "camera": "head_camera", "uv": (0.3, 0.6)}}}), "APPROACH_MUG")

        def reject_pixel(command, _obs):
            if command.action == "move_to":
                raise ValueError("grounding_pixel_behind_local_surface")

        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=logs.append,
                save_observation=store.save, episode_id="pixel-retry", seed=3,
                max_turns=4, prepare=reject_pixel).run()
        results = [event for event in logs if event.get("event_type") == "planner_turn_result"]
        self.assertEqual(feedback_seen[0], None)
        self.assertIn("lower outer mug wall", feedback_seen[1])
        self.assertNotIn("0.3", feedback_seen[1])
        self.assertEqual([item["grounding_rejected"] for item in results[:1]], [True])
        self.assertFalse(results[0]["skill_executed"])
        self.assertEqual(summary["grounding_reselections"], 1)
        self.assertEqual(summary["planner_turns"], 2)
        self.assertEqual((run.calls, env.ticks, worker.chunks), (1, 0, 0))
        self.assertEqual(summary["outcome"], "pixel_rejected")

    def test_missing_symbolic_target_returns_grounding_feedback_to_planner(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand
        env = AcquisitionEnvironment()
        gates = PredicateRegistry(HangingMugPredicates(env.evidence, source="development_oracle"))
        run, worker, events = executor(env, predicates=gates, oracle=True)
        contexts, logs = [], []

        class Planner:
            def select(self, context):
                contexts.append(context)
                if context.grounding_feedback:
                    return PlannerSelection(SkillCommand(action="abort", reason="no grounded target"),
                                            "FAILED")
                return PlannerSelection(SkillCommand.model_validate({"action": "move_to", "arm": "left",
                    "parameters": {"target": "table_approach"}}), "HANDOFF")

        def missing_table_anchor(_command, _observation):
            if _command.action == "move_to":
                raise ValueError("grounded_target_unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "observations")
            summary = SkillOrchestrator(run, Planner(), log=logs.append,
                save_observation=store.save, episode_id="missing-symbolic-target", seed=9,
                max_turns=3, prepare=missing_table_anchor).run()

        results = [item for item in logs if item.get("event_type") == "planner_turn_result"]
        self.assertTrue(results[0]["grounding_rejected"])
        self.assertEqual(results[0]["grounding_rejection_code"],
                         "symbolic_target_grounding_unavailable")
        self.assertIn("table_approach", contexts[1].grounding_feedback)
        self.assertEqual(summary["planner_turns"], 2)
        self.assertEqual(summary["outcome"], "no grounded target")
        self.assertEqual((run.calls, worker.chunks), (1, 0))

    def test_rack_pixel_reselection_feedback_is_rack_specific(self):
        from harness.runtime.orchestrator import grounding_reselection_feedback
        message = grounding_reselection_feedback("rack_approach")
        self.assertIn("hanger or support bar", message)
        self.assertIn("background", message)
        self.assertNotIn("mug wall", message)

    def test_grounding_reselection_is_bounded_after_second_rejection(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand
        env = AcquisitionEnvironment()
        gates = PredicateRegistry(HangingMugPredicates(env.evidence, source="development_oracle"))
        run, worker, events = executor(env, predicates=gates, oracle=True)
        contexts = []

        class Planner:
            def select(self, context):
                contexts.append(context)
                return PlannerSelection(SkillCommand.model_validate({"action": "move_to", "arm": "left",
                    "parameters": {"target": "mug_pregrasp", "grounding": {
                        "camera": "head_camera", "uv": (0.3, 0.6)}}}), "APPROACH_MUG")

        def reject_pixel(command, _obs):
            raise ValueError("grounding_pixel_behind_local_surface")

        logs = []
        summary = SkillOrchestrator(run, Planner(), log=logs.append,
            save_observation=lambda obs: {"sequence": obs.sequence}, episode_id="pixel-stop", seed=3,
            max_turns=5, prepare=reject_pixel).run()
        self.assertEqual(summary["outcome"], "grounding_reselection_budget_exhausted")
        self.assertEqual(summary["grounding_reselections"], 3)
        self.assertEqual(summary["planner_turns"], 3)
        self.assertEqual((run.calls, env.ticks, worker.chunks), (0, 0, 0))
        rejections = [event for event in logs if event.get("grounding_rejected")]
        self.assertEqual(len(rejections), 3)
        self.assertEqual([item["grounding_rejection_code"] for item in rejections], [
            "grounding_pixel_behind_local_surface", "grounding_pixel_behind_local_surface",
            "grounding_camera_not_changed"])
        self.assertEqual(rejections[-1]["failure_type"], "grounding_reselection_budget_exhausted")
        self.assertEqual(contexts[2].grounding_camera_to_avoid, "head_camera")
        self.assertIn("different camera view", contexts[2].grounding_feedback)
        self.assertEqual([context.remaining_budget["grounding_reselections"] for context in contexts],
                         [2, 2, 1])
        self.assertEqual(len([event for event in events if event.get("event_type") == "skill_requested"]), 0)

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
