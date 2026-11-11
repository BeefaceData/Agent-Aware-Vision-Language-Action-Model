"""Phase composition and evidence accounting without native simulation or APIs."""

from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness.jsonl import append_jsonl
from harness.planner.handoff import HandoffPlanner
from harness.planner.scripted import GraspRetryPlanner, GraspRetrySettings
from harness.predicates.hanging_mug import HangingMugPredicates
from harness.predicates.registry import PredicateRegistry
from harness.runtime.orchestrator import SkillOrchestrator
from harness.skills.schemas import VLAParameters
from harness.skills.schemas import MoveParameters, SkillCommand
from harness.perception.anchors import Anchor
from development.simulator.skill_oracle import oracle_anchor_preparer
from harness.evaluation.manifest import EvaluationSettings
from harness.exploration.candidates import candidates_from_attempt
from harness.exploration.controller import ExplorationController, write_json
from harness.exploration.config import ExploreSettings
from harness.exploration.process import AttemptExit
from run_skills import development_anchor_fallback
from tests.test_analytic_skills import evidence, executor
from tests.test_skill_orchestrator import AcquisitionEnvironment
from tests.scenario_replay import scenario


class TransferEnvironment(AcquisitionEnvironment):
    def __init__(self, *, transfers=True, acquires=True, uncertain=False):
        super().__init__()
        self.transfers, self.acquires, self.uncertain = transfers, acquires, uncertain
        self.home_pose = self.arms["left"]["pose"].copy()
        self.home_pose[0] += 0.2

    @property
    def native_success(self):
        return self.transfers and self.policy_steps >= 10

    def step(self, action):
        if self.uncertain and self.policy_steps == 2:
            raise ConnectionError('lost response')
        return super().step(action)

    def evidence(self, obs):
        held = self.acquires and self.policy_steps >= 1
        staged = held and self.policy_steps >= 4
        right_held = self.acquires and self.policy_steps >= 7
        left = held and not staged and not right_held
        return evidence(obs.sequence, left_contact=left, left_bilateral=left,
            right_contact=right_held, right_bilateral=right_held,
            table_contact=not held or (staged and not right_held),
            left_gripper=1.0 if staged else 0.0, right_gripper=0.0 if right_held else 1.0,
            mug_position=(-0.2, 0.0, .80 if (held and not staged) or right_held else .75),
            stage_distance_m=0.02 if staged else 0.4)


def run_transfer(output, env, *, completion_predicate='task_success'):
    output.mkdir()
    gates = PredicateRegistry(HangingMugPredicates(env.evidence, source='development_oracle'))
    runtime, worker, events = executor(env, predicates=gates, oracle=True)
    def prepare(command, observation):
        if command.action == "move_to" and command.parameters.target == "left_home":
            runtime.anchors.bind("left_home", "left",
                Anchor(tuple(env.home_pose), observation.sequence, source="development_oracle"))
        if command.action == "move_to" and command.parameters.target == "rack_approach":
            pose = env.arm_state("right")["pose"].copy()
            pose[0] += 0.02
            runtime.anchors.bind("rack_approach", "right",
                Anchor(tuple(pose), observation.sequence, source="development_oracle"))
    bounds = VLAParameters(max_chunks=1, chunk_size=4, max_policy_actions=4)
    planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False, max_retries=0, vla=bounds), bounds)
    summary = SkillOrchestrator(runtime, planner, completion_predicate=completion_predicate,
        log=partial(append_jsonl, output/'assessments.jsonl'), save_observation=lambda o: {'sequence':o.sequence},
        episode_id=output.name, seed=3, prepare=prepare).run()
    write_json(output/'summary.json',summary)
    return summary, worker, events


class HandoffTests(unittest.TestCase):
    def test_development_anchor_fallback_is_limited_to_unavailable_mug_grounding(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillCommand

        calls, logs = [], []
        def public_prepare(command, observation):
            raise TypeError("perception output unavailable for this mug pose")
        def oracle_prepare(command, observation):
            calls.append((command.parameters.target, observation.sequence))
            return "oracle_anchor"
        prepare = development_anchor_fallback(public_prepare, oracle_prepare, logs.append)
        command = SkillCommand(action="move_to", arm="left",
            parameters={"target": "current_ee", "path_mode": "reverse_approach",
                        "retreat_distance_m": 0.05, "orientation_mode": "preserve_current"})
        observation = SimpleNamespace(sequence=17)
        self.assertEqual(prepare(command, observation), "oracle_anchor")
        self.assertEqual(calls, [("current_ee", 17)])
        self.assertEqual(logs[0]["information_source"], "development_oracle")
        self.assertEqual(logs[0]["error_type"], "TypeError")

        def unavailable_oracle(command, observation):
            raise TypeError("no valid grasp pose")

        prepare_with_unavailable_oracle = development_anchor_fallback(
            public_prepare, unavailable_oracle, logs.append)
        with self.assertRaisesRegex(ValueError, "grounded_target_unavailable"):
            prepare_with_unavailable_oracle(command, observation)
        self.assertEqual(logs[-1]["event_type"], "development_oracle_anchor_unavailable")
        self.assertEqual(logs[-1]["error_type"], "TypeError")

        command = SkillCommand(action="move_to", arm="right",
            parameters={"target": "rack_approach"})
        with self.assertRaisesRegex(TypeError, "perception output unavailable"):
            prepare(command, observation)

    def test_full_task_routes_left_restage_planning_failure_to_retry_fallback(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        planner = HandoffPlanner(GraspRetrySettings(
            stage_before_first_vla=False, max_retries=2),
            VLAParameters(max_chunks=1, chunk_size=4, max_policy_actions=4))
        planner.acquisition.phase = "APPROACH_MUG"
        planner.acquisition.retries = 2
        planner.acquisition.reorientation_attempted = True
        previous = SimpleNamespace(episode_terminal=False, execution_uncertain=False,
            completed=False, skill=SkillType.MOVE_TO, stop_reason="planning_failure")
        context = SimpleNamespace(previous=previous, value=lambda name: None)

        selection = planner.select(context)
        self.assertEqual(selection.phase, "RETRY_RESTAGE_FALLBACK")
        self.assertEqual(selection.command.action, SkillType.MOVE_TO)
        self.assertEqual(selection.command.parameters.orientation_mode, "preserve_current")

    def test_retry_vla_can_finish_small_grounded_pregrasp_residual(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        planner = GraspRetryPlanner(GraspRetrySettings(
            stage_before_first_vla=False, max_retries=2,
            vla={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 4}))
        values = {"mug_lifted": False, "mug_grasp_confirmed": False,
                  "release_ready": True}
        visual = {"status": {"mug": "grounded"}}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name), visual_context=visual)

        def result(skill, *, completed=True, stop="completed", position=None, orientation=None):
            return SimpleNamespace(skill=SkillType(skill), completed=completed,
                episode_terminal=False, execution_uncertain=False, stop_reason=stop,
                position_error_m=position, orientation_error_rad=orientation)

        planner.phase = "APPROACH_MUG"
        planner.retries = 1
        near_miss = result("move_to", completed=False, stop="target_not_reached",
                           position=0.0274, orientation=0.064)
        selection = planner.select(context(near_miss))
        self.assertEqual(selection.command.action, SkillType.VLA_ACT)
        self.assertEqual(selection.phase, "GRASP_MUG")

        visual["status"]["mug"] = "not_detected"
        planner = GraspRetryPlanner(GraspRetrySettings(
            stage_before_first_vla=False, max_retries=2,
            vla={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 4}))
        planner.phase = "APPROACH_MUG"
        planner.retries = 1
        planner.reorientation_attempted = False
        self.assertEqual(planner.select(context(near_miss)).command.action, SkillType.ABORT)

    def test_first_right_retry_retreats_without_wrist_rotation(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=1, vla=bounds), bounds)
        values = {"mug_lifted": False, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}

        def context(previous):
            return SimpleNamespace(previous=previous, value=lambda name: values.get(name),
                visual_context={"status": {"mug": "grounded"}})

        def result(skill, *, completed=True, stop="completed", objective=True):
            state = SimpleNamespace(command_state={"arm": "unchanged"})
            return SimpleNamespace(episode_terminal=False, completed=completed,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=None,
                executed_physics_ticks=0, executed_policy_actions=0,
                vla_chunks_requested=0, state_before=state, state_after=state)

        transfer = result(SkillType.VLA_ACT)
        transfer.predicate = SimpleNamespace(name="left_transfer_complete", value=True)
        planner.select(context(transfer))
        failed_grasp = result(SkillType.VLA_ACT, stop="chunk_budget_exhausted", objective=False)
        planner.select(context(failed_grasp))
        retreat = planner.select(context(result(SkillType.SET_GRIPPER)))
        self.assertEqual((retreat.phase, retreat.command.parameters.target),
                         ("RIGHT_RETRY_RETREAT", "current_ee"))
        self.assertEqual(retreat.command.parameters.path_mode, "reverse_approach")
        self.assertEqual(retreat.command.parameters.retreat_distance_m, 0.05)
        self.assertEqual(retreat.command.parameters.orientation_mode, "preserve_current")
        self.assertEqual(retreat.command.parameters.max_physics_ticks,
                         HandoffPlanner.RIGHT_RESTAGE_MAX_PHYSICS_TICKS)
        retry = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual(retry.phase, "RIGHT_GRASP_RETRY")
        self.assertEqual(retry.command.action.value, "vla_act")

    def test_public_false_lift_evidence_allows_retry_when_skill_result_has_no_objective(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=2, vla=bounds), bounds)
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name))
        result = lambda skill, **kwargs: SimpleNamespace(
            episode_terminal=False, completed=True, execution_uncertain=False,
            objective_satisfied=kwargs.get("objective"),
            stop_reason=kwargs.get("stop", "completed"), skill=skill,
            predicate=None, executed_physics_ticks=0, executed_policy_actions=0,
            vla_chunks_requested=0)

        transfer = result(SkillType.VLA_ACT)
        transfer.predicate = SimpleNamespace(name="left_transfer_complete", value=True)
        planner.select(context(transfer))
        failed_vla = result(SkillType.VLA_ACT, stop="policy_action_budget_exhausted")
        retry = planner.select(context(failed_vla))

        self.assertEqual(retry.phase, "RIGHT_RETRY_OPEN")
        self.assertEqual(retry.command.action, SkillType.SET_GRIPPER)

    def test_second_right_retry_reorients_only_after_retreat(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=2, vla=bounds), bounds)
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name))
        result = lambda skill, **kwargs: SimpleNamespace(
            episode_terminal=False, completed=True, execution_uncertain=False,
            objective_satisfied=kwargs.get("objective", True),
            stop_reason=kwargs.get("stop", "completed"), skill=skill,
            predicate=kwargs.get("predicate"), executed_physics_ticks=0,
            executed_policy_actions=0, vla_chunks_requested=0)

        transfer = result(SkillType.VLA_ACT,
            predicate=SimpleNamespace(name="left_transfer_complete", value=True))
        planner.select(context(transfer))
        failed = result(SkillType.VLA_ACT, stop="policy_action_budget_exhausted", objective=False)
        opened1 = planner.select(context(failed))
        retreat1 = planner.select(context(result(SkillType.SET_GRIPPER)))
        retry1 = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual([opened1.command.action.value, retreat1.command.action.value,
                          retry1.command.action.value], ["set_gripper", "move_to", "vla_act"])
        self.assertEqual(retreat1.command.parameters.retreat_distance_m, 0.05)

        opened2 = planner.select(context(failed))
        retreat2 = planner.select(context(result(SkillType.SET_GRIPPER)))
        reorient2 = planner.select(context(result(SkillType.MOVE_TO)))
        retry2 = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual([opened2.command.action.value, retreat2.command.action.value,
                          reorient2.command.action.value, retry2.command.action.value],
                         ["set_gripper", "move_to", "move_to", "vla_act"])
        self.assertEqual(opened2.phase, "RIGHT_RETRY_OPEN")
        self.assertEqual(retreat2.command.parameters.retreat_distance_m, 0.10)
        self.assertEqual(retreat2.command.parameters.orientation_mode, "preserve_current")
        self.assertEqual(reorient2.phase, "RIGHT_RETRY_REORIENT")
        self.assertEqual(reorient2.command.parameters.position_mode, "preserve_current")
        self.assertEqual(retry2.phase, "RIGHT_GRASP_RETRY")

    def test_failed_second_right_reorientation_falls_back_to_preserved_wrist(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=2, vla=bounds), bounds)
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name))

        def result(skill, *, completed=True, stop="completed", objective=True, predicate=None):
            return SimpleNamespace(episode_terminal=False, completed=completed,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=predicate,
                executed_physics_ticks=0, executed_policy_actions=0, vla_chunks_requested=0)

        transfer = result(SkillType.VLA_ACT,
            predicate=SimpleNamespace(name="left_transfer_complete", value=True))
        planner.select(context(transfer))
        failed_attempt = result(SkillType.VLA_ACT,
            stop="policy_action_budget_exhausted", objective=False)
        planner.select(context(failed_attempt))
        planner.select(context(result(SkillType.SET_GRIPPER)))
        planner.select(context(result(SkillType.MOVE_TO)))
        planner.select(context(failed_attempt))
        planner.select(context(result(SkillType.SET_GRIPPER)))
        reorient = planner.select(context(result(SkillType.MOVE_TO)))

        fallback = planner.select(context(result(SkillType.MOVE_TO,
            completed=False, stop="trajectory_limits", objective=False)))
        self.assertEqual(fallback.phase, "RIGHT_RETRY_RESTAGE_FALLBACK")
        self.assertEqual(fallback.command.parameters.target, "mug_pregrasp")
        self.assertEqual(fallback.command.parameters.position_mode, "target")
        self.assertEqual(fallback.command.parameters.orientation_mode, "preserve_current")

        retry = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual(retry.phase, "RIGHT_GRASP_RETRY")
        self.assertEqual(retry.command.action, SkillType.VLA_ACT)
        self.assertEqual(reorient.phase, "RIGHT_RETRY_REORIENT")

    def test_right_reorientation_motion_failures_use_preserved_wrist_fallback(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name))

        def result(skill, *, completed=True, stop="completed", objective=True, predicate=None):
            return SimpleNamespace(episode_terminal=False, completed=completed,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=predicate,
                executed_physics_ticks=0, executed_policy_actions=0, vla_chunks_requested=0)

        for stop_reason in ("planning_failure", "tracking_error"):
            with self.subTest(stop_reason=stop_reason):
                planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
                    max_retries=2, vla=bounds), bounds)
                transfer = result(SkillType.VLA_ACT,
                    predicate=SimpleNamespace(name="left_transfer_complete", value=True))
                planner.select(context(transfer))
                failed_attempt = result(SkillType.VLA_ACT,
                    stop="policy_action_budget_exhausted", objective=False)
                planner.select(context(failed_attempt))
                planner.select(context(result(SkillType.SET_GRIPPER)))
                planner.select(context(result(SkillType.MOVE_TO)))
                planner.select(context(failed_attempt))
                planner.select(context(result(SkillType.SET_GRIPPER)))
                planner.select(context(result(SkillType.MOVE_TO)))
                failed_reorientation = result(SkillType.MOVE_TO, completed=False,
                                              stop=stop_reason, objective=False)
                fallback = planner.select(context(failed_reorientation))
                self.assertEqual(fallback.phase, "RIGHT_RETRY_RESTAGE_FALLBACK")
                self.assertEqual(fallback.command.parameters.orientation_mode,
                                 "preserve_current")
                retry = planner.select(context(result(SkillType.MOVE_TO)))
                self.assertEqual(retry.command.action, SkillType.VLA_ACT)
                self.assertEqual(retry.phase, "RIGHT_GRASP_RETRY")

    def test_injected_right_wrist_residual_uses_handoff_planner_fallback(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillResult, SkillType, TrackingError

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=2, vla=bounds), bounds)
        planner.left_transfer_confirmed = True
        planner.right_retry_phase = "reorientation_attempt"
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name))
        state = {"command_state": [0.0] * 14}
        measured_residual = 0.251
        diagnostic = TrackingError(arm="right", comparison="controlled_arm",
            joint_index=2, measured_position_rad=measured_residual,
            commanded_position_rad=0.0, residual_rad=measured_residual,
            threshold_rad=0.25)
        failure = SkillResult.model_validate({
            "call_id": 1, "skill": SkillType.MOVE_TO, "completed": False,
            "stop_reason": "tracking_error", "executed_policy_actions": 0,
            "vla_chunks_requested": 0, "discarded_actions": 0,
            "executed_physics_ticks": 320, "state_before": state,
            "state_after": state, "observation_before": 7, "observation_after": 8,
            "episode_terminal": False, "task_success": False,
            "controlled_arms": "right", "elapsed_seconds": 0.1,
            "tracking_error": diagnostic,
        })

        fallback = planner.select(context(failure))
        self.assertEqual(fallback.phase, "RIGHT_RETRY_RESTAGE_FALLBACK")
        self.assertEqual(fallback.command.action, SkillType.MOVE_TO)
        self.assertEqual(fallback.command.parameters.target, "mug_pregrasp")
        self.assertEqual(fallback.command.parameters.orientation_mode, "preserve_current")
        self.assertEqual(fallback.command.parameters.minimum_translation_m, 0.05)

        completed_restage = SimpleNamespace(skill=SkillType.MOVE_TO, completed=True,
            episode_terminal=False, execution_uncertain=False, predicate=None)
        retry = planner.select(context(completed_restage))
        self.assertEqual(retry.phase, "RIGHT_GRASP_RETRY")
        self.assertEqual(retry.command.action, SkillType.VLA_ACT)

    def test_failed_right_retreat_restages_before_vla_and_aborts_if_fallback_fails(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillResult, SkillType

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=2, vla=bounds), bounds)
        planner.left_transfer_confirmed = True
        planner.right_retry_phase = "reorient"
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name))
        state = {"command_state": [0.0] * 14}

        def move_result(*, completed, stop, ticks):
            return SkillResult.model_validate({
                "call_id": 1, "skill": SkillType.MOVE_TO, "completed": completed,
                "stop_reason": stop, "executed_policy_actions": 0,
                "vla_chunks_requested": 0, "discarded_actions": 0,
                "executed_physics_ticks": ticks, "state_before": state,
                "state_after": state, "observation_before": 7,
                "observation_after": 8, "episode_terminal": False,
                "task_success": False, "controlled_arms": "right",
                "information_source": "development_oracle", "elapsed_seconds": 0.1,
            })

        fallback = planner.select(context(move_result(
            completed=False, stop="planning_failure", ticks=40)))
        self.assertEqual(fallback.phase, "RIGHT_RETRY_RESTAGE_FALLBACK")
        self.assertEqual(fallback.command.action, SkillType.MOVE_TO)
        self.assertEqual(fallback.command.parameters.orientation_mode, "preserve_current")
        self.assertEqual(fallback.command.parameters.minimum_translation_m, 0.05)

        completed_restage = planner.select(context(move_result(
            completed=True, stop="completed", ticks=100)))
        self.assertEqual(completed_restage.phase, "RIGHT_GRASP_RETRY")
        self.assertEqual(completed_restage.command.action, SkillType.VLA_ACT)

        planner.right_retry_phase = "retry_vla"
        fallback = planner.select(context(move_result(
            completed=False, stop="trajectory_limits", ticks=50)))
        self.assertEqual(fallback.command.action, SkillType.MOVE_TO)
        abort = planner.select(context(move_result(
            completed=False, stop="trajectory_limits", ticks=50)))
        self.assertEqual(abort.command.action, SkillType.ABORT)

    def test_failed_second_left_or_right_reorientation_falls_back_to_preserved_wrist(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                planner = GraspRetryPlanner(GraspRetrySettings(
                    arm=arm, stage_before_first_vla=False, max_retries=2,
                    vla={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 4}))
                values = {"mug_lifted": True, "mug_grasp_confirmed": False,
                          "release_ready": True}
                context = lambda previous: SimpleNamespace(previous=previous,
                    value=lambda name: values.get(name))

                def result(skill, *, completed=True, stop="completed"):
                    return SimpleNamespace(skill=SkillType(skill), completed=completed,
                        episode_terminal=False, execution_uncertain=False, stop_reason=stop)

                failed_vla = result("vla_act", stop="policy_action_budget_exhausted")
                planner.select(context(failed_vla))
                planner.select(context(result("set_gripper")))
                first_restage = planner.select(context(result("move_to")))
                self.assertEqual(first_restage.command.parameters.orientation_mode,
                                 "preserve_current")
                planner.select(context(result("move_to")))
                planner.select(context(failed_vla))
                planner.select(context(result("set_gripper")))
                reoriented_restage = planner.select(context(result("move_to")))
                self.assertEqual(reoriented_restage.command.parameters.orientation_mode, "target")

                fallback = planner.select(context(result("move_to", completed=False,
                    stop="trajectory_limits")))
                self.assertEqual(fallback.phase, "RETRY_RESTAGE_FALLBACK")
                self.assertEqual(fallback.command.arm, arm)
                self.assertEqual(fallback.command.parameters.orientation_mode,
                                 "preserve_current")
                retry = planner.select(context(result("move_to")))
                self.assertEqual(retry.command.action, SkillType.VLA_ACT)
                self.assertEqual(retry.phase, "GRASP_MUG")

    def test_partial_reorientation_planning_failure_falls_back_without_more_rotation(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                planner = GraspRetryPlanner(GraspRetrySettings(
                    arm=arm, stage_before_first_vla=False, max_retries=2))
                values = {"mug_lifted": True, "mug_grasp_confirmed": False,
                          "release_ready": True}
                context = lambda previous: SimpleNamespace(previous=previous,
                    value=lambda name: values.get(name))

                def result(skill, *, completed=True, stop="completed"):
                    return SimpleNamespace(skill=SkillType(skill), completed=completed,
                        episode_terminal=False, execution_uncertain=False, stop_reason=stop)

                failed_vla = result("vla_act", stop="policy_action_budget_exhausted")
                planner.select(context(failed_vla))
                planner.select(context(result("set_gripper")))
                planner.select(context(result("move_to")))
                planner.select(context(result("move_to")))
                planner.select(context(failed_vla))
                planner.select(context(result("set_gripper")))
                planner.select(context(result("move_to")))
                reorientation_failure = result("move_to", completed=False,
                                               stop="planning_failure")
                fallback = planner.select(context(reorientation_failure))
                self.assertEqual(fallback.phase, "RETRY_RESTAGE_FALLBACK")
                self.assertEqual(fallback.command.parameters.orientation_mode,
                                 "preserve_current")
                retry = planner.select(context(result("move_to")))
                self.assertEqual(retry.command.action, SkillType.VLA_ACT)

    def test_unconfirmed_right_grasp_retries_after_safe_open_and_bounded_retreat(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=2, chunk_size=4, max_policy_actions=8)
        recorded = scenario("right_grasp_unconfirmed_retry")
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=recorded["retry_count"], vla=bounds), bounds)
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}

        def result(skill, *, stop="completed", objective=True):
            return SimpleNamespace(episode_terminal=False, completed=True,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=None)

        def context(previous):
            return SimpleNamespace(previous=previous, value=lambda name: values.get(name),
                visual_context={"status": {"mug": "grounded"}})

        previous = result(SkillType.VLA_ACT)
        previous.predicate = SimpleNamespace(name="left_transfer_complete", value=True)
        first = planner.select(context(previous))
        self.assertEqual((first.phase, first.command.stop), ("RIGHT_GRASP", "right_mug_lifted"))

        failed = result(SkillType.VLA_ACT, stop="failure_predicate_satisfied", objective=False)
        opened = planner.select(context(failed))
        self.assertEqual((opened.phase, opened.command.action.value, opened.command.arm),
                         ("RIGHT_RETRY_OPEN", "set_gripper", "right"))
        self.assertEqual(opened.command.parameters.value, 1.0)

        retreat = planner.select(context(result(SkillType.SET_GRIPPER)))
        self.assertEqual(retreat.phase, "RIGHT_RETRY_RETREAT")
        self.assertEqual(retreat.command.parameters.target, "current_ee")
        self.assertEqual(retreat.command.parameters.path_mode, "reverse_approach")
        self.assertEqual(retreat.command.parameters.retreat_distance_m, 0.05)
        self.assertEqual(retreat.command.parameters.orientation_mode, "preserve_current")

        retry = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual((retry.phase, retry.command.stop),
                         ("RIGHT_GRASP_RETRY", "right_mug_lifted"))
        self.assertLessEqual(retry.command.parameters.max_policy_actions,
                             HandoffPlanner.RIGHT_GRASP_RETRY_ACTIONS)
        self.assertEqual([opened.command.action.value, retreat.command.action.value,
                          retry.command.action.value],
                         recorded["expected_recovery_skills"])
        self.assertEqual(retreat.command.parameters.target, recorded["retreat_target"])
        self.assertTrue(recorded["requires_release_ready"])

        values["right_mug_lifted"] = True
        stage = planner.select(context(result(SkillType.VLA_ACT)))
        self.assertEqual(stage.phase, "RACK_STAGING")
        self.assertEqual(stage.command.parameters.target, "rack_approach")

    def test_right_grasp_retry_requires_safe_open_and_stops_after_one_retry(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=1, chunk_size=4, max_policy_actions=4)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=1, vla=bounds), bounds)
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True, "task_success": False}
        visual_context = {"status": {"mug": "grounded"}}

        def context(previous):
            return SimpleNamespace(previous=previous, value=lambda name: values.get(name),
                visual_context=visual_context)

        def result(skill, *, stop="completed", objective=True):
            return SimpleNamespace(episode_terminal=False, completed=True,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=None)

        previous = result(SkillType.VLA_ACT)
        previous.predicate = SimpleNamespace(name="left_transfer_complete", value=True)
        planner.select(context(previous))
        failed = result(SkillType.VLA_ACT, stop="policy_action_budget_exhausted", objective=False)
        values["release_ready"] = False
        values["mug_lifted"] = False
        retry_open = planner.select(context(failed))
        self.assertEqual(retry_open.command.action.value, "set_gripper")
        self.assertEqual(retry_open.phase, "RIGHT_RETRY_OPEN")

        values["mug_lifted"] = True
        visual_context["status"]["mug"] = "not_detected"
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=1, vla=bounds), bounds)
        planner.select(context(previous))
        blocked = planner.select(context(failed))
        self.assertEqual(blocked.command.action.value, "abort")
        self.assertEqual(blocked.command.reason, "right_grasp_retry_open_not_authorized")

        values["release_ready"] = True
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=1, vla=bounds), bounds)
        planner.select(context(previous))
        planner.select(context(failed))
        planner.select(context(result(SkillType.SET_GRIPPER)))
        planner.select(context(result(SkillType.MOVE_TO)))
        exhausted = planner.select(context(failed))
        self.assertEqual(exhausted.command.action.value, "abort")
        self.assertEqual(exhausted.command.reason, "right_grasp_retry_budget_exhausted")

    def test_right_side_replans_from_latched_transfer_when_global_lift_is_false(self):
        from types import SimpleNamespace
        from harness.predicates.base import PredicateResult
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=1, chunk_size=4, max_policy_actions=4)
        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=0, vla=bounds), bounds)
        values = {"mug_lifted": False, "left_transfer_complete": False,
                  "right_mug_lifted": False, "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name))
        previous = SimpleNamespace(episode_terminal=False, completed=True,
            execution_uncertain=False, objective_satisfied=True,
            skill=SkillType.VLA_ACT,
            predicate=PredicateResult(name="left_transfer_complete", value=True,
                source="development_oracle", observation_sequence=8, reason="verified"))

        grasp = planner.select(context(previous))
        self.assertEqual(grasp.phase, "RIGHT_GRASP")
        self.assertEqual(grasp.command.stop, "right_mug_lifted")

        values["right_mug_lifted"] = True
        previous = SimpleNamespace(episode_terminal=False, completed=True,
            execution_uncertain=False, objective_satisfied=True,
            skill=SkillType.VLA_ACT,
            predicate=PredicateResult(name="right_mug_lifted", value=True,
                source="development_oracle", observation_sequence=10, reason="verified"))
        stage = planner.select(context(previous))
        self.assertEqual(stage.phase, "RACK_STAGING")
        self.assertEqual(stage.command.parameters.target, "rack_approach")

        previous = SimpleNamespace(episode_terminal=False, completed=True,
            execution_uncertain=False, objective_satisfied=None,
            skill=SkillType.MOVE_TO, predicate=None)
        hang = planner.select(context(previous))
        self.assertEqual(hang.phase, "RACK_HANG")
        self.assertEqual(hang.command.stop, "task_success")

    def test_right_lift_evidence_requires_right_bilateral_contact_after_baseline(self):
        class Frame:
            def __init__(self, sequence):
                self.sequence = sequence

        frames = {0: evidence(0),
                  1: evidence(1, mug_position=(-.2, 0.0, .80), table_contact=False,
                      right_contact=True, right_bilateral=True, right_gripper=0.0),
                  2: evidence(2, mug_position=(-.2, 0.0, .80), table_contact=False,
                      right_contact=True, right_bilateral=True, right_gripper=0.0)}
        gate = HangingMugPredicates(lambda obs: frames[obs.sequence], source="development_oracle")
        self.assertFalse(gate.evaluate("right_mug_lifted", Frame(0)).value)
        self.assertFalse(gate.evaluate("right_mug_lifted", Frame(1)).value)
        self.assertTrue(gate.evaluate("right_mug_lifted", Frame(2)).value)

    def test_runner_treats_configured_left_transfer_milestone_as_success(self):
        from run_skills import successful_outcome
        self.assertTrue(successful_outcome({"outcome":"left_transfer_complete"}))
        self.assertTrue(successful_outcome({"outcome":"task_success"}))
        self.assertFalse(successful_outcome({"outcome":"left_transfer_unconfirmed"}))

    def test_episode_home_anchor_is_available_for_retreat(self):
        class Task:
            class Robot:
                def get_left_ee_pose(self):
                    return [0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0]
            robot = Robot()

        env = TransferEnvironment()
        observation = env.observe()
        anchors = executor(env, oracle=True)[0].anchors
        prepare = oracle_anchor_preparer(Task(), anchors, log=lambda event: None)
        command = SkillCommand(action="move_to", arm="left",
                               parameters=MoveParameters(target="left_home"))
        prepare(command, observation)
        anchor = anchors.resolve("left_home", "left", observation, allow_oracle=True)
        self.assertEqual(anchor.pose[:3], (0.1, 0.2, 0.9))
        self.assertEqual(anchor.observation_sequence, observation.sequence)

    def test_oracle_retry_retreat_uses_longer_authored_pregrasp_standoff(self):
        class Task:
            class Robot:
                def get_left_ee_pose(self):
                    return [0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0]
            robot = Robot()
            mug = object()

            def __init__(self):
                self.pre_distances = []

            def get_grasp_pose(self, mug, arm, *, contact_point_id, pre_dis):
                self.pre_distances.append(pre_dis)
                return [-0.2 - pre_dis, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]

        task = Task()
        env = TransferEnvironment()
        observation = env.observe()
        anchors = executor(env, oracle=True)[0].anchors
        logs = []
        prepare = oracle_anchor_preparer(task, anchors, log=logs.append)
        retreat = SkillCommand(action="move_to", arm="left",
            parameters=MoveParameters(target="current_ee", path_mode="reverse_approach",
                                     orientation_mode="preserve_current"))
        prepare(retreat, observation)
        anchor = anchors.resolve("mug_pregrasp", "left", observation, allow_oracle=True)
        self.assertEqual(task.pre_distances, [0.12])
        self.assertEqual(anchor.pose[0], -0.32)
        self.assertEqual(logs[-1]["pre_dis_m"], 0.12)

        staging = SkillCommand(action="move_to", arm="left",
            parameters=MoveParameters(target="mug_pregrasp"))
        prepare(staging, observation)
        self.assertEqual(task.pre_distances, [0.12, 0.05])
        self.assertEqual(logs[-1]["event_type"], "grasp_orientation_candidate")
        self.assertEqual(logs[-1]["selection"], "shortest_successful_joint_trajectory")

    def test_vla_controls_both_arms_through_transfer_grasp_and_hang(self):
        recorded = scenario("left_transfer_then_hang")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'attempt_0001'
            env = TransferEnvironment(**{key: recorded[key] for key in ('transfers', 'acquires', 'uncertain')})
            s,w,events=run_transfer(p,env)
            self.assertEqual(s['outcome'],recorded['expected_outcome'])
            self.assertTrue(s['acquisition_confirmed'])
            self.assertTrue(s['milestone_reached'])
            self.assertTrue(s['task_success'])
            self.assertEqual(s['phase'],'COMPLETE')
            self.assertEqual((w.policy_resets,w.chunks),(2,4))
            results=[e['result'] for e in events if e.get('event_type')=='skill_result']
            # RoboTwin success ends VLA execution immediately; the received
            # chunk tail is discarded before another command is applied.
            self.assertEqual([r['executed_policy_actions'] for r in results],[2,3,3,0,2])
            self.assertEqual([r['discarded_actions'] for r in results],[2,1,1,0,2])
            self.assertEqual(results[-1]['post_success_actions'], 0)
            self.assertEqual([r['policy_history_reset'] for r in results],[True,False,False,False,True])
            selections=[e['command'] for e in events if e.get('event_type')=='skill_requested']
            self.assertEqual([c.get('stop') for c in selections],
                             ['mug_lifted','left_transfer_complete','right_mug_lifted',None,'task_success'])
            self.assertEqual([c['parameters'].get('lock_arm') for c in selections],
                             [None,None,None,None,None])
            self.assertEqual([c['action'] for c in selections],
                             ['vla_act','vla_act','vla_act','move_to','vla_act'])
            self.assertEqual(s['skill_calls'],5)
            catalog=candidates_from_attempt(p,attempt_id=p.name,seed=3,execution_kind='synthetic')
            self.assertEqual(catalog.task_memory[0].objective,'task_success')
            self.assertEqual(catalog.task_memory[0].evidence[0].outcome,'success')
            self.assertFalse(catalog.task_memory[0].planner_reviewed)

    def test_left_transfer_only_stop_prevents_right_hang_attempt(self):
        recorded = scenario("left_transfer_then_hang")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'attempt_0001'
            env=TransferEnvironment(**{key: recorded[key] for key in ('transfers','acquires','uncertain')})
            s,w,events=run_transfer(p,env,completion_predicate='left_transfer_complete')
            self.assertEqual(s['outcome'],'left_transfer_complete')
            self.assertEqual(s['phase'],'RIGHT_PICK_AND_HANG')
            self.assertTrue(s['milestone_reached'])
            self.assertTrue(s['acquisition_confirmed'])
            self.assertFalse(s['task_success'])
            self.assertEqual(s['skill_calls'],2)
            self.assertEqual(w.policy_resets,1)
            results=[e['result'] for e in events if e.get('event_type')=='skill_result']
            self.assertEqual([r['executed_policy_actions'] for r in results],[2,3])
            self.assertEqual([r['predicate']['name'] for r in results],
                             ['mug_lifted','left_transfer_complete'])

    def test_unconfirmed_rack_task_aborts_after_one_attempt(self):
        recorded = scenario("left_transfer_unconfirmed")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'attempt_0001'
            env=TransferEnvironment(**{key: recorded[key] for key in ('transfers','acquires','uncertain')})
            s,w,events=run_transfer(p,env)
            self.assertEqual(s['outcome'],recorded['expected_outcome'])
            self.assertFalse(s['milestone_reached'])
            self.assertEqual(w.chunks,4)
            self.assertEqual(events[-1]['result']['skill'],recorded['expected_terminal_skill'])
            abort = next(r for r in events if r.get('event_type') == 'skill_requested'
                         and r['command']['action'] == 'abort')
            self.assertEqual(abort['command']['reason'], 'rack_hang_unconfirmed')
            c=candidates_from_attempt(p,attempt_id=p.name,seed=3,execution_kind='synthetic')
            evidence = {(item.metric, item.outcome) for item in c.task_memory[0].evidence}
            self.assertIn(('mug_lifted','success'), evidence)
            self.assertIn(('task_success','failure'), evidence)

    def test_no_lift_never_invokes_handoff_and_uncertainty_never_retries(self):
        failed = scenario("failed_acquisition")
        ambiguous = scenario("ambiguous_action_response")
        for spec, expected, chunks in [(failed, failed['expected_outcome'], failed['expected_policy_chunks']),
                                      (ambiguous, ambiguous['expected_outcome'], 2)]:
            with tempfile.TemporaryDirectory() as tmp:
                env=TransferEnvironment(**{key: spec[key] for key in ('transfers','acquires','uncertain')})
                s,w,events=run_transfer(Path(tmp)/'attempt_0001',env)
                self.assertEqual(s['outcome'],expected)
                self.assertFalse(s['milestone_reached'])
                self.assertEqual(w.chunks,chunks)
                if spec is ambiguous:
                    self.assertTrue(events[-1]['result']['execution_uncertain'])
                    self.assertTrue(spec['expected_terminal_uncertain'])

    def test_campaign_reports_distinct_milestones(self):
        class Runner:
            execution_kind='synthetic'
            def __call__(self, *, output, **kwargs):
                run_transfer(output,TransferEnvironment())
                return AttemptExit(0)
        with tempfile.TemporaryDirectory() as tmp:
            settings=ExploreSettings(variants=[{'name':'handoff'}],max_attempts=1)
            s=ExplorationController(settings,runner=Runner(),output=Path(tmp)/'campaign').run()
            self.assertEqual((s['acquisition_count'],s['handoff_count'],s['full_task_success_count']),(1,0,1))

    def test_acquisition_evaluation_and_anthropic_reject_handoff_before_startup(self):
        with self.assertRaises(ValueError):
            EvaluationSettings(seeds=[1001],development_seeds=[3],episode={'handoff':{}})
        import run_skills
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'config.yaml').write_text('handoff: {}\n')
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                run_skills.main(['--config',str(p/'config.yaml'),'--output',str(p/'out'),
                    '--planner','anthropic','--development-oracle'])
            self.assertFalse((p/'out').exists())

    def test_full_task_anthropic_evaluation_rejects_before_output_creation(self):
        import run_skills
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'config.yaml').write_text(
                'anthropic:\n  model: claude-test-model\n  task_scope: full_task\n')
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                run_skills.main(['--config',str(p/'config.yaml'),'--output',str(p/'out'),
                    '--planner','anthropic','--development-oracle',
                    '--evaluation-manifest',str(p/'manifest.json')])
            self.assertFalse((p/'out').exists())

    def test_left_transfer_stop_requires_handoff_recipe(self):
        from harness.runtime.settings import RunSettings
        with self.assertRaisesRegex(ValueError, 'left_transfer_stop_requires_handoff_recipe'):
            RunSettings(stop_after_left_transfer=True)
        settings=RunSettings.model_validate({"stop_after_left_transfer":True,"handoff":{
            "max_chunks":1,"chunk_size":8,"max_policy_actions":8}})
        self.assertEqual(settings.completion_predicate,'left_transfer_complete')
