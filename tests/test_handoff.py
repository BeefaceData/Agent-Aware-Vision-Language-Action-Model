"""Phase composition and evidence accounting without native simulation or APIs."""

from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from harness.jsonl import append_jsonl
from harness.planner.handoff import HandoffPlanner
from harness.planner.scripted import (GraspRetryPlanner, GraspRetrySettings,
                                      retry_open_authorized)
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
        return self.transfers and self.acquires and self.policy_steps >= 10

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
    def test_retry_open_requires_current_support_and_upright_evidence(self):
        values = {"release_ready": True, "mug_toppled": False}
        context = SimpleNamespace(value=lambda name: values.get(name))

        self.assertTrue(retry_open_authorized(context))
        values["release_ready"] = None
        self.assertFalse(retry_open_authorized(context))
        values.update(release_ready=True, mug_toppled=True)
        self.assertFalse(retry_open_authorized(context))

    def test_oracle_recovery_clearance_is_fixed_above_current_mug(self):
        class Robot:
            def get_left_ee_pose(self):
                return (0.1, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0)

            def get_right_ee_pose(self):
                return (0.12, -0.04, 0.95, 1.0, 0.0, 0.0, 0.0)

            def get_right_tcp_pose(self):
                return (0.24, -0.04, 0.95, 1.0, 0.0, 0.0, 0.0)

        class Task:
            robot = Robot()
            mug = SimpleNamespace(get_pose=lambda: SimpleNamespace(
                p=(0.03, 0.02, 0.74)))

        env = TransferEnvironment()
        observation = env.observe()
        anchors = executor(env, oracle=True)[0].anchors
        prepare = oracle_anchor_preparer(Task(), anchors, log=lambda event: None)
        command = SkillCommand(action="move_to", arm="right",
            parameters=MoveParameters(target="mug_recovery_clearance"))

        prepare(command, observation)
        anchor = anchors.resolve("mug_recovery_clearance", "right", observation,
                                 allow_oracle=True)

        # RoboTwin's TCP is 12 cm along the planning frame's local +X.
        # Compensate that transform so the physical TCP is above the mug.
        self.assertAlmostEqual(anchor.pose[0], -0.09)
        self.assertAlmostEqual(anchor.pose[1], 0.02)
        self.assertAlmostEqual(anchor.pose[2], 0.84)
        self.assertEqual(anchor.pose[3:], (1.0, 0.0, 0.0, 0.0))

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
            parameters={"target": "mug_recovery_clearance",
                        "orientation_mode": "preserve_current"})
        observation = SimpleNamespace(sequence=17)
        self.assertEqual(prepare(command, observation), "oracle_anchor")
        self.assertEqual(calls, [("mug_recovery_clearance", 17)])
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

    def test_development_anchor_fallback_binds_reverse_target_when_public_prepare_returns_none(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillCommand

        calls, logs = [], []
        def oracle_prepare(command, observation):
            calls.append((command.parameters.target, command.arm, observation.sequence))
            return "oracle_preapproach"

        prepare = development_anchor_fallback(
            lambda command, observation: None, oracle_prepare, logs.append)
        command = SkillCommand(action="move_to", arm="right",
            parameters={"target": "mug_recovery_clearance",
                        "orientation_mode": "preserve_current"})

        self.assertEqual(prepare(command, SimpleNamespace(sequence=23)),
                         "oracle_preapproach")
        self.assertEqual(calls, [("mug_recovery_clearance", "right", 23)])
        self.assertEqual(logs, [{
            "event_type": "perception_anchor_oracle_fallback",
            "target": "mug_recovery_clearance", "arm": "right",
            "observation_sequence": 23,
            "reason": "public_recovery_clearance_unavailable",
            "error_type": None,
            "information_source": "development_oracle",
        }])

    def test_development_anchor_fallback_passes_non_move_commands_through(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillCommand

        calls = []
        def public_prepare(command, observation):
            calls.append((command.action.value, observation.sequence))
            return "public_result"
        def oracle_prepare(command, observation):
            raise AssertionError("oracle should not be called")

        prepare = development_anchor_fallback(
            public_prepare, oracle_prepare, lambda event: None)
        command = SkillCommand(action="vla_act", arm="both",
            parameters={"max_chunks": 1, "chunk_size": 1,
                        "max_policy_actions": 1})

        self.assertEqual(prepare(command, SimpleNamespace(sequence=31)),
                         "public_result")
        self.assertEqual(calls, [("vla_act", 31)])

    def test_left_grasp_retry_is_one_retreat_then_fresh_vla(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=1, chunk_size=4, max_policy_actions=4)
        planner = GraspRetryPlanner(GraspRetrySettings(
            stage_before_first_vla=False, max_retries=2, vla=bounds))
        values = {"mug_lifted": False, "mug_grasp_confirmed": False,
                  "release_ready": True, "mug_toppled": False}

        def context(previous):
            return SimpleNamespace(previous=previous,
                value=lambda name: values.get(name),
                visual_context={"status": {"mug": "grounded"}})

        def result(skill, *, completed=True, stop="completed", objective=False):
            is_clearance = skill == SkillType.MOVE_TO
            is_motion = is_clearance or skill == SkillType.SET_GRIPPER
            return SimpleNamespace(episode_terminal=False, completed=completed,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=None,
                executed_physics_ticks=28 if is_motion else 0,
                position_error_m=0.001 if is_clearance else None,
                orientation_error_rad=0.01 if is_clearance else None,
                executed_policy_actions=4,
                vla_chunks_requested=1)

        first = planner.select(context(None))
        self.assertEqual(first.command.action, SkillType.VLA_ACT)
        failed = result(SkillType.VLA_ACT, stop="policy_action_budget_exhausted")
        retreat = planner.select(context(failed))
        self.assertEqual((retreat.phase, retreat.command.action),
                         ("RETRY_RETREAT", SkillType.MOVE_TO))
        self.assertEqual(retreat.command.parameters.target, "mug_recovery_clearance")
        self.assertEqual(retreat.command.parameters.orientation_mode, "preserve_current")

        opened = planner.select(context(result(SkillType.MOVE_TO, objective=True)))
        self.assertEqual((opened.phase, opened.command.action),
                         ("RETRY_OPEN", SkillType.SET_GRIPPER))
        self.assertEqual(opened.command.parameters.value, 1.0)

        retry = planner.select(context(result(SkillType.SET_GRIPPER, objective=True)))
        self.assertEqual(retry.command.action, SkillType.VLA_ACT)
        self.assertEqual(retry.command.stop, "mug_lifted")

    def test_right_grasp_retry_is_one_retreat_then_fresh_vla_for_each_retry(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        bounds = VLAParameters(max_chunks=1, chunk_size=4, max_policy_actions=4)
        planner = HandoffPlanner(GraspRetrySettings(
            stage_before_first_vla=False, max_retries=2, vla=bounds), bounds)
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": True,
                  "mug_toppled": False,
                  "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name),
            visual_context={"status": {"mug": "grounded"}})

        def result(skill, *, stop="completed", objective=True, predicate=None):
            is_clearance = skill == SkillType.MOVE_TO
            is_motion = is_clearance or skill == SkillType.SET_GRIPPER
            return SimpleNamespace(episode_terminal=False, completed=True,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=predicate,
                executed_physics_ticks=28 if is_motion else 0,
                position_error_m=0.001 if is_clearance else None,
                orientation_error_rad=0.01 if is_clearance else None,
                executed_policy_actions=4,
                vla_chunks_requested=1)

        transfer = result(SkillType.VLA_ACT,
            predicate=SimpleNamespace(name="left_transfer_complete", value=True))
        first = planner.select(context(transfer))
        self.assertEqual((first.phase, first.command.action),
                         ("RIGHT_GRASP", SkillType.VLA_ACT))

        failed = result(SkillType.VLA_ACT, stop="policy_action_budget_exhausted",
                        objective=False)
        retreat1 = planner.select(context(failed))
        self.assertEqual((retreat1.phase, retreat1.command.action),
                         ("RIGHT_RETRY_RETREAT", SkillType.MOVE_TO))
        self.assertEqual(retreat1.command.arm, "right")
        self.assertEqual(retreat1.command.parameters.target, "mug_recovery_clearance")
        self.assertEqual(retreat1.command.parameters.orientation_mode, "preserve_current")
        open1 = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual((open1.phase, open1.command.action),
                         ("RIGHT_RETRY_OPEN", SkillType.SET_GRIPPER))
        self.assertEqual(open1.command.arm, "right")
        self.assertEqual(open1.command.parameters.value, 1.0)
        retry1 = planner.select(context(result(SkillType.SET_GRIPPER)))
        self.assertEqual((retry1.phase, retry1.command.action),
                         ("RIGHT_GRASP_RETRY", SkillType.VLA_ACT))

        retreat2 = planner.select(context(failed))
        self.assertEqual(retreat2.command.action, SkillType.MOVE_TO)
        self.assertEqual(retreat2.command.parameters, retreat1.command.parameters)
        open2 = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual(open2.command.action, SkillType.SET_GRIPPER)
        retry2 = planner.select(context(result(SkillType.SET_GRIPPER)))
        self.assertEqual(retry2.command.action, SkillType.VLA_ACT)

    def test_failed_retry_retreat_aborts_instead_of_handing_back_to_vla(self):
        from types import SimpleNamespace
        from harness.skills.schemas import SkillType

        planner = HandoffPlanner(GraspRetrySettings(stage_before_first_vla=False,
            max_retries=1), VLAParameters(max_chunks=1, chunk_size=4,
                                           max_policy_actions=4))
        values = {"mug_lifted": True, "left_transfer_complete": True,
                  "right_mug_lifted": False, "release_ready": False,
                  "task_success": False}
        context = lambda previous: SimpleNamespace(previous=previous,
            value=lambda name: values.get(name),
            visual_context={"status": {"mug": "grounded"}})
        result = lambda skill, **kwargs: SimpleNamespace(
            episode_terminal=False, completed=kwargs.get("completed", True),
            execution_uncertain=False, objective_satisfied=kwargs.get("objective", False),
            stop_reason=kwargs.get("stop", "policy_action_budget_exhausted"),
            skill=skill, predicate=kwargs.get("predicate"),
            executed_physics_ticks=0, executed_policy_actions=4, vla_chunks_requested=1)

        transfer = result(SkillType.VLA_ACT,
            predicate=SimpleNamespace(name="left_transfer_complete", value=True))
        planner.select(context(transfer))
        failed = result(SkillType.VLA_ACT)
        retreat = planner.select(context(failed))
        self.assertEqual(retreat.command.action, SkillType.MOVE_TO)
        rejected = result(SkillType.MOVE_TO, completed=False, stop="planning_failure")
        abort = planner.select(context(rejected))
        self.assertEqual(abort.command.action, SkillType.ABORT)
        self.assertEqual(abort.command.reason, "previous_skill_failed")

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
        class FakeArmTag:
            def __init__(self, arm):
                self.arm = arm

        class Task:
            class Robot:
                def get_left_ee_pose(self):
                    return [0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0]
            robot = Robot()
            mug = object()

            def __init__(self):
                self.pre_distances = []

            def get_grasp_pose(self, mug, arm, *, contact_point_id, pre_dis):
                if not isinstance(arm, FakeArmTag):
                    raise TypeError("arm_tag_required")
                self.pre_distances.append(pre_dis)
                return [-0.2 - pre_dis, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]

        arm_tag_patch = patch.dict(Task.get_grasp_pose.__globals__,
                                   {"ArmTag": FakeArmTag})
        arm_tag_patch.start()
        self.addCleanup(arm_tag_patch.stop)
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
            self.assertTrue(all('lock_arm' not in c['parameters'] for c in selections))
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
