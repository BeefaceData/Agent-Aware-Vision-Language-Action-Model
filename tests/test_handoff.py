"""Phase composition and evidence accounting without native simulation or APIs."""

from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness.jsonl import append_jsonl
from harness.planner.handoff import HandoffPlanner
from harness.planner.scripted import GraspRetrySettings
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
    def test_unconfirmed_right_grasp_retries_after_safe_open_retreat_and_fresh_restage(self):
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
            return SimpleNamespace(previous=previous, value=lambda name: values.get(name))

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
        self.assertEqual(retreat.command.parameters.offset_xyz, (0.0, 0.0, 0.05))

        restage = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual(restage.phase, "RIGHT_RETRY_STAGE")
        self.assertEqual(restage.command.parameters.target, "mug_pregrasp")

        retry = planner.select(context(result(SkillType.MOVE_TO)))
        self.assertEqual((retry.phase, retry.command.stop),
                         ("RIGHT_GRASP_RETRY", "right_mug_lifted"))
        self.assertLessEqual(retry.command.parameters.max_policy_actions,
                             HandoffPlanner.RIGHT_GRASP_RETRY_ACTIONS)
        self.assertEqual([opened.command.action.value, retreat.command.action.value,
                          restage.command.action.value, retry.command.action.value],
                         recorded["expected_recovery_skills"])
        self.assertEqual(list(retreat.command.parameters.offset_xyz), recorded["retreat_offset_xyz"])
        self.assertEqual(restage.command.parameters.target, recorded["restage_target"])
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

        def context(previous):
            return SimpleNamespace(previous=previous, value=lambda name: values.get(name))

        def result(skill, *, stop="completed", objective=True):
            return SimpleNamespace(episode_terminal=False, completed=True,
                execution_uncertain=False, objective_satisfied=objective,
                stop_reason=stop, skill=skill, predicate=None)

        previous = result(SkillType.VLA_ACT)
        previous.predicate = SimpleNamespace(name="left_transfer_complete", value=True)
        planner.select(context(previous))
        failed = result(SkillType.VLA_ACT, stop="policy_action_budget_exhausted", objective=False)
        values["release_ready"] = False
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

    def test_left_vla_coordinates_bimanual_handoff_then_locks_left_for_hang(self):
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
            # Native hang success is retained as a candidate signal; this
            # fixture's four-action command allowance permits two trailing
            # VLA actions before the ordinary per-command cap wins.
            self.assertEqual([r['executed_policy_actions'] for r in results],[2,3,3,0,4])
            self.assertEqual([r['discarded_actions'] for r in results],[2,1,1,0,0])
            self.assertEqual(results[-1]['post_success_actions'], 2)
            self.assertEqual([r['policy_history_reset'] for r in results],[True,False,False,False,True])
            selections=[e['command'] for e in events if e.get('event_type')=='skill_requested']
            self.assertEqual([c.get('stop') for c in selections],
                             ['mug_lifted','left_transfer_complete','right_mug_lifted',None,'task_success'])
            self.assertEqual([c['parameters'].get('lock_arm') for c in selections],
                             [None,None,'left',None,'left'])
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
