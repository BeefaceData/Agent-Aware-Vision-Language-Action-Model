"""R9 checks with fake episodes; no native startup, network or credentials."""

import json
from pathlib import Path
import tempfile
import unittest

from harness.evaluation.manifest import EvaluationSettings, FrozenEvaluation, freeze
from harness.evaluation.guard import SingleResetEnvironment, episode_guard
from harness.evaluation.controller import EvaluationController
from harness.memory import MemoryCatalog, MemorySnapshot
from tests.test_memory import recipe, evidence
from tests.test_exploration import FakeAttemptRunner


def settings(**overrides):
    data = dict(seeds=[1001, 1002], development_seeds=[3])
    data.update(overrides)
    return EvaluationSettings.model_validate(data)


class EvalRunner(FakeAttemptRunner):
    def __init__(self, mutate=None):
        super().__init__()
        self.mutate = mutate

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        path = kwargs["output"] / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(mode="eval", evaluation_sha256=self.evaluation[1], condition="oracle_assisted_acquisition")
        path.write_text(json.dumps(summary))
        if self.mutate:
            self.mutate()
        return result


class EvaluationTests(unittest.TestCase):
    def test_declared_explored_and_memory_evidence_seeds_excluded(self):
        with self.assertRaises(ValueError): settings(seeds=[3])
        with self.assertRaises(ValueError): settings(seeds=[1001, 1001])
        with self.assertRaises(ValueError): settings(planner="anthropic")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explored = root/"explored.json"
            explored.write_text(json.dumps({"mode": "explore", "attempts": [{"seed": 1001}]}))
            with self.assertRaises(ValueError): freeze(settings(), root=root, exploration_summaries=[explored])
            memory = MemorySnapshot(MemoryCatalog(task_memory=(recipe(evidence=[evidence(seed=1002)]),)))
            with self.assertRaises(ValueError): freeze(settings(), root=root, memory=memory)

    def test_manifest_tampering_code_changes_and_added_skills_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"harness").mkdir()
            code = root/"harness"/"runtime.py"; code.write_text("original")
            frozen = freeze(settings(), root=root)
            path = root/"frozen.json"; frozen.save(path)
            FrozenEvaluation.load(path, expected_sha256=frozen.digest).verify(root)
            with self.assertRaises(FileExistsError): frozen.save(path)
            code.write_text("modified")
            with self.assertRaises(ValueError): frozen.verify(root)
            code.write_text("original")
            (root/"harness"/"new_skill.py").write_text("new")
            with self.assertRaises(ValueError): frozen.verify(root)
            data = json.loads(path.read_text()); data["manifest"]["settings"]["seeds"] = [2000]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): FrozenEvaluation.load(path)

    def test_reset_is_single_use_even_after_ambiguous_reset(self):
        class Environment:
            calls = 0
            def reset(self, seed):
                self.calls += 1
                raise ConnectionError("unknown reset outcome")
        original = Environment(); wrapped = SingleResetEnvironment(original)
        with self.assertRaises(ConnectionError): wrapped.reset(1001)
        with self.assertRaises(RuntimeError): wrapped.reset(1001)
        self.assertEqual(original.calls, 1)

    def test_episode_contract_rejects_seed_override_and_mutated_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = freeze(settings(), root=root)
            path = root/"frozen.json"; frozen.save(path)
            memory = root/"memory.json"; MemorySnapshot(frozen.memory).export(memory)
            args = dict(root=root, settings=frozen.settings.episode, planner="scripted", model_override=None,
                        memory_path=memory, manifest_path=path, robotwin_root=root/"RoboTwin")
            with self.assertRaises(ValueError): episode_guard(frozen, seed=3, **args)
            check = episode_guard(frozen, seed=1001, **args)
            memory.write_text("{}")
            with self.assertRaises(ValueError): check()

    def test_complete_fixed_schedule_has_no_candidate_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = freeze(settings(), root=root)
            runner = EvalRunner()
            output = root/"evaluation"
            summary = EvaluationController(frozen, root=root, output=output, runner=runner).run()
            self.assertTrue(summary["protocol_completed"])
            self.assertEqual(summary["acquisition_rate"], 1.0)
            self.assertEqual(summary["execution_kind"], "synthetic")
            self.assertIsNone(summary["strict_hanging_success"])
            self.assertFalse(summary["remote_policy_weights_verified"])
            self.assertEqual([r["seed"] for r in summary["episodes"]], [1001, 1002])
            self.assertFalse((output/"candidates").exists())
            self.assertFalse((output/"candidate_catalog.json").exists())
            self.assertEqual([w.policy_resets for w in runner.workers], [1, 1])

    def test_drift_during_episode_invalidates_and_stops_following_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"harness").mkdir()
            code = root/"harness"/"runtime.py"; code.write_text("original")
            frozen = freeze(settings(), root=root)
            summary = EvaluationController(frozen, root=root, output=root/"evaluation",
                runner=EvalRunner(mutate=lambda: code.write_text("changed"))).run()
            self.assertEqual(summary["stop_reason"], "evaluation_invalid")
            self.assertEqual(summary["started_count"], 1)
            self.assertFalse(summary["protocol_completed"])
            self.assertIsNone(summary["acquisition_rate"])

    def test_invariant_check_blocks_motion_after_planner_returns(self):
        from harness.planner.base import PlannerSelection
        from harness.skills.schemas import SkillCommand
        from harness.skill_orchestrator import SkillOrchestrator
        from tests.test_skills import setup_runtime, COMMAND
        class Planner:
            def select(self, context):
                return PlannerSelection(SkillCommand.model_validate(COMMAND), "GRASP_MUG")
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("code_changed_while_planning")
        runtime, executor, _ = setup_runtime()
        summary = SkillOrchestrator(executor, Planner(), log=lambda _: None,
            save_observation=lambda obs: {}, episode_id="test", seed=1001, invariant_check=check).run()
        self.assertEqual(summary["outcome"], "controller_error")
        self.assertEqual((runtime.chunks, executor.calls, runtime.actions), (0, 0, []))
