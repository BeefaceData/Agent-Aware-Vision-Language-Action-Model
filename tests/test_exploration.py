"""Focused R8 lifecycle checks; no native simulator or paid inference."""

from functools import partial
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

from harness.jsonl import append_jsonl
from harness.exploration.config import ExploreSettings
from harness.exploration.controller import ExplorationController, write_json
from harness.exploration.process import AttemptExit, ProcessAttemptRunner
from harness.exploration.candidates import candidates_from_attempt
from harness.exploration.candidates import _memory_safe_step, _append_compact_step
from harness.memory.task_memory import RecipeStep
from harness.memory import MemoryCatalog, MemorySnapshot
from harness.planner.scripted import GraspRetryPlanner, GraspRetrySettings
from harness.predicates.hanging_mug import HangingMugPredicates
from harness.predicates.registry import PredicateRegistry
from harness.perception.anchors import Anchor
from harness.runtime.orchestrator import SkillOrchestrator
from tests.test_analytic_skills import executor as make_executor
from tests.test_skill_orchestrator import AcquisitionEnvironment


class FakeAttemptRunner:
    execution_kind = "synthetic"

    def __init__(self, *, success=True, timeout=False, crash=False):
        self.environments, self.workers, self.timeouts = [], [], []
        self.success, self.timeout, self.crash = success, timeout, crash

    def __call__(self, *, seed, config, output, timeout):
        self.timeouts.append(timeout)
        if self.timeout:
            return AttemptExit(None, True)
        output.mkdir()
        if self.crash:
            write_json(output / "summary.json", dict(outcome="runner_error", error_type="RuntimeError"))
            return AttemptExit(1)
        env = AcquisitionEnvironment(1 if self.success else None)
        self.environments.append(env)
        gates = PredicateRegistry(HangingMugPredicates(env.evidence, source="development_oracle"))
        executor, worker, _ = make_executor(env, predicates=gates, oracle=True)
        self.workers.append(worker)
        def prepare(command, obs):
            if command.action == "move_to" and command.parameters.target == "mug_pregrasp":
                executor.anchors.bind("mug_pregrasp", "left",
                    Anchor(tuple(env.arm_state("left")["pose"]), obs.sequence, "development_oracle"))
        summary = SkillOrchestrator(executor, GraspRetryPlanner(GraspRetrySettings(max_retries=0)),
            log=partial(append_jsonl, output / "assessments.jsonl"), save_observation=lambda obs: {"sequence": obs.sequence},
            episode_id=output.name, seed=seed, prepare=prepare).run()
        write_json(output / "summary.json", summary)
        return AttemptExit(0 if summary["milestone_reached"] else 1)


def settings(**overrides):
    data = dict(seeds=[3], repeats=2, variants=[{"name": "stage"}], max_attempts=2)
    data.update(overrides)
    return ExploreSettings.model_validate(data)


class ExplorationTests(unittest.TestCase):
    def test_memory_recipe_discards_old_pixel_and_keeps_symbolic_target(self):
        step = RecipeStep.model_validate({"phase": "APPROACH_RACK", "command": {
            "action": "move_to", "arm": "right", "parameters": {
                "target": "rack_approach", "offset_xyz": [0.01, 0.0, 0.02],
                "grounding": {"camera": "head_camera", "uv": [0.42, 0.61]}}}})

        saved = _memory_safe_step(step)

        self.assertEqual(saved.command.parameters.target, "rack_approach")
        self.assertEqual(saved.command.parameters.offset_xyz, (0.01, 0.0, 0.02))
        self.assertIsNone(saved.command.parameters.grounding)
        self.assertIsNone(saved.command.reason)
        trace = []
        _append_compact_step(trace, saved)
        _append_compact_step(trace, saved)
        self.assertEqual(len(trace), 1)

    def test_two_attempts_fresh_state_and_unreviewed_private_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "campaign"
            runner = FakeAttemptRunner()
            controller = ExplorationController(settings(), runner=runner, output=output)
            summary = controller.run()
            self.assertEqual((summary["attempt_count"], summary["acquisition_count"]), (2, 2))
            self.assertEqual([w.policy_resets for w in runner.workers], [1, 1])
            self.assertIsNot(runner.environments[0], runner.environments[1])
            catalog = MemoryCatalog.model_validate_json((output / "candidate_catalog.json").read_bytes())
            self.assertEqual(len(catalog.task_memory), 2)
            for record in (*catalog.task_memory, *catalog.global_memory):
                self.assertFalse(record.planner_reviewed)
                self.assertEqual(record.source, "development_oracle")
                self.assertEqual(record.evidence_level, "hypothesis")
                self.assertEqual(record.evidence[0].execution, "synthetic")
            self.assertEqual(MemorySnapshot(catalog).retrieve(task="hanging_mug", phase="APPROACH_MUG").planner_payload(), [])
            self.assertEqual(summary["explored_seeds"], [3])
            self.assertFalse(summary["evaluation_admitted"])
            with self.assertRaises(RuntimeError): controller.run()

    def test_attempt_budget_and_success_stop(self):
        for config, expected in ((settings(max_attempts=1), "attempt_budget_exhausted"),
                                 (settings(stop_on_acquisition=True), "acquisition_stop_requested")):
            with tempfile.TemporaryDirectory() as tmp:
                summary = ExplorationController(config, runner=FakeAttemptRunner(), output=Path(tmp)/"run").run()
                self.assertEqual(summary["stop_reason"], expected)
                self.assertEqual(summary["attempt_count"], 1)

    def test_schedule_offset_skips_completed_campaign_slots_without_extra_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_settings = settings(seeds=[3, 4], repeats=2, max_attempts=4,
                                    schedule_offset=1)
            runner = FakeAttemptRunner()
            summary = ExplorationController(run_settings, runner=runner,
                output=Path(tmp)/"run").run()
        self.assertEqual(summary["attempt_count"], 3)
        self.assertEqual([record["seed"] for record in summary["attempts"]], [3, 4, 4])
        self.assertEqual([record["repeat"] for record in summary["attempts"]], [1, 0, 1])
        self.assertEqual(summary["schedule_offset"], 1)

    def test_failed_attempts_retained_but_not_success_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"run"
            summary = ExplorationController(settings(), runner=FakeAttemptRunner(success=False), output=output).run()
            self.assertEqual(summary["attempt_count"], 2)
            self.assertEqual(summary["acquisition_count"], 0)
            catalog = MemoryCatalog.model_validate_json((output/"candidate_catalog.json").read_bytes())
            self.assertEqual(catalog.task_memory[0].evidence[0].outcome, "failure")
            self.assertEqual(catalog.task_memory[0].evidence_level, "hypothesis")
            self.assertEqual(len(list((output/"attempts").glob("*/summary.json"))), 2)

    def test_timeout_or_infrastructure_failure_stops_campaign(self):
        for runner, expected in ((FakeAttemptRunner(timeout=True), "attempt_timeout"),
                                 (FakeAttemptRunner(crash=True), "attempt_infrastructure_failure")):
            with tempfile.TemporaryDirectory() as tmp:
                summary = ExplorationController(settings(), runner=runner, output=Path(tmp)/"run").run()
                self.assertEqual(summary["stop_reason"], expected)
                self.assertEqual(summary["attempt_count"], 1)
                self.assertEqual(summary["task_candidates"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            times = iter([0.0, 20.0, 20.0])
            summary = ExplorationController(settings(campaign_timeout_seconds=10), runner=FakeAttemptRunner(),
                output=Path(tmp)/"run", clock=lambda: next(times)).run()
            self.assertEqual(summary["stop_reason"], "campaign_timeout")
            self.assertEqual(summary["attempt_count"], 0)

    def test_process_timeout_kills_group_and_does_not_capture_credentials(self):
        process = Mock(pid=4321)
        process.wait.side_effect = [subprocess.TimeoutExpired("child", 1), -9]
        with patch("harness.exploration.process.subprocess.Popen", return_value=process) as launch, \
             patch("harness.exploration.process.os.killpg") as kill:
            result = ProcessAttemptRunner(robotwin_root="RoboTwin", planner="anthropic", model="claude-test-model")(
                seed=3, config="config.yaml", output="out", timeout=1)
        self.assertTrue(result.timed_out)
        kill.assert_called_once()
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertNotIn("env", launch.call_args.kwargs)
        self.assertNotIn("shell", launch.call_args.kwargs)
        self.assertNotIn("ANTHROPIC_API_KEY", str(launch.call_args))

    def test_mismatched_journal_cannot_become_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"attempt_0001"
            FakeAttemptRunner()(seed=3, config=None, output=output, timeout=10)
            with self.assertRaises(ValueError):
                candidates_from_attempt(output, attempt_id="attempt_0001", seed=4, execution_kind="synthetic")
            catalog = candidates_from_attempt(output, attempt_id="attempt_0001", seed=3, execution_kind="live")
            self.assertEqual(catalog.task_memory[0].evidence_level, "single_case")
            self.assertFalse(catalog.task_memory[0].planner_reviewed)

    def test_failure_candidate_retains_visual_context_for_review_without_auto_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "attempt_0001"
            FakeAttemptRunner(success=False)(seed=3, config=None, output=output, timeout=10)
            path = output / "assessments.jsonl"
            events = [json.loads(line) for line in path.read_text().splitlines()]
            selection = next(event for event in events if event.get("event_type") == "planner_selection")
            selection["visual_context"] = {"source": "public_rgbd", "objects": {"mug": {
                "camera": "head_camera", "normalized_bbox": [0.1, 0.2, 0.4, 0.8],
                "median_rgb_0_255": [122, 65, 48],
                "visible_surface_extent_m": [0.04, 0.05, 0.09]}}}
            path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            catalog = candidates_from_attempt(output, attempt_id="attempt_0001", seed=3,
                                               execution_kind="synthetic")
            candidate = catalog.global_memory[0]
            condition = next(value for value in candidate.operating_conditions
                             if value.startswith("Public mug visual descriptor"))
            self.assertIn('"median_rgb_0_255":[122.0,65.0,48.0]', condition)
            self.assertIn("not a diagnosis", condition)
            self.assertFalse(candidate.planner_reviewed)
            self.assertEqual(candidate.evidence_level, "hypothesis")

    def test_input_memory_is_archived_without_automatic_candidate_promotion(self):
        from tests.test_memory import rule
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"run"
            memory = MemorySnapshot(MemoryCatalog(global_memory=(rule(),)))
            runner = FakeAttemptRunner()
            ExplorationController(settings(), runner=runner, output=output, memory=memory).run()
            archived = MemorySnapshot.load(output/"input_memory.json", expected_sha256=memory.sha256)
            self.assertEqual(runner.memory, (output/"input_memory.json", memory.sha256))
            self.assertEqual(len(archived.retrieve(task="hanging_mug", phase="APPROACH_MUG").global_memory), 1)
