"""End-to-end run_skills wiring with CPU fakes for the simulator, evidence and policy."""

import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from development.replays.latency import analyze_episode
from harness.predicates.hanging_mug import HangingMugPredicates
from harness.predicates.registry import PredicateRegistry
from harness.vla.modal_client import ModalVLAClient
import run_skills
from tests.test_skill_orchestrator import AcquisitionEnvironment
from tests.test_skills import Runtime

CONFIG = """\
max_turns: 3
recipe:
  stage_before_first_vla: false
  max_retries: 0
  vla: {max_chunks: 3, chunk_size: 8, max_policy_actions: 24}
budget: {max_skill_calls: 3, max_analytic_ticks: 1000, max_policy_actions: 24, max_vla_chunks: 3}
"""


class ResettableEnvironment(AcquisitionEnvironment):
    def __init__(self):
        super().__init__(succeeds_after=1)
        self.reset_seeds = []

    def reset(self, seed):
        self.reset_seeds.append(seed)
        return self.observe()


class FakeSimulator:
    instances = []

    def __init__(self, robotwin_root, *, rgbd_enabled=False, diagnostics=None):
        self.task = None
        self.closed = False
        FakeSimulator.instances.append(self)

    def close(self):
        self.closed = True


class RunSkillsEpisodeTests(unittest.TestCase):
    def setUp(self):
        FakeSimulator.instances = []

    def run_main(self, directory, environment):
        config = directory / "config.yaml"
        config.write_text(CONFIG)
        oracle = PredicateRegistry(HangingMugPredicates(environment.evidence,
                                                        source="development_oracle"))
        with patch("development.simulator.local_simulator.LocalSimulator", FakeSimulator), \
             patch.object(run_skills, "open_environment", lambda *_, **__: environment), \
             patch.object(run_skills, "build_evidence", lambda *_, **__: (oracle, None)), \
             patch("harness.vla.modal_client.ModalVLAClient",
                   lambda: ModalVLAClient(Runtime())), \
             contextlib.redirect_stdout(io.StringIO()):
            return run_skills.main(["--config", str(config), "--output", str(directory / "out"),
                                    "--seed", "5", "--development-oracle"])

    def test_scripted_acquisition_episode_writes_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, environment = Path(tmp), ResettableEnvironment()
            exit_code = self.run_main(directory, environment)
            output = directory / "out"
            summary = json.loads((output / "summary.json").read_text())

            self.assertEqual(exit_code, 0)
            self.assertEqual(environment.reset_seeds, [5])
            self.assertEqual(summary["outcome"], "acquisition_confirmed")
            self.assertEqual([sim.closed for sim in FakeSimulator.instances], [True])
            for name in ("manifest.json", "assessments.jsonl", "skill_events.jsonl",
                         "candidate_catalog.json"):
                self.assertTrue((output / name).exists(), name)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["seed"], 5)
            provenance = json.loads((output / "manifest.json").read_text())["latency_provenance"]
            self.assertEqual(provenance["config_name"], "config.yaml")
            self.assertEqual(provenance["config_sha256"], sha256(CONFIG.encode()).hexdigest())
            report = analyze_episode(output)
            self.assertEqual(report["metadata"]["outcome"], "acquisition_confirmed")
            self.assertEqual(report["metadata"]["seed"], 5)
            self.assertEqual(report["reconciliation"]["status"], "partial coverage")
            self.assertGreaterEqual(report["reconciliation"]["unexplained_seconds"], 0.)
            measured = {stage["name"]: stage for stage in report["stages"]}
            self.assertEqual(measured["skill_execution"]["count"], summary["skill_calls"])
            self.assertEqual(measured["planner_selection"]["count"], summary["planner_turns"])

    def test_manifest_provenance_handles_unavailable_git_without_fabricating_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(CONFIG)
            with patch.object(run_skills.subprocess, "run", side_effect=FileNotFoundError):
                provenance = run_skills.latency_provenance(config)
            self.assertIsNone(provenance["source_commit"])
            self.assertIsNone(provenance["source_dirty"])
            self.assertEqual(provenance["config_sha256"], sha256(CONFIG.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
