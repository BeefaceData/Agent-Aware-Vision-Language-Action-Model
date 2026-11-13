"""Seed evaluation driver: serial, resumable, never overwrites a run folder; CPU only."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from development.hanging import seed_evaluation


class FakeLauncher:
    """Writes the files a finished Modal run leaves, without starting anything."""

    def __init__(self, fail=()):
        self.commands, self.fail = [], set(fail)

    def __call__(self, command, *, cwd, env, stdout, stderr):
        self.commands.append(command)
        target = Path(command[command.index("--output")+1])
        seed = int(command[command.index("--seed")+1])
        kind = "policy_only" if command[4].endswith("::baseline") else "supervised"
        if (seed, kind) in self.fail:
            self.fail.discard((seed, kind))
            target.mkdir(parents=True)  # A crashed launch leaves a partial folder.
            return subprocess.CompletedProcess(command, 1)
        target.mkdir(parents=True)
        if kind == "policy_only":
            (target/"summary.json").write_text(json.dumps(dict(outcome="native_success", native_success=True,
                policy_actions=500, elapsed_seconds=1200.)))
        else:
            usage = dict(total=dict(requests=10, failed=0, input_tokens=100000, output_tokens=5000,
                cache_read_input_tokens=0, cache_creation_input_tokens=0, latency_seconds_mean=3.,
                latency_seconds_max=9., latency_seconds_total=30.),
                by_tool={"report_public_hanging": dict(requests=10, input_tokens=100000, output_tokens=5000,
                                                       latency_seconds_total=30.)})
            (target/"report.json").write_text(json.dumps(dict(status="action_limit", model_requests=10,
                perception_requests=4, elapsed_seconds=2000., claude_usage=usage,
                summary=dict(task_success=False, policy_actions=900, planner_turns=57))))
            (target/"supervisor_events.jsonl").write_text(json.dumps(dict(event_type="hanging_supervisor_decision",
                decision="hand_back_to_policy", reason="visibility_budget_exhausted", entry_sequence=420,
                observation_sequence=600))+"\n")
        return subprocess.CompletedProcess(command, 0)


class SeedEvaluationTests(unittest.TestCase):
    def test_runs_pairs_serially_resumes_and_sets_aside_partial_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"eval"
            launcher = FakeLauncher(fail={(16, "supervised")})
            seed_evaluation.run_all(root, [15, 16], model="claude-test", gpu="L4", launcher=launcher)
            kinds = [("::baseline" in c[4], int(c[c.index("--seed")+1])) for c in launcher.commands]
            self.assertEqual(kinds, [(True, 15), (False, 15), (True, 16), (False, 16), (False, 16)])
            self.assertTrue((root/"seed16"/"supervised.incomplete-1").is_dir())
            supervised = next(c for c in launcher.commands if "::supervised_hanging" in c[4])
            for flag in ("--perception-view", "--complementary-view", "--max-turns"):
                self.assertIn(flag, supervised)
            self.assertEqual(supervised[supervised.index("--model")+1], "claude-test")
            baseline = launcher.commands[0]
            self.assertEqual(baseline[baseline.index("--perception-camera")+1], "close")
            # A second invocation finds every run complete and launches nothing.
            again = FakeLauncher()
            seed_evaluation.run_all(root, [15, 16], model="claude-test", gpu="L4", launcher=again)
            self.assertEqual(again.commands, [])
            for name in ("report.pdf", "metrics.json", "metrics.csv", "README.md", "launches.jsonl"):
                self.assertTrue((root/name).is_file(), name)
            metrics = json.loads((root/"metrics.json").read_text())
            self.assertEqual(metrics["totals"]["policy_only"]["successes"], 2)
            self.assertEqual(metrics["totals"]["supervised"]["successes"], 0)
            # 100k input at $4/M plus 5k output at $20/M, per supervised run.
            self.assertAlmostEqual(metrics["totals"]["supervised"]["cost_claude_usd"], 2*(0.4+0.1))
            run = next(r for r in metrics["runs"] if r["kind"] == "supervised")
            self.assertEqual(run["handed_back_reason"], "visibility_budget_exhausted")
            self.assertEqual(run["folder"], "seed15/supervised")


if __name__ == "__main__":
    unittest.main()
