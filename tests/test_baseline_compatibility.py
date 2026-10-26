"""Exercise the real episode loop without GPU, network, or policy dependencies."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_inference_chunked as runner
from harness.legacy.pass_through import PassThroughHarness
from tests.helpers import config, images


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class FakeRuntime:
    def __init__(self, *, stop=None, error=None, empty=False):
        self.stop = stop
        self.error = error
        self.empty = empty
        self.actions = []
        self.calls = []
        self.proposals = []

    def reset(self):
        self.calls.append(("policy_reset",))
        return {"chunk_limit": 24}, 0.0

    def act_chunk(self, rgb, state, chunk_size):
        step = len(self.actions)
        self.calls.append(("chunk", step, chunk_size, tuple(state)))
        actions = np.array([
            np.arange(14, dtype=np.float32) / 10 + step + i for i in range(chunk_size)
        ]) if not self.empty else np.empty((0, 14), dtype=np.float32)
        self.proposals.append((actions, actions.copy()))
        return actions, {
            "modal_rpc_ms": 1.0, "modal_compute_ms": 0.5, "modal_overhead_ms": 0.5,
            "cloud_inference_ms": 0.25, "num_actions": len(actions),
        }

    def sim(self, host, port, path, payload):
        self.calls.append((path, payload))
        if self.error == path:
            raise RuntimeError("transport failure")
        if path == "/step":
            self.actions.append(payload["action"])
        step = len(self.actions)
        # Additional oracle values must never escape via public harness inputs.
        observation = {
            **images(step), "state": np.full(14, step, dtype=np.float32),
            "success": self.stop is not None and step >= self.stop,
            "mug_pose": "PRIVILEGED_MARKER", "contacts": "PRIVILEGED_MARKER",
        }
        return observation, {key: 0.0 for key in (
            "sim_http_ms", "sim_action_ms", "sim_observation_ms", "sim_encode_ms",
        )}


def run_fake(runtime, directory, harness=None, max_steps=5, seed=7):
    with patch.object(runner, "call_sim", runtime.sim), contextlib.redirect_stdout(io.StringIO()):
        result = runner.run_episode(
            runtime, seed, max_steps, 3, None, "fake", 0, directory, harness=harness,
        )
    with (Path(directory) / f"episode_{seed}_steps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for original, snapshot in runtime.proposals:
        np.testing.assert_array_equal(original, snapshot)
    return result, rows


class BaselineCompatibilityTests(unittest.TestCase):
    def test_disabled_and_pass_match_baseline_calls_actions_and_termination(self):
        for stop, horizon in ((None, 5), (2, 5), (None, 1), (4, 8)):
            with self.subTest(stop=stop, horizon=horizon), tempfile.TemporaryDirectory() as td:
                records = []
                for mode in ("absent", "disabled", "pass"):
                    root = Path(td) / mode
                    harness = None if mode == "absent" else PassThroughHarness(config(root, mode == "pass"))
                    runtime = FakeRuntime(stop=stop)
                    result, rows = run_fake(runtime, root / "timing", harness, horizon)
                    indices = [(r["step"], r["chunk_id"], r["chunk_pos"]) for r in rows]
                    records.append((result, runtime.calls, runtime.actions, indices))
                    if mode == "disabled":
                        self.assertFalse((root / "traces").exists())
                        self.assertFalse((root / "assessments").exists())
                    if mode == "pass":
                        assessments = read_jsonl(harness.assessments.path)
                        self.assertEqual(len(assessments), 2 * len(runtime.actions))
                        for step in range(len(runtime.actions)):
                            pre, post = assessments[step * 2:step * 2 + 2]
                            self.assertEqual(pre["event_id"], post["event_id"])
                            self.assertEqual(pre["policy_action"], runtime.actions[step])
                            self.assertEqual(post["executed_action"], runtime.actions[step])
                            self.assertEqual(pre["pre_state_summary"]["command_state"], [step] * 14)
                            self.assertEqual(post["post_state_summary"]["command_state"], [step + 1] * 14)
                            self.assertEqual(pre["decision"], "pass")
                            self.assertEqual(post["outcome"], "unknown")
                        end = read_jsonl(harness.evaluation_path)[0]
                        self.assertEqual(end["success"], result)
                        self.assertEqual(end["executed_steps"], len(runtime.actions))
                        self.assertIsNone(end["pending_event_id"])
                self.assertEqual(records[0], records[1])
                self.assertEqual(records[0], records[2])

    def test_fresh_keyframes_and_oracle_secret_isolation(self):
        with tempfile.TemporaryDirectory() as td, patch.dict("os.environ", {"ANTHROPIC_API_KEY": "KEY_MUST_NOT_APPEAR"}):
            harness = PassThroughHarness(config(td))
            run_fake(FakeRuntime(), Path(td) / "timing", harness)
            traces = list(harness.trace_dir.glob("episode_*/trace.jsonl"))
            events = read_jsonl(traces[0])
            proposals = [e for e in events if e["event_type"] == "decision"]
            self.assertEqual([e["step"] for e in proposals if e["keyframe_path"]], [0, 2, 4])
            for event in proposals:
                if event["keyframe_path"]:
                    with np.load(traces[0].parent / event["keyframe_path"], allow_pickle=False) as data:
                        self.assertEqual(set(data.files), set(runner.CAMERAS))
                        for camera in runner.CAMERAS:
                            np.testing.assert_array_equal(data[camera], images(event["step"])[camera])
            for path in Path(td).rglob("*.jsonl"):
                text = path.read_text()
                self.assertNotIn("KEY_MUST_NOT_APPEAR", text)
                self.assertNotIn("PRIVILEGED_MARKER", text)
                if "evaluation" not in path.parts:
                    self.assertNotIn('"success"', text)

    def test_failure_finalizes_and_never_claims_unconfirmed_execution(self):
        for failure in ("/reset", "/step", "empty"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                harness = PassThroughHarness(config(td))
                runtime = FakeRuntime(error=failure, empty=failure == "empty")
                with self.assertRaises(RuntimeError):
                    run_fake(runtime, Path(td) / "timing", harness)
                self.assertFalse(harness._active)
                end = read_jsonl(harness.evaluation_path)[0]
                self.assertEqual(end["reason"], "error")
                self.assertEqual(end["executed_steps"], 0)
                self.assertIsNone(end["success"])
                if failure == "/step":
                    self.assertIsNotNone(end["pending_event_id"])
                    self.assertEqual(len(read_jsonl(harness.assessments.path)), 1)
                else:
                    self.assertIsNone(end["pending_event_id"])

    def test_repeated_seeds_do_not_overwrite_harness_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            harness = PassThroughHarness(config(td))
            for i in range(2):
                run_fake(FakeRuntime(stop=1), Path(td) / f"timing{i}", harness)
            rows = read_jsonl(harness.evaluation_path)
            self.assertEqual(len({r["episode_id"] for r in rows}), 2)
            self.assertEqual(len(list(harness.trace_dir.glob("episode_*/trace.jsonl"))), 2)
            other = PassThroughHarness(config(td))
            self.assertNotEqual(other.run_id, harness.run_id)

    def test_baseline_runner_does_not_load_runtime_validation_stack(self):
        # The frozen baseline shares RoboTwin/Modal helpers with the harness but
        # must stay runnable with only the simulator/policy dependencies.
        code = "import sys, run_inference_chunked; sys.exit('pydantic' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0)

    def test_original_identity_hook_still_works(self):
        action = np.arange(14, dtype=np.float32)
        self.assertIs(runner.supervise(0, images(), np.zeros(14), action), action)
