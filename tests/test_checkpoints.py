"""Checkpoint capture, storage and replay tests without GPU/network dependencies."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from development.capture.failures import capture_episode
from development.checkpoints.bank import CheckpointBank, sha256
from development.checkpoints.provenance import local_provenance
from development.checkpoints.replay import ReplayEnvironment, load_prefix
from development.checkpoints.schemas import FailureCheckpoint, ReplayOptions
from harness.legacy.schemas import FailureType, TaskPhase
from tests.test_baseline_compatibility import FakeRuntime, run_fake


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bank = CheckpointBank(self.root / "bank")

    def capture(self, runtime=None, steps=5, history=2):
        runtime = runtime or FakeRuntime()
        with contextlib.redirect_stdout(io.StringIO()):
            result = capture_episode(runtime, self.bank, seed=7, max_steps=steps, chunk_size=3,
                                     sim_client=runtime.sim, history=history, image_interval=3)
        record = self.bank.get(result["checkpoint_id"]) if result["checkpoint_id"] else None
        return record, result, runtime

    def rewrite_asset(self, checkpoint, field, text):
        relative = getattr(checkpoint, field)
        path = self.bank.resolve_artifact(relative)
        path.write_text(text)
        data = checkpoint.model_dump()
        data["artifact_hashes"][relative] = sha256(path)
        return FailureCheckpoint.model_validate(data)

    def test_round_trip_full_prefix_bounded_histories_latest_three_images(self):
        cp, result, runtime = self.capture()
        self.assertEqual(result["reason"], "horizon")
        self.assertEqual(cp.step, 5)
        self.assertEqual(cp.state, (5.0,) * 14)
        self.assertEqual(len(cp.recent_states), 2)
        self.assertEqual(len(cp.recent_actions), 2)
        self.assertEqual([list(f.action) for f in load_prefix(self.bank, cp)[1:]], runtime.actions)
        self.bank.verify_artifacts(cp)
        for image in self.bank.load_images(cp).values():
            np.testing.assert_array_equal(image, np.full((2, 3, 3), 5, dtype=np.uint8))
        self.assertEqual(len(list(self.bank.path.rglob("images_*.npz"))), 3)  # reset, 3, terminal 5
        self.assertEqual(cp, FailureCheckpoint.model_validate_json(cp.model_dump_json()))
        self.assertIsNone(cp.simulator_snapshot_path)

    def test_capture_preserves_baseline_actions_requests_and_termination(self):
        for stop in (None, 2):
            original = FakeRuntime(stop=stop)
            result, _ = run_fake(original, self.root / str(stop), max_steps=5)
            cp, captured, wrapped = self.capture(FakeRuntime(stop=stop))
            self.assertEqual(original.calls, wrapped.calls)
            self.assertEqual(original.actions, wrapped.actions)
            self.assertEqual(captured["reason"] == "success", result)
            if stop:
                self.assertIsNone(cp)

    def test_repeated_seeds_and_duplicate_publication(self):
        first, _, _ = self.capture()
        second, _, _ = self.capture()
        self.assertNotEqual(first.checkpoint_id, second.checkpoint_id)
        with self.assertRaises(FileExistsError):
            self.bank.add(first)
        self.assertEqual(len(self.bank.query()), 2)

    def test_query_and_relabel_preserve_assets_and_prior_labels(self):
        cp, _, _ = self.capture()
        self.assertEqual(self.bank.query(phase=TaskPhase.GRASP_MUG), [])
        updated = self.bank.relabel(cp.checkpoint_id, phase=TaskPhase.GRASP_MUG,
                                   failure_type=FailureType.MISSED_GRASP, notes="manually inspected")
        self.assertEqual(updated.label_revision, 1)
        self.assertEqual(updated.artifact_hashes, cp.artifact_hashes)
        self.assertEqual(self.bank.query(phase="grasp_mug", failure_type="missed_grasp"), [updated])
        history = self.bank.path / "label_history" / cp.checkpoint_id / "0.json"
        self.assertEqual(FailureCheckpoint.model_validate_json(history.read_text()), cp)

    def test_public_assets_exclude_oracle_values_and_secrets(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "DO_NOT_LOG_THIS_KEY"}):
            cp, _, _ = self.capture()
        trace = self.bank.resolve_artifact(cp.trace_path).read_text()
        self.assertNotIn("success", trace)
        self.assertEqual(set(json.loads(trace.splitlines()[0])), {"schema_version", "step", "action", "command_state"})
        for path in self.bank.path.rglob("*.json*"):
            self.assertNotIn("PRIVILEGED_MARKER", path.read_text())
            self.assertNotIn("DO_NOT_LOG_THIS_KEY", path.read_text())
        evaluation = json.loads(self.bank.resolve_artifact(cp.evaluation_path).read_text())
        self.assertFalse(evaluation["success_at_last_confirmed_frame"])

    def test_transport_failure_records_only_last_confirmed_frame(self):
        cp, result, _ = self.capture(FakeRuntime(error="/step"))
        self.assertEqual(cp.step, 0)
        self.assertEqual(len(load_prefix(self.bank, cp)), 1)
        self.assertEqual(result["reason"], "error")
        evaluation = json.loads(self.bank.resolve_artifact(cp.evaluation_path).read_text())
        self.assertEqual(evaluation["error_type"], "RuntimeError")
        self.assertIsNotNone(evaluation["unconfirmed_action"])
        self.assertEqual(cp.recent_actions, ())

    def test_reset_failure_saves_evaluation_without_inventing_checkpoint(self):
        cp, result, _ = self.capture(FakeRuntime(error="/reset"))
        self.assertIsNone(cp)
        self.assertIsNone(result["confirmed_steps"])
        self.assertEqual(self.bank.query(), [])
        self.assertEqual(len(list(self.bank.path.rglob("evaluation.json"))), 1)

    def test_paths_ids_hashes_shapes_and_finite_values_rejected(self):
        cp, _, _ = self.capture()
        invalid = [
            {"checkpoint_id": "../escape"}, {"trace_path": "../outside"},
            {"trace_path": "/absolute"}, {"trace_path": "a//b"},
            {"artifact_hashes": {}}, {"images": {}}, {"state": [float("nan")] * 14},
            {"simulator_snapshot_path": "pretend.snapshot"}, {"state": [0.0]},
            {"recent_states": []}, {"step": -1},
        ]
        for updates in invalid:
            with self.subTest(updates=list(updates)), self.assertRaises(ValidationError):
                FailureCheckpoint.model_validate({**cp.model_dump(), **updates})
        (self.bank.path / "escape").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.bank.resolve_artifact("escape/outside")

    def test_restore_uses_complete_prefix_without_policy_calls(self):
        cp, _, original = self.capture()
        fresh = FakeRuntime()
        with patch("harness.vla.modal_client.ModalPolicyClient", side_effect=AssertionError("must not query policy")):
            report = cp.restore(ReplayEnvironment(self.bank, fresh.sim))
        self.assertEqual(report.status, "matched")
        self.assertEqual(report.executed_steps, 5)
        self.assertEqual(report.state_max_abs_error, 0.0)
        self.assertFalse(report.full_state_restored)
        self.assertEqual(fresh.actions, original.actions)
        self.assertEqual([c[0] for c in fresh.calls], ["/reset"] + ["/step"] * 5)

    def test_checksum_corruption_blocks_before_reset(self):
        cp, _, _ = self.capture()
        self.bank.resolve_artifact(cp.trace_path).write_text("tampered\n")
        fresh = FakeRuntime()
        report = cp.restore(ReplayEnvironment(self.bank, fresh.sim))
        self.assertEqual(report.status, "unavailable")
        self.assertEqual(fresh.calls, [])

    def test_incomplete_or_reordered_prefix_blocks_even_with_updated_hash(self):
        for edit in (lambda f: f[1:], lambda f: f[:2] + f[3:], lambda f: f[:-1],
                     lambda f: [f[0], f[2], f[1], *f[3:]]):
            cp, _, _ = self.capture()
            lines = self.bank.resolve_artifact(cp.trace_path).read_text().splitlines()
            altered = self.rewrite_asset(cp, "trace_path", "\n".join(edit(lines)) + "\n")
            fresh = FakeRuntime()
            self.assertEqual(altered.restore(ReplayEnvironment(self.bank, fresh.sim)).status, "unavailable")
            self.assertEqual(fresh.calls, [])

    def test_checkpoint_history_mismatch_blocks_before_reset(self):
        cp, _, _ = self.capture()
        altered = FailureCheckpoint.model_validate({**cp.model_dump(), "recent_actions": [[0.0] * 14]})
        fresh = FakeRuntime()
        self.assertEqual(altered.restore(ReplayEnvironment(self.bank, fresh.sim)).status, "unavailable")
        self.assertEqual(fresh.calls, [])

    def test_provenance_mismatch_blocks_before_reset(self):
        cp, _, _ = self.capture()
        fresh = FakeRuntime()
        with patch("development.checkpoints.replay.local_provenance", return_value={"different": True}):
            report = cp.restore(ReplayEnvironment(self.bank, fresh.sim))
        self.assertEqual(report.status, "unavailable")
        self.assertEqual(report.local_provenance, "mismatch")
        self.assertEqual(fresh.calls, [])

    def test_malformed_provenance_blocks_before_reset(self):
        cp, _, _ = self.capture()
        cp = self.rewrite_asset(cp, "provenance_path", "[]")
        fresh = FakeRuntime()
        self.assertEqual(cp.restore(ReplayEnvironment(self.bank, fresh.sim)).status, "unavailable")
        self.assertEqual(fresh.calls, [])

    def test_truncated_image_archive_blocks_even_with_matching_hash(self):
        cp, _, _ = self.capture()
        relative = cp.images["head_camera"]
        path = self.bank.resolve_artifact(relative)
        path.write_bytes(path.read_bytes()[:50])
        data = cp.model_dump()
        data["artifact_hashes"][relative] = sha256(path)
        cp = FailureCheckpoint.model_validate(data)
        fresh = FakeRuntime()
        self.assertEqual(cp.restore(ReplayEnvironment(self.bank, fresh.sim)).status, "unavailable")
        self.assertEqual(fresh.calls, [])

    def test_source_provenance_includes_real_simulator_config(self):
        # CPU tests must also run in a fresh clone without downloaded simulator assets.
        source_root = self.root / "source"
        files = ("run_inference_chunked.py", "harness/environment/robotwin.py",
                 "RoboTwin/env_cfg/task_config/demo_clean.yml",
                 "RoboTwin/envs/_base_task.py", "RoboTwin/assets/embodiments/aloha-agilex/config.yml")
        for relative in files:
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        with patch("development.checkpoints.provenance.ROOT", source_root):
            paths = local_provenance()["source_hashes"]
        self.assertEqual(paths, {relative: sha256(source_root / relative) for relative in files})

    def test_state_drift_stops_at_first_mismatched_frame(self):
        for drift_step in (0, 2):
            cp, _, _ = self.capture()
            fresh = FakeRuntime()
            def drift(*args):
                obs, timing = fresh.sim(*args)
                if len(fresh.actions) == drift_step:
                    obs["state"] += 0.5
                return obs, timing
            report = cp.restore(ReplayEnvironment(self.bank, drift))
            self.assertEqual(report.status, "drifted")
            self.assertEqual(report.executed_steps, drift_step)
            self.assertEqual(report.state_max_abs_error, 0.5)

    def test_camera_drift_and_tolerance(self):
        cp, _, _ = self.capture()
        for tolerance, expected in ((5.0, "drifted"), (10.0, "matched")):
            fresh = FakeRuntime()
            def drift(*args):
                obs, timing = fresh.sim(*args)
                obs["head_camera"] += 10
                return obs, timing
            report = cp.restore(ReplayEnvironment(self.bank, drift, options=ReplayOptions(image_mae_tolerance=tolerance)))
            self.assertEqual(report.status, expected)
            self.assertEqual(report.image_mean_abs_errors["head_camera"], 10.0)
            self.assertFalse(report.full_state_restored)

    def test_replay_error_reports_confirmed_count_without_secret_exception(self):
        cp, _, _ = self.capture()
        report = cp.restore(ReplayEnvironment(self.bank, FakeRuntime(error="/step").sim))
        self.assertEqual(report.status, "error")
        self.assertEqual(report.executed_steps, 0)
        self.assertEqual(report.error_type, "RuntimeError")
        self.assertNotIn("transport failure", report.model_dump_json())

    def test_bank_cli_list_show_verify_label_and_help_without_services(self):
        cp, _, _ = self.capture()
        for command in ("list", "show", "verify", "label"):
            args = [sys.executable, "-m", "development.checkpoints.bank", command]
            if command != "list":
                args.append(cp.checkpoint_id)
            args += ["--path", str(self.bank.path)]
            if command == "label":
                args += ["--phase", "grasp_mug", "--failure-type", "missed_grasp"]
            result = subprocess.run(args, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout))
        for module in ("capture.failures", "checkpoints.replay"):
            result = subprocess.run([sys.executable, "-m", f"development.{module}", "--help"], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
