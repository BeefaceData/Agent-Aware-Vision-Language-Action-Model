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

import run_inference_chunked as runner
from harness.orchestrator import PassThroughHarness
from tests.helpers import config, images


class OrchestratorTests(unittest.TestCase):
    def test_identity_no_random_state_change_and_lifecycle_guards(self):
        with tempfile.TemporaryDirectory() as td:
            state_before = np.random.get_state()
            harness = PassThroughHarness(config(td))
            action = np.arange(14, dtype=np.float32)
            with harness.episode(seed=0, max_steps=2, chunk_size=2) as episode:
                self.assertIs(episode.before_action(0, images(), np.zeros(14), action, chunk_id=0, chunk_pos=0), action)
                with self.assertRaises(RuntimeError):
                    episode.before_action(1, images(), np.zeros(14), action, chunk_id=0, chunk_pos=1)
                with self.assertRaises(RuntimeError):
                    episode.complete(success=False)
                with self.assertRaises(RuntimeError):
                    harness.episode(seed=1, max_steps=2, chunk_size=2)
                episode.after_action(0, action, np.ones(14))
                episode.complete(success=True)
                self.assertFalse(hasattr(episode.trace.context(), "success"))
            with self.assertRaises(RuntimeError):
                episode.after_action(0, action, np.ones(14))
            state_after = np.random.get_state()
            self.assertEqual(state_before[0], state_after[0])
            np.testing.assert_array_equal(state_before[1], state_after[1])
            self.assertEqual(state_before[2:], state_after[2:])

    def test_no_silent_continuation_on_logging_failure(self):
        with tempfile.TemporaryDirectory() as td:
            harness = PassThroughHarness(config(td))
            with self.assertRaises(OSError):
                with harness.episode(seed=0, max_steps=2, chunk_size=2) as episode:
                    with patch.object(harness.assessments, "append", side_effect=OSError("disk unavailable")):
                        episode.before_action(0, images(), np.zeros(14), np.ones(14), chunk_id=0, chunk_pos=0)
            end = json.loads(harness.evaluation_path.read_text())
            self.assertEqual(end["error_type"], "OSError")
            self.assertEqual(end["executed_steps"], 0)

    def test_imports_do_not_load_policy_simulator_or_provider(self):
        script = (
            "import harness.orchestrator, sys; "
            "assert not set(['torch', 'sapien', 'lerobot', 'modal', 'anthropic']) & set(sys.modules)"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_incomplete_lifecycle_is_logged_and_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            harness = PassThroughHarness(config(td))
            with self.assertRaisesRegex(RuntimeError, "without a terminal"):
                with harness.episode(seed=0, max_steps=2, chunk_size=2):
                    pass
            self.assertFalse(harness._active)
            end = json.loads(harness.evaluation_path.read_text())
            self.assertEqual(end["reason"], "error")
            self.assertEqual(end["error_type"], "RuntimeError")

    def test_interrupt_and_finalization_failure_preserve_original_error(self):
        with tempfile.TemporaryDirectory() as td:
            harness = PassThroughHarness(config(td))
            with self.assertRaises(KeyboardInterrupt):
                with harness.episode(seed=0, max_steps=2, chunk_size=2):
                    raise KeyboardInterrupt
            self.assertEqual(json.loads(harness.evaluation_path.read_text())["error_type"], "KeyboardInterrupt")
            with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(ValueError, "original"), \
                 patch("harness.orchestrator.append_jsonl", side_effect=OSError):
                with harness.episode(seed=1, max_steps=2, chunk_size=2):
                    raise ValueError("original")

    def test_cli_rejects_bad_configuration_before_worker_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "invalid.yaml"
            path.write_text("supervisor:\n  mode: anthropic\n")
            with patch.object(sys, "argv", ["runner", "policy", "--harness-config", str(path)]), \
                 patch.object(runner, "ModalPolicyClient") as worker, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                runner.main()
            self.assertEqual(exc.exception.code, 2)
            worker.assert_not_called()
