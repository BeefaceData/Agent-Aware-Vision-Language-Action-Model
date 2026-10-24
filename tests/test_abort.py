import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_inference_chunked as runner
from harness.legacy.pass_through import PassThroughHarness
from tests.helpers import config
from tests.test_baseline_compatibility import FakeRuntime, read_jsonl, run_fake


class AbortTests(unittest.TestCase):
    def test_abort_before_first_or_mid_chunk_skips_action_and_remaining_chunk(self):
        for stop in (0, 1, 2, 3):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                harness = PassThroughHarness(config(root))
                runtime = FakeRuntime()
                original = runner.supervise

                def hook(step, images, state, action, **kwargs):
                    if step == stop:
                        kwargs["harness"].request_abort("trusted local abort test")
                    return original(step, images, state, action, **kwargs)

                with patch.object(runner, "supervise", hook), \
                     patch.object(runner, "call_sim", runtime.sim), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = runner.run_episode(runtime, 7, 6, 3, None, "fake", 0, root / "timing", harness)
                self.assertFalse(result)
                self.assertEqual(len(runtime.actions), stop)
                self.assertEqual(sum(call[0] == "chunk" for call in runtime.calls), stop // 3 + 1)
                records = read_jsonl(harness.assessments.path)
                self.assertEqual(len(records), stop * 2 + 1)
                self.assertEqual(records[-1]["decision"], "abort")
                self.assertEqual(records[-1]["step"], stop)
                self.assertIsNone(records[-1]["executed_action"])
                end = read_jsonl(harness.evaluation_path)[0]
                self.assertEqual(end["reason"], "abort")
                self.assertEqual(end["executed_steps"], stop)
                self.assertIsNone(end["error_type"])
                self.assertIsNone(end["pending_event_id"])
                trace = read_jsonl(next(harness.trace_dir.glob("episode_*/trace.jsonl")))
                self.assertEqual(trace[-1]["event_type"], "chunk_discarded")
                self.assertEqual(trace[-1]["discarded_actions"], 3 - stop % 3)
                self.assertEqual(trace[-1]["event_id"], records[-1]["event_id"])
                # A new episode has no inherited abort or safety-budget state.
                run_fake(FakeRuntime(stop=1), root / "next", harness, max_steps=2)
                self.assertEqual(read_jsonl(harness.evaluation_path)[1]["reason"], "success")

    def test_independent_safety_abort_is_applied_and_logged(self):
        with tempfile.TemporaryDirectory() as td:
            harness = PassThroughHarness(config(td))
            runtime = FakeRuntime()
            original = runner.supervise

            def hook(step, images, state, action, **kwargs):
                from harness.legacy.schemas import TaskPhase
                kwargs["harness"].safety.reserve({}, current_phase=TaskPhase.UNKNOWN, safety_abort=True)
                return original(step, images, state, action, **kwargs)

            with patch.object(runner, "supervise", hook), patch.object(runner, "call_sim", runtime.sim), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_episode(runtime, 7, 5, 3, None, "fake", 0, Path(td) / "timing", harness)
            self.assertEqual(runtime.actions, [])
            record = read_jsonl(harness.assessments.path)[0]
            self.assertEqual(record["decision"], "abort")
            self.assertEqual(record["notes"], "independent_safety_abort")
