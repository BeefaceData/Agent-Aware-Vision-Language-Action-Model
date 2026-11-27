import json
import tempfile
import time
import unittest
from pathlib import Path

from modal_apps.deployed_supervised_hanging import build_parser, launch, resume


class DeployedSupervisedHangingTests(unittest.TestCase):
    def test_launch_preserves_supervised_options_and_downloads_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed3"
            args = build_parser().parse_args(
                [
                    "--model", "claude-opus-5",
                    "--seed", "3",
                    "--time-limit-minutes", "75",
                    "--complementary-view",
                    "--perception-view", "close",
                    "--development-oracle-fallback",
                    "--output", str(output),
                ]
            )
            calls = []

            class Call:
                object_id = "fc-test"

                def get(self):
                    calls.append(("get", self.object_id))
                    return {"status": "complete"}

            def spawn(run_id, options):
                calls.append(("spawn", run_id, options))
                return Call()

            def wait_for_volume(run_id):
                calls.append(("wait", run_id))
                return 3

            def download_run(run_id, target, *, skip_observations, resume):
                calls.append(("download", run_id, target, skip_observations, resume))

            report = launch(
                args,
                spawn=spawn,
                wait_for_volume=wait_for_volume,
                download_run=download_run,
                now=time.gmtime(0),
                nonce=7,
            )
            state = json.loads(
                output.with_name(f"{output.name}.modal-call.json").read_text()
            )

        run_id = "supervised_hanging_seed3_19700101T000000Z_7"
        self.assertEqual(report, {"status": "complete"})
        self.assertEqual(calls[0][0:2], ("spawn", run_id))
        self.assertEqual(calls[0][2]["seed"], 3)
        self.assertEqual(calls[0][2]["chunk_size"], 16)
        self.assertEqual(calls[0][2]["time_limit_minutes"], 75)
        self.assertTrue(calls[0][2]["complementary_view"])
        self.assertEqual(calls[0][2]["perception_view"], "close")
        self.assertTrue(calls[0][2]["development_oracle_fallback"])
        self.assertEqual(calls[1], ("get", "fc-test"))
        self.assertEqual(calls[2], ("wait", run_id))
        self.assertEqual(calls[3], ("download", run_id, output.resolve(), False, True))
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["call_id"], "fc-test")

    def test_existing_output_is_rejected_before_remote_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                ["--model", "claude-opus-5", "--output", directory]
            )
            with self.assertRaisesRegex(ValueError, "supervised_output_already_exists"):
                launch(
                    args,
                    spawn=lambda *_: self.fail("remote function should not run"),
                    wait_for_volume=lambda _: 0,
                    download_run=lambda *_args, **_kwargs: None,
                )

    def test_interrupted_client_leaves_detached_call_state_for_resume(self):
        class InterruptedCall:
            object_id = "fc-detached"

            def get(self):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed3"
            args = build_parser().parse_args(
                ["--model", "claude-opus-5", "--output", str(output)]
            )
            with self.assertRaises(KeyboardInterrupt):
                launch(
                    args,
                    spawn=lambda *_: InterruptedCall(),
                    wait_for_volume=lambda _: self.fail("unfinished call has no artifacts"),
                    download_run=lambda *_args, **_kwargs: None,
                    now=time.gmtime(0),
                    nonce=9,
                )
            state_path = output.with_name(f"{output.name}.modal-call.json")
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "detached")
            self.assertEqual(state["call_id"], "fc-detached")

    def test_terminal_remote_failure_is_recorded_as_failed_not_detached(self):
        class FailedCall:
            object_id = "fc-failed"

            def get(self):
                raise TypeError("remote options mismatch")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed3"
            args = build_parser().parse_args(
                ["--model", "claude-opus-5", "--output", str(output)]
            )
            with self.assertRaisesRegex(TypeError, "remote options mismatch"):
                launch(
                    args,
                    spawn=lambda *_: FailedCall(),
                    wait_for_volume=lambda _: self.fail("failed call has no artifacts"),
                    download_run=lambda *_args, **_kwargs: None,
                    now=time.gmtime(0),
                    nonce=10,
                )
            state_path = output.with_name(f"{output.name}.modal-call.json")
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["local_error_type"], "TypeError")
            with self.assertRaisesRegex(ValueError, "supervised_remote_call_failed"):
                resume(
                    state_path,
                    lookup_call=lambda _: self.fail("failed call must not be resumed"),
                    wait_for_volume=lambda _: 0,
                    download_run=lambda *_args, **_kwargs: None,
                )

    def test_resume_waits_for_saved_call_and_resumes_download(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed3"
            output.mkdir()
            state_path = output.with_name(f"{output.name}.modal-call.json")
            state_path.write_text(json.dumps({
                "schema_version": 1,
                "status": "detached",
                "call_id": "fc-resume",
                "run_id": "remote-seed3",
                "output": str(output),
                "skip_observations": False,
            }))
            calls = []

            class FinishedCall:
                def get(self):
                    return {"status": "task_success"}

            report = resume(
                state_path,
                lookup_call=lambda call_id: calls.append(("lookup", call_id)) or FinishedCall(),
                wait_for_volume=lambda run_id: calls.append(("wait", run_id)) or 3,
                download_run=lambda run_id, target, **kwargs:
                    calls.append(("download", run_id, target, kwargs)),
            )

            self.assertEqual(report["status"], "task_success")
            self.assertEqual(calls[0], ("lookup", "fc-resume"))
            self.assertEqual(calls[1], ("wait", "remote-seed3"))
            self.assertEqual(calls[2], (
                "download", "remote-seed3", output,
                {"skip_observations": False, "resume": True},
            ))


if __name__ == "__main__":
    unittest.main()
