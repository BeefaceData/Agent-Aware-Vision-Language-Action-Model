import tempfile
import time
import unittest
from pathlib import Path

from modal_apps.deployed_supervised_hanging import build_parser, launch


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

            def invoke(run_id, options):
                calls.append(("invoke", run_id, options))
                return {"status": "complete"}

            def wait_for_volume(run_id):
                calls.append(("wait", run_id))
                return 3

            def download_run(run_id, target, *, skip_observations):
                calls.append(("download", run_id, target, skip_observations))

            report = launch(
                args,
                invoke=invoke,
                wait_for_volume=wait_for_volume,
                download_run=download_run,
                now=time.gmtime(0),
                nonce=7,
            )

        run_id = "supervised_hanging_seed3_19700101T000000Z_7"
        self.assertEqual(report, {"status": "complete"})
        self.assertEqual(calls[0][0:2], ("invoke", run_id))
        self.assertEqual(calls[0][2]["seed"], 3)
        self.assertEqual(calls[0][2]["chunk_size"], 16)
        self.assertEqual(calls[0][2]["time_limit_minutes"], 75)
        self.assertTrue(calls[0][2]["complementary_view"])
        self.assertEqual(calls[0][2]["perception_view"], "close")
        self.assertTrue(calls[0][2]["development_oracle_fallback"])
        self.assertEqual(calls[1], ("wait", run_id))
        self.assertEqual(calls[2], ("download", run_id, output.resolve(), False))

    def test_existing_output_is_rejected_before_remote_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                ["--model", "claude-opus-5", "--output", directory]
            )
            with self.assertRaisesRegex(ValueError, "supervised_output_already_exists"):
                launch(
                    args,
                    invoke=lambda *_: self.fail("remote function should not run"),
                    wait_for_volume=lambda _: 0,
                    download_run=lambda *_args, **_kwargs: None,
                )


if __name__ == "__main__":
    unittest.main()
