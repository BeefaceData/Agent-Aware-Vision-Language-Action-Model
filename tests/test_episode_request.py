"""Modal episode requests keep run_skills arguments and redirect container-owned paths."""

from pathlib import Path
import tempfile
import unittest

from modal_apps.episode_request import (DEFAULT_TIME_LIMIT_MINUTES, build_request,
                                        set_aside_previous_attempt)


def request(*argv):
    return build_request(argv, input_dir="/tmp/in", runs_dir="/runs", robotwin_root="/opt/RoboTwin")


class EpisodeRequestTests(unittest.TestCase):
    def test_config_is_uploaded_and_its_path_rewritten(self):
        result = request("--config", "configs/a.yaml", "--output", "data/runs/x")

        self.assertEqual(result.uploads, {"config.yaml": Path("configs/a.yaml")})
        self.assertEqual(result.remote_argv[:2], ("--config", "/tmp/in/config.yaml"))

    def test_output_and_robotwin_root_point_into_the_container(self):
        result = request("--output", "data/runs/x")

        self.assertEqual(result.run_id, "x")
        self.assertEqual(result.local_output, Path("data/runs/x"))
        self.assertEqual(result.remote_argv, ("--output", "/runs/x",
                                              "--robotwin-root", "/opt/RoboTwin"))

    def test_other_options_are_forwarded_unchanged(self):
        result = request("--planner", "anthropic", "--development-oracle", "--output", "o")

        self.assertEqual(result.remote_argv[:3], ("--planner", "anthropic", "--development-oracle"))

    def test_local_argv_keeps_local_paths_for_validation(self):
        result = request("--config=c.yaml", "--seed", "4", "--output", "o")

        self.assertEqual(result.local_argv, ("--config", "c.yaml", "--seed", "4", "--output", "o"))

    def test_launcher_options_are_consumed_not_forwarded(self):
        result = request("--time-limit-minutes", "30", "--skip-observations", "--output", "o")

        self.assertEqual(result.time_limit_minutes, 30)
        self.assertTrue(result.skip_observations)
        self.assertNotIn("--time-limit-minutes", result.remote_argv)
        self.assertNotIn("--skip-observations", result.remote_argv)

    def test_time_limit_has_a_default(self):
        self.assertEqual(request("--output", "o").time_limit_minutes, DEFAULT_TIME_LIMIT_MINUTES)

    def test_output_is_required(self):
        with self.assertRaisesRegex(ValueError, "--output is required"):
            request("--planner", "scripted")

    def test_time_limit_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "between 1 and"):
            request("--time-limit-minutes", "0", "--output", "o")

    def test_frozen_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            request("--evaluation-manifest", "m.json", "--output", "o")

    def test_option_missing_its_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "needs a value"):
            request("--output")


class PreviousAttemptTests(unittest.TestCase):
    def test_fresh_output_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(set_aside_previous_attempt(Path(directory) / "run"))

    def test_interrupted_attempt_is_kept_beside_a_clean_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            (output / "summary.json").write_text("{}")

            kept = set_aside_previous_attempt(output)

            self.assertEqual(kept.name, "run.preempted-1")
            self.assertTrue((kept / "summary.json").is_file())
            self.assertFalse(output.exists())

    def test_repeated_preemptions_get_increasing_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            set_aside_previous_attempt(output)
            output.mkdir()

            self.assertEqual(set_aside_previous_attempt(output).name, "run.preempted-2")


if __name__ == "__main__":
    unittest.main()
