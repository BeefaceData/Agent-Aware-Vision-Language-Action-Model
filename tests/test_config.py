import os
from pathlib import Path
import tempfile
import unittest

from harness.config import load_config


class ConfigTests(unittest.TestCase):
    def test_paths_are_relative_to_yaml_and_survive_cwd_change(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "configs"
            folder.mkdir()
            path = folder / "harness.yaml"
            path.write_text("harness:\n  enabled: true\nlogging:\n  trace_dir: ../traces\n")
            previous = Path.cwd()
            try:
                os.chdir("/tmp")
                config = load_config(path)
                self.assertTrue(config.harness.enabled)
                self.assertEqual(config.logging.trace_dir, Path(td) / "traces")
                self.assertTrue(config.logging.assessment_dir.is_absolute())
            finally:
                os.chdir(previous)
            self.assertFalse(config.logging.trace_dir.exists())

    def test_invalid_configs_fail_closed(self):
        values = ["", "[]", "harness: [", "unknown: true", "harness:\n  enabled: 'true'",
                  "harness:\n  enabled: 1", "supervisor:\n  mode: anthropic",
                  "supervisor:\n  api_key: secret", "trace:\n  state_history: 0",
                  "trace:\n  keyframe_interval: true", "trace:\n  keyframes: 1.5",
                  "harness:\n  enabled: true\n  enabled: false", "1: value",
                  "task: another_task", "!!python/object:bad {}",
                  "safety:\n  max_skill_steps: 0", "safety:\n  max_interventions_per_episode: -1",
                  "safety:\n  abort_on_invalid_skill_request: 'false'",
                  "skills:\n  minimum_recovery_rate: 1.1",
                  "skills:\n  minimum_validation_cases: true",
                  "supervisor:\n  minimum_confidence: .nan"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.yaml"
            for value in values:
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_config(path)

    def test_checked_in_configs(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse(load_config(root / "configs/baseline.yaml").harness.enabled)
        self.assertTrue(load_config(root / "configs/harness.yaml").harness.enabled)
