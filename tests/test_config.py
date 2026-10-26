import os
from pathlib import Path
import tempfile
import unittest

from harness.legacy.config import load_config
from harness.runtime.settings import RunSettings


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

    def test_rgbd_sensor_capture_is_opt_in_and_strict(self):
        self.assertFalse(RunSettings().perception.rgbd_enabled)
        self.assertTrue(RunSettings.model_validate({"perception": {"rgbd_enabled": True}})
                        .perception.rgbd_enabled)
        self.assertEqual(RunSettings().perception.rack_prompt_profile, "default")
        self.assertEqual(RunSettings().perception.rack_grounding_camera, "head_camera")
        self.assertEqual(RunSettings.model_validate({"perception": {
            "rgbd_enabled": True, "rack_prompt_profile": "explore_mug_rack"}})
            .perception.rack_prompt_profile, "explore_mug_rack")
        with self.assertRaises(ValueError):
            RunSettings.model_validate({"perception": {"rgbd_enabled": 1}})
        with self.assertRaises(ValueError):
            RunSettings.model_validate({"perception": {
                "rack_prompt_profile": "unbounded_user_prompt"}})
        with self.assertRaisesRegex(ValueError, "opus_rack_grounding_requires_rgbd"):
            RunSettings.model_validate({"perception": {"opus_rack_grounding": True}})
        with self.assertRaisesRegex(ValueError, "opus_rack_grounding_requires_full_task_anthropic"):
            RunSettings.model_validate({"perception": {"rgbd_enabled": True,
                "opus_rack_grounding": True}, "anthropic": {"model": "claude-test-model"}})
        with self.assertRaisesRegex(ValueError, "opus_mug_lift_grounding_requires_rgbd"):
            RunSettings.model_validate({"perception": {"opus_mug_lift_grounding": True},
                "anthropic": {"model": "claude-test-model"}})
        enabled = RunSettings.model_validate({"perception": {
            "rgbd_enabled": True, "opus_mug_lift_grounding": True},
            "anthropic": {"model": "claude-test-model"}})
        self.assertTrue(enabled.perception.opus_mug_lift_grounding)

    def test_anthropic_full_task_scope_selects_full_task_completion(self):
        settings = RunSettings.model_validate({"anthropic": {
            "model": "claude-test-model", "task_scope": "full_task"}})
        self.assertEqual(settings.completion_predicate, "task_success")
        acquisition = RunSettings.model_validate({"anthropic": {"model": "claude-test-model"}})
        self.assertEqual(acquisition.completion_predicate, "mug_lifted")
        with self.assertRaisesRegex(ValueError, "handoff_recipe_requires_full_task_anthropic_scope"):
            RunSettings.model_validate({"handoff": {"max_chunks": 1, "chunk_size": 8,
                "max_policy_actions": 8}, "anthropic": {"model": "claude-test-model"}})
        stop = RunSettings.model_validate({"stop_after_left_transfer": True, "anthropic": {
            "model": "claude-test-model", "task_scope": "full_task"}})
        self.assertEqual(stop.completion_predicate, "left_transfer_complete")

        root = Path(__file__).resolve().parents[1]
        import yaml
        with (root / "configs/skill_runtime_opus_full_task.yaml").open() as stream:
            checked_in = RunSettings.model_validate(yaml.safe_load(stream))
        self.assertEqual(checked_in.completion_predicate, "task_success")
        self.assertEqual(checked_in.anthropic.task_scope, "full_task")
        self.assertFalse(hasattr(checked_in.anthropic, "max_vla_attempts"))
        with (root / "configs/skill_runtime_opus_full_task_rgbd.yaml").open() as stream:
            rgbd = RunSettings.model_validate(yaml.safe_load(stream))
        self.assertTrue(rgbd.perception.rgbd_enabled)
        self.assertTrue(rgbd.perception.opus_rack_grounding)
        self.assertEqual(rgbd.perception.rack_grounding_camera, "right_camera")
        self.assertFalse(hasattr(rgbd.anthropic, "hang_vla_attempts"))
        self.assertEqual(rgbd.completion_predicate, "task_success")
        self.assertEqual(rgbd.anthropic.task_scope, "full_task")
        with (root / "configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml").open() as stream:
            table_lift = RunSettings.model_validate(yaml.safe_load(stream))
        self.assertTrue(table_lift.perception.opus_mug_lift_grounding)
        self.assertTrue(table_lift.perception.opus_rack_grounding)
        self.assertFalse(hasattr(table_lift.anthropic, "hang_vla_attempts"))
