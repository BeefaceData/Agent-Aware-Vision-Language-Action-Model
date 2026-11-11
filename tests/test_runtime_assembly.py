"""CPU tests for episode assembly, CLI option checks and turn-decision helpers."""

from argparse import Namespace
from types import SimpleNamespace
import unittest

from harness.planner.handoff import HandoffPlanner
from harness.planner.scripted import GraspRetryPlanner
from harness.predicates.base import PredicateResult
from harness.runtime import assembly
from harness.runtime.orchestrator import (grounding_rejection_code, reselection_feedback,
                                          turn_failure)
from harness.runtime.settings import RunSettings
from harness.skills.schemas import SkillCommand
from run_skills import option_error

HANDOFF = {"max_chunks": 1, "chunk_size": 8, "max_policy_actions": 8}


def cli(**overrides):
    values = dict(planner="scripted", anthropic_model=None, perception_factory=None,
                  evaluation_manifest=None,
                  planner_pixel_grounding=False, memory_sha256=None, memory_snapshot=None,
                  development_oracle=False, replay_prefix=None, development_start_phase=None)
    return Namespace(**{**values, **overrides})


def move(target, *, camera=None):
    parameters = {"target": target}
    if camera is not None:
        parameters["grounding"] = {"camera": camera, "uv": (0.5, 0.5)}
    return SkillCommand.model_validate({"action": "move_to", "arm": "left", "parameters": parameters})


def predicate(name, value):
    return PredicateResult(name=name, value=value, source="perception",
                           observation_sequence=1, reason="test")


class OptionErrorTests(unittest.TestCase):
    def test_default_scripted_run_is_valid(self):
        self.assertIsNone(option_error(cli(), RunSettings()))

    def test_perception_factory_requires_rgbd_capture(self):
        self.assertIn("rgbd_enabled", option_error(cli(perception_factory="m:f"), RunSettings()))

    def test_full_task_scope_requires_anthropic_planner(self):
        settings = RunSettings.model_validate(
            {"anthropic": {"model": "claude-test", "task_scope": "full_task"}})
        self.assertEqual(option_error(cli(), settings), "Full-task scope requires --planner anthropic")

    def test_anthropic_planner_requires_an_explicit_model(self):
        self.assertIn("no fallback model", option_error(cli(planner="anthropic"), RunSettings()))
        self.assertIsNone(option_error(cli(planner="anthropic", anthropic_model="claude-x"),
                                       RunSettings()))

    def test_memory_digest_requires_snapshot(self):
        self.assertIn("--memory-snapshot", option_error(cli(memory_sha256="0" * 64), RunSettings()))

    def test_development_resume_requires_oracle_prefix_and_scripted_planner(self):
        self.assertIn("requires --development-oracle", option_error(
            cli(development_start_phase="left_grasp_retry", replay_prefix="prefix.npz"), RunSettings()))
        self.assertIn("requires --replay-prefix", option_error(
            cli(development_oracle=True, development_start_phase="left_grasp_retry"), RunSettings()))
        self.assertIn("limited to --planner scripted", option_error(
            cli(planner="anthropic", development_oracle=True, replay_prefix="prefix.npz",
                development_start_phase="left_grasp_retry"), RunSettings()))

    def test_left_resume_requires_acquisition_recipe(self):
        settings = RunSettings.model_validate({"handoff": HANDOFF})
        self.assertIn("acquisition-only", option_error(cli(development_oracle=True,
            replay_prefix="prefix.npz", development_start_phase="left_grasp_retry"), settings))

    def test_right_resume_requires_handoff_recipe(self):
        self.assertIn("scripted handoff", option_error(cli(development_oracle=True,
            replay_prefix="prefix.npz", development_start_phase="right_grasp"), RunSettings()))

    def test_hang_recovery_requires_handoff_recipe(self):
        self.assertIn("scripted handoff", option_error(cli(development_oracle=True,
            replay_prefix="prefix.npz", development_start_phase="hang_recovery"), RunSettings()))


class AnthropicSettingsTests(unittest.TestCase):
    def test_missing_model_is_rejected_without_fallback(self):
        with self.assertRaisesRegex(ValueError, "anthropic_model_required"):
            assembly.anthropic_settings(RunSettings())

    def test_cli_model_overrides_config_model(self):
        settings = RunSettings.model_validate({"anthropic": {"model": "claude-config"}})
        self.assertEqual(assembly.anthropic_settings(settings, "claude-cli").model, "claude-cli")

    def test_handoff_recipe_without_planner_config_implies_full_task(self):
        settings = RunSettings.model_validate({"handoff": HANDOFF})
        resolved = assembly.anthropic_settings(settings, "claude-cli")
        self.assertEqual(resolved.task_scope, "full_task")
        self.assertEqual(resolved.vla.max_policy_actions, 8)


class BuildPlannerTests(unittest.TestCase):
    def test_scripted_planner_follows_recipe(self):
        self.assertIsInstance(assembly.build_planner("scripted", RunSettings()), GraspRetryPlanner)

    def test_scripted_handoff_recipe_uses_handoff_planner(self):
        settings = RunSettings.model_validate({"handoff": HANDOFF})
        self.assertIsInstance(assembly.build_planner("scripted", settings), HandoffPlanner)

    def test_hang_recovery_uses_development_only_sequence(self):
        from development.planner.hanging_recovery import HangingRecoveryPlanner
        settings = RunSettings.model_validate({"handoff": HANDOFF})
        self.assertIsInstance(assembly.build_planner("scripted", settings,
            development_start_phase="hang_recovery"), HangingRecoveryPlanner)

    def test_development_resume_starts_left_recovery_open(self):
        planner = assembly.build_planner("scripted", RunSettings.model_validate({
            "recipe": {"max_retries": 1}}), development_start_phase="left_grasp_retry")
        self.assertEqual(planner.phase, "RETRY_OPEN")
        self.assertEqual(planner.retries, 1)
        selected = planner.select(SimpleNamespace(previous=None,
            value=lambda name: False if name == "mug_lifted" else True))
        self.assertEqual((selected.phase, selected.command.action.value),
                         ("RETRY_OPEN", "set_gripper"))

    def test_development_resume_skips_left_transfer_for_right_grasp(self):
        settings = RunSettings.model_validate({"handoff": HANDOFF})
        planner = assembly.build_planner("scripted", settings, development_start_phase="right_grasp")
        self.assertTrue(planner.left_transfer_confirmed)
        self.assertTrue(planner.stage_attempted)
        self.assertFalse(planner.right_lift_attempted)
        selected = planner.select(SimpleNamespace(previous=None,
            value=lambda name: {"task_success": False, "right_mug_lifted": False}.get(name)))
        self.assertEqual((selected.phase, selected.command.action.value, selected.command.stop),
                         ("RIGHT_GRASP", "vla_act", "right_mug_lifted"))

    def test_unknown_planner_kind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported_planner"):
            assembly.build_planner("random", RunSettings())


class FailingPerception:
    tracker = "unset"

    def observe(self, observation):
        return None

    def detection_mask(self, label, observation, *, camera):
        raise ValueError("detection_mask_not_detected")

    def set_mug_lift_evidence_tracker(self, tracker):
        self.tracker = tracker


class InitializationFailureTests(unittest.TestCase):
    def setUp(self):
        self.environment = SimpleNamespace(observe=lambda: SimpleNamespace(sequence=0))
        self.planner = SimpleNamespace(client=None, settings=SimpleNamespace(model="claude-test"))
        self.events = []

    def test_mug_lift_failure_disables_tracker_instead_of_falling_back(self):
        perception = FailingPerception()
        result = assembly.initialize_mug_lift_evidence(
            self.environment, perception, self.planner, log=self.events.append)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(perception.tracker)
        self.assertEqual(self.events[0]["event_type"], "table_plane_initialized")

    def test_rack_frame_failure_returns_no_cache_and_logs_error_type(self):
        cache, report = assembly.initialize_rack_frame(
            self.environment, FailingPerception(), self.planner, camera="head_camera",
            log=self.events.append)
        self.assertIsNone(cache)
        self.assertEqual(report["status"], "error")
        self.assertEqual(self.events[0]["event_type"], "rack_grounding_initialized")
        self.assertEqual(self.events[0]["error_code"], "structured_tool_client_required")


class TurnDecisionTests(unittest.TestCase):
    def test_hang_target_grounding_errors_become_recoverable_rejections(self):
        code = grounding_rejection_code(move("rack_seat"), "rack_frame_not_cached")
        self.assertEqual(code, "hang_target_grounding_unavailable")

    def test_unrecognized_grounding_error_is_not_swallowed(self):
        self.assertIsNone(grounding_rejection_code(move("mug_pregrasp"), "unexpected_error"))

    def test_second_pixel_rejection_requires_a_different_camera(self):
        feedback, avoid = reselection_feedback(move("mug_pregrasp", camera="left_camera"), 2)
        self.assertEqual(avoid, "left_camera")
        self.assertIn("different camera view", feedback)

    def test_vla_chunk_boundary_returns_control_for_planner_feedback(self):
        command = SkillCommand(action="vla_act", stop="mug_lifted")
        result = SimpleNamespace(completed=True, stop_reason="chunk_budget_exhausted")
        self.assertIsNone(turn_failure(command, result, [predicate("mug_lifted", False)]))
        self.assertIsNone(turn_failure(command, result, [predicate("mug_lifted", True)]))


if __name__ == "__main__":
    unittest.main()
