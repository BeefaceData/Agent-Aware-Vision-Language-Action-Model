"""Runner evidence modes stay explicit and gate-only oracle use is supported."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from run_skills import build_parser, parse_arguments, validate_evidence_mode


class RunnerEvidenceTests(unittest.TestCase):
    def test_rgbd_harness_config_defaults_to_merged_public_perception(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        args, settings = parse_arguments(build_parser(root), [
            "--config", str(root / "configs/skill_runtime_rgbd.yaml")])
        self.assertEqual(args.perception_factory,
                         "harness.perception.modal_sam3:create_detector")
        self.assertEqual(settings.perception.perception_camera, "default")

    def test_rgbd_oracle_run_keeps_public_perception_as_primary_evidence(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        args, _ = parse_arguments(build_parser(root), [
            "--config", str(root / "configs/skill_runtime_rgbd.yaml"),
            "--development-oracle"])
        self.assertEqual(args.perception_factory,
                         "harness.perception.modal_sam3:create_detector")

    def test_pixel_grounding_can_use_private_development_gates(self):
        validate_evidence_mode(development_oracle=True, perception_factory=None,
                               planner_pixel_grounding=True)

    def test_public_perception_can_be_combined_with_pixel_anchor_fallback(self):
        validate_evidence_mode(development_oracle=False, perception_factory="pkg:factory",
                               planner_pixel_grounding=True)

    def test_oracle_can_fill_public_unknowns_in_development(self):
        validate_evidence_mode(development_oracle=True, perception_factory="pkg:factory",
                               planner_pixel_grounding=False)

    def test_registry_uses_oracle_only_when_public_predicate_is_unknown(self):
        from types import SimpleNamespace
        from harness.predicates.base import PredicateResult
        from harness.predicates.registry import PredicateRegistry

        class Provider:
            source = "perception"

            def __init__(self, value, source):
                self.value, self.result_source, self.calls = value, source, 0

            def evaluate(self, name, observation, *, arm="auto"):
                self.calls += 1
                return PredicateResult(name=name, value=self.value,
                    source=self.result_source, observation_sequence=observation.sequence,
                    reason="fixture")

        observation = SimpleNamespace(sequence=7)
        public = Provider(True, "perception")
        oracle = Provider(False, "development_oracle")
        registry = PredicateRegistry(perception=public, fallback_oracle=oracle)
        confirmed = registry.evaluate("mug_grasp_confirmed", observation, allow_oracle=True)
        self.assertEqual((confirmed.value, confirmed.source), (True, "perception"))
        self.assertEqual(oracle.calls, 0)

        public.value = None
        unknown = registry.evaluate("mug_grasp_confirmed", observation, allow_oracle=False)
        self.assertIsNone(unknown.value)
        self.assertEqual(oracle.calls, 0)
        recovered = registry.evaluate("mug_grasp_confirmed", observation, allow_oracle=True)
        self.assertEqual((recovered.value, recovered.source), (False, "development_oracle"))
        self.assertEqual(oracle.calls, 1)

    def test_runner_wires_oracle_as_unknown_only_public_fallback(self):
        import run_skills
        from harness.predicates.registry import PredicateRegistry

        public = SimpleNamespace(source="perception")
        oracle = SimpleNamespace(source="development_oracle")
        registry = PredicateRegistry(perception=public)
        args = SimpleNamespace(development_oracle=True, development_start_phase=None,
                               planner_pixel_grounding=False)
        from harness.runtime.settings import RunSettings
        settings = RunSettings.model_validate({"perception": {
            "rgbd_enabled": True, "rack_grounding_camera": "perception_camera",
            "depth_pose_anchors": True}})
        with patch.object(run_skills.assembly, "public_perception_evidence",
                          return_value=(registry, object())) as public_evidence, \
             patch("development.simulator.skill_oracle.oracle_predicates",
                   return_value=SimpleNamespace(hanging_mug=oracle)):
            predicates, _ = run_skills.build_evidence(args, settings,
                environment=object(), simulator=SimpleNamespace(task=object()),
                anchors=object(), perception=object(), planner=object(),
                rack_frame_cache=None, oracle_log=lambda _: None)
        self.assertIs(predicates, registry)
        self.assertIs(registry.fallback_oracle, oracle)
        self.assertTrue(public_evidence.call_args.kwargs["depth_pose_anchors"])

    def test_evidence_source_remains_required(self):
        with self.assertRaisesRegex(ValueError, "evidence_provider_required"):
            validate_evidence_mode(development_oracle=False, perception_factory=None,
                                   planner_pixel_grounding=False)


if __name__ == "__main__":
    unittest.main()
