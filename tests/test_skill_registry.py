import json
from pathlib import Path
import tempfile
import unittest

from pydantic import StrictFloat, ValidationError

from harness.legacy.config import load_config
from harness.schemas import Model
from harness.legacy.skill_registry import SkillMetadata, SkillRejected, VerifiedSkillRegistry
from harness.legacy.recovery_skills import candidate_skills
from harness.legacy.recovery_skills.retry_grasp import RetryGrasp
from tests.skill_fixtures import TestSkill, registry_with_verified_skill, verified_metadata


class RegistryTests(unittest.TestCase):
    def setUp(self):
        from tests.skill_fixtures import isolate_simulator_dependencies
        isolate_simulator_dependencies(self)

    def test_verified_only_catalog_and_duplicate_rejection(self):
        registry = registry_with_verified_skill()
        candidate = RetryGrasp()
        registry.register(candidate)
        self.assertEqual([s["skill_id"] for s in registry.catalog()], ["test_only"])
        self.assertEqual(registry.metadata("test_only").version, 2)
        self.assertEqual(registry.metadata("test_only").validation.recoveries, 8)
        with self.assertRaises(ValueError):
            registry.register(candidate)
        with self.assertRaisesRegex(SkillRejected, "unverified"):
            registry.get_verified("retry_grasp")
        with self.assertRaisesRegex(SkillRejected, "unknown"):
            registry.get_verified("invented")

    def test_production_candidates_hidden_and_require_development_context(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/harness.yaml")
        self.assertEqual(config.skills.registry, root / "configs/skills.yaml")
        registry = VerifiedSkillRegistry.load(config.skills.registry, candidate_skills(), config.skills)
        self.assertEqual(registry.catalog(), [])
        for candidate in candidate_skills():
            with self.assertRaises((NotImplementedError, ValueError)):
                candidate.execute(None, {})
            with self.assertRaisesRegex(SkillRejected, "candidate_only"):
                VerifiedSkillRegistry().register(candidate, verified_metadata(candidate))

    def test_evidence_required_and_thresholds_checked(self):
        skill = TestSkill()
        with self.assertRaises(ValidationError):
            SkillMetadata(version=2, verified=True)
        for changes in (
            {"recoveries": 6},
            {"harmful": 1, "catastrophic_failures": 1},
            {"attempts": 9, "checkpoint_ids": [f"case_{i}" for i in range(9)]},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(SkillRejected, "threshold"):
                VerifiedSkillRegistry().register(skill, verified_metadata(skill, **changes))

    def test_evidence_integrity_and_version_binding(self):
        skill = TestSkill()
        for changes in (
            {"checkpoint_ids": ["duplicate"] * 10}, {"recoveries": 11},
            {"catastrophic_failures": 1, "harmful": 0}, {"parameter_settings": []},
            {"validated_at": "2026-10-08T20:00:00"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                verified_metadata(skill, **changes)
        for changes in ({"code_hash": "a" * 64}, {"skill_version": 1},
                        {"parameter_settings": [{"distance": 1.0}]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                VerifiedSkillRegistry().register(skill, verified_metadata(skill, **changes))

    def test_stale_definition_or_dependency_loses_eligibility(self):
        with tempfile.TemporaryDirectory() as td:
            dependency = Path(td) / "controller.txt"
            dependency.write_text("version one")
            skill = TestSkill()
            skill.dependency_files = (dependency,)
            registry = VerifiedSkillRegistry()
            registry.register(skill, verified_metadata(skill))
            self.assertEqual(len(registry.catalog()), 1)
            dependency.write_text("version two")
            self.assertEqual(registry.catalog(), [])
            with self.assertRaisesRegex(SkillRejected, "stale_skill_code"):
                registry.get_verified(skill.skill_id)
        registry = registry_with_verified_skill()
        registry.get_verified("test_only").version = 3
        self.assertEqual(registry.catalog(), [])

    def test_metadata_and_catalog_copies_do_not_change_registry(self):
        registry = registry_with_verified_skill()
        registry.metadata("test_only").validation.parameter_settings[0]["distance"] = 99
        catalog = registry.catalog()
        catalog[0]["parameters"]["properties"]["distance"]["maximum"] = 99
        self.assertEqual(registry.metadata("test_only").validation.parameter_settings[0]["distance"], 0.05)
        with self.assertRaises(ValueError):
            registry.validate_parameters("test_only", {"distance": 99.0})

    def test_loader_rejects_arbitrary_imports_unknown_ids_and_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.yaml"
            for text in (
                "skills:\n  unknown:\n    version: 1\n",
                "skills:\n  retry_grasp:\n    version: 1\n    module: arbitrary_code\n",
                "skills:\n  retry_grasp:\n    version: 1\n  retry_grasp:\n    version: 1\n",
            ):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    VerifiedSkillRegistry.load(path, candidate_skills())

    def test_metadata_yaml_roundtrip(self):
        skill = TestSkill()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.yaml"
            # JSON is valid YAML; retains exact timestamp and scalar types.
            path.write_text(json.dumps({"skills": {skill.skill_id: verified_metadata(skill).model_dump(mode="json")}}))
            loaded = VerifiedSkillRegistry.load(path, [skill])
            self.assertEqual(loaded.metadata(skill.skill_id), verified_metadata(skill))
            self.assertEqual(loaded.validate_parameters(skill.skill_id, {}), {"distance": 0.05})

    def test_privileged_skills_and_unbounded_parameters_rejected(self):
        skill = TestSkill()
        skill.runtime_observation_compatible = False
        with self.assertRaisesRegex(SkillRejected, "privileged"):
            VerifiedSkillRegistry().register(skill, verified_metadata(skill))

        class Unbounded(Model):
            distance: StrictFloat

        skill = TestSkill()
        skill.parameter_model = Unbounded
        with self.assertRaisesRegex(ValueError, "bounds"):
            VerifiedSkillRegistry().register(skill)
