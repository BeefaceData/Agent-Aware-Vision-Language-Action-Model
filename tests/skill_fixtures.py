"""Synthetic CPU-only verification fixtures. Never part of the runtime catalog."""

from typing import Annotated
from pydantic import Field, StrictFloat

from harness.schemas import FailureType, Model, SkillResult, TaskPhase
from harness.skill_registry import SkillMetadata, ValidationEvidence, VerifiedSkillRegistry
from harness.legacy.recovery_skills.base import RecoverySkill


class TestParameters(Model):
    distance: Annotated[StrictFloat, Field(ge=0.01, le=0.10)] = 0.05


class TestSkill(RecoverySkill):
    skill_id = "test_only"
    version = 2
    description = "Synthetic contract used by unit tests; no robot movement."
    supported_phases = frozenset({TaskPhase.GRASP_MUG})
    supported_failures = frozenset({FailureType.MISSED_GRASP})
    parameter_model = TestParameters
    max_steps = 10
    interruption_conditions = ("step_budget", "safety_violation")

    def execute(self, context, parameters):
        self.validate_parameters(parameters)
        return SkillResult(skill_id=self.skill_id, executed_steps=0, termination_reason="test_only")


def verified_metadata(skill, **report_updates):
    fingerprint = skill.code_hash()
    report = ValidationEvidence.model_validate({
        "skill_version": skill.version, "code_hash": fingerprint,
        "checkpoint_ids": [f"synthetic_case_{i}" for i in range(10)],
        "attempts": 10, "recoveries": 8, "harmful": 0, "catastrophic_failures": 0,
        "parameter_settings": [{"distance": 0.05}] if skill.skill_id == "test_only" else [{}],
        "validated_at": "2026-10-08T20:00:00+02:00", **report_updates,
    })
    return SkillMetadata(version=skill.version, verified=True, code_hash=fingerprint, validation=report)


def registry_with_verified_skill():
    registry = VerifiedSkillRegistry()
    skill = TestSkill()
    registry.register(skill, verified_metadata(skill))
    return registry


def skill_decision(**updates):
    from harness.schemas import SupervisorDecision
    return SupervisorDecision.model_validate({
        "decision": "execute_skill", "skill_id": "test_only", "phase": "grasp_mug",
        "failure_type": "missed_grasp", "confidence": 0.9,
        "reason": "Synthetic test request", "parameters": {"distance": 0.05}, **updates,
    })


def isolate_simulator_dependencies(test_case):
    """Hash synthetic native files in CPU tests, keeping real harness dependencies.

    This never changes runtime hashing or produces offline admission evidence.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from harness.legacy.recovery_skills.retry_grasp import RetryGrasp
    from harness.legacy.recovery_skills.backoff_reapproach_rack import BackoffReapproachRack

    temporary = TemporaryDirectory()
    test_case.addCleanup(temporary.cleanup)
    for skill_type in (RetryGrasp, BackoffReapproachRack):
        dependencies = []
        for source in skill_type.dependency_files:
            if "RoboTwin" in source.parts:
                relative = Path(*source.parts[source.parts.index("RoboTwin"):])
                fixture = Path(temporary.name) / relative
                fixture.parent.mkdir(parents=True, exist_ok=True)
                fixture.write_text("Synthetic CPU-only dependency: " + relative.as_posix())
                dependencies.append(fixture)
            else:
                dependencies.append(source)
        scoped = patch.object(skill_type, "dependency_files", tuple(dependencies))
        scoped.start()
        test_case.addCleanup(scoped.stop)
