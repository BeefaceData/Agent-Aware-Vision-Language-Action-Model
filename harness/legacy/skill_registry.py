"""Legacy recovery verification catalog, separate from harness.skills.registry.

This consumes offline evidence; it neither runs validation nor admits a skill
automatically. Test fixtures must never be copied into the production catalog.
"""

from datetime import datetime
from typing import Annotated

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from harness.legacy.config import SkillOptions, load_yaml_mapping
from harness.schemas import Identifier, Model, NonnegativeInt, Parameter
from harness.legacy.recovery_skills.base import RecoverySkill


Version = Annotated[StrictInt, Field(gt=0)]
CodeHash = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]


class ValidationEvidence(Model):
    skill_version: Version
    code_hash: CodeHash
    checkpoint_ids: tuple[Identifier, ...]
    attempts: Annotated[StrictInt, Field(gt=0)]
    recoveries: NonnegativeInt
    harmful: NonnegativeInt
    catastrophic_failures: NonnegativeInt
    parameter_settings: tuple[dict[StrictStr, Parameter], ...]
    validated_at: StrictStr

    @model_validator(mode="after")
    def consistency(self):
        if len(set(self.checkpoint_ids)) != self.attempts or len(self.checkpoint_ids) != self.attempts:
            raise ValueError("validation requires one unique case ID per attempt")
        if self.recoveries + self.harmful > self.attempts or self.catastrophic_failures > self.harmful:
            raise ValueError("inconsistent validation outcome counts")
        if not self.parameter_settings:
            raise ValueError("validation must record parameter settings")
        if datetime.fromisoformat(self.validated_at.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("validation timestamp must include a timezone")
        return self


class SkillMetadata(Model):
    version: Version
    verified: StrictBool = False
    code_hash: CodeHash | None = None
    validation: ValidationEvidence | None = None

    @model_validator(mode="after")
    def verified_requires_evidence(self):
        if self.verified and (self.code_hash is None or self.validation is None):
            raise ValueError("verified skills require a code hash and validation evidence")
        return self


class RegistryDocument(Model):
    skills: dict[Identifier, SkillMetadata]


class SkillRejected(ValueError):
    """A stable rejection reason safe to put in a structured decision log."""


class VerifiedSkillRegistry:
    def __init__(self, options: SkillOptions | None = None):
        self.options = options or SkillOptions()
        self._entries: dict[str, tuple[RecoverySkill, SkillMetadata]] = {}

    @classmethod
    def load(cls, path, implementations, options=None):
        document = RegistryDocument.model_validate(load_yaml_mapping(path))
        definitions = {}
        for skill in implementations:
            if skill.skill_id in definitions:
                raise ValueError("duplicate implementation ID")
            definitions[skill.skill_id] = skill
        registry = cls(options)
        for skill_id, metadata in document.skills.items():
            if skill_id not in definitions:
                raise ValueError(f"registry references an unknown implementation: {skill_id}")
            registry.register(definitions[skill_id], metadata)
        return registry

    def register(self, skill: RecoverySkill, metadata: SkillMetadata | None = None):
        if skill.skill_id in self._entries:
            raise ValueError(f"duplicate skill ID: {skill.skill_id}")
        # Validate the trusted definition as well as external metadata.
        RegistryDocument(skills={skill.skill_id: SkillMetadata(version=skill.version)})
        if type(skill.max_steps) is not int or skill.max_steps <= 0:
            raise ValueError("skill max_steps must be a positive integer")
        if not skill.supported_phases or not skill.supported_failures or not skill.interruption_conditions:
            raise ValueError("skill must declare phases, failures and interruption conditions")
        skill.parameter_schema()
        metadata = SkillMetadata.model_validate(
            metadata.model_dump(warnings=False) if metadata is not None
            else {"version": skill.version},
        )
        if metadata.version != skill.version:
            raise ValueError("skill definition/metadata version mismatch")
        if metadata.verified:
            self._require_eligible(skill, metadata)
        self._entries[skill.skill_id] = (skill, metadata.model_copy(deep=True))

    def _require_eligible(self, skill, metadata):
        if not metadata.verified:
            raise SkillRejected("unverified_skill")
        if skill.candidate_only:
            raise SkillRejected("candidate_only_skill")
        if not skill.runtime_observation_compatible:
            raise SkillRejected("privileged_skill")
        report = metadata.validation
        if report is None or metadata.code_hash is None:
            raise SkillRejected("missing_validation_evidence")
        if metadata.version != skill.version or report.skill_version != skill.version:
            raise SkillRejected("stale_skill_version")
        if metadata.code_hash != skill.code_hash() or report.code_hash != metadata.code_hash:
            raise SkillRejected("stale_skill_code")
        options = self.options
        if (report.attempts < options.minimum_validation_cases
                or report.recoveries / report.attempts < options.minimum_recovery_rate
                or report.catastrophic_failures / report.attempts > options.maximum_catastrophic_rate):
            raise SkillRejected("validation_thresholds_not_met")
        for parameters in report.parameter_settings:
            skill.validate_parameters(parameters)

    def get_verified(self, skill_id: str) -> RecoverySkill:
        if skill_id not in self._entries:
            raise SkillRejected("unknown_skill")
        skill, metadata = self._entries[skill_id]
        # Recheck identity at every eligibility boundary, not only at startup.
        self._require_eligible(skill, metadata)
        return skill

    def validate_parameters(self, skill_id: str, parameters: dict) -> dict:
        return self.get_verified(skill_id).validate_parameters(parameters)

    def metadata(self, skill_id: str) -> SkillMetadata:
        """Development inspection only; supervisor callers should use catalog()."""
        return self._entries[skill_id][1].model_copy(deep=True)

    def catalog(self) -> list[dict]:
        result = []
        for skill_id in sorted(self._entries):
            try:
                skill = self.get_verified(skill_id)
            except (ValueError, OSError):
                continue
            result.append(skill.descriptor())
        return result
