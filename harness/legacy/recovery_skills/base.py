"""Skill contract and bounded parameter validation, independent of RoboTwin."""

from abc import ABC, abstractmethod
import hashlib
import inspect
import json
import math
from pathlib import Path

from harness.schemas import FailureType, Model, SkillResult, TaskPhase


class RecoverySkill(ABC):
    skill_id: str
    version: int
    description: str
    supported_phases: frozenset[TaskPhase]
    supported_failures: frozenset[FailureType]
    parameter_model: type[Model]
    max_steps: int
    interruption_conditions: tuple[str, ...]
    returns_to_policy: bool = True
    runtime_observation_compatible: bool = True
    candidate_only: bool = False
    # Trusted author must list additional implementation dependencies for hashing.
    dependency_files: tuple[Path, ...] = ()

    def parameter_schema(self) -> dict:
        schema = self.parameter_model.model_json_schema()
        if schema.get("additionalProperties") is not False:
            raise ValueError("skill parameter models must reject unknown fields")
        for field in schema.get("properties", {}).values():
            kind = field.get("type")
            if kind in ("number", "integer"):
                lower = field.get("minimum", field.get("exclusiveMinimum"))
                upper = field.get("maximum", field.get("exclusiveMaximum"))
                if (lower is None or upper is None or not math.isfinite(lower)
                        or not math.isfinite(upper) or lower > upper):
                    raise ValueError("numeric skill parameters require lower and upper bounds")
            elif kind == "string":
                if not field.get("enum") and "const" not in field:
                    raise ValueError("string skill parameters require fixed choices")
            elif kind != "boolean":
                raise ValueError("skill parameters must be bounded scalars, never arrays or objects")
        return schema

    def validate_parameters(self, parameters: dict) -> dict:
        # Reject out-of-range requests; never silently clamp model decisions.
        return self.parameter_model.model_validate(parameters, strict=True).model_dump(mode="json")

    def descriptor(self) -> dict:
        return {
            "skill_id": self.skill_id, "version": self.version,
            "description": self.description,
            "supported_phases": sorted(x.value for x in self.supported_phases),
            "supported_failures": sorted(x.value for x in self.supported_failures),
            "parameters": self.parameter_schema(), "max_steps": self.max_steps,
            "interruption_conditions": list(self.interruption_conditions),
            "returns_to_policy": self.returns_to_policy,
            "runtime_observation_compatible": self.runtime_observation_compatible,
            "candidate_only": self.candidate_only,
        }

    def code_hash(self) -> str:
        digest = hashlib.sha256(json.dumps(self.descriptor(), sort_keys=True, allow_nan=False).encode())
        files = [Path(__file__), Path(inspect.getfile(Model)),
                 Path(inspect.getfile(type(self))), *self.dependency_files]
        for file in dict.fromkeys(Path(p).resolve() for p in files):
            payload = file.read_bytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    @abstractmethod
    def execute(self, context, parameters: dict) -> SkillResult:
        """Execute deterministic code through a bounded adapter, never LLM actions.

        Context/physics-budget instrumentation is implemented with real skills in
        later phases. No Phase 2 runtime entry point calls this method.
        """
