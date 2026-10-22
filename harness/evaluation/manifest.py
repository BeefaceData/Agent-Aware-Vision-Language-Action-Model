"""Freeze declared inputs and reject seed contamination or local runtime drift."""

from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator
from harness.schemas import Model
from harness.skill_config import RunSettings
from harness.memory.snapshot import MemoryCatalog, MemorySnapshot, Digest, canonical
from harness.skills.registry import SkillRegistry


class EvaluationSettings(Model):
    seeds: Annotated[tuple[Annotated[StrictInt, Field(ge=0)], ...], Field(min_length=1, max_length=100)]
    development_seeds: tuple[Annotated[StrictInt, Field(ge=0)], ...]
    planner: Literal["scripted", "anthropic"] = "scripted"
    condition: Literal["oracle_assisted_acquisition"] = "oracle_assisted_acquisition"
    episode: RunSettings = Field(default_factory=RunSettings)
    episode_timeout_seconds: Annotated[StrictInt, Field(ge=1, le=3600)] = 900
    campaign_timeout_seconds: Annotated[StrictInt, Field(ge=1, le=86400)] = 3600

    @model_validator(mode="after")
    def fixed_condition(self):
        if self.episode.handoff is not None:
            raise ValueError("handoff_not_part_of_acquisition_evaluation")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate_evaluation_seed")
        if set(self.seeds) & set(self.development_seeds):
            raise ValueError("evaluation_seed_was_used_in_development")
        if self.planner == "anthropic" and self.episode.anthropic is None:
            raise ValueError("evaluation_requires_explicit_anthropic_settings_and_model")
        return self


def runtime_identity(root):
    root = Path(root).resolve()
    paths = set()
    # Scope deliberately excludes data, secrets, docs/tests and native generated
    # *_tmp configs. External packages/assets/remote weights are not byte-attested.
    patterns = ("harness/**/*.py", "harness/prompts/*.md", "development/*.py",
        "run_skills.py", "run_evaluation.py", "run_inference_chunked.py", "modal_policy*.py",
        "requirements-*.txt", "RoboTwin/envs/*.py", "RoboTwin/envs/robot/**/*.py",
        "RoboTwin/envs/utils/**/*.py", "RoboTwin/envs/camera/**/*.py",
        "RoboTwin/env_cfg/**/*.yml", "RoboTwin/assets/embodiments/aloha-agilex/*.yml")
    for pattern in patterns:
        paths.update(p for p in root.glob(pattern) if p.is_file() and not p.name.endswith("_tmp.yml"))
    files = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    packages = {}
    for name in ("numpy", "pydantic", "PyYAML", "Pillow", "anthropic", "modal", "torch", "sapien"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return dict(files=files, packages=packages,
                skills=sorted(action.value for action in SkillRegistry().available()))


class FrozenEvaluation(Model):
    schema_version: Literal[2] = 2
    settings: EvaluationSettings
    memory: MemoryCatalog
    excluded_seeds: tuple[Annotated[StrictInt, Field(ge=0)], ...]
    exploration_evidence: dict[str, Digest]
    files: dict[str, Digest]
    packages: dict[str, str]
    skills: tuple[str, ...]
    remote_policy_weights_verified: Literal[False] = False

    @model_validator(mode="after")
    def split(self):
        excluded = set(self.excluded_seeds)
        memory_seeds = {e.seed for r in (*self.memory.task_memory, *self.memory.global_memory) for e in r.evidence}
        if not (memory_seeds | set(self.settings.development_seeds)) <= excluded:
            raise ValueError("incomplete_evaluation_exclusion_set")
        if excluded & set(self.settings.seeds):
            raise ValueError("held_out_seed_contamination")
        return self

    @property
    def digest(self):
        return sha256(canonical(self.model_dump(mode="json"))).hexdigest()

    def verify(self, root):
        identity = runtime_identity(root)
        if identity != dict(files=self.files, packages=self.packages, skills=list(self.skills)):
            raise ValueError("frozen_runtime_changed")

    def save(self, path):
        with Path(path).open("xb") as stream:
            stream.write(canonical(dict(sha256=self.digest, manifest=self.model_dump(mode="json"))) + b"\n")

    @classmethod
    def load(cls, path, *, expected_sha256=None):
        with Path(path).open("rb") as stream:
            data = stream.read(4_000_001)
        if len(data) > 4_000_000:
            raise ValueError("evaluation_manifest_too_large")
        envelope = json.loads(data)
        if set(envelope) != {"sha256", "manifest"}:
            raise ValueError("invalid_evaluation_envelope")
        result = cls.model_validate(envelope["manifest"])
        if result.digest != envelope["sha256"] or (expected_sha256 is not None and result.digest != expected_sha256):
            raise ValueError("evaluation_manifest_hash_mismatch")
        return result


def freeze(settings, *, root, memory=None, exploration_summaries=()):
    settings = EvaluationSettings.model_validate(settings.model_dump())
    # Load a validated canonical copy, never keep mutable caller models.
    catalog = memory.catalog() if memory else MemoryCatalog()
    excluded = set(settings.development_seeds)
    excluded.update(e.seed for r in (*catalog.task_memory, *catalog.global_memory) for e in r.evidence)
    evidence = {}
    for index, path in enumerate(exploration_summaries):
        with Path(path).open("rb") as stream:
            raw = stream.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("exploration_summary_too_large")
        summary = json.loads(raw)
        if summary.get("mode") != "explore" or not isinstance(summary.get("attempts"), list):
            raise ValueError("expected_exploration_summary")
        for attempt in summary["attempts"]:
            seed = attempt.get("seed")
            if type(seed) is not int or seed < 0:
                raise ValueError("invalid_explored_seed")
            excluded.add(seed)
        evidence[f"exploration_{index:04d}"] = sha256(raw).hexdigest()
    return FrozenEvaluation(settings=settings, memory=catalog, excluded_seeds=tuple(sorted(excluded)),
                            exploration_evidence=evidence, **runtime_identity(root))
