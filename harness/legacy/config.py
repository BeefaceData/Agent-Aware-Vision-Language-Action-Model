"""Strict configuration. Paths are relative to the YAML file, not simulator cwd."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, StrictBool, StrictInt

from harness.schemas import Confidence, Model, NonnegativeInt


PositiveInt = Annotated[StrictInt, Field(gt=0)]


class HarnessOptions(Model):
    enabled: StrictBool = False


class SupervisorOptions(Model):
    mode: Literal["pass_through"] = "pass_through"
    minimum_confidence: Confidence = 0.70


class SafetyOptions(Model):
    max_interventions_per_episode: NonnegativeInt = 5
    max_consecutive_skill_failures: PositiveInt = 2
    max_skill_steps: PositiveInt = 150
    abort_on_invalid_skill_request: StrictBool = False


class SkillOptions(Model):
    registry: Path | None = None
    minimum_validation_cases: PositiveInt = 10
    minimum_recovery_rate: Confidence = 0.70
    maximum_catastrophic_rate: Confidence = 0.0


class TraceOptions(Model):
    state_history: PositiveInt = 40
    action_history: PositiveInt = 40
    keyframes: PositiveInt = 6
    keyframe_interval: PositiveInt = 24


class LoggingOptions(Model):
    assessment_dir: Path = Path("../data/assessments")
    trace_dir: Path = Path("../data/traces")
    evaluation_dir: Path = Path("../data/evaluation")


class HarnessConfig(Model):
    task: Literal["hanging_mug"] = "hanging_mug"
    harness: HarnessOptions = Field(default_factory=HarnessOptions)
    supervisor: SupervisorOptions = Field(default_factory=SupervisorOptions)
    trace: TraceOptions = Field(default_factory=TraceOptions)
    logging: LoggingOptions = Field(default_factory=LoggingOptions)
    safety: SafetyOptions = Field(default_factory=SafetyOptions)
    skills: SkillOptions = Field(default_factory=SkillOptions)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject ambiguous YAML instead of silently overwriting a setting."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("configuration keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping,
)


def load_yaml_mapping(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        try:
            raw = yaml.load(stream, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError("invalid configuration YAML") from exc
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    return raw


def load_config(path: str | Path) -> HarnessConfig:
    path = Path(path).resolve()
    config = HarnessConfig.model_validate(load_yaml_mapping(path))
    resolved = {}
    for name, value in config.logging:
        resolved[name] = (path.parent / value).resolve()
    skills = config.skills
    if skills.registry is not None:
        skills = skills.model_copy(update={"registry": (path.parent / skills.registry).resolve()})
    return config.model_copy(update={"logging": LoggingOptions(**resolved), "skills": skills})
