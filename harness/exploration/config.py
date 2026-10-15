"""Explicit exploration variations and independent campaign bounds."""

from typing import Annotated
from pydantic import Field, StrictBool, StrictInt, model_validator
from harness.schemas import Model
from harness.memory.base import Name
from harness.skill_config import RunSettings


class Variant(Model):
    name: Name
    settings: RunSettings = Field(default_factory=RunSettings)


class ExploreSettings(Model):
    seeds: Annotated[tuple[Annotated[StrictInt, Field(ge=0)], ...], Field(min_length=1, max_length=20)] = (3,)
    repeats: Annotated[StrictInt, Field(ge=1, le=5)] = 1
    variants: Annotated[tuple[Variant, ...], Field(min_length=1, max_length=8)]
    max_attempts: Annotated[StrictInt, Field(ge=1, le=40)] = 8
    attempt_timeout_seconds: Annotated[StrictInt, Field(ge=1, le=3600)] = 900
    campaign_timeout_seconds: Annotated[StrictInt, Field(ge=1, le=14400)] = 3600
    stop_on_acquisition: StrictBool = False

    @model_validator(mode="after")
    def unique_variants(self):
        if len({v.name for v in self.variants}) != len(self.variants):
            raise ValueError("duplicate_exploration_variant")
        return self

    def schedule(self):
        for seed in self.seeds:
            for repeat in range(self.repeats):
                for variant in self.variants:
                    yield seed, repeat, variant
