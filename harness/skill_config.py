"""Shared single-episode settings for the runner and exploration lifecycle."""

from typing import Annotated
from pydantic import Field, StrictInt
from harness.schemas import Model
from harness.planner.scripted import GraspRetrySettings
from harness.planner.anthropic import AnthropicSettings
from harness.skills.schemas import EpisodeBudget


class RunSettings(Model):
    max_turns: Annotated[StrictInt, Field(ge=1, le=1000)] = 20
    recipe: GraspRetrySettings = Field(default_factory=GraspRetrySettings)
    budget: EpisodeBudget = Field(default_factory=EpisodeBudget)
    anthropic: AnthropicSettings | None = None
