"""Execution state shared by fixed, authored skill implementations."""

from dataclasses import dataclass
from typing import Callable, Protocol

from harness.environment.base import Environment, Observation
from harness.vla.base import VLAClient
from .schemas import SkillCommand


@dataclass
class ExecutionContext:
    environment: Environment
    vla: VLAClient
    observation: Observation
    log: Callable[[dict], None]
    cancelled: Callable[[], bool]
    action_limit: int
    chunk_limit: int
    refresh_history: bool
    executed: int = 0
    chunks: int = 0
    discarded: int = 0
    history_reset: bool = False
    ambiguous: bool = False
    tick_limit: int = 0
    physics_ticks: int = 0
    motion_started: bool = False
    predicate_check: Callable | None = None
    last_predicate: object = None
    anchors: object = None
    allow_oracle: bool = False
    oracle_used: bool = False
    position_error: float | None = None
    orientation_error: float | None = None

    def check_predicate(self, name, arm="auto"):
        result = self.predicate_check(name, self.observation, arm)
        self.last_predicate = result
        self.oracle_used |= result.source == "development_oracle"
        self.log({"event_type": "predicate_result", "predicate": result.model_dump(mode="json")})
        return result.value


class Skill(Protocol):
    def execute(self, context: ExecutionContext, command: SkillCommand) -> str: ...
