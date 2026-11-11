"""Single-episode skill dispatcher; no planner, scene reset, or auto-resume.

Use exclusively with one environment and one policy session. Local locking stops
overlapping calls on this executor; it is not a distributed service lease.
"""

import threading
import time
from collections import deque

from harness.schemas import StateSummary
from harness.predicates.registry import PredicateRegistry
from harness.perception.anchors import AnchorRegistry
from .base import ExecutionContext
from .registry import SkillRegistry
from .schemas import (EpisodeBudget, SkillCommand, SkillResult, SkillType,
                      is_recoverable_release_support_gap)


TCP_HISTORY_LIMIT = 256


class SkillExecutor:
    def __init__(self, environment, vla, *, log, budget=None, cancelled=lambda: False,
                 predicates=None, anchors=None, allow_development_oracle=False):
        self.environment, self.vla, self.log = environment, vla, log
        self.budget = EpisodeBudget.model_validate((budget or EpisodeBudget()).model_dump())
        self.cancelled = cancelled
        self.registry = SkillRegistry()
        if type(allow_development_oracle) is not bool:
            raise ValueError("oracle permission must be Boolean")
        self.predicates = predicates or PredicateRegistry()
        self.anchors = anchors or AnchorRegistry()
        self.allow_oracle = allow_development_oracle
        self.calls = self.executed_policy_actions = self.vla_chunks_requested = 0
        self.executed_physics_ticks = 0
        self.terminal = False
        self._refresh_history = True
        self.measured_tcp_history = {
            arm: deque(maxlen=TCP_HISTORY_LIMIT) for arm in ("left", "right")}
        self._lock = threading.Lock()
        initial = environment.observe()  # Requires an already-reset, confirmed scene.
        self.predicates.prime(initial, allow_oracle=self.allow_oracle)

    def execute(self, command, *, confirmed_observation=None,
                allow_recoverable_tracking_error=False):
        """Execute one command from a confirmed, unchanged simulator state.

        The orchestrator may pass the observation used for its planner turn. No
        simulator action occurs while the planner selects a skill, so rendering
        the same state again here is redundant and can block before dispatch.
        """
        # Validate before reserving budgets, requesting inference or writing drives.
        command = SkillCommand.model_validate(
            command.model_dump() if isinstance(command, SkillCommand) else command)
        skill = self.registry.get(command.action)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("skill_already_active")
        try:
            if self.terminal:
                raise RuntimeError("episode_is_terminal")
            if self.calls >= self.budget.max_skill_calls:
                self.terminal = True
                raise RuntimeError("skill_call_budget_exhausted")
            call_id = self.calls + 1
            source = ("planner_confirmed_observation" if confirmed_observation is not None
                      else "environment_refresh")
            preflight = time.perf_counter()

            def emit(event):
                self.log({"schema_version": 2, "call_id": call_id, **event})

            emit({"event_type": "skill_preflight_started", "command": command.model_dump(mode="json"),
                  "observation_source": source,
                  "candidate_observation": (confirmed_observation.sequence
                                            if confirmed_observation is not None else None)})
            try:
                before = (confirmed_observation if confirmed_observation is not None
                          else self.environment.observe())
            except Exception as exc:
                emit({"event_type": "skill_preflight_failed", "error_type": type(exc).__name__,
                      "elapsed_seconds": time.perf_counter() - preflight})
                raise
            emit({"event_type": "skill_preflight_completed", "observation": before.sequence,
                  "observation_source": source,
                  "elapsed_seconds": time.perf_counter() - preflight})
            self.calls += 1
            started = time.perf_counter()

            context = ExecutionContext(
                self.environment, self.vla, before, emit, self.cancelled,
                self.budget.max_policy_actions - self.executed_policy_actions,
                self.budget.max_vla_chunks - self.vla_chunks_requested,
                self._refresh_history,
                tick_limit=self.budget.max_analytic_ticks-self.executed_physics_ticks,
                anchors=self.anchors, allow_oracle=self.allow_oracle,
                measured_tcp_history=self.measured_tcp_history,
                feedback_boundary_predicates=self.predicates.feedback_boundary_predicates(),
            )
            context.predicate_check = lambda name, observation, arm: self.predicates.evaluate(
                name, observation, arm=arm, native_success=self.environment.native_success,
                budget_exhausted=context.chunks >= command.parameters.max_chunks
                    if command.action == SkillType.VLA_ACT else False,
                allow_oracle=self.allow_oracle,
            )
            context.predicate_attempt_start = lambda name, observation, arm: self.predicates.begin_attempt(
                name, observation, arm=arm, allow_oracle=self.allow_oracle)
            error_type = None
            try:
                emit({"event_type": "skill_requested", "command": command.model_dump(mode="json"),
                      "observation": before.sequence, "command_state": before.command_state})
                reason = skill.execute(context, command)
            except Exception as exc:
                reason, error_type = "execution_error", type(exc).__name__
            except BaseException:
                self.terminal = True
                raise
            finally:
                self.executed_policy_actions += context.executed
                self.vla_chunks_requested += context.chunks
                self.executed_physics_ticks += context.physics_ticks
            if context.history_reset:
                self._refresh_history = False
            # A gripper ramp does not invalidate the measured TCP approach
            # path. Preserve it across the open-before-retreat recovery step;
            # clear it after analytic arm motion, which could make the
            # recorded path stale for a later retry.
            if context.motion_started and command.action in (SkillType.MOVE_TO,
                                                              SkillType.ROTATE_WRIST):
                self._refresh_history = True  # Never reuse pre-staging history.
            recoverable_tracking_error = (
                allow_recoverable_tracking_error
                and command.action == SkillType.MOVE_TO
                and reason == "tracking_error"
                and context.tracking_error is not None
                and context.tracking_error.arm == command.arm
                and context.tracking_error.comparison == "controlled_arm"
                and not context.ambiguous
            )
            self.terminal = (reason in {"aborted", "cancelled", "execution_error", "task_success",
                                        "episode_budget_exhausted", "tracking_error", "safety_violation"}
                             and not recoverable_tracking_error
                             or self.calls >= self.budget.max_skill_calls)
            success = None if context.ambiguous else self.environment.native_success
            self.terminal |= success is True
            result = SkillResult(
                call_id=call_id, skill=command.action,
                completed=reason in {"completed", "predicate_satisfied", "failure_predicate_satisfied",
                                     "chunk_budget_exhausted", "policy_action_budget_exhausted",
                                     "task_success", "aborted"},
                stop_reason=reason, executed_policy_actions=context.executed,
                vla_chunks_requested=context.chunks, discarded_actions=context.discarded,
                executed_physics_ticks=context.physics_ticks,
                state_before=StateSummary(command_state=before.command_state),
                state_after=None if context.ambiguous else StateSummary(command_state=context.observation.command_state),
                observation_before=before.sequence,
                observation_after=None if context.ambiguous else context.observation.sequence,
                episode_terminal=self.terminal,
                task_success=success,
                native_success_seen=context.native_success_seen,
                post_success_actions=context.post_success_actions,
                objective_satisfied=(context.last_predicate.value if command.stop and context.last_predicate
                                     and context.last_predicate.name == command.stop
                                     else success if command.stop == "task_success" else None),
                controlled_arms=("both" if command.action == SkillType.VLA_ACT
                                 else "none" if command.action == SkillType.ABORT else command.arm),
                information_source="development_oracle" if context.oracle_used else "public",
                predicate=context.last_predicate,
                tracking_error=context.tracking_error,
                position_error_m=context.position_error, orientation_error_rad=context.orientation_error,
                policy_history_reset=context.history_reset,
                execution_uncertain=context.ambiguous,
                elapsed_seconds=time.perf_counter() - started, error_type=error_type,
            )
            if (reason == "release_support_lost"
                    and not is_recoverable_release_support_gap(result)):
                self.terminal = True
                result = result.model_copy(update={"episode_terminal": True})
            try:
                emit({"event_type": "skill_result", "result": result.model_dump(mode="json")})
            except BaseException:
                self.terminal = True
                raise
            return result
        finally:
            self._lock.release()
