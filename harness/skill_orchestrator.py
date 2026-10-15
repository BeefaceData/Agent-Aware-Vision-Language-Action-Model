"""One-episode skill turns; additive to the legacy supervision runner.

No reset, model access, object geometry, or memory writes. A trusted caller may
refresh symbolic anchors before execution, outside the planner interface.
"""

import time

from harness.planner.base import PlannerContext
from harness.skills.schemas import SkillCommand


class SkillOrchestrator:
    def __init__(self, executor, planner, *, log, save_observation, episode_id,
                 seed, max_turns=20, prepare=None, completion_predicate="mug_lifted", memory=None,
                 invariant_check=lambda: None):
        if type(max_turns) is not int or not 1 <= max_turns <= 1000:
            raise ValueError("invalid_planner_turn_budget")
        self.executor, self.planner = executor, planner
        self.log, self.save_observation, self.prepare = log, save_observation, prepare
        self.episode_id, self.seed, self.max_turns = episode_id, seed, max_turns
        if completion_predicate not in ("mug_lifted", "task_success"):
            raise ValueError("unsupported_completion_predicate")
        self.completion_predicate = completion_predicate
        self.memory = memory
        self.invariant_check = invariant_check
        self._started = False

    def _predicates(self, observation):
        executor = self.executor
        # Acquisition accepts either arm because Neotix drives both. Opening
        # permission is checked by the skill again for its selected arm.
        return tuple(executor.predicates.evaluate(name, observation,
            native_success=executor.environment.native_success,
            allow_oracle=executor.allow_oracle) for name in
            ("mug_grasp_confirmed", "mug_lifted", "release_ready"))

    def run(self):
        if self._started:
            raise RuntimeError("orchestrator_is_single_episode")
        self._started = True
        executor = self.executor
        started = time.perf_counter()
        previous = None
        retries = turns = 0
        phase, outcome, error_type = "APPROACH_MUG", "planner_turn_budget_exhausted", None
        success = None
        milestone = False
        try:
            for turn in range(1, self.max_turns + 1):
                self.invariant_check()
                observation = executor.environment.observe()
                success = executor.environment.native_success
                predicates = self._predicates(observation)
                if success:
                    phase, outcome = "COMPLETE", "task_success"
                    break
                if self.completion_predicate == "mug_lifted" and any(
                        p.name == "mug_lifted" and p.value is True for p in predicates):
                    milestone = True
                    phase, outcome = "TRANSFER", "acquisition_confirmed"
                    break
                before_ref = self.save_observation(observation)
                retrieved = self.memory.retrieve(task="hanging_mug", phase=phase) if self.memory else None
                tick = time.perf_counter()
                selection = self.planner.select(PlannerContext(turn, observation.copy(), predicates,
                    previous.model_copy(deep=True) if previous else None,
                    remaining_budget={"planner_turns": self.max_turns-turn+1,
                        "skill_calls": executor.budget.max_skill_calls-executor.calls,
                        "policy_actions": executor.budget.max_policy_actions-executor.executed_policy_actions,
                        "vla_chunks": executor.budget.max_vla_chunks-executor.vla_chunks_requested,
                        "analytic_ticks": executor.budget.max_analytic_ticks-executor.executed_physics_ticks},
                    memory=retrieved))
                command = SkillCommand.model_validate(selection.command.model_dump())
                latency = time.perf_counter() - tick
                phase, retries = selection.phase, selection.retry_count
                base = dict(schema_version=2, episode=self.episode_id, seed=self.seed,
                            planner_turn=turn, phase=phase, retry_count=retries)
                # Durable intent precedes anchor preparation and all motion.
                self.log({**base, "event_type": "planner_selection", "observation_before": before_ref,
                    "predicates_before": [p.model_dump(mode="json") for p in predicates],
                    "memory_retrieved": retrieved.references() if retrieved else [],
                    "memory_snapshot_sha256": self.memory.sha256 if self.memory else None,
                    "command": command.model_dump(mode="json"),
                    "planner_latency_seconds": latency, "planner_metadata": selection.metadata})
                if self.prepare:
                    self.prepare(command, observation)
                self.invariant_check()  # Recheck after the potentially slow provider/planner call.
                previous = executor.execute(command)
                turns = turn
                success = previous.task_success
                after_ref, post = None, ()
                if not previous.execution_uncertain:
                    after = executor.environment.observe()
                    after_ref, post = self.save_observation(after), self._predicates(after)
                milestone = (previous.completed and not previous.execution_uncertain
                             and any(p.name == "mug_lifted" and p.value is True for p in post))
                failure = (command.reason if command.action == "abort" else
                           previous.stop_reason if not previous.completed else
                           "acquisition_unconfirmed" if command.action == "vla_act" and not milestone else None)
                self.log({**base, "event_type": "planner_turn_result", "observation_after": after_ref,
                    "post_condition": [p.model_dump(mode="json") for p in post],
                    "result": previous.model_dump(mode="json"), "failure_type": failure,
                    "task_success": success, "milestone_reached": milestone})
                if previous.episode_terminal or not previous.completed:
                    if success:
                        outcome = "task_success"
                    elif previous.completed and executor.calls >= executor.budget.max_skill_calls:
                        outcome = "skill_call_budget_exhausted"
                    else:
                        outcome = failure or previous.stop_reason
                    phase = "COMPLETE" if success else "FAILED"
                    break
                if milestone and self.completion_predicate == "mug_lifted":
                    phase, outcome = "TRANSFER", "acquisition_confirmed"
                    break
            else:
                phase = "FAILED"
        except Exception as exc:
            # Do not include provider exception text, credentials or raw observations.
            phase, outcome, error_type = "FAILED", "controller_error", type(exc).__name__
            success = None
            milestone = False
        finally:
            # A completed subtask or turn limit cannot accidentally resume this executor.
            executor.terminal = True
        summary = dict(schema_version=2, episode=self.episode_id, seed=self.seed,
            phase=phase, outcome=outcome, task_success=success, milestone_reached=milestone,
            planner_turns=turns, retries=retries, skill_calls=executor.calls,
            vla_chunks=executor.vla_chunks_requested, policy_actions=executor.executed_policy_actions,
            analytic_ticks=executor.executed_physics_ticks,
            development_oracle=executor.allow_oracle, elapsed_seconds=time.perf_counter()-started,
            completion_predicate=self.completion_predicate,
            memory_snapshot_sha256=self.memory.sha256 if self.memory else None,
            error_type=error_type)
        self.log({"event_type": "episode_summary", **summary})
        return summary
