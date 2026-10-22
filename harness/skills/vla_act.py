"""A bounded attempt, never an episode runner or an environment reset."""

import time

import numpy as np

from harness.trace import vector14
from .base import ExecutionContext
from .schemas import SkillCommand


# Once RoboTwin's native checker first reports success during the full-task
# VLA hang call, let the frozen policy execute a short, fixed follow-through
# for release/retreat. This is deliberately not planner-selectable.
NATIVE_SUCCESS_FOLLOW_THROUGH_ACTIONS = 16


class VLAAct:
    def execute(self, context: ExecutionContext, command: SkillCommand) -> str:
        env, parameters = context.environment, command.parameters
        if context.cancelled():
            return "cancelled"
        if env.native_success:
            return "task_success"
        # These limits bound only learned-policy work. Analytic skills remain
        # available so a planner can recover after VLA budget exhaustion.
        if context.action_limit < 1:
            return "policy_action_budget_exhausted"
        if context.chunk_limit < 1:
            return "chunk_budget_exhausted"
        # A task-success check may turn true even when Opus selected an ordinary
        # feedback-boundary stop (for example, chunk_budget_exhausted). Native
        # success itself, not the requested chunk stop, starts the fixed tail.
        success_follow_through = True
        native_success_seen = False
        if parameters.failure_stop is not None:
            if not context.begin_predicate_attempt(parameters.failure_stop, parameters.failure_arm):
                return "predicate_unavailable"
            context.log({"event_type": "failure_monitor_started",
                         "predicate": parameters.failure_stop,
                         "arm": parameters.failure_arm,
                         "observation": context.observation.sequence})
        def predicate_stop():
            if command.stop not in (None, "task_success", "chunk_budget_exhausted"):
                value = context.check_predicate(command.stop)
                if value is None: return "predicate_unavailable"
                if value: return "predicate_satisfied"
            if parameters.failure_stop is not None:
                value = context.check_predicate(parameters.failure_stop, parameters.failure_arm)
                if value is None: return "predicate_unavailable"
                if value: return "failure_predicate_satisfied"
            return None
        stopped = predicate_stop()
        if stopped:
            return stopped
        if context.refresh_history:
            context.vla.reset_history()
            context.history_reset = True
            context.log({"event_type": "policy_history_reset"})
        configured_chunk_limit = min(parameters.max_chunks, context.chunk_limit)
        success_chunk_limit = None
        while context.chunks < (success_chunk_limit or configured_chunk_limit):
            if context.cancelled():
                return "cancelled"
            remaining = min(parameters.max_policy_actions, context.action_limit) - context.executed
            if remaining <= 0:
                break
            observation = env.observe()
            context.chunks += 1  # Failed requests consume the request budget too.
            context.log({"event_type": "chunk_requested", "chunk": context.chunks,
                         "observation": observation.sequence, "requested_actions": parameters.chunk_size})
            inference_started = time.perf_counter()
            actions = np.asarray(context.vla.infer(observation, parameters.chunk_size))
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            context.log({"event_type": "chunk_received", "chunk": context.chunks,
                         "observation": observation.sequence, "inference_ms": inference_ms,
                         "returned_actions": len(actions) if actions.ndim else 0})
            if (actions.ndim != 2 or actions.shape[1:] != (14,)
                    or not 1 <= len(actions) <= parameters.chunk_size
                    or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all()):
                raise ValueError("invalid_vla_chunk")
            # Validate the complete chunk before any simulator mutation.
            actions = [vector14(action) for action in actions]
            pending = len(actions)
            try:
                for action in actions:
                    if context.cancelled():
                        return "cancelled"
                    if env.native_success:
                        if not native_success_seen:
                            native_success_seen = True
                            context.native_success_seen = True
                            context.log({"event_type": "native_success_candidate",
                                         "observation": context.observation.sequence,
                                         "post_success_action_allowance": NATIVE_SUCCESS_FOLLOW_THROUGH_ACTIONS})
                        if not success_follow_through:
                            return "task_success"
                        if context.post_success_actions >= NATIVE_SUCCESS_FOLLOW_THROUGH_ACTIONS:
                            context.log({"event_type": "native_success_follow_through_complete",
                                         "post_success_actions": context.post_success_actions})
                            return "task_success"
                    stopped = predicate_stop()
                    if stopped:
                        return stopped
                    if context.executed >= min(parameters.max_policy_actions, context.action_limit):
                        break
                    if parameters.lock_arm is not None:
                        # Hold the inactive arm at its latest confirmed target.
                        # Refresh after each acknowledged action; stale chunk
                        # tails are still discarded by the stop/budget checks.
                        start = 0 if parameters.lock_arm == "left" else 7
                        stop = start + 7
                        held = np.asarray(context.observation.command_state[start:stop], dtype=float)
                        if held.shape != (7,) or not np.isfinite(held).all():
                            raise ValueError("invalid_locked_arm_state")
                        action = np.concatenate((action[:start], held, action[stop:])).tolist()
                    action_started = time.perf_counter()
                    context.log({"event_type": "action_requested", "action": action,
                                 "observation": context.observation.sequence,
                                 "action_index": context.executed + 1,
                                 "requested_at_monotonic_seconds": action_started})
                    # The command may execute even if its response is lost. Never retry it.
                    context.ambiguous = True
                    pending -= 1
                    context.observation = (env.step_after_success(action) if native_success_seen
                                           else env.step(action))
                    step_ms = (time.perf_counter() - action_started) * 1000.0
                    context.executed += 1
                    context.ambiguous = False
                    context.log({"event_type": "action_confirmed", "action": action,
                                 "observation": context.observation.sequence,
                                 "command_state": context.observation.command_state,
                                 "action_index": context.executed,
                                 "environment_step_ms": step_ms})
                    if native_success_seen:
                        context.post_success_actions += 1
                    overlap_check = getattr(env, "arm_overlap_detected", None)
                    if callable(overlap_check) and overlap_check():
                        context.log({"event_type": "inter_arm_contact_guard",
                                     "observation": context.observation.sequence,
                                     "action_index": context.executed,
                                     "stopped": True})
                        return "arm_overlap_detected"
                    if env.native_success and not native_success_seen:
                        native_success_seen = True
                        context.native_success_seen = True
                        context.log({"event_type": "native_success_candidate",
                                     "observation": context.observation.sequence,
                                     "post_success_action_allowance": NATIVE_SUCCESS_FOLLOW_THROUGH_ACTIONS})
                    if (native_success_seen and success_follow_through
                            and context.post_success_actions >= NATIVE_SUCCESS_FOLLOW_THROUGH_ACTIONS):
                        context.log({"event_type": "native_success_follow_through_complete",
                                     "post_success_actions": context.post_success_actions})
                        return "task_success"
                    stopped = predicate_stop()
                    if stopped:
                        return stopped
            finally:
                context.discarded += pending
                if pending:
                    context.log({"event_type": "chunk_discarded", "discarded_actions": pending})
            if native_success_seen and success_follow_through:
                # The command's ordinary chunk allowance bounds planner work;
                # a checker-positive hang gets only enough additional requests
                # to execute the fixed follow-through allowance. Recompute
                # against returned chunk length, since Modal may return fewer
                # actions than requested.
                remaining_follow_through = (NATIVE_SUCCESS_FOLLOW_THROUGH_ACTIONS
                                            - context.post_success_actions)
                if remaining_follow_through > 0:
                    needed_chunks = ((remaining_follow_through + parameters.chunk_size - 1)
                                     // parameters.chunk_size)
                    proposed_limit = max(configured_chunk_limit, context.chunks + needed_chunks)
                    success_chunk_limit = min(context.chunk_limit,
                        max(success_chunk_limit or 0, proposed_limit))
        if command.stop in ("chunk_budget_exhausted", "task_success"):
            context.check_predicate(command.stop)
        if (context.executed >= context.action_limit
                or context.executed >= parameters.max_policy_actions):
            return "policy_action_budget_exhausted"
        if context.chunks >= context.chunk_limit:
            return "chunk_budget_exhausted"
        return "chunk_budget_exhausted"


class Abort:
    def execute(self, context, command):
        return "aborted"
