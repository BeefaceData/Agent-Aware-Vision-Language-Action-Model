"""A VLA attempt that can run until evidence stops it; never resets an episode."""

import time

import numpy as np

from harness.environment.base import vector14
from .base import ExecutionContext
from .schemas import SkillCommand


class VLAAct:
    def execute(self, context: ExecutionContext, command: SkillCommand) -> str:
        env, parameters = context.environment, command.parameters
        if context.cancelled():
            return "cancelled"
        if env.native_success:
            context.native_success_seen = True
            return "task_success"
        # The episode ceilings remain a watchdog. A monitored grasp uses its
        # lift/failure predicates as normal stop conditions, not the per-call
        # horizon selected for ordinary planner feedback.
        if context.action_limit < 1:
            return "policy_action_budget_exhausted"
        if context.chunk_limit < 1:
            return "chunk_budget_exhausted"
        failure_monitor_active = False
        if parameters.failure_stop is not None:
            failure_monitor_active = context.begin_predicate_attempt(
                parameters.failure_stop, parameters.failure_arm)
            if not failure_monitor_active:
                context.log({"event_type": "failure_monitor_unavailable",
                             "predicate": parameters.failure_stop,
                             "arm": parameters.failure_arm,
                             "observation": context.observation.sequence})
            else:
                context.log({"event_type": "failure_monitor_started",
                             "predicate": parameters.failure_stop,
                             "arm": parameters.failure_arm,
                             "observation": context.observation.sequence})

        if parameters.failure_stop is not None and not failure_monitor_active:
            return "failure_monitor_unavailable"

        def predicate_stop():
            if command.stop not in (None, "task_success", "chunk_budget_exhausted"):
                # Public visual predicates are checked once when an ordinary
                # VLA segment returns to the planner. Polling them on every
                # action would cause repeated Opus/SAM requests and make the
                # planner's configured feedback horizon meaningless.
                if command.stop not in context.feedback_boundary_predicates:
                    value = context.check_predicate(command.stop)
                    if value is None: return "predicate_unavailable"
                    if value: return "predicate_satisfied"
            if failure_monitor_active:
                lifted_predicate = ("right_mug_lifted" if parameters.failure_arm == "right"
                                    else "mug_lifted")
                value = context.check_predicate(lifted_predicate, parameters.failure_arm)
                if value is None: return "predicate_unavailable"
                if value: return "predicate_satisfied"
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
            if context.measured_tcp_history is not None:
                for poses in context.measured_tcp_history.values():
                    poses.clear()
            context.log({"event_type": "policy_history_reset"})
        self._remember_tcp_poses(context, context.observation)
        # When the runtime has started a local grasp monitor, continue across
        # planner-feedback-sized chunks until lift/failure evidence arrives.
        # The episode-wide action/chunk ceilings remain emergency watchdogs.
        configured_chunk_limit = (context.chunk_limit if failure_monitor_active else
                                  min(parameters.max_chunks, context.chunk_limit))
        configured_action_limit = (context.action_limit if failure_monitor_active else
                                   min(parameters.max_policy_actions, context.action_limit))
        while context.chunks < configured_chunk_limit:
            if context.cancelled():
                return "cancelled"
            remaining = configured_action_limit - context.executed
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
                        context.native_success_seen = True
                        return "task_success"
                    stopped = predicate_stop()
                    if stopped:
                        return stopped
                    if context.executed >= configured_action_limit:
                        break
                    action_started = time.perf_counter()
                    context.log({"event_type": "action_requested", "action": action,
                                 "observation": context.observation.sequence,
                                 "action_index": context.executed + 1,
                                 "requested_at_monotonic_seconds": action_started})
                    # The command may execute even if its response is lost. Never retry it.
                    context.ambiguous = True
                    pending -= 1
                    context.observation = env.step(action)
                    self._remember_tcp_poses(context, context.observation)
                    step_ms = (time.perf_counter() - action_started) * 1000.0
                    context.executed += 1
                    context.ambiguous = False
                    context.log({"event_type": "action_confirmed", "action": action,
                                 "observation": context.observation.sequence,
                                 "command_state": context.observation.command_state,
                                 "action_index": context.executed,
                                 "environment_step_ms": step_ms})
                    if env.native_success:
                        context.native_success_seen = True
                        context.log({"event_type": "native_success_candidate",
                                     "observation": context.observation.sequence,
                                     "post_success_action_allowance": 0})
                        return "task_success"
                    stopped = predicate_stop()
                    if stopped:
                        return stopped
            finally:
                context.discarded += pending
                if pending:
                    context.log({"event_type": "chunk_discarded", "discarded_actions": pending})
        if command.stop in ("chunk_budget_exhausted", "task_success"):
            context.check_predicate(command.stop)
        if failure_monitor_active and (context.executed >= context.action_limit
                                      or context.chunks >= context.chunk_limit):
            return "grasp_monitor_watchdog_exhausted"
        if (context.executed >= context.action_limit
                or context.executed >= parameters.max_policy_actions):
            return "policy_action_budget_exhausted"
        if context.chunks >= context.chunk_limit:
            return "chunk_budget_exhausted"
        return "chunk_budget_exhausted"

    @staticmethod
    def _remember_tcp_poses(context, observation):
        if context.measured_tcp_history is None or observation.measured_tcp_poses is None:
            return
        for arm, pose in observation.measured_tcp_poses.items():
            history = context.measured_tcp_history.get(arm)
            if history is None:
                continue
            value = tuple(float(component) for component in pose)
            if not history or history[-1] != value:
                history.append(value)

class Abort:
    def execute(self, context, command):
        return "aborted"
