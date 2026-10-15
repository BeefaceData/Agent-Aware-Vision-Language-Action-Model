"""A bounded attempt, never an episode runner or an environment reset."""

import numpy as np

from harness.trace import vector14
from .base import ExecutionContext
from .schemas import SkillCommand


class VLAAct:
    def execute(self, context: ExecutionContext, command: SkillCommand) -> str:
        env, parameters = context.environment, command.parameters
        if context.cancelled():
            return "cancelled"
        if env.native_success:
            return "task_success"
        if context.action_limit < 1 or context.chunk_limit < 1:
            return "episode_budget_exhausted"
        def predicate_stop():
            if command.stop not in (None, "task_success", "chunk_budget_exhausted"):
                value = context.check_predicate(command.stop)
                if value is None: return "predicate_unavailable"
                if value: return "predicate_satisfied"
            return None
        stopped = predicate_stop()
        if stopped:
            return stopped
        if context.refresh_history:
            context.vla.reset_history()
            context.history_reset = True
            context.log({"event_type": "policy_history_reset"})
        while context.chunks < min(parameters.max_chunks, context.chunk_limit):
            if context.cancelled():
                return "cancelled"
            remaining = min(parameters.max_policy_actions, context.action_limit) - context.executed
            if remaining <= 0:
                break
            observation = env.observe()
            context.chunks += 1  # Failed requests consume the request budget too.
            context.log({"event_type": "chunk_requested", "chunk": context.chunks,
                         "observation": observation.sequence, "requested_actions": parameters.chunk_size})
            actions = np.asarray(context.vla.infer(observation, parameters.chunk_size))
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
                        return "task_success"
                    stopped = predicate_stop()
                    if stopped:
                        return stopped
                    if context.executed >= min(parameters.max_policy_actions, context.action_limit):
                        break
                    context.log({"event_type": "action_requested", "action": action,
                                 "observation": context.observation.sequence})
                    # The command may execute even if its response is lost. Never retry it.
                    context.ambiguous = True
                    pending -= 1
                    context.observation = env.step(action)
                    context.executed += 1
                    context.ambiguous = False
                    context.log({"event_type": "action_confirmed", "action": action,
                                 "observation": context.observation.sequence,
                                 "command_state": context.observation.command_state})
                    if env.native_success:
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
        if context.executed >= context.action_limit or context.chunks >= context.chunk_limit:
            return "episode_budget_exhausted"
        if context.executed >= parameters.max_policy_actions:
            return "policy_action_budget_exhausted"
        return "chunk_budget_exhausted"


class Abort:
    def execute(self, context, command):
        return "aborted"
