"""Host-owned current-scene gate for corrections advised by retained experience."""

from copy import deepcopy
from dataclasses import replace
from math import isfinite

from adaptation_memory import AdaptationMemory
from decision_memory import DecisionMemory
from fixed_memory import FixedMemory
from episode_harness import (ActionResolution, check_observation_freshness,
                             supervisor_observation)
from reopen_retreat import ReopenRetreatExecutor
from single_action_adjustment import SingleActionAdjustment


class MemoryCorrectionExecutor:
    """Compose retrieval with existing executors, never replay historical commands.

    request_for(proposal, memory) supplies a new request using detached advice.
    scene_check(proposal, resolution) must affirm the complete proposed correction
    against current deployable geometry. It receives no historical evidence or
    evaluator state. These are trusted host callbacks, not model-authored code.
    Recovery additionally requires its existing proposal-bound scene assessment.
    Use at run_episode's action_selector seam; episode budgets/expiry still apply.
    """

    def __init__(self, memory, executor, request_for, scene_check, *, required_sources):
        if type(memory) not in (DecisionMemory, FixedMemory, AdaptationMemory):
            raise ValueError('explicit cross-episode memory required')
        if type(executor) not in (SingleActionAdjustment, ReopenRetreatExecutor):
            raise ValueError('bounded correction executor required')
        if not callable(request_for) or not callable(scene_check):
            raise ValueError('host request and current-scene callbacks required')
        if (type(required_sources) is not dict or not required_sources or
                set(required_sources) - {'main', 'wrist', 'robot_state'} or
                any(type(age) not in (int, float) or not isfinite(age) or age < 0
                    for age in required_sources.values())):
            raise ValueError('current sensor age limits required')
        self._memory, self._executor = memory, executor
        self._request_for, self._scene_check = request_for, scene_check
        self._required_sources = deepcopy(required_sources)

    def resolve(self, proposal, *, now_monotonic, scene=None, window=None):
        """Reject unknown geometry, stale sensing and incompatible old requests.

        No rebinding of historical identity or frame transformation is performed.
        A geometrically rejected request may consume the executor's single-use
        identity; retry only with a new observation/proposal, never a cached permit.
        Retrieval corruption propagates before requesting or executing a command.
        """
        current = replace(deepcopy(proposal), observation=supervisor_observation(
            proposal.observation))
        memory = self._memory.prepare(current)
        try:
            freshness = check_observation_freshness(
                current.observation, now_monotonic, self._required_sources)
            if any(item.status != 'fresh' or
                   item.observation_sequence != current.observation.sequence for item in freshness):
                raise ValueError('memory correction requires fresh current scene evidence')
            request = self._request_for(deepcopy(current), deepcopy(memory))
            if request is None:
                result = ActionResolution('pass')
            elif type(self._executor) is SingleActionAdjustment:
                result = self._executor.resolve(current, request)
            else:
                result = self._executor.resolve(current, request, scene=scene,
                    now_monotonic=now_monotonic, window=window)
            if result.kind in ('override', 'recovery') and self._scene_check(
                    deepcopy(current), deepcopy(result)) is not True:
                raise ValueError('memory correction geometry is unknown or incompatible')
        except ValueError as exc:
            result = ActionResolution('reject', reason=str(exc))
        return replace(result, memory_context=deepcopy(memory))
