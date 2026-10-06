"""Host-owned admission checks for executing a baseline after supervision fails."""

from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from types import MappingProxyType

from episode_harness import check_observation_freshness, supervisor_observation


class BaselineFallback:
    """Require measured required inputs and affirmative adapter checks.

    controller_healthy() checks the current controller/transport state.
    proposal_valid(proposal) checks native shape, finite values, bounds and
    adapter-specific validity for this exact proposal. Both must return True
    (not a truthy value); exceptions mean unknown and refuse dispatch. These
    are trusted host callbacks, never supervisor claims. The harness supplies
    its clock and checks sensor age after the callbacks have completed.
    """

    def __init__(self, max_age_seconds, controller_healthy, proposal_valid):
        limits = dict(max_age_seconds)
        if (not limits or set(limits) - {'main', 'wrist', 'robot_state'} or
                any(type(v) not in (int, float) or not isfinite(v) or v < 0
                    for v in limits.values())):
            raise ValueError('fallback requires declared sensor age limits')
        if not callable(controller_healthy) or not callable(proposal_valid):
            raise ValueError('fallback requires controller and native proposal checks')
        self._limits = MappingProxyType(limits)
        self._controller = controller_healthy
        self._proposal = proposal_valid

    def assess(self, proposal, cause, clock):
        def check(callback, *args):
            try:
                value = callback(*args)
                return value if type(value) is bool else None
            except Exception:
                return None

        healthy = check(self._controller)
        valid = check(self._proposal, deepcopy(proposal))
        now = clock()
        reports = [asdict(item) for item in check_observation_freshness(
            supervisor_observation(proposal.observation), now, self._limits)]
        return fallback_evidence(proposal, cause, now, reports, healthy, valid)


def fallback_evidence(proposal, cause, now, reports=(), healthy=None, valid=None):
    failures = []
    if not reports:
        failures.append('required inputs undeclared')
    elif any(item['status'] != 'fresh' or
             item['observation_sequence'] != proposal.observation.sequence for item in reports):
        failures.append('required inputs not fresh for current proposal')
    if healthy is not True:
        failures.append('controller health not confirmed')
    if valid is not True:
        failures.append('current proposal validity not confirmed')
    return dict(episode_id=proposal.observation.episode_id,
                observation_sequence=proposal.observation.sequence,
                proposal_id=proposal.proposal_id, cause=cause, checked_at=now,
                inputs=list(reports), controller_healthy=healthy, proposal_valid=valid,
                selected='refuse' if failures else 'baseline',
                reason='; '.join(failures) if failures else 'all fallback prerequisites satisfied')


def validate_fallback(evidence, proposal):
    """Recompute the recorded sensor/decision checks without a live controller."""
    if type(evidence) is not dict:
        raise ValueError('invalid fallback evidence')
    now = evidence['checked_at']
    if (type(now) not in (int, float) or not isfinite(now) or
            now < proposal.observation.captured_monotonic or
            type(evidence['observation_sequence']) is not int or
            type(evidence['cause']) is not str or not evidence['cause'].strip() or
            any(evidence[key] is not None and type(evidence[key]) is not bool
                for key in ('controller_healthy', 'proposal_valid'))):
        raise ValueError('invalid fallback checks')
    inputs = evidence['inputs']
    if type(inputs) is not list:
        raise ValueError('invalid fallback inputs')
    limits = {item['name']: item['max_age_seconds'] for item in inputs}
    reports = ([asdict(item) for item in check_observation_freshness(
        supervisor_observation(proposal.observation), now, limits)]
        if inputs else [])
    expected = fallback_evidence(proposal, evidence['cause'], now, reports,
                                 evidence['controller_healthy'], evidence['proposal_valid'])
    if evidence != expected:
        raise ValueError('fallback does not match recorded prerequisites')
