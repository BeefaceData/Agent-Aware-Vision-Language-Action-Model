"""Required sensing gate for any synchronous supervisor decision callable."""

from copy import deepcopy
from math import isfinite
from time import monotonic
from types import MappingProxyType

from episode_harness import (ActionProposal, SupervisorAbstention,
                             check_observation_freshness, supervisor_observation)


class ObservationEligibleSupervisor:
    """Abstain before calling a supervisor unless every required input is fresh.

    Required names and age limits are caller-owned; omitted inputs are optional.
    Return-time timestamps cannot establish sensor freshness. This gate grants
    no action authority and does not replace execution-time health validation.
    The clock must share the observation capture's monotonic time basis.
    """

    def __init__(self, supervisor, max_age_seconds, *, clock=monotonic):
        if not callable(supervisor) or not callable(clock):
            raise ValueError('supervisor and clock must be callable')
        limits = dict(max_age_seconds)
        if not limits or set(limits) - {'main', 'wrist', 'robot_state'}:
            raise ValueError('required inputs must name known sensors')
        if any(type(age) not in (int, float) or not isfinite(age) or age < 0
               for age in limits.values()):
            raise ValueError('age limits must be finite and nonnegative')
        self._limits = MappingProxyType(limits)
        self._supervisor = supervisor
        self._clock = clock

    def __call__(self, proposal):
        packet = supervisor_observation(proposal.observation)
        reports = check_observation_freshness(packet, self._clock(), self._limits)
        unavailable = [item for item in reports if item.status != 'fresh']
        if unavailable:
            availability = {name: 'unknown' for name in ('main', 'wrist', 'robot_state')}
            for item in reports:
                availability[item.name] = {
                    'fresh': 'available', 'missing': 'missing', 'stale': 'stale',
                    'unverified': 'unknown', 'invalid_metadata': 'unknown',
                }[item.status]
            reason = 'Insufficient required evidence: ' + '; '.join(
                f'{item.name} ({item.status}): {item.diagnostic}' for item in unavailable)
            return SupervisorAbstention(packet.episode_id, packet.sequence,
                                        proposal.proposal_id, reason, availability)
        return self._supervisor(ActionProposal(proposal.proposal_id, packet,
                                               deepcopy(proposal.action)))
