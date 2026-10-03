"""Decode recorded recovery responses as data, without execution authority."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from episode_harness import ActionProposal


def _name(value):
    return type(value) is str and 0 < len(value) <= 128 and value.isidentifier()


def _number(value):
    try:
        return type(value) in (int, float) and isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class RecoveryParameter:
    """Caller-owned inclusive scalar bounds; no physical readiness implied."""

    name: str
    minimum: float
    maximum: float

    def __post_init__(self):
        if (not _name(self.name) or not _number(self.minimum) or
                not _number(self.maximum) or self.minimum > self.maximum):
            raise ValueError('invalid recovery parameter bounds')


@dataclass(frozen=True)
class RecoveryEvidenceReference:
    observation_sequence: int
    source: str


@dataclass(frozen=True)
class SupervisorRecoveryRequest:
    episode_id: str
    observation_sequence: int
    proposal_id: str
    decision_id: str
    tool_name: str
    parameters: tuple[tuple[str, float], ...]
    evidence: tuple[RecoveryEvidenceReference, ...]
    kind: Literal['recovery'] = field(default='recovery', init=False)


@dataclass(frozen=True)
class RecoveryRequestDecoder:
    """One configured tool's wire interface, not a registry or executor.

    Construct from trusted configuration, never from the provider response.
    Evidence references are structurally checked; existence and diagnostic
    validity require a later evidence resolver. Results cannot authorize actions.
    """

    tool_name: str
    parameter_bounds: tuple[RecoveryParameter, ...]

    def __post_init__(self):
        bounds = tuple(self.parameter_bounds)
        if (not _name(self.tool_name) or not bounds or
                any(type(p) is not RecoveryParameter for p in bounds) or
                len({p.name for p in bounds}) != len(bounds)):
            raise ValueError('invalid recovery request configuration')
        object.__setattr__(self, 'parameter_bounds', bounds)

    def decode(self, response: dict, proposal: ActionProposal) -> SupervisorRecoveryRequest:
        """Return detached immutable request data from a JSON-decoded response.

        All configured parameters are required. Unknown fields at every level,
        nonfinite values, booleans and foreign/stale proposal identities fail.
        No callback, environment, tool code or instruction is accepted here.
        """
        fields = {'kind', 'episode_id', 'observation_sequence', 'proposal_id',
                  'decision_id', 'tool_name', 'parameters', 'evidence'}
        if type(response) is not dict or set(response) != fields:
            raise ValueError('invalid recovery response fields')
        sequence = response['observation_sequence']
        if (response['kind'] != 'recovery' or
                type(sequence) is not int or sequence < 0 or
                type(response['episode_id']) is not str or
                not response['episode_id'] or
                type(response['proposal_id']) is not str or
                not response['proposal_id'] or
                response['episode_id'] != proposal.observation.episode_id or
                sequence != proposal.observation.sequence or
                response['proposal_id'] != proposal.proposal_id):
            raise ValueError('recovery response must reference the current proposal')
        decision = response['decision_id']
        if type(decision) is not str or not decision.strip() or len(decision) > 128:
            raise ValueError('invalid recovery decision identity')
        if response['tool_name'] != self.tool_name:
            raise ValueError('unknown recovery tool')
        parameters = response['parameters']
        if (type(parameters) is not dict or
                set(parameters) != {p.name for p in self.parameter_bounds}):
            raise ValueError('invalid recovery parameter fields')
        for bound in self.parameter_bounds:
            value = parameters[bound.name]
            if not _number(value) or not bound.minimum <= value <= bound.maximum:
                raise ValueError('recovery parameter outside configured bounds')
        evidence = response['evidence']
        if type(evidence) is not list or not 1 <= len(evidence) <= 32:
            raise ValueError('recovery evidence references required (maximum 32)')
        references = []
        for ref in evidence:
            if (type(ref) is not dict or set(ref) != {'observation_sequence', 'source'} or
                    type(ref['observation_sequence']) is not int or
                    not 0 <= ref['observation_sequence'] <= sequence or
                    not _name(ref['source'])):
                raise ValueError('invalid recovery evidence reference')
            references.append(RecoveryEvidenceReference(**ref))
        if len(set(references)) != len(references):
            raise ValueError('duplicate recovery evidence reference')
        return SupervisorRecoveryRequest(
            response['episode_id'], sequence, response['proposal_id'], decision,
            self.tool_name, tuple((p.name, parameters[p.name]) for p in self.parameter_bounds),
            tuple(references))
