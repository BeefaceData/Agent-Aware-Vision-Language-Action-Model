"""Decode single-proposal numerical adjustment requests without executing them."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from episode_harness import ActionProposal


def _label(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 128


def _number(value):
    try:
        return type(value) in (int, float) and isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class AdjustmentComponent:
    """Caller-owned component semantics and inclusive residual bounds."""

    name: str
    unit: str
    minimum: float
    maximum: float

    def __post_init__(self):
        if (not _label(self.name) or not _label(self.unit) or
                not _number(self.minimum) or not _number(self.maximum) or
                self.minimum > self.maximum):
            raise ValueError('invalid adjustment component configuration')


@dataclass(frozen=True)
class SupervisorAdjustmentRequest:
    episode_id: str
    observation_sequence: int
    proposal_id: str
    decision_id: str
    target: str
    frame: str
    units: tuple[tuple[str, str], ...]
    residual: tuple[tuple[str, float], ...]
    kind: Literal['adjustment'] = field(default='adjustment', init=False)
    scope: Literal['single_action'] = field(default='single_action', init=False)


@dataclass(frozen=True)
class AdjustmentRequestDecoder:
    """Trusted contract for one arm or coordinated group in one declared frame.

    Components describe residuals, not replacements for the policy action.
    This interface does not convert units, map native action indices, validate
    final actions or authorize execution. Those require an intervention executor.
    """

    target: str
    frame: str
    components: tuple[AdjustmentComponent, ...]

    def __post_init__(self):
        components = tuple(self.components)
        if (not _label(self.target) or not _label(self.frame) or not components or
                any(type(c) is not AdjustmentComponent for c in components) or
                len({c.name for c in components}) != len(components)):
            raise ValueError('invalid adjustment request configuration')
        object.__setattr__(self, 'components', components)

    def decode(self, response: dict, proposal: ActionProposal) -> SupervisorAdjustmentRequest:
        """Return immutable, JSON-recordable data bound to exactly this proposal."""
        fields = {'kind', 'scope', 'episode_id', 'observation_sequence',
                  'proposal_id', 'decision_id', 'target', 'frame', 'units', 'residual'}
        if type(response) is not dict or set(response) != fields:
            raise ValueError('invalid adjustment response fields')
        sequence = response['observation_sequence']
        if (response['kind'] != 'adjustment' or response['scope'] != 'single_action' or
                type(sequence) is not int or sequence < 0 or
                not _label(response['episode_id']) or
                not _label(response['proposal_id']) or
                response['episode_id'] != proposal.observation.episode_id or
                sequence != proposal.observation.sequence or
                response['proposal_id'] != proposal.proposal_id):
            raise ValueError('adjustment response must reference one current proposal')
        if not _label(response['decision_id']):
            raise ValueError('invalid adjustment decision identity')
        if response['target'] != self.target or response['frame'] != self.frame:
            raise ValueError('unsupported adjustment target or frame')
        units, residual = response['units'], response['residual']
        if (type(units) is not dict or units != {c.name: c.unit for c in self.components} or
                type(residual) is not dict or set(residual) != {c.name for c in self.components}):
            raise ValueError('invalid adjustment components or units')
        for component in self.components:
            value = residual[component.name]
            if (not _number(value) or
                    not component.minimum <= value <= component.maximum):
                raise ValueError('adjustment residual outside configured bounds')
        return SupervisorAdjustmentRequest(
            response['episode_id'], sequence, response['proposal_id'],
            response['decision_id'], self.target, self.frame,
            tuple((c.name, c.unit) for c in self.components),
            tuple((c.name, residual[c.name]) for c in self.components))
