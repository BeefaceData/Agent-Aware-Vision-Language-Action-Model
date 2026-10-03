"""Executor-owned parameter checks; validation alone never authorizes actuation."""

from dataclasses import asdict, dataclass

from episode_harness import ActionProposal
from recovery_registry import RecoveryRegistry, ResolvedRecovery
from supervisor_adjustment import AdjustmentRequestDecoder, SupervisorAdjustmentRequest
from supervisor_recovery import RecoveryEvidenceReference, SupervisorRecoveryRequest


def _pairs(values):
    """Reject ambiguous typed requests before converting their fields to JSON."""
    if (type(values) not in (tuple, list) or
            any(type(pair) not in (tuple, list) or len(pair) != 2 or
                type(pair[0]) is not str for pair in values)):
        raise ValueError('invalid correction parameter pairs')
    names = [pair[0] for pair in values]
    if len(set(names)) != len(names):
        raise ValueError('duplicate correction parameter')
    return dict(values)


def _wire(request):
    if type(request) is dict:
        return request
    if type(request) is SupervisorAdjustmentRequest:
        response = asdict(request)
        response['units'] = _pairs(request.units)
        response['residual'] = _pairs(request.residual)
        return response
    if type(request) is SupervisorRecoveryRequest:
        if (type(request.evidence) not in (tuple, list) or
                any(type(ref) is not RecoveryEvidenceReference for ref in request.evidence)):
            raise ValueError('invalid correction evidence')
        response = asdict(request)
        response['parameters'] = _pairs(request.parameters)
        response['evidence'] = [asdict(ref) for ref in request.evidence]
        return response
    raise ValueError('unsupported correction request')


@dataclass(frozen=True)
class CorrectionValidator:
    """Immutable executor contract constructed from trusted host configuration.

    The adjustment contract names exactly one permitted arm/group, frame and
    component set. Recovery targets/units are fixed by each tool's required
    semantic capabilities. Adapter capabilities must come from the host, never
    from a VLM response. No configurations here establish physical readiness.
    """

    recovery: RecoveryRegistry
    capabilities: tuple[str, ...] = ()
    adjustment: AdjustmentRequestDecoder | None = None

    def __post_init__(self):
        if (type(self.recovery) is not RecoveryRegistry or
                (self.adjustment is not None and
                 type(self.adjustment) is not AdjustmentRequestDecoder)):
            raise ValueError('invalid executor correction configuration')
        if type(self.capabilities) not in (tuple, list):
            raise ValueError('invalid executor capabilities')
        capabilities = tuple(self.capabilities)
        if (any(type(name) is not str or not name.isidentifier() or
                len(name) > 128 for name in capabilities) or
                len(set(capabilities)) != len(capabilities)):
            raise ValueError('invalid executor capabilities')
        object.__setattr__(self, 'capabilities', capabilities)

    def validate(self, request: dict | SupervisorAdjustmentRequest |
                 SupervisorRecoveryRequest, proposal: ActionProposal) -> (
                     SupervisorAdjustmentRequest | ResolvedRecovery):
        """Recheck wire or decoded data against this executor's inclusive limits.

        A permissive upstream decoder, manually constructed request or cached
        selection cannot grant more authority. Invalid input raises ValueError;
        returned data is detached and immutable. The proposal is never modified.
        Native conversion, final-action bounds, evidence/scene eligibility,
        expiry and operational budgets remain separate execution-time gates.
        """
        response = _wire(request)
        kind = response.get('kind')
        if type(kind) is not str:
            raise ValueError('unsupported correction kind')
        if kind == 'recovery':
            return self.recovery.resolve(response, proposal,
                                         capabilities=self.capabilities)
        if kind == 'adjustment':
            if self.adjustment is None:
                raise ValueError('executor adjustments are disabled')
            return self.adjustment.decode(response, proposal)
        raise ValueError('unsupported correction kind')
