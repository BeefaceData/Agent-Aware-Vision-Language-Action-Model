"""Adapter-owned hold/stop contracts; never synthesize a zero-action command."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class InterruptionContract:
    """One operation selected by the adapter, with documented acknowledgement.

    The synchronous interrupt(request) method must return exactly True only
    after this contract's condition holds. False/None/exception is unconfirmed.
    A hold ends this harness episode too; resumption requires a new reset.
    Physical adapters must supply their own bounded transport/deadline behavior.
    """

    identity: str
    operation: str
    acknowledgement: str

    def __post_init__(self):
        if (type(self.operation) is not str or self.operation not in ('hold', 'stop') or
                any(type(v) is not str or not v.strip()
                    for v in (self.identity, self.acknowledgement))):
            raise ValueError('documented hold/stop contract required')


@dataclass(frozen=True)
class InterruptionRequest:
    episode_id: str
    proposal_id: str
    observation_sequence: int
    operation: str
    reason: str


def require_interruption(environment):
    contract = getattr(environment, 'interruption_contract', None)
    if type(contract) is not InterruptionContract or not callable(
            getattr(environment, 'interrupt', None)):
        raise ValueError('active execution requires an adapter hold/stop contract')
    return contract


def interrupt(environment, contract, proposal, reason, clock):
    request = InterruptionRequest(proposal.observation.episode_id, proposal.proposal_id,
                                  proposal.observation.sequence, contract.operation, reason)
    started = clock()
    error = None
    try:
        confirmed = environment.interrupt(request) is True
    except Exception as exc:
        confirmed = False
        error = type(exc).__name__  # No private controller/transport payloads.
    return dict(contract=asdict(contract), request=asdict(request), confirmed=confirmed,
                error_type=error, requested_at=started, finished_at=clock())


def validate_interruption(evidence, proposal, reason):
    if type(evidence) is not dict or set(evidence) != {
        'contract', 'request', 'confirmed', 'error_type', 'requested_at', 'finished_at'
    }:
        raise ValueError('invalid interruption evidence')
    contract = InterruptionContract(**evidence['contract'])
    expected = asdict(InterruptionRequest(proposal.observation.episode_id,
        proposal.proposal_id, proposal.observation.sequence, contract.operation, reason))
    if (type(evidence['request']) is not dict or evidence['request'] != expected or
        type(evidence['request']['observation_sequence']) is not int or
        type(evidence['confirmed']) is not bool or
        (evidence['error_type'] is not None and
         (type(evidence['error_type']) is not str or not evidence['error_type'].isidentifier()
          or evidence['confirmed'])) or
        any(type(evidence[k]) not in (int, float) or not isfinite(evidence[k])
            for k in ('requested_at', 'finished_at')) or
        not proposal.observation.captured_monotonic <= evidence['requested_at'] <= evidence['finished_at']):
        raise ValueError('invalid interruption acknowledgement or identity')
