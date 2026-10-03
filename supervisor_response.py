"""Strict structured-response boundary shared by recorded/provider adapters."""

from copy import deepcopy
from dataclasses import dataclass

from temporal_diagnosis import valid_temporal_diagnosis

from episode_harness import (ActionProposal, SupervisorAbstention, SupervisorPass,
                             SupervisorResponseError, valid_abstention_details)
from supervisor_adjustment import AdjustmentRequestDecoder, SupervisorAdjustmentRequest
from supervisor_recovery import RecoveryRequestDecoder, SupervisorRecoveryRequest


@dataclass(frozen=True)
class SupervisorResponseDecoder:
    """Decode JSON objects with caller-owned correction contracts.

    Errors contain schema reasons, never raw provider payloads. Recovery and
    adjustment results remain request data and cannot authorize execution.
    Pass/abstain results can be returned from run_episode's supervisor_decider.
    """

    recovery: RecoveryRequestDecoder | None = None
    adjustment: AdjustmentRequestDecoder | None = None

    def __post_init__(self):
        if ((self.recovery is not None and type(self.recovery) is not RecoveryRequestDecoder) or
                (self.adjustment is not None and
                 type(self.adjustment) is not AdjustmentRequestDecoder)):
            raise ValueError('invalid supervisor decoder configuration')

    def decode(self, response: dict, proposal: ActionProposal) -> (
            SupervisorPass | SupervisorAbstention | SupervisorRecoveryRequest |
            SupervisorAdjustmentRequest):
        """Reject unsupported fields and malformed values with a recorded reason.

        Input is a JSON-decoded object, not executable code or a typed decision.
        The harness records SupervisorResponseError as a rejected proposal and
        terminates the attempt through its existing failure path.
        """
        if type(response) is not dict:
            raise SupervisorResponseError('supervisor response must be an object')
        kind = response.get('kind')
        if type(kind) is not str or kind not in ('pass', 'abstain', 'recovery', 'adjustment'):
            raise SupervisorResponseError('unknown supervisor decision type')
        if kind in ('recovery', 'adjustment'):
            decoder = self.recovery if kind == 'recovery' else self.adjustment
            if decoder is None:
                raise SupervisorResponseError(f'{kind} requests are not configured')
            try:
                return decoder.decode(response, proposal)
            except ValueError as exc:
                raise SupervisorResponseError(str(exc)) from exc
        fields = {'kind', 'episode_id', 'observation_sequence', 'proposal_id'}
        if kind == 'abstain':
            fields |= {'diagnosis', 'reason', 'evidence_availability'}
        if 'temporal_diagnosis' in response:
            fields.add('temporal_diagnosis')
        temporal = response.get('temporal_diagnosis')
        if not valid_temporal_diagnosis(temporal, kind):
            raise SupervisorResponseError('invalid temporal diagnosis')
        if set(response) != fields:
            raise SupervisorResponseError(f'invalid {kind} response fields')
        sequence = response['observation_sequence']
        if (type(sequence) is not int or sequence < 0 or
                type(response['episode_id']) is not str or not response['episode_id'] or
                type(response['proposal_id']) is not str or not response['proposal_id'] or
                response['episode_id'] != proposal.observation.episode_id or
                sequence != proposal.observation.sequence or
                response['proposal_id'] != proposal.proposal_id):
            raise SupervisorResponseError(f'{kind} response must reference the current proposal')
        identity = (response['episode_id'], sequence, response['proposal_id'])
        if kind == 'pass':
            return SupervisorPass(*identity, temporal_diagnosis=deepcopy(temporal))
        if (response['diagnosis'] != 'unknown' or
                not valid_abstention_details(response['reason'], response['evidence_availability'])):
            raise SupervisorResponseError('invalid abstention diagnosis or evidence availability')
        return SupervisorAbstention(*identity, response['reason'],
                                    deepcopy(response['evidence_availability']),
                                    temporal_diagnosis=deepcopy(temporal))
