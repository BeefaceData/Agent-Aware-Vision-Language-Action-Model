"""Strict structured-response boundary shared by recorded/provider adapters."""

from copy import deepcopy
from dataclasses import dataclass, replace

from temporal_diagnosis import valid_temporal_diagnosis
from observation_window import ObservationWindow
from decision_memory import disabled_memory, validate_memory

from episode_harness import (ActionProposal, SupervisorAbstention, SupervisorPass,
                             SupervisorResponseError, valid_abstention_details,
                             supervisor_observation)
from supervisor_adjustment import AdjustmentRequestDecoder, SupervisorAdjustmentRequest
from supervisor_recovery import RecoveryRequestDecoder, SupervisorRecoveryRequest


@dataclass(frozen=True)
class WindowedSupervisorResponse:
    """Adapter-owned request context; never deserialize this from model JSON."""

    response: dict
    window: ObservationWindow
    memory_context: dict | None = None


def _resolve_temporal_evidence(temporal, proposal, window):
    if temporal is None:
        return
    current = supervisor_observation(proposal.observation)
    observations = (current,)
    if window is not None:
        if type(window) is ObservationWindow:
            current.observation.setdefault('task', window.task)
        if (type(window) is not ObservationWindow or
                window.episode_id != current.episode_id or not window.observations or
                len(window.observations) > window.settings.max_observations or
                any(packet.episode_id != current.episode_id for packet in window.observations) or
                any(left.sequence >= right.sequence for left, right in
                    zip(window.observations, window.observations[1:])) or
                window.observations[-1] != current):
            raise SupervisorResponseError('invalid active diagnosis observation window')
        observations = window.observations
    packets = {packet.sequence: supervisor_observation(packet) for packet in observations}
    for index, evidence in enumerate(temporal['evidence']):
        packet = packets.get(evidence['observation_sequence'])
        reason = None
        if packet is None:
            reason = 'observation outside active window'
        elif evidence['source'] == 'robot_state':
            state = packet.observation.get('robot_state')
            capture = packet.robot_state_capture
            def has_measurement(value):
                if isinstance(value, dict):
                    return any(has_measurement(item) for item in value.values())
                if isinstance(value, (list, tuple)):
                    return any(has_measurement(item) for item in value)
                return type(value) in (int, float, bool)

            if (not has_measurement(state) or
                    (capture is not None and capture.observation_sequence != packet.sequence)):
                reason = 'robot state unavailable or mismatched'
        else:
            source = evidence['source']
            refs = [ref for ref in packet.frame_references if ref.camera == source]
            key = 'image' if source == 'main' else 'image2'
            if (len(refs) != 1 or refs[0].observation_sequence != packet.sequence or
                    refs[0].availability != 'available' or
                    packet.observation.get('pixels', {}).get(key) is None):
                reason = 'camera frame unavailable or mismatched'
        if reason:
            # Retain an inspectable reason without echoing arbitrary model text.
            raise SupervisorResponseError(f'temporal evidence reference {index}: {reason}')


def _conflict_abstention(temporal, proposal):
    """Conflict presence dominates the requested category and correction kind."""
    diagnosis = deepcopy(temporal)
    diagnosis['category'] = 'unknown'
    diagnosis['summary'] = ('Conflicting temporal evidence; original assessment: ' +
                            temporal['summary'])[:2000]
    return SupervisorAbstention(
        proposal.observation.episode_id, proposal.observation.sequence,
        proposal.proposal_id, 'Conflicting temporal evidence requires abstention',
        {source: 'unknown' for source in ('main', 'wrist', 'robot_state')},
        temporal_diagnosis=diagnosis)


@dataclass(frozen=True)
class SupervisorResponseDecoder:
    """Decode JSON objects with caller-owned correction contracts.

    Errors contain schema reasons, never raw provider payloads. Recovery and
    adjustment results remain request data and cannot authorize execution.
    Pass/abstain results can be returned from run_episode's supervisor_decider.
    With observation_only=True, configured valid correction requests become
    unchanged-policy passes with a suppressed-kind marker. Temporal conflicts
    still force abstention; malformed or stale responses remain errors.
    """

    recovery: RecoveryRequestDecoder | None = None
    adjustment: AdjustmentRequestDecoder | None = None
    observation_only: bool = False

    def __post_init__(self):
        if type(self.observation_only) is not bool:
            raise ValueError('observation_only must be bool')
        if ((self.recovery is not None and type(self.recovery) is not RecoveryRequestDecoder) or
                (self.adjustment is not None and
                 type(self.adjustment) is not AdjustmentRequestDecoder)):
            raise ValueError('invalid supervisor decoder configuration')

    def decode(self, response, proposal, *, window=None):
        memory = disabled_memory()
        if type(response) is WindowedSupervisorResponse and response.memory_context is not None:
            memory = deepcopy(response.memory_context)
        try:
            validate_memory(memory)
        except ValueError as exc:
            raise SupervisorResponseError(str(exc)) from exc
        decision = self._decode(response, proposal, window=window)
        if type(decision) in (SupervisorPass, SupervisorAbstention):
            return replace(decision, memory_context=memory)
        if memory['retrieval'] != 'disabled':
            raise SupervisorResponseError('memory-informed corrections require current-scene integration')
        return decision

    def _decode(self, response: dict | WindowedSupervisorResponse, proposal: ActionProposal,
               *, window: ObservationWindow | None = None) -> (
            SupervisorPass | SupervisorAbstention | SupervisorRecoveryRequest |
            SupervisorAdjustmentRequest):
        """Reject unsupported fields and malformed values with a recorded reason.

        Input is model JSON, optionally paired with adapter-owned window context.
        Without a window, temporal references can cite only the current packet.
        The harness records SupervisorResponseError as a rejected proposal and
        terminates the attempt through its existing failure path.
        """
        if type(response) is WindowedSupervisorResponse:
            if window is not None:
                raise SupervisorResponseError('duplicate diagnosis observation window')
            window, response = response.window, response.response
        if type(response) is not dict:
            raise SupervisorResponseError('supervisor response must be an object')
        kind = response.get('kind')
        if type(kind) is not str or kind not in ('pass', 'abstain', 'recovery', 'adjustment'):
            raise SupervisorResponseError('unknown supervisor decision type')
        # Select one mode before invoking either correction decoder. Even an
        # empty/zero residual on a recovery request is an ambiguous decision.
        recovery_fields = {'tool_name', 'parameters', 'evidence'}
        adjustment_fields = {'scope', 'target', 'frame', 'units', 'residual'}
        foreign_fields = (adjustment_fields if kind == 'recovery' else
                          recovery_fields if kind == 'adjustment' else
                          recovery_fields | adjustment_fields)
        if foreign_fields.intersection(response):
            raise SupervisorResponseError('conflicting supervisor decision fields')
        if kind in ('recovery', 'adjustment'):
            decoder = self.recovery if kind == 'recovery' else self.adjustment
            if decoder is None:
                raise SupervisorResponseError(f'{kind} requests are not configured')
            try:
                request = decoder.decode(
                    {key: value for key, value in response.items()
                     if key != 'temporal_diagnosis'}, proposal)
                temporal = response.get('temporal_diagnosis')
                if self.observation_only:
                    # Validate diagnosis against the actual request window as usual.
                    # No correction data can reach the action selection seam.
                    assessment = self.decode({
                        'kind': 'pass', 'episode_id': request.episode_id,
                        'observation_sequence': request.observation_sequence,
                        'proposal_id': request.proposal_id,
                        'temporal_diagnosis': temporal,
                    }, proposal, window=window)
                    if type(assessment) is SupervisorPass:
                        return replace(assessment, suppressed_correction=kind)
                    return assessment
                if temporal is not None:
                    if not valid_temporal_diagnosis(temporal, 'pass'):
                        raise SupervisorResponseError('invalid temporal diagnosis')
                    _resolve_temporal_evidence(temporal, proposal, window)
                    if temporal.get('conflicts'):
                        return _conflict_abstention(temporal, proposal)
                    raise SupervisorResponseError('correction temporal diagnosis requires conflicts')
                return request
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
        _resolve_temporal_evidence(temporal, proposal, window)
        if kind == 'abstain' and (response['diagnosis'] != 'unknown' or
                not valid_abstention_details(response['reason'], response['evidence_availability'])):
            raise SupervisorResponseError('invalid abstention diagnosis or evidence availability')
        if temporal is not None and temporal.get('conflicts'):
            return _conflict_abstention(temporal, proposal)
        if kind == 'pass':
            return SupervisorPass(*identity, temporal_diagnosis=deepcopy(temporal))
        return SupervisorAbstention(*identity, response['reason'],
                                    deepcopy(response['evidence_availability']),
                                    temporal_diagnosis=deepcopy(temporal))
