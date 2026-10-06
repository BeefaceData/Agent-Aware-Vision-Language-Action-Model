"""Chronological camera requests for the Anthropic Messages API.

Image conversion and transport are injected. No provider is contacted on import
or construction; callers remain responsible for resource and data-use gates.
"""

from base64 import b64encode
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import http.client
import json
import os
from time import monotonic

from observation_window import ObservationWindowBuilder, WindowSettings
from model_usage import ModelReply
from supervisor_response import WindowedSupervisorResponse
from supervisor_retry import RecoverableProviderError
from decision_memory import DecisionMemory, NoMemory
from fixed_memory import FixedMemory


PROVIDER_API_VERSION = '2023-06-01'
SUPERVISOR_PROMPT = ('Assess only the supplied observations. Return one JSON object, '
    'without markdown. Copy the request identity fields exactly. '
    'Assess temporal progress: progress, suspected_missed_grasp, '
    'suspected_lost_grasp, stall, or unknown. Include temporal_diagnosis '
    'with category, summary, and evidence: a list of objects with '
    'observation_sequence, source (main, wrist, robot_state), and '
    'description of visible evidence. Cite supplied observations; '
    'non-unknown categories require evidence. Explain temporal changes '
    'and limitations; missing history is not proof of a stall. '
    'Report contradictions in temporal_diagnosis.conflicts as objects '
    'with case and evidence_indices (at least two distinct zero-based '
    'indices into evidence). Use grasp_state_conflict when apparent '
    'grasp success conflicts with later state; use ambiguous_object_motion '
    'when views or camera motion leave object movement ambiguous. '
    'Preserve both sides as evidence and abstain with unknown. '
    'Confidence cannot override a conflict. '
    'Do not claim ground-truth task success or infer hidden object state. '
    'For non-unknown assessments use kind="pass"; suspected failures '
    'are observation-only and leave the policy action unchanged. '
    'For uncertainty use kind="abstain", diagnosis="unknown", '
    'a nonempty reason, and evidence_availability mapping main, wrist '
    'and robot_state to available, missing, stale or unknown. '
    'Do not generate code, corrections or instruction changes.')


@dataclass(frozen=True)
class VlmSettings:
    model: str
    max_tokens: int = 512
    max_observations: int = 8
    immutable_model_version: bool = False
    version_limitation: str = (
        "Provider model immutability is unverified; the selected identifier may drift.")

    def __post_init__(self):
        if type(self.model) is not str or not self.model.strip():
            raise ValueError('explicit provider model required')
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError('positive output token limit required')
        WindowSettings(self.max_observations, 0)
        if type(self.immutable_model_version) is not bool:
            raise ValueError('model immutability must be explicit boolean')
        if type(self.version_limitation) is not str:
            raise ValueError('version limitation must be text')
        if not self.immutable_model_version and not self.version_limitation.strip():
            raise ValueError('mutable or unverified versions require a limitation')


def supervisor_identity(settings):
    """Declared selection, not proof that a remote provider preserves weights."""
    if type(settings) is not VlmSettings:
        raise ValueError('VlmSettings required')
    return {
        'schema_version': 1, 'provider': 'anthropic',
        'provider_api_version': PROVIDER_API_VERSION,
        'model': settings.model,
        'immutable_model_version': settings.immutable_model_version,
        'version_limitation': settings.version_limitation,
        'prompt_template_sha256': sha256(SUPERVISOR_PROMPT.encode('utf-8')).hexdigest(),
        'generation_settings': {'max_tokens': settings.max_tokens,
                                'temperature': 'provider_default',
                                'top_p': 'provider_default', 'top_k': 'provider_default'},
        'observation_settings': {'max_observations': settings.max_observations},
    }


def _json(value):
    return json.dumps(value, allow_nan=False, default=lambda item:
                      item.isoformat() if type(item) is datetime else _unsupported())


def _unsupported():
    raise ValueError('unsupported request value')


class AnthropicMessagesTransport:
    """Single HTTPS request; credentials are read only into the auth header.

    No retries or redirects. Wrap the adapter in BoundedSupervisorProvider:
    socket timeouts alone do not bound an entire response or cancel remote work.
    """

    def __call__(self, payload, deadline, cancellation):
        remaining = deadline - monotonic()
        if cancellation.is_set() or remaining <= 0:
            raise RuntimeError('request no longer current')
        key = os.environ.get('ANTHROPIC_API_KEY')
        if not key:
            raise RuntimeError('provider credential unavailable')
        connection = http.client.HTTPSConnection('api.anthropic.com', timeout=remaining)
        try:
            connection.request('POST', '/v1/messages', body=_json(payload).encode(),
                               headers={'content-type': 'application/json',
                                        'anthropic-version': PROVIDER_API_VERSION,
                                        'x-api-key': key})
            response = connection.getresponse()
            if response.status in (429, 503):
                raise RecoverableProviderError(
                    'rate_limited' if response.status == 429 else 'unavailable')
            if response.status != 200:
                raise RuntimeError('provider request failed')
            data = response.read(1_048_577)
            if len(data) > 1_048_576:
                raise RuntimeError('provider response exceeds limit')
            if cancellation.is_set() or monotonic() >= deadline:
                raise RuntimeError('request no longer current')
            return json.loads(data)
        finally:
            connection.close()


class ChronologicalVlmAdapter:
    """Provider callback for BoundedSupervisorProvider, one serial episode stream.

    encode_png receives a sanitized numeric image and returns PNG bytes. Camera
    dimensions/preprocessing belong to that caller-supplied converter. History
    contains observations only: past proposals are never labeled executed actions.
    A new episode ID resets history; sequence/time reversal is rejected.
    """

    def __init__(self, settings: VlmSettings, encode_png, transport=None, *,
                 call_journal=None, frozen_manifest=None, decision_memory=None):
        if type(settings) is not VlmSettings or not callable(encode_png):
            raise ValueError('VlmSettings and image encoder required')
        if transport is not None and not callable(transport):
            raise ValueError('transport must be callable')
        self.settings = settings
        self._initial_identity = supervisor_identity(settings)
        self._frozen_manifest = frozen_manifest
        if frozen_manifest is not None:
            frozen_manifest.verify(supervisor_identity(settings))
        self._encode_png = encode_png
        self._transport = transport if transport is not None else AnthropicMessagesTransport()
        self._history = None
        self._call_journal = call_journal
        self._retry_proposal = None
        if decision_memory is not None and type(decision_memory) not in (DecisionMemory, FixedMemory, NoMemory):
            raise ValueError('DecisionMemory, FixedMemory or NoMemory required')
        self._decision_memory = NoMemory() if decision_memory is None else decision_memory

    def __call__(self, proposal, deadline, cancellation):
        if cancellation.is_set() or monotonic() >= deadline:
            raise RuntimeError('request no longer current')
        if self._frozen_manifest is not None:
            self._frozen_manifest.verify(supervisor_identity(self.settings))
        if supervisor_identity(self.settings) != self._initial_identity:
            raise ValueError('supervisor selection changed during adapter lifetime')
        packet = proposal.observation
        if self._history is None or self._history.episode_id != packet.episode_id:
            self._history = ObservationWindowBuilder(
                packet.episode_id, packet.observation.get('task'),
                WindowSettings(self.settings.max_observations, 0))
        if self._retry_proposal != proposal:
            self._history.append(packet)
        self._retry_proposal = None
        window = self._history.snapshot()
        identity = dict(episode_id=packet.episode_id,
                        observation_sequence=packet.sequence,
                        proposal_id=proposal.proposal_id)
        memory = self._decision_memory.prepare(proposal)
        content = [{'type': 'text', 'text': _json({
            'request': identity, 'task': window.task,
            'proposed_action': proposal.action,
            'ordering': window.ordering,
            'max_observations': window.settings.max_observations,
            'omitted_prefix': asdict(window.omitted_prefix) if window.omitted_prefix else None,
            'missing_intervals': [asdict(gap) for gap in window.missing_intervals],
            'executed_action_history': 'not supplied',
        })}]
        if memory['retrieval'] == 'enabled':
            content.append({'type': 'text', 'text': memory['context_json']})
        for observation in window.observations:
            content.append({'type': 'text', 'text': _json({
                'observation_sequence': observation.sequence,
                'captured_at': observation.captured_at,
                'captured_monotonic': observation.captured_monotonic,
                'robot_state': observation.observation.get('robot_state'),
                'robot_state_capture': (asdict(observation.robot_state_capture)
                                        if observation.robot_state_capture else None),
            })})
            references = {ref.camera: ref for ref in observation.frame_references}
            if len(references) != 2 or len(observation.frame_references) != 2:
                raise ValueError('main and wrist camera metadata required')
            pixels = observation.observation.get('pixels', {})
            for camera, key in (('main', 'image'), ('wrist', 'image2')):
                ref = references[camera]
                frame = pixels.get(key)
                if (ref.availability == 'available') != (frame is not None):
                    raise ValueError('camera availability does not match payload')
                content.append({'type': 'text', 'text': _json(asdict(ref))})
                if frame is not None:
                    encoded = self._encode_png(frame)
                    if type(encoded) is not bytes or not encoded.startswith(b'\x89PNG\r\n\x1a\n'):
                        raise ValueError('image encoder must return PNG bytes')
                    content.append({'type': 'image', 'source': {
                        'type': 'base64', 'media_type': 'image/png',
                        'data': b64encode(encoded).decode('ascii')}})
        payload = dict(model=self.settings.model, max_tokens=self.settings.max_tokens,
                       system=SUPERVISOR_PROMPT,
                       messages=[{'role': 'user', 'content': content}])
        if cancellation.is_set() or monotonic() >= deadline:
            raise RuntimeError('request no longer current')
        def send():
            response = self._transport(payload, deadline, cancellation)
            if self._call_journal is None:
                return response
            # Keep only numeric usage counters, never arbitrary provider fields.
            usage = response.get('usage') if type(response) is dict else None
            reported = None
            if type(usage) is dict:
                reported = {key: usage[key] for key in (
                    'input_tokens', 'output_tokens', 'cache_creation_input_tokens',
                    'cache_read_input_tokens') if key in usage}
                reported = reported or None
            return ModelReply(response, usage=reported)

        try:
            response = (send() if self._call_journal is None else
                        self._call_journal.call(proposal, 'anthropic',
                                                self.settings.model, send))
        except RecoverableProviderError:
            self._retry_proposal = deepcopy(proposal)
            raise
        if (type(response) is not dict or response.get('type') != 'message' or
                response.get('stop_reason') != 'end_turn'):
            raise ValueError('provider did not return a complete message')
        blocks = response.get('content')
        if (type(blocks) is not list or len(blocks) != 1 or
                type(blocks[0]) is not dict or blocks[0].get('type') != 'text' or
                type(blocks[0].get('text')) is not str):
            raise ValueError('provider must return one structured text response')
        # The bounded provider applies the shared strict decision decoder next.
        return WindowedSupervisorResponse(json.loads(blocks[0]['text']), window, memory)
