"""Offline, append-only intervention memory derived from pinned episode evidence.

This is evaluator storage, not a supervisor observation or retrieval policy.
Callers retain diagnosis/request context at decision time and pin its digest.
No caller-supplied outcome is accepted. File hashes detect changes, not forgery.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
import re

from action_capabilities import ActionCapabilities, ActionComponent
from recorded_replay import TraceError, load_recorded_replay, _read
from temporal_diagnosis import valid_temporal_diagnosis


def _bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def _require(condition, reason):
    if not condition:
        raise TraceError(reason)


class InterventionMemory:
    """One exclusively created record per episode/proposal, with verified reads.

    append takes four (path, expected_sha256) pairs: a sealed trace manifest,
    pre-action attempt identity, frozen supervisor identity, and decision context.
    Context is {episode_id, proposal_id, observation_sequence, diagnosis, request}.
    Diagnosis uses the temporal-diagnosis schema. Evidence paths are relative to
    this store, so moving their common parent preserves the record references.
    The API has no update/delete operation. Filesystem owners can still tamper;
    retain returned record digests separately for stronger audit provenance.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def _reference(self, pair):
        path, digest = pair
        path = Path(path).resolve()
        _require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest),
                 'expected evidence digest required')
        return {'path': os.path.relpath(path, self.directory).replace('\\', '/'),
                'sha256': digest}

    def _read_reference(self, reference):
        path = self.directory / reference['path']
        raw = path.read_bytes()
        _require(sha256(raw).hexdigest() == reference['sha256'],
                 'memory evidence digest mismatch')
        return path, _read(raw)

    def _build(self, provenance, version=2):
        _require(type(version) is int and version in (1, 2),
                 'unsupported memory record version')
        _require(set(provenance) == {'trace', 'attempt', 'supervisor', 'context'},
                 'complete memory provenance required')
        loaded = {name: self._read_reference(ref) for name, ref in provenance.items()}
        trace_path, _ = loaded['trace']
        _require(trace_path.name == 'manifest.json', 'sealed trace manifest required')
        replay = load_recorded_replay(trace_path.parent)
        evidence = replay.evidence()
        attempt, supervisor, context = (loaded[name][1]
                                        for name in ('attempt', 'supervisor', 'context'))
        episode = evidence['source_episode_id']
        _require(type(attempt['version']) is int and attempt['version'] == 1 and
                 attempt['episode_id'] == episode and
                 attempt['episode_config'] == evidence['config'],
                 'attempt does not match sealed episode')
        _require(attempt['supervisor_sha256'] == provenance['supervisor']['sha256'] and
                 attempt['supervisor_manifest'] == loaded['supervisor'][0].name,
                 'supervisor identity does not match attempt')
        for value in (attempt['task'], attempt['policy_assets'], attempt['settings'], supervisor):
            _require(type(value) is dict and bool(value), 'missing task/model/configuration identity')
        _require(isinstance(attempt['task']['instruction'], str) and
                 bool(attempt['task']['instruction'].strip()), 'missing task instruction')
        capabilities = attempt['settings']['action_capabilities']
        ActionCapabilities(
            tuple(ActionComponent(**item) for item in capabilities['components']),
            capabilities['control_frequency_hz'], tuple(capabilities['operations']),
            capabilities['layout'])
        _require(type(context) is dict and set(context) == {
            'episode_id', 'proposal_id', 'observation_sequence', 'diagnosis', 'request'},
            'complete decision context required')
        sequence = context['observation_sequence']
        _require(context['episode_id'] == episode and type(sequence) is int and
                 0 <= sequence < len(evidence['decisions']), 'foreign decision context')
        first = evidence['decisions'][sequence]
        record = first['action_record']
        _require(record['proposal_id'] == context['proposal_id'] and
                 record['disposition'] == 'overridden', 'completed intervention required')
        request = context['request']
        _require(type(request) is dict and all(request.get(key) == context[key]
                 for key in ('episode_id', 'proposal_id', 'observation_sequence')) and
                 isinstance(request.get('decision_id'), str) and bool(request['decision_id']),
                 'request provenance does not match intervention')
        diagnosis = context['diagnosis']
        _require(type(diagnosis) is dict and valid_temporal_diagnosis(diagnosis,
                 'abstain' if diagnosis.get('category') == 'unknown' else 'pass'),
                 'missing or invalid diagnosis provenance')
        for ref in diagnosis['evidence']:
            index = ref['observation_sequence']
            _require(index <= sequence, 'diagnosis references future evidence')
            packet = evidence['observations'][index]
            if ref['source'] == 'robot_state':
                _require(packet['robot_state_capture'] is not None,
                         'diagnosis references missing robot state')
            else:
                _require(any(frame['camera'] == ref['source'] and
                             frame['availability'] == 'available'
                             for frame in packet['frame_references']),
                         'diagnosis references missing camera evidence')
        rows = [first]
        recovery = record.get('recovery')
        if recovery is not None:
            _require(request == recovery['sequence']['request'] and
                     recovery['action_index'] == 0, 'recovery request provenance mismatch')
            for row in evidence['decisions'][sequence + 1:]:
                continuation = row['action_record'].get('recovery')
                if continuation is None or continuation['sequence'] != recovery['sequence']:
                    break
                rows.append(row)
        else:
            _require(request.get('kind') == 'adjustment' and
                     request.get('scope') == 'single_action' and
                     all(request.get(key) for key in ('target', 'frame', 'units', 'residual')),
                     'missing adjustment request provenance')
        _require(all(row['result'] is not None and
                     row['action_record']['execution_acknowledgement'] is not None
                     for row in rows), 'unconfirmed intervention execution')
        # Check pinned files again after derivation; never publish an observed drift.
        for reference in provenance.values():
            self._read_reference(reference)
        identity = sha256(_bytes([episode, context['proposal_id']])).hexdigest()
        record = {'version': version, 'record_id': identity, 'episode_id': episode,
                'proposal_id': context['proposal_id'], 'task': attempt['task'],
                'robot_capabilities': capabilities,
                'models': {'policy': attempt['policy_assets'], 'supervisor': supervisor},
                'configuration': {'episode': evidence['config'], 'settings': attempt['settings']},
                'diagnosis': diagnosis, 'request': request, 'execution': rows,
                'episode_outcome': evidence['outcome'], 'provenance': provenance}
        if version == 2:
            # Execution completion is not evidence of causal task benefit.
            local = {'status': 'unknown', 'reason': 'no_local_assessment'}
            if recovery is not None:
                check = rows[-1]['action_record']['recovery']['check']
                local = {'status': check['status'], 'reason': check['reason']}
                if check['status'] == 'continuing':
                    local = {'status': 'unknown', 'reason': 'no_terminal_local_check'}
            record['local_outcome'] = local
        return record

    def append(self, *, trace, attempt, supervisor, context):
        """Verify evidence and exclusively append; duplicate identity is an error."""
        try:
            provenance = {key: self._reference(value) for key, value in
                          dict(trace=trace, attempt=attempt, supervisor=supervisor,
                               context=context).items()}
            record = self._build(provenance)
            raw = _bytes(record)
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise TraceError(f'invalid intervention provenance: {exc}') from exc
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (record['record_id'] + '.json')
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return {'record_id': record['record_id'], 'sha256': sha256(raw).hexdigest()}

    def read(self, record_id, *, expected_sha256):
        """Return detached data only after verifying the record and source evidence."""
        try:
            _require(type(record_id) is str and re.fullmatch('[0-9a-f]{64}', record_id),
                     'invalid memory record identity')
            _, record = self._read_reference({'path': record_id + '.json',
                                              'sha256': expected_sha256})
            _require(record['record_id'] == record_id and
                     _bytes(record) == _bytes(self._build(record['provenance'], record['version'])),
                     'memory record differs from source evidence')
            return record
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise TraceError(f'invalid intervention memory: {exc}') from exc
