"""Portable, lossless-value episode traces and model-free public-interface replay.

JSON arrays preserve values/shapes, not NumPy/Torch types. Raw observations are
private evaluator evidence. No pickle, video decoding, inference or simulator is
needed. Only normally returned episodes (including rejected proposals) are sealed.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from episode_harness import (
    ActionResolution, EpisodeConfig, FrameReference, ObservationIngestor,
    ObservationPacket, RobotStateCapture, StepResult, run_episode,
    supervisor_observation,
)
from replay_adapters import ReplayRecorder
from execution_progress import ExecutionJournal


class TraceError(ValueError):
    """Missing, corrupt, inconsistent or unsupported recorded evidence."""


def _encode(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported trace value: {type(value).__name__}')


def _json(value):
    return json.dumps(value, default=_encode, allow_nan=False, sort_keys=True)


def _plain(value):
    return json.loads(_json(value))


def _read(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise TraceError(f'duplicate JSON field: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise TraceError(f'nonfinite JSON value: {value}')

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _json(value)  # Also catches overflow such as 1e999.
    return value


def _summary(outcome):
    return {key: _plain(getattr(outcome, key)) for key in
            ('success', 'steps', 'stop_reason', 'sum_rewards')}


class TraceRecorder:
    """Wrap any EpisodeRecorder; seal(outcome) publishes replay only after finish.

    The destination must be new. Failed/interrupted recordings retain partial
    files but have no manifest, and cannot masquerade as replayable episodes.
    """

    def __init__(self, directory, config, recorder):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.recorder = recorder
        self._files = {}
        self._finished = False
        self._failed = False
        self._episode_id = None
        self._journal = None

    def _write(self, name, value):
        self._files[name].write(_json(value) + '\n')
        self._files[name].flush()

    def begin(self, observation):
        self._episode_id = observation.episode_id
        self._journal = ExecutionJournal(self.directory / 'execution.jsonl',
                                         self._episode_id, asdict(self.config))
        for name in ('observations.jsonl', 'decisions.jsonl'):
            self._files[name] = (self.directory / name).open('x', encoding='utf-8', newline='\n')
        self._write('observations.jsonl', asdict(observation))
        self.recorder.begin(observation)

    def record_step(self, step, source, action, result, ingestion):
        # Preserve the acknowledgement before writing bulky observations or
        # invoking video/other sinks, which can fail independently.
        self._journal.acknowledge(step, source.episode_id, source.sequence,
                                  _plain(asdict(result.action_record)))
        if not ingestion.accepted:
            self._failed = True
        self._write('observations.jsonl', asdict(result.observation))
        self._write('decisions.jsonl', {
            'step': step, 'source_episode_id': source.episode_id,
            'source_sequence': source.sequence,
            'action_record': asdict(result.action_record),
            'result': {key: getattr(result, key) for key in
                       ('reward', 'success', 'terminated', 'truncated')},
        })
        self.recorder.record_step(step, source, action, result, ingestion)

    def record_failure(self, step, source, action, failure):
        if failure.action_record is None or failure.action_record.disposition != 'rejected':
            self._failed = True
        self._write('decisions.jsonl', {
            'step': step, 'source_episode_id': source.episode_id,
            'source_sequence': source.sequence,
            'action_record': (asdict(failure.action_record)
                              if failure.action_record else None), 'result': None,
        })
        self.recorder.record_failure(step, source, action, failure)

    def finish(self):
        errors = []
        artifacts = {}
        try:
            artifacts = dict(self.recorder.finish())
        except BaseException as exc:
            errors.append(exc)
        streams = list(self._files.values())
        if self._journal is not None:
            streams.append(self._journal)
        for stream in streams:
            try:
                stream.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            self._failed = True
            for error in errors:
                if not isinstance(error, Exception):
                    raise error
            raise RuntimeError('; '.join(f'{type(e).__name__}: {e}' for e in errors))
        self._finished = True
        return artifacts

    def seal(self, outcome):
        if (not self._finished or self._failed or
                outcome.artifact_status != 'completed' or
                outcome.episode_id != self._episode_id):
            raise TraceError('cannot seal incomplete or foreign episode evidence')
        manifest = {
            'version': 1, 'source_episode_id': self._episode_id,
            'config': asdict(self.config), 'outcome': _summary(outcome),
            'files': {name: sha256((self.directory / name).read_bytes()).hexdigest()
                      for name in ('observations.jsonl', 'decisions.jsonl')},
        }
        # Validate our own output before publishing the completion marker.
        _load(self.directory, manifest)
        path = self.directory / 'manifest.json'
        with path.open('x', encoding='utf-8') as stream:
            stream.write(_json(manifest) + '\n')
        return str(path.resolve())


def _packet(row):
    row = deepcopy(row)
    row['captured_at'] = datetime.fromisoformat(row['captured_at'])
    refs = []
    for ref in row['frame_references']:
        if ref['captured_at'] is not None:
            ref['captured_at'] = datetime.fromisoformat(ref['captured_at'])
        refs.append(FrameReference(**ref))
    row['frame_references'] = tuple(refs)
    capture = row['robot_state_capture']
    if capture is not None:
        capture['captured_at'] = datetime.fromisoformat(capture['captured_at'])
        row['robot_state_capture'] = RobotStateCapture(**capture)
    if 'observation' not in row or row['observation'] is None:
        raise TraceError('observation payload missing')
    packet = ObservationPacket(**row)
    supervisor_observation(packet)  # Validate typed sensor metadata too.
    if packet.frame_references:
        refs = {ref.camera: ref for ref in packet.frame_references}
        if len(refs) != len(packet.frame_references):
            raise TraceError('duplicate camera reference')
        pixels = packet.observation.get('pixels', {})
        for ref in packet.frame_references:
            available = pixels.get(ref.image_key.split('.')[1]) is not None
            if available != (ref.availability == 'available'):
                raise TraceError('camera payload does not match availability')
    return packet


def _require(condition, message):
    if not condition:
        raise TraceError(message)


def _load(directory, manifest):
    _require(type(manifest['version']) is int and manifest['version'] == 1,
             'unsupported trace version')
    episode_id = manifest['source_episode_id']
    config = EpisodeConfig(**manifest['config'])
    _require(type(config.seed) is int and type(config.max_steps) is int and
             config.max_steps > 0, 'invalid episode configuration')
    _require(set(manifest['files']) == {'observations.jsonl', 'decisions.jsonl'},
             'required replay artifacts missing')
    data = {}
    for name, digest in manifest['files'].items():
        path = directory / name
        _require(path.resolve().parent == directory.resolve(), 'artifact escapes bundle')
        raw = path.read_bytes()
        _require(sha256(raw).hexdigest() == digest, f'artifact checksum mismatch: {name}')
        data[name] = [_read(line) for line in raw.decode('utf-8').splitlines()]
    packets = tuple(_packet(row) for row in data['observations.jsonl'])
    _require(bool(packets), 'initial observation missing')
    ingestor = ObservationIngestor(episode_id)
    for packet in packets:
        ingestion = ingestor.ingest(packet)
        _require(ingestion.accepted, f'invalid observation ordering/identity: {ingestion.code}')
    decisions = data['decisions.jsonl']
    _require(0 < len(decisions) <= config.max_steps, 'invalid decision count')
    executed = 0
    reward = 0.0
    success = False
    stop = 'step_limit'
    for index, row in enumerate(decisions):
        _require(type(row['step']) is int and type(row['source_sequence']) is int and
                 row['step'] == index + 1 and row['source_sequence'] == index and
                 row['source_episode_id'] == episode_id, 'invalid decision source/order')
        _require(index < len(packets), 'decision source observation missing')
        record = row['action_record']
        _require(record['proposal_id'] == f'{episode_id}:{index + 1}', 'foreign proposal')
        _require(record['proposed_action'] is not None, 'proposal missing')
        response = record.get('supervisor_pass')
        if response is not None:
            _require(type(response) is dict and
                     type(response.get('observation_sequence')) is int and
                     response == {'kind': 'pass', 'episode_id': episode_id,
                                  'observation_sequence': index,
                                  'proposal_id': record['proposal_id']} and
                     record['disposition'] == 'unmodified',
                     'invalid supervisor pass evidence')
        result = row['result']
        last = index == len(decisions) - 1
        if record['disposition'] == 'rejected':
            _require(last and result is None and bool(record['rejection_reason']) and
                     all(record[key] is None for key in ('selected_action', 'executed_action',
                                                        'execution_acknowledgement')),
                     'invalid rejected decision')
            stop = 'proposal_rejected'
            break
        _require(record['disposition'] in ('unmodified', 'overridden') and
                 record['selected_action'] is not None and
                 record['executed_action'] == record['selected_action'] and
                 record['rejection_reason'] is None, 'invalid execution evidence')
        if record['disposition'] == 'unmodified':
            _require(record['executed_action'] == record['proposed_action'],
                     'pass decision changed action')
        _require(record['execution_acknowledgement'] == {
            'proposal_id': record['proposal_id'], 'result_episode_id': episode_id,
            'result_sequence': index + 1}, 'invalid execution acknowledgement')
        _require(result is not None and set(result) == {
            'reward', 'success', 'terminated', 'truncated'}, 'result missing/invalid')
        _require(all(type(result[key]) is bool for key in ('success', 'terminated', 'truncated'))
                 and type(result['reward']) in (int, float), 'invalid evaluator values')
        executed += 1
        reward += result['reward']
        success = result['success']
        if success or result['terminated'] or result['truncated']:
            _require(last, 'decisions after terminal result')
            stop = 'success' if success else 'terminated' if result['terminated'] else 'truncated'
    _require(len(packets) == executed + 1, 'missing or extra observation artifacts')
    _require(stop != 'step_limit' or executed == config.max_steps, 'unfinished trace')
    _require(manifest['outcome'] == dict(success=success, steps=executed,
                                       stop_reason=stop, sum_rewards=reward),
             'outcome does not match recorded sequence')
    return RecordedReplay(episode_id, config, packets, decisions, manifest['outcome'])


def load_recorded_replay(directory):
    """Validate the entire bundle before returning usable replay adapters.

    Hashes detect changed artifacts, not authenticity against a malicious author.
    Legacy video/action-only bundles and incomplete traces are explicitly refused.
    """
    try:
        directory = Path(directory)
        return _load(directory, _read((directory / 'manifest.json').read_bytes()))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise TraceError(f'invalid replay bundle: {exc}') from exc


class RecordedReplay:
    """Validated trace; run() creates fresh replay-only adapters on every call."""

    def __init__(self, episode_id, config, packets, decisions, outcome):
        self.source_episode_id = episode_id
        self.config = config
        self._packets = packets
        self._decisions = deepcopy(decisions)
        self._outcome = deepcopy(outcome)

    def evidence(self):
        """Return detached, JSON-compatible historical evidence for inspection.

        Includes private evaluator observations; this is not a supervisor input.
        Timestamps retain their original meaning, not replay execution timing.
        """
        return _plain({
            'source_episode_id': self.source_episode_id,
            'config': asdict(self.config),
            'observations': [asdict(packet) for packet in self._packets],
            'decisions': self._decisions,
            'outcome': self._outcome,
        })

    def run(self, recorder=None):
        packets, decisions = deepcopy((self._packets, self._decisions))
        config = self.config
        # Historical timestamps are evidence, not measurements of replay latency.
        now = max(packet.captured_monotonic for packet in packets)

        class Policy:
            def reset(self):
                pass

            def resume(self, packet):
                # Recorded proposals are indexed by accepted observation, not queued.
                _require(packet.observation == packets[packet.sequence].observation,
                         'replay resume observation diverged')

            def act(self, packet):
                _require(packet.observation == packets[packet.sequence].observation,
                         'replay observation diverged')
                return deepcopy(decisions[packet.sequence]['action_record']['proposed_action'])

        class Environment:
            def reset(self, seed, episode_id):
                _require(seed == config.seed, 'replay seed mismatch')
                self.episode_id, self.index = episode_id, 0
                return replace(deepcopy(packets[0]), episode_id=episode_id)

            def step(self, action):
                row = decisions[self.index]
                _require(action == row['action_record']['executed_action'],
                         'replay action diverged')
                self.index += 1
                return StepResult(replace(deepcopy(packets[self.index]),
                                          episode_id=self.episode_id), **row['result'])

        def select(proposal):
            record = decisions[proposal.observation.sequence]['action_record']
            if record['disposition'] == 'rejected':
                return ActionResolution('reject', reason=record['rejection_reason'])
            if record['disposition'] == 'overridden':
                return ActionResolution('override', deepcopy(record['selected_action']))
            return ActionResolution('pass')

        outcome = run_episode(config, Policy(), Environment(),
                              recorder if recorder is not None else ReplayRecorder(),
                              clock=lambda: now, action_selector=select)
        _require(_summary(outcome) == self._outcome, 'replayed outcome diverged')
        return outcome


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='sealed replay directory containing manifest.json')
    args = parser.parse_args()
    replay = load_recorded_replay(args.directory)
    print(_json({'source_episode_id': replay.source_episode_id,
                 'replay_outcome': _summary(replay.run())}))
