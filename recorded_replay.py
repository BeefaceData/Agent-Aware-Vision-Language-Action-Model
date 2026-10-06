"""Portable, lossless-value episode traces and model-free public-interface replay.

JSON arrays preserve values/shapes, not NumPy/Torch types. Raw observations are
private evaluator evidence. No pickle, video decoding, inference or simulator is
needed. Only normally returned episodes (including rejected proposals) are sealed.
"""

from temporal_diagnosis import valid_temporal_diagnosis

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
from math import isfinite
import json
from pathlib import Path

from episode_harness import (
    ActionProposal, ActionResolution, EpisodeConfig, FrameReference, ObservationIngestor, RecoverySequence,
    ObservationPacket, RobotStateCapture, StepResult, StepTiming, run_episode,
    supervisor_observation, valid_abstention_details,
)
from replay_adapters import ReplayRecorder
from execution_progress import ExecutionJournal
from intervention_budget import InterventionBudget
from supervisor_retry import RECOVERABLE_ERRORS


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
            'timing': asdict(result.timing),
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
            'supervisor_call_budget': outcome.supervisor_call_budget,
            'wall_clock_limit': outcome.wall_clock_limit,
            'episode_clock': outcome.episode_clock,
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


def _supervisor_budget(evidence, config, previous, episode_clock):
    """Validate reservations, including all attempts within a retried decision."""
    _require(type(evidence) is dict and type(evidence.get('admitted')) is bool and
             type(evidence.get('attempted')) is int and type(evidence.get('limit')) is int,
             'invalid supervisor call budget')
    attempts = evidence.get('retry_attempts') if type(evidence) is dict else None
    if attempts is None:
        admitted = previous < config.max_supervisor_calls
        _require(not (config.supervisor_max_retries and admitted),
                 'missing supervisor retry attempts')
        count = previous + int(admitted)
        _require(evidence == dict(limit=config.max_supervisor_calls, attempted=count,
                 admitted=admitted, policy=config.supervisor_exhaustion_policy),
                 'supervisor call budget does not match allowance')
        return count, not admitted
    _require(config.supervisor_max_retries > 0 and type(attempts) is list and
             1 <= len(attempts) <= config.supervisor_max_retries + 1,
             'invalid supervisor retry count')
    count = previous + len(attempts)
    _require(count <= config.max_supervisor_calls, 'supervisor retry exceeds allowance')
    last = None
    for attempt in attempts:
        _require(type(attempt) is dict and set(attempt) ==
                 {'started_at', 'finished_at', 'status', 'error_class'},
                 'invalid supervisor retry evidence')
        start, end = attempt['started_at'], attempt['finished_at']
        _require(all(type(t) in (int, float) and isfinite(t) for t in (start, end)) and
                 episode_clock['started_at'] <= start < episode_clock['deadline'] and
                 start <= end and attempt['status'] in
                 ('response', 'error', 'timeout', 'cancelled', 'busy', 'rejected') and
                 (attempt['error_class'] is None or
                  (attempt['status'] == 'error' and
                   attempt['error_class'] in RECOVERABLE_ERRORS)),
                 'invalid supervisor retry timing or disposition')
        if last is not None:
            _require(last['status'] == 'error' and
                     last['error_class'] in config.supervisor_retry_errors and
                     start >= last['finished_at'] + config.supervisor_retry_delay_seconds,
                     'ineligible supervisor retry or missing delay')
        last = attempt
    admitted = evidence.get('admitted')
    _require(type(admitted) is bool and evidence == dict(
        limit=config.max_supervisor_calls, attempted=count, admitted=admitted,
        policy=config.supervisor_exhaustion_policy, retry_attempts=attempts),
        'invalid supervisor retry budget')
    if not admitted:
        _require(count == config.max_supervisor_calls and
                 len(attempts) <= config.supervisor_max_retries and
                 last['status'] == 'error' and last['error_class'] in config.supervisor_retry_errors,
                 'invalid supervisor retry allowance refusal')
    return count, not admitted


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
    wall_limit = manifest.get('wall_clock_limit')
    episode_clock = manifest.get('episode_clock')
    if config.max_episode_seconds is not None:
        _require(type(episode_clock) is dict and set(episode_clock) == {'started_at', 'deadline'} and
                 all(type(v) in (int, float) and isfinite(v) for v in episode_clock.values()) and
                 packets[0].captured_monotonic <= episode_clock['started_at'] < episode_clock['deadline'] and
                 episode_clock['deadline'] == episode_clock['started_at'] + config.max_episode_seconds,
                 'invalid episode wall-clock declaration')
    else:
        _require(episode_clock is None, 'episode clock without configured cap')
    _require(0 <= len(decisions) <= config.max_steps and (decisions or wall_limit is not None),
             'invalid decision count')
    executed = 0
    reward = 0.0
    success = False
    stop = 'step_limit'
    budget = InterventionBudget(config.max_interventions, config.recovery_attempt_limits)
    budget_required = 'max_interventions' in manifest['config']
    cooldown_remaining = 0
    supervisor_calls = 0
    supervisor_exhausted = False
    for index, row in enumerate(decisions):
        _require(type(row['step']) is int and type(row['source_sequence']) is int and
                 row['step'] == index + 1 and row['source_sequence'] == index and
                 row['source_episode_id'] == episode_id, 'invalid decision source/order')
        _require(index < len(packets), 'decision source observation missing')
        record = row['action_record']
        _require(record['proposal_id'] == f'{episode_id}:{index + 1}', 'foreign proposal')
        _require(record['proposed_action'] is not None, 'proposal missing')
        call_budget = record.get('supervisor_call_budget')
        if call_budget is not None:
            supervisor_calls, refused = _supervisor_budget(
                call_budget, config, supervisor_calls, episode_clock)
            admitted = not refused
            supervisor_exhausted = supervisor_exhausted or refused
            attempts = call_budget.get('retry_attempts')
            if attempts:
                returned = record.get('supervisor_pass') or record.get('supervisor_abstention')
                _require(not returned or attempts[-1]['status'] == 'response',
                         'supervisor response without a returned attempt')
                _require(attempts[-1]['finished_at'] < episode_clock['deadline'],
                         'supervisor retry completed outside episode deadline')
            if not admitted:
                fallback = record.get('fallback')
                _require(fallback is not None and
                         fallback['cause'] == 'supervisor call allowance exhausted' and
                         record.get('supervisor_pass') is None and
                         record.get('supervisor_abstention') is None and
                         (config.supervisor_exhaustion_policy != 'stop' or
                          fallback['selected'] == 'refuse'),
                         'supervisor allowance refusal does not match policy')
        elif 'max_supervisor_calls' in manifest['config']:
            _require(record.get('supervisor_pass') is None and
                     record.get('supervisor_abstention') is None and
                     (record.get('fallback') or {}).get('cause') !=
                         'supervisor call allowance exhausted',
                     'missing supervisor call budget')
        recovery = record.get('recovery')
        _require(not (config.correction_mode == 'adjustment_only' and recovery is not None),
                 'recovery disallowed in adjustment_only mode')
        _require(not (config.correction_mode == 'recovery_only' and
                      record['disposition'] == 'overridden' and recovery is None),
                 'adjustment disallowed in recovery_only mode')
        expiry = record.get('correction_expiry')
        expired = expiry is not None and expiry.get('valid') is False
        previous = decisions[index - 1]['action_record'].get('recovery') if index else None
        pending = (previous is not None and
                   previous['check']['status'] == 'continuing' and
                   previous['action_index'] + 1 < len(previous['sequence']['actions']))
        if recovery is not None:
            _require(type(recovery) is dict and set(recovery) == {'sequence', 'action_index', 'check'} and
                     type(recovery['action_index']) is int and
                     type(recovery['sequence']) is dict, 'invalid recovery evidence')
            try:
                plan = RecoverySequence(**recovery['sequence'])
            except (ValueError, TypeError) as exc:
                raise TraceError('invalid recovery sequence') from exc
            offset = recovery['action_index']
            _require(0 <= offset < len(plan.actions) and offset <= index and
                     record['disposition'] == 'overridden' and
                     record['selected_action'] == plan.actions[offset], 'invalid recovery command')
            source = index - offset
            _require(plan.request['episode_id'] == episode_id and
                     plan.request['proposal_id'] == f'{episode_id}:{source + 1}' and
                     plan.request['observation_sequence'] == source and
                     record['proposed_action'] == decisions[source]['action_record']['proposed_action'],
                     'invalid recovery source')
            _require((offset == 0 and not pending) or
                     (pending and previous['sequence'] == recovery['sequence'] and
                      offset == previous['action_index'] + 1), 'invalid recovery ordering')
            _require(source + len(plan.actions) <= config.max_steps, 'recovery exceeds horizon')
            from recovery_monitor import RecoveryAssessment, aborted, check_recovery
            check = recovery['check']
            _require(type(check) is dict and index + 1 < len(packets),
                     'missing recovery check or result observation')
            try:
                assessment = (RecoveryAssessment(**check['assessment'])
                              if check['assessment'] is not None else None)
                if check.get('reason') == 'episode_wall_clock':
                    _require(wall_limit is not None and index == len(decisions) - 1,
                             'recovery cancelled without episode expiry')
                    expected = aborted('episode_wall_clock', offset + 1)
                elif check.get('reason') == 'episode_terminated':
                    _require(type(row['result']) is dict and any(
                        row['result'].get(key) is True
                        for key in ('success', 'terminated', 'truncated')),
                        'recovery cancelled without terminal result')
                    expected = aborted('episode_terminated', offset + 1)
                else:
                    # Historical traces may retain a local check at termination.
                    expected = check_recovery(plan, offset, packets[index + 1], assessment,
                                              check['checked_at'])
            except (ValueError, TypeError, KeyError) as exc:
                raise TraceError('invalid recovery check') from exc
            _require(check == expected, 'recovery check does not match evidence')
        else:
            _require(not pending or (expired and record['disposition'] == 'rejected'),
                     'missing recovery continuation')
        if record['disposition'] == 'overridden' and not pending:
            _require(cooldown_remaining == 0, 'intervention during recovery cooldown')
        reason = record.get('rejection_reason')
        if isinstance(reason, str) and reason.startswith('recovery cooldown:'):
            _require(cooldown_remaining > 0 and reason ==
                     f'recovery cooldown: {cooldown_remaining} baseline actions remaining',
                     'cooldown rejection does not match acknowledged actions')
        accounting = record.get('intervention_budget')
        if budget_required and record['disposition'] == 'rejected' and (
            record.get('rejection_reason') == 'episode intervention limit exhausted' or
            str(record.get('rejection_reason', '')).startswith('recovery attempt limit exhausted: ')
        ):
            _require(accounting is not None, 'missing intervention exhaustion evidence')
        expected_budget = None
        if recovery is not None:
            expected_budget = (budget.admit('recovery', plan.request['tool_name'])
                               if offset == 0 else
                               decisions[index - 1]['action_record'].get('intervention_budget'))
        elif record['disposition'] == 'overridden':
            expected_budget = budget.admit('override')
        elif pending and expired:
            expected_budget = decisions[index - 1]['action_record'].get('intervention_budget')
        elif accounting is not None:
            _require(type(accounting) is dict and record['disposition'] == 'rejected' and
                     accounting.get('kind') in ('recovery', 'override') and
                     ((accounting['kind'] == 'recovery' and
                       type(accounting.get('tool')) is str and accounting['tool'].isidentifier()) or
                      (accounting['kind'] == 'override' and accounting.get('tool') is None)),
                     'invalid exhausted intervention evidence')
            expected_budget = budget.admit(accounting['kind'], accounting['tool'])
            _require(not expected_budget['admitted'] and
                     record['rejection_reason'] == expected_budget['reason'],
                     'intervention exhaustion does not match counts')
        if expected_budget is not None and record['disposition'] != 'rejected':
            _require(expected_budget['admitted'], 'execution exceeds intervention limits')
        if budget_required or accounting is not None:
            _require(_json(accounting) == _json(expected_budget), 'invalid intervention accounting')
        if config.correction_timeout_seconds is not None and (
                record['disposition'] == 'overridden' or
                record.get('rejection_reason') in (
                    'correction request expired', 'correction observation too old')):
            _require(expiry is not None, 'missing correction expiry evidence')
        if expiry is not None:
            from correction_expiry import check_expiry
            _require(config.correction_timeout_seconds is not None,
                     'expiry without declared limits')
            try:
                expected_expiry = check_expiry(config, expiry['requested_at'],
                    expiry['captured_at'], expiry['source_proposal_id'], expiry['checked_at'])
            except (KeyError, TypeError, ValueError) as exc:
                raise TraceError('invalid correction expiry evidence') from exc
            _require(expiry == expected_expiry, 'correction expiry does not match limits')
            if pending:
                previous_expiry = decisions[index - 1]['action_record'].get('correction_expiry')
                _require(previous_expiry is not None and all(
                    expiry[key] == previous_expiry[key] for key in
                    ('requested_at', 'captured_at', 'deadline', 'source_proposal_id')),
                    'recovery renewed correction validity')
                _require(expiry['checked_at'] >= previous_expiry['checked_at'],
                         'recovery expiry clock moved backwards')
            else:
                _require(expiry['source_proposal_id'] == record['proposal_id'] and
                         expiry['captured_at'] == packets[index].captured_monotonic,
                         'expiry does not reference source proposal')
            if expired:
                _require(record['disposition'] == 'rejected' and
                         record['rejection_reason'] == expiry['reason'],
                         'expired correction dispatched')
            elif record['disposition'] == 'rejected':
                _require(accounting is not None and not accounting['admitted'],
                         'valid expiry rejection lacks budget refusal')
            else:
                _require(record['disposition'] == 'overridden' and
                         row['timing']['execution_started_at'] == expiry['checked_at'] and
                         (pending or row['timing']['request_at'] == expiry['requested_at']),
                         'expiry check is not at correction dispatch')
        response = record.get('supervisor_pass')
        if response is not None:
            _require(type(response) is dict and
                     type(response.get('observation_sequence')) is int and
                     valid_temporal_diagnosis(response.get('temporal_diagnosis'), 'pass') and
                     response.get('suppressed_correction') in (None, 'recovery', 'adjustment') and
                     {k: v for k, v in response.items()
                      if k not in ('temporal_diagnosis', 'suppressed_correction')} ==
                     {'kind': 'pass', 'episode_id': episode_id,
                      'observation_sequence': index,
                      'proposal_id': record['proposal_id']} and
                     record['disposition'] == 'unmodified',
                     'invalid supervisor pass evidence')
        abstention = record.get('supervisor_abstention')
        if abstention is not None:
            _require(type(abstention) is dict and response is None and
                     valid_temporal_diagnosis(abstention.get('temporal_diagnosis'), 'abstain') and
                     set(abstention) - {'temporal_diagnosis'} == {'kind', 'diagnosis', 'episode_id',
                                         'observation_sequence', 'proposal_id',
                                         'reason', 'evidence_availability'} and
                     abstention['kind'] == 'abstain' and
                     abstention['diagnosis'] == 'unknown' and
                     abstention['episode_id'] == episode_id and
                     type(abstention['observation_sequence']) is int and
                     abstention['observation_sequence'] == index and
                     abstention['proposal_id'] == record['proposal_id'] and
                     valid_abstention_details(abstention['reason'],
                                              abstention['evidence_availability']) and
                     record['disposition'] in ('unmodified', 'rejected'),
                     'invalid supervisor abstention evidence')
        fallback = record.get('fallback')
        # Historical traces lack the field; new abstentions must retain admission.
        if abstention is not None and 'fallback' in record:
            _require(fallback is not None, 'missing fallback admission evidence')
        if fallback is not None:
            from baseline_fallback import validate_fallback
            try:
                validate_fallback(fallback, ActionProposal(
                    record['proposal_id'], packets[index], record['proposed_action']))
            except (ValueError, TypeError, KeyError) as exc:
                raise TraceError('invalid fallback evidence') from exc
            _require(response is None and recovery is None and accounting is None and
                     ((fallback['selected'] == 'baseline' and record['disposition'] == 'unmodified') or
                      (fallback['selected'] == 'refuse' and record['disposition'] == 'rejected' and
                       record['rejection_reason'] == fallback['reason'])),
                     'fallback selection does not match dispatch')
            if abstention is not None:
                _require(fallback['cause'] == 'supervisor abstained: ' + abstention['reason'],
                         'fallback cause does not match abstention')
            if 'timing' in row:
                _require(row['timing']['request_at'] <= fallback['checked_at'] <=
                         row['timing']['execution_started_at'], 'invalid fallback check time')
        result = row['result']
        interruption = record.get('interruption')
        if interruption is not None:
            from environment_interruption import validate_interruption
            try:
                validate_interruption(interruption, ActionProposal(
                    record['proposal_id'], packets[index], record['proposed_action']),
                    record['rejection_reason'])
            except (ValueError, TypeError, KeyError) as exc:
                raise TraceError('invalid interruption evidence') from exc
            _require(record['disposition'] == 'rejected', 'interruption without refusal')
            if expiry is not None:
                _require(interruption['requested_at'] >= expiry['checked_at'],
                         'interruption precedes expiry check')
            if fallback is not None:
                _require(interruption['requested_at'] >= fallback['checked_at'],
                         'interruption precedes fallback check')
        if record['disposition'] == 'rejected' and 'interruption' in record:
            _require(interruption is not None, 'missing interruption evidence')
        if 'timing' in row:
            timing = StepTiming(**row['timing'])
            if episode_clock is not None:
                _require(episode_clock['started_at'] <= timing.execution_started_at <
                         episode_clock['deadline'], 'dispatch outside episode wall-clock limit')
            _require(timing.capture_at == packets[index].captured_monotonic,
                     'timing does not reference source capture')
        last = index == len(decisions) - 1
        if record['disposition'] == 'rejected':
            _require(last and result is None and bool(record['rejection_reason']) and
                     all(record[key] is None for key in ('selected_action', 'executed_action',
                                                        'execution_acknowledgement')),
                     'invalid rejected decision')
            stop = ('interruption_failed' if interruption is not None and
                    not interruption['confirmed'] else 'proposal_rejected')
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
        if recovery is not None and recovery['check']['status'] == 'completed':
            cooldown_remaining = config.recovery_cooldown_actions
        elif record['disposition'] == 'unmodified':
            cooldown_remaining = max(0, cooldown_remaining - 1)
        executed += 1
        reward += result['reward']
        success = result['success']
        if success or result['terminated'] or result['truncated']:
            _require(last, 'decisions after terminal result')
            stop = 'success' if success else 'terminated' if result['terminated'] else 'truncated'
        elif recovery is not None and recovery['check']['status'] == 'aborted':
            _require(last, 'decisions after recovery abort')
            stop = 'recovery_aborted'
    if wall_limit is not None:
        from environment_interruption import validate_interruption
        _require(type(wall_limit) is dict and set(wall_limit) == {
            'limit_seconds', 'started_at', 'deadline', 'expired_at', 'stage',
            'interruption', 'pending_supervisor_call'}, 'invalid wall-clock evidence')
        _require(config.max_episode_seconds is not None and
                 type(wall_limit['limit_seconds']) in (int, float) and
                 wall_limit['limit_seconds'] == config.max_episode_seconds and
                 wall_limit['started_at'] == episode_clock['started_at'] and
                 wall_limit['deadline'] == episode_clock['deadline'] and
                 all(type(wall_limit[k]) in (int, float) and isfinite(wall_limit[k])
                     for k in ('started_at', 'deadline', 'expired_at')) and
                 wall_limit['deadline'] == wall_limit['started_at'] + config.max_episode_seconds and
                 packets[0].captured_monotonic <= wall_limit['started_at'] <
                 wall_limit['deadline'] <= wall_limit['expired_at'] and
                 wall_limit['stage'] in ('policy', 'assessment_trigger', 'supervisor',
                     'selection', 'fallback', 'dispatch', 'recovery', 'after_execution',
                     'on_step', 'resume'), 'wall-clock evidence does not match limit')
        _require(executed == len(decisions) and not success and
                 (not decisions or not any(decisions[-1]['result'][k]
                    for k in ('success', 'terminated', 'truncated'))),
                 'wall-clock expiry follows terminal decision')
        for row in decisions:
            _require('timing' in row and
                     wall_limit['started_at'] <= row['timing']['execution_started_at'] <
                     wall_limit['deadline'], 'dispatch outside episode wall-clock limit')
        evidence = wall_limit['interruption']
        validate_interruption(evidence, ActionProposal(f'{episode_id}:{executed + 1}',
                              packets[-1], None), 'episode wall-clock limit exhausted')
        _require(wall_limit['deadline'] <= evidence['requested_at'] <=
                 evidence['finished_at'] <= wall_limit['expired_at'],
                 'interruption precedes episode wall-clock expiry')
        pending_call = wall_limit['pending_supervisor_call']
        if pending_call is not None:
            supervisor_calls, refused = _supervisor_budget(
                pending_call, config, supervisor_calls, episode_clock)
            supervisor_exhausted = supervisor_exhausted or refused
        stop = 'wall_clock_limit' if evidence['confirmed'] else 'interruption_failed'
    _require(len(packets) == executed + 1, 'missing or extra observation artifacts')
    _require(stop != 'step_limit' or executed == config.max_steps, 'unfinished trace')
    _require(manifest['outcome'] == dict(success=success, steps=executed,
                                       stop_reason=stop, sum_rewards=reward),
             'outcome does not match recorded sequence')
    summary = dict(limit=config.max_supervisor_calls, attempted=supervisor_calls,
                   exhausted=supervisor_exhausted, policy=config.supervisor_exhaustion_policy)
    if 'max_supervisor_calls' in manifest['config']:
        _require(manifest.get('supervisor_call_budget') == summary,
                 'supervisor call budget outcome mismatch')
    return RecordedReplay(episode_id, config, packets, decisions, manifest['outcome'],
                          manifest['config'], manifest.get('supervisor_call_budget'), wall_limit, episode_clock)


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

    def __init__(self, episode_id, config, packets, decisions, outcome, recorded_config=None,
                 supervisor_call_budget=None, wall_clock_limit=None, episode_clock=None):
        self._episode_clock = deepcopy(episode_clock)
        self._wall_clock_limit = deepcopy(wall_clock_limit)
        self.source_episode_id = episode_id
        self.config = config
        self._packets = packets
        self._decisions = deepcopy(decisions)
        self._outcome = deepcopy(outcome)
        self._supervisor_call_budget = deepcopy(supervisor_call_budget)
        self._recorded_config = asdict(config)
        # Historical annotation fingerprints include pre-existing config defaults,
        # but must not acquire fields introduced after the trace was sealed.
        if recorded_config is not None:
            for name in ('max_interventions', 'recovery_attempt_limits', 'recovery_cooldown_actions',
                         'correction_timeout_seconds', 'correction_max_age_seconds',
                         'max_supervisor_calls', 'supervisor_exhaustion_policy', 'max_episode_seconds',
                         'supervisor_max_retries', 'supervisor_retry_delay_seconds',
                         'supervisor_retry_errors', 'correction_mode'):
                if name not in recorded_config:
                    self._recorded_config.pop(name)

    def evidence(self):
        """Return detached, JSON-compatible historical evidence for inspection.

        Includes private evaluator observations; this is not a supervisor input.
        Timestamps retain their original meaning, not replay execution timing.
        """
        result = {
            'source_episode_id': self.source_episode_id,
            'config': self._recorded_config,
            'observations': [asdict(packet) for packet in self._packets],
            'decisions': self._decisions,
            'outcome': self._outcome,
        }
        if self._episode_clock is not None:
            result['episode_clock'] = self._episode_clock
        if self._wall_clock_limit is not None:
            result['wall_clock_limit'] = self._wall_clock_limit
        return _plain(result)

    def report(self):
        """Execute validated replay and label diagnostic scope explicitly."""
        return dict(source_episode_id=self.source_episode_id,
                    correction_mode=self.config.correction_mode,
                    optional_ablation=self.config.correction_mode != 'combined',
                    primary_acceptance_evidence=False,
                    replay_outcome=_summary(self.run()))

    def run(self, recorder=None):
        packets, decisions = deepcopy((self._packets, self._decisions))
        config = self.config
        # Historical timestamps are evidence, not measurements of replay latency.
        wall_limit = self._wall_clock_limit
        now = (self._episode_clock['started_at'] if self._episode_clock else
               packets[0].captured_monotonic)
        from environment_interruption import InterruptionContract
        interruption = (wall_limit['interruption'] if wall_limit else
                        decisions[-1]['action_record'].get('interruption'))
        contract = (InterruptionContract(**interruption['contract']) if interruption else
                    InterruptionContract('recorded-replay-stop-v1', 'stop',
                        'Stop scripted playback without a controller.'))

        class Policy:
            def reset(self):
                pass

            def resume(self, packet):
                # Recorded proposals are indexed by accepted observation, not queued.
                _require(packet.observation == packets[packet.sequence].observation,
                         'replay resume observation diverged')

            def act(self, packet):
                nonlocal now
                if wall_limit and packet.sequence == len(decisions):
                    now = wall_limit['expired_at']
                    return None
                expiry = decisions[packet.sequence]['action_record'].get('correction_expiry')
                if expiry is not None:
                    now = expiry['requested_at']
                _require(packet.observation == packets[packet.sequence].observation,
                         'replay observation diverged')
                return deepcopy(decisions[packet.sequence]['action_record']['proposed_action'])

        class Environment:
            interruption_contract = contract

            def interrupt(self, request):
                _require(request.episode_id == self.episode_id and
                         request.operation == contract.operation and
                         request.observation_sequence == self.index,
                         'replay interruption diverged')
                record = decisions[self.index]['action_record'] if self.index < len(decisions) else {}
                evidence = wall_limit['interruption'] if wall_limit else record.get('interruption')
                if evidence is not None:
                    _require(request.reason == evidence['request']['reason'],
                             'replay interruption reason diverged')
                return evidence['confirmed'] if evidence is not None else True

            def reset(self, seed, episode_id):
                _require(seed == config.seed, 'replay seed mismatch')
                self.episode_id, self.index = episode_id, 0
                return replace(deepcopy(packets[0]), episode_id=episode_id)

            def step(self, action):
                nonlocal now
                row = decisions[self.index]
                _require(action == row['action_record']['executed_action'],
                         'replay action diverged')
                self.index += 1
                recovery = row['action_record'].get('recovery')
                now = (recovery['check']['checked_at'] if recovery is not None
                       and recovery['check']['checked_at'] is not None
                       else max(now, packets[self.index].captured_monotonic))
                return StepResult(replace(deepcopy(packets[self.index]),
                                          episode_id=self.episode_id), **row['result'])

        def select(proposal):
            nonlocal now
            record = decisions[proposal.observation.sequence]['action_record']
            if record.get('correction_expiry') is not None:
                now = record['correction_expiry']['checked_at']
            if record.get('recovery') is not None:
                data = deepcopy(record['recovery']['sequence'])
                data['request']['episode_id'] = proposal.observation.episode_id
                data['request']['proposal_id'] = proposal.proposal_id
                return ActionResolution('recovery', recovery=RecoverySequence(**data))
            if record['disposition'] == 'rejected':
                return ActionResolution('reject', reason=record['rejection_reason'])
            if record['disposition'] == 'overridden':
                return ActionResolution('override', deepcopy(record['selected_action']))
            return ActionResolution('pass')

        def observe(plan, offset, packet):
            nonlocal now
            if wall_limit and packet.sequence == len(decisions) and wall_limit['stage'] == 'recovery':
                now = wall_limit['expired_at']
                return None
            from recovery_monitor import RecoveryAssessment
            data = deepcopy(decisions[packet.sequence - 1]['action_record']['recovery']
                            ['check']['assessment'])
            if data is None:
                return None
            if data['episode_id'] == self.source_episode_id:
                data['episode_id'] = packet.episode_id
            return RecoveryAssessment(**data)

        def after_step(step, result):
            nonlocal now
            if wall_limit and step == len(decisions):
                now = wall_limit['expired_at']
            if step < len(decisions):
                expiry = decisions[step]['action_record'].get('correction_expiry')
                if expiry is not None:
                    now = expiry['checked_at']

        outcome = run_episode(replace(config, supervisor_max_retries=0), Policy(), Environment(),
                              recorder if recorder is not None else ReplayRecorder(),
                              clock=lambda: now, action_selector=select,
                              recovery_observer=observe, on_step=after_step)
        _require(_summary(outcome) == self._outcome, 'replayed outcome diverged')
        return replace(outcome, supervisor_call_budget=deepcopy(self._supervisor_call_budget),
                       wall_clock_limit=deepcopy(self._wall_clock_limit),
                       episode_clock=deepcopy(self._episode_clock))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='sealed replay directory containing manifest.json')
    args = parser.parse_args()
    replay = load_recorded_replay(args.directory)
    print(_json(replay.report()))
