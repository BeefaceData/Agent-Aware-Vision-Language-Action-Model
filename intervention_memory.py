"""Offline, append-only intervention memory derived from pinned episode evidence.

Raw records are evaluator storage; retrieve_context prepares bounded summaries.
Exact-context candidate lookup never filters on success or supplies authority.
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
from temporal_diagnosis import CATEGORIES, valid_temporal_diagnosis


def _bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def _require(condition, reason):
    if not condition:
        raise TraceError(reason)


class InterventionMemory:
    """One exclusively created record per episode/proposal, with verified reads.

    append takes four (path, expected_sha256) pairs: a sealed trace manifest,
    pre-action attempt identity, frozen supervisor identity, and decision context.
    Context is {episode_id, proposal_id, observation_sequence, diagnosis, request},
    optionally with a host-declared progress_context (nonempty string mapping).
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

    def _build(self, provenance, version=4):
        _require(type(version) is int and version in (1, 2, 3, 4),
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
        required_context = {
            'episode_id', 'proposal_id', 'observation_sequence', 'diagnosis', 'request'}
        allowed_contexts = [required_context]
        if version >= 4:
            allowed_contexts.append(required_context | {'progress_context'})
        _require(type(context) is dict and set(context) in allowed_contexts,
            'complete decision context required')
        if 'progress_context' in context:
            _validate_progress(context['progress_context'])
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
        if version >= 2:
            # Execution completion is not evidence of causal task benefit.
            local = {'status': 'unknown', 'reason': 'no_local_assessment'}
            if recovery is not None:
                check = rows[-1]['action_record']['recovery']['check']
                local = {'status': check['status'], 'reason': check['reason']}
                if check['status'] == 'continuing':
                    local = {'status': 'unknown', 'reason': 'no_terminal_local_check'}
            record['local_outcome'] = local
        if version >= 3:
            outcome = evidence['outcome']
            # Match EpisodeOutcome.task_status: early stops do not establish
            # task failure, and successful tasks do not establish local benefit.
            status = ('success' if outcome['success'] else
                      'failure' if outcome['stop_reason'] in ('terminated', 'step_limit')
                      else 'unknown')
            record['task_outcome'] = {'status': status, 'reason': outcome['stop_reason']}
            limitations = ['causal_benefit_unverified']
            if diagnosis['category'] == 'unknown':
                limitations.append('diagnosis_unknown')
            if diagnosis.get('conflicts'):
                limitations.append('diagnosis_conflicting_evidence')
            if local['status'] == 'unknown':
                limitations.append('local_outcome_unverified')
            if recovery is not None:
                check = rows[-1]['action_record']['recovery']['check']
                if check['assessment'] is None:
                    limitations.append('local_assessment_missing')
                if check['reason'] == 'stale_observation':
                    limitations.append('local_assessment_stale')
            if status == 'unknown':
                limitations.append('task_outcome_unverified')
            record['evidence_limitations'] = limitations
        if version >= 4:
            record['progress_context'] = context.get('progress_context')
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

    def candidates(self, references, *, task, robot_capabilities):
        """Read pinned candidates for an exact task/control context, in input order.

        Outcome-neutral evaluator lookup, not supervisor-ready retrieval. The
        caller supplies the permitted reference set; every reference is verified
        before filtering. Model compatibility, splits, ranking and redaction
        belong to the retrieval policy. Legacy records retain their own schema.
        """
        _require(type(task) is dict and bool(task) and
                 type(robot_capabilities) is dict and bool(robot_capabilities),
                 'explicit task and robot context required')
        records = []
        for reference in references:
            try:
                record = self.read(reference['record_id'],
                                   expected_sha256=reference['sha256'])
            except (TypeError, KeyError) as exc:
                raise TraceError('invalid memory candidate reference') from exc
            if (record['task'] == task and
                    record['robot_capabilities'] == robot_capabilities):
                records.append(record)
        return records

    def filter_candidates(self, references, *, task, robot_capabilities, progress_context):
        """Return verified candidates and pinned exclusions with ordered reasons.

        Conservative exact compatibility: no implicit frame conversion, arm
        remapping, range widening or task synonym inference. Progress must be
        explicitly declared at decision time; missing evidence is excluded.
        This outcome-neutral result is evaluator data, not supervisor input.
        """
        _require(type(task) is dict and bool(task) and
                 isinstance(task.get('instruction'), str) and
                 bool(task['instruction'].strip()), 'explicit task instruction required')
        _validate_progress(progress_context)
        active = _capabilities(robot_capabilities)
        candidates, excluded = [], []
        for reference in references:
            try:
                record = self.read(reference['record_id'],
                                   expected_sha256=reference['sha256'])
            except (TypeError, KeyError) as exc:
                raise TraceError('invalid memory candidate reference') from exc
            reasons = []
            if record['task'] != task:
                reasons.append('task_mismatch')
            progress = record.get('progress_context')
            if progress is None:
                reasons.append('progress_context_missing')
            elif progress != progress_context:
                reasons.append('progress_context_mismatch')
            retained = _capabilities(record['robot_capabilities'])
            for field in ('layout', 'control_frequency_hz'):
                if getattr(retained, field) != getattr(active, field):
                    reasons.append(field + '_mismatch')
            if set(retained.operations) != set(active.operations):
                reasons.append('operations_mismatch')
            if len(retained.components) != len(active.components):
                reasons.append('component_count_mismatch')
            else:
                for field in ('name', 'arm', 'group', 'frame', 'representation',
                              'unit', 'scale', 'minimum', 'maximum'):
                    if any(getattr(old, field) != getattr(new, field)
                           for old, new in zip(retained.components, active.components)):
                        reasons.append(field + '_mismatch')
            if reasons:
                excluded.append({'record_id': reference['record_id'],
                                 'sha256': reference['sha256'], 'reasons': reasons})
            else:
                candidates.append(record)
        return {'candidates': candidates, 'excluded': excluded}

    def rank_candidates(self, references, *, task, robot_capabilities,
                        progress_context, failure_category):
        """Rank compatible pinned evidence without preferring positive outcomes.

        Task/progress/control compatibility is a hard gate. Within that context,
        exact diagnosis-category matches precede other categories; record ID is
        the ascending tie-breaker. Unknown is a category, never a wildcard.
        Return evaluator records, exclusions and the reproducible ranking rule.
        """
        _require(type(failure_category) is str and failure_category in CATEGORIES,
                 'declared diagnosis category required for ranking')
        filtered = self.filter_candidates(references, task=task,
            robot_capabilities=robot_capabilities, progress_context=progress_context)
        # Verification above includes duplicates and excluded records. A repeated
        # reference must not occupy multiple places in the relevance ordering.
        unique = {record['record_id']: record for record in filtered['candidates']}
        candidates = sorted(unique.values(), key=lambda record: (
            record['diagnosis']['category'] != failure_category, record['record_id']))
        return {'candidates': candidates,
                'excluded': sorted(filtered['excluded'], key=lambda item: item['record_id']),
                'ranking': {
                    'policy': 'exact-context-failure-v1',
                    'failure_category': failure_category,
                    'progress_context': dict(progress_context),
                    'compatibility': 'exact-task-progress-control',
                    'order': ['failure_category_match_desc', 'record_id_asc'],
                    'outcome_preference': 'none'}}

    def retrieve_context(self, references, *, task, robot_capabilities,
                         progress_context, failure_category, max_entries,
                         max_summary_bytes, max_context_bytes):
        """Build bounded historical summaries; only context_json is model input.

        Budgets count UTF-8 bytes of compact, ASCII-escaped JSON, including list
        brackets and commas. Whole summaries are skipped, never truncated.
        Caller must supply permitted development references, not held-out data.
        This prepares context without enabling retrieval in a runtime policy.
        """
        for name, value, minimum in (
                ('max_entries', max_entries, 0),
                ('max_summary_bytes', max_summary_bytes, 0),
                ('max_context_bytes', max_context_bytes, 2)):
            _require(type(value) is int and value >= minimum,
                     f'{name} must be an integer >= {minimum}')
        references = list(references)
        ranked = self.rank_candidates(references, task=task,
            robot_capabilities=robot_capabilities, progress_context=progress_context,
            failure_category=failure_category)
        pins = {ref['record_id']: ref['sha256'] for ref in references}
        selected, omitted = [], []
        used = 2  # The empty JSON list is part of the allowance.
        for record in ranked['candidates']:
            summary = {
                'record_id': record['record_id'], 'sha256': pins[record['record_id']],
                'evidence_scope': 'historical_experience',
                'failure_category': record['diagnosis']['category'],
                'intervention_kind': ('recovery' if record['execution'][0][
                    'action_record'].get('recovery') is not None else 'adjustment'),
                'local_outcome': record.get('local_outcome', {
                    'status': 'unknown', 'reason': 'legacy_outcome_unavailable'}),
                'task_outcome': {'status': 'withheld', 'reason': 'evaluator_only'},
                'evidence_limitations': record.get('evidence_limitations', [
                    'causal_benefit_unverified', 'legacy_qualifications_unavailable'])}
            size = len(_context_json(summary).encode('utf-8'))
            addition = size + bool(selected)
            reason = ('entry_limit' if len(selected) >= max_entries else
                      'summary_size_limit' if size > max_summary_bytes else
                      'context_size_limit' if used + addition > max_context_bytes else None)
            if reason:
                omitted.append({'record_id': summary['record_id'],
                                'sha256': summary['sha256'], 'reason': reason})
            else:
                selected.append(summary)
                used += addition
        return {'context_json': _context_json(selected),
                'context_bytes': used, 'selected': selected, 'omitted': omitted,
                'excluded': ranked['excluded'], 'ranking': ranked['ranking'],
                'settings': {'policy': 'whole-historical-summary-v1',
                    'max_entries': max_entries, 'max_summary_bytes': max_summary_bytes,
                    'max_context_bytes': max_context_bytes,
                    'size_unit': 'utf8_bytes', 'oversize_policy': 'skip_in_rank_order'}}


def _context_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(',', ':'))


def _validate_progress(value):
    _require(type(value) is dict and bool(value) and
             all(type(key) is str and key.strip() and
                 type(item) is str and item.strip() for key, item in value.items()),
             'explicit progress context requires nonempty string identifiers')


def _capabilities(value):
    try:
        _require(type(value) is dict and set(value) == {
            'components', 'control_frequency_hz', 'operations', 'layout'},
            'complete robot capabilities required')
        _require(isinstance(value['operations'], (list, tuple)),
                 'robot operations must be a sequence')
        return ActionCapabilities(
            tuple(ActionComponent(**item) for item in value['components']),
            value['control_frequency_hz'], tuple(value['operations']), value['layout'])
    except (ValueError, TypeError, KeyError) as exc:
        raise TraceError(f'invalid robot context: {exc}') from exc
