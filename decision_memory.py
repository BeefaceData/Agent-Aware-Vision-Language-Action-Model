"""Host-owned memory provenance, separate from model-authored response JSON."""

from copy import deepcopy
from hashlib import sha256
import json


def disabled_memory():
    return {'retrieval': 'disabled', 'snapshot': None, 'selected': [],
            'context_json': '[]', 'context_sha256': sha256(b'[]').hexdigest(),
            'settings': None, 'ranking': None, 'query': None,
            'references': [], 'excluded': [], 'omitted': []}


class DecisionMemory:
    """Explicit permitted references and host-declared query for each proposal.

    query(proposal) returns retrieve_context keyword arguments. snapshot is an
    optional externally pinned identity, not verification of snapshot membership.
    The caller owns permission, split selection and snapshot policy.
    """

    def __init__(self, store, references, query, *, snapshot=None):
        if not callable(query):
            raise ValueError('memory query callback required')
        _validate_snapshot(snapshot)
        self._store = store
        self._references = deepcopy(list(references))
        self._query = query
        self._snapshot = deepcopy(snapshot)

    def prepare(self, proposal):
        query = deepcopy(self._query(deepcopy(proposal)))
        result = self._store.retrieve_context(self._references, **query)
        record = dict(retrieval='enabled', snapshot=deepcopy(self._snapshot),
            query=query, references=deepcopy(self._references),
            **{key: deepcopy(result[key]) for key in (
                'selected', 'context_json', 'settings', 'ranking', 'excluded', 'omitted')})
        record['context_sha256'] = sha256(record['context_json'].encode('utf-8')).hexdigest()
        validate_memory(record)
        return record


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate_snapshot(snapshot):
    if snapshot is not None and (type(snapshot) is not dict or
            set(snapshot) != {'snapshot_id', 'sha256'} or
            type(snapshot['snapshot_id']) is not str or not snapshot['snapshot_id'] or
            not _digest(snapshot['sha256'])):
        raise ValueError('invalid declared snapshot pin')


def validate_memory(record):
    """Validate retained context integrity without blessing source permissions."""
    try:
        if type(record) is not dict or set(record) != set(disabled_memory()):
            raise ValueError('invalid memory fields')
        if record['retrieval'] == 'disabled':
            if record != disabled_memory():
                raise ValueError('disabled memory contains retrieval evidence')
            return
        if record['retrieval'] != 'enabled':
            raise ValueError('invalid retrieval mode')
        _validate_snapshot(record['snapshot'])
        if any(type(record[key]) is not list for key in ('references', 'excluded', 'omitted')):
            raise ValueError('invalid retrieval audit lists')
        context = record['context_json']
        selected = record['selected']
        if (type(context) is not str or type(selected) is not list or
                json.dumps(selected, sort_keys=True, ensure_ascii=True, allow_nan=False,
                           separators=(',', ':')) != context or
                sha256(context.encode('utf-8')).hexdigest() != record['context_sha256']):
            raise ValueError('memory context mismatch')
        settings = record['settings']
        if type(settings) is not dict or type(record['query']) is not dict or type(record['ranking']) is not dict:
            raise ValueError('missing retrieval settings')
        for key, minimum in (('max_entries', 0), ('max_summary_bytes', 0), ('max_context_bytes', 2)):
            if type(settings[key]) is not int or settings[key] < minimum or record['query'][key] != settings[key]:
                raise ValueError('invalid memory budget')
        if len(selected) > settings['max_entries'] or len(context.encode()) > settings['max_context_bytes']:
            raise ValueError('memory budget exceeded')
        pins = set()
        for ref in record['references']:
            if (type(ref) is not dict or set(ref) != {'record_id', 'sha256'} or
                    type(ref['record_id']) is not str or not ref['record_id'] or
                    not _digest(ref['sha256'])):
                raise ValueError('invalid memory reference')
            pins.add((ref['record_id'], ref['sha256']))
        seen = set()
        for summary in selected:
            if (set(summary) != {'record_id', 'sha256', 'evidence_scope', 'failure_category',
                    'intervention_kind', 'local_outcome', 'task_outcome', 'evidence_limitations'} or
                    (summary['record_id'], summary['sha256']) not in pins or
                    summary['record_id'] in seen or
                    summary['evidence_scope'] != 'historical_experience' or
                    summary['task_outcome'] != {'status': 'withheld', 'reason': 'evaluator_only'} or
                    'causal_benefit_unverified' not in summary['evidence_limitations'] or
                    len(json.dumps(summary, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                   separators=(',', ':')).encode()) > settings['max_summary_bytes']):
                raise ValueError('invalid selected memory summary')
            seen.add(summary['record_id'])
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('invalid decision memory') from exc
