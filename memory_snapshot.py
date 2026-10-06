"""Immutable manifests over explicitly selected, evidence-backed memory records.

Retain the manifest pin separately and preserve the common evidence/store tree.
This seals membership and configuration; it does not certify development splits
or enforce fixed-evaluation access policy.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
import re

from intervention_memory import InterventionMemory
from recorded_replay import TraceError


def _require(condition, message):
    if not condition:
        raise TraceError(message)


def _encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _validate(document):
    _require(type(document) is dict and set(document) == {
        'version', 'store', 'records', 'metadata', 'retrieval'}, 'invalid snapshot fields')
    _require(type(document['version']) is int and document['version'] == 1,
             'unsupported snapshot version')
    location = document['store']
    _require(type(location) is str and bool(location) and
             not Path(location).is_absolute() and '\\' not in location and ':' not in location,
             'snapshot requires a relative store path')
    _require(type(document['metadata']) is dict and bool(document['metadata']),
             'nonempty snapshot metadata required')
    configuration = document['retrieval']
    _require(type(configuration) is dict and set(configuration) == {
        'ranking_policy', 'summary_policy', 'max_entries', 'max_summary_bytes',
        'max_context_bytes'}, 'complete snapshot retrieval configuration required')
    _require(configuration['ranking_policy'] == 'exact-context-failure-v1' and
             configuration['summary_policy'] == 'whole-historical-summary-v1',
             'unsupported snapshot retrieval policy')
    for key, minimum in (('max_entries', 0), ('max_summary_bytes', 0), ('max_context_bytes', 2)):
        _require(type(configuration[key]) is int and configuration[key] >= minimum,
                 f'invalid snapshot retrieval budget: {key}')
    records = document['records']
    _require(type(records) is list, 'snapshot records must be a list')
    previous = ''
    for reference in records:
        _require(type(reference) is dict and set(reference) == {'record_id', 'sha256'} and
                 _digest(reference['record_id']) and _digest(reference['sha256']),
                 'invalid snapshot record pin')
        _require(reference['record_id'] > previous, 'snapshot records must be sorted and unique')
        previous = reference['record_id']
    _encode(document)  # Reject non-JSON metadata and nonfinite numbers.


def _verify_records(path, document):
    store = InterventionMemory(path.parent / document['store'])
    for reference in document['records']:
        try:
            store.read(reference['record_id'], expected_sha256=reference['sha256'])
        except TraceError as exc:
            raise TraceError(f"snapshot record {reference['record_id']}: {exc}") from exc


class MemorySnapshot:
    """Create once and verify on every read, requiring an externally retained pin.

    The content-derived snapshot_id covers sorted record pins (which bind all
    record metadata and source evidence), declared metadata, retrieval settings
    and the relative store location. sha256 additionally pins the manifest bytes.
    No filesystem scan, copying, update operation or runtime mode is implied.
    """

    def __init__(self, path, *, expected_reference):
        _require(type(expected_reference) is dict and
                 set(expected_reference) == {'snapshot_id', 'sha256'} and
                 all(_digest(value) for value in expected_reference.values()),
                 'external snapshot identity and digest required')
        self._path = Path(path).resolve()
        self._reference = dict(expected_reference)
        self.read()

    @classmethod
    def freeze(cls, path, store, references, *, metadata, retrieval):
        """Verify all supplied pins, then exclusively write a canonical manifest."""
        path = Path(path).resolve()
        try:
            references = list(references)
            # Validate before sorting, including duplicates and malformed pins.
            for ref in references:
                _require(type(ref) is dict and set(ref) == {'record_id', 'sha256'} and
                         _digest(ref['record_id']) and _digest(ref['sha256']),
                         'invalid snapshot record pin')
            document = json.loads(_encode({
                'version': 1,
                'store': os.path.relpath(store.directory, path.parent).replace('\\', '/'),
                'records': sorted(references, key=lambda ref: ref['record_id']),
                'metadata': metadata, 'retrieval': retrieval}))
            _validate(document)
            _verify_records(path, document)
            identity = sha256(_encode(document)).hexdigest()
            raw = _encode(dict(document, snapshot_id=identity))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise TraceError(f'invalid memory snapshot: {exc}') from exc
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return cls(path, expected_reference={
            'snapshot_id': identity, 'sha256': sha256(raw).hexdigest()})

    @property
    def reference(self):
        return dict(self._reference)

    def read(self):
        """Return detached manifest data after checking every member's evidence."""
        try:
            raw = self._path.read_bytes()
            _require(sha256(raw).hexdigest() == self._reference['sha256'],
                     'snapshot manifest digest mismatch')
            document = json.loads(raw)
            _require(type(document) is dict, 'invalid snapshot manifest')
            identity = document.pop('snapshot_id')
            _validate(document)
            _require(identity == self._reference['snapshot_id'] ==
                     sha256(_encode(document)).hexdigest(), 'snapshot identity mismatch')
            _verify_records(self._path, document)
            _require(self._path.read_bytes() == raw, 'snapshot changed during verification')
            return dict(document, snapshot_id=identity)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise TraceError(f'invalid memory snapshot: {exc}') from exc
