"""Freeze and verify a declared evaluation protocol before using its results."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re


class ProtocolError(ValueError):
    """A protocol is incomplete or differs from its retained reference."""


_FIELDS = {'conditions', 'splits', 'seeds', 'allocation', 'horizons',
           'analysis', 'outcomes'}


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def _digest(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _validate_splits(splits):
    _require(set(splits) == {'development', 'held_out'},
             'development and held_out splits required')
    seen_ids, seen_digests = {}, {}
    for split, members in splits.items():
        _require(type(members) is list and bool(members),
                 f'nonempty {split} membership required')
        for member in members:
            _require(type(member) is dict and set(member) == {
                'suite', 'task_id', 'state_id', 'selected_sha256', 'digest_encoding'},
                f'invalid {split} task/state identity')
            suite, task, state = member['suite'], member['task_id'], member['state_id']
            _require(suite == 'libero_10' and type(task) is int and 0 <= task < 10
                     and type(state) is int and state >= 0
                     and _digest(member['selected_sha256'])
                     and member['digest_encoding'] == 'little-endian-float64-vector-v1',
                     f'invalid {split} task/state identity')
            by_id = (suite, task, state)
            by_digest = (suite, task, member['selected_sha256'])
            _require(by_id not in seen_ids and by_digest not in seen_digests,
                     f'duplicate or overlapping task/state membership: {split}')
            seen_ids[by_id] = split
            seen_digests[by_digest] = split
    covered = {member['task_id'] for member in splits['held_out']}
    _require(covered == set(range(10)),
             f'held_out missing LIBERO-10 tasks: {sorted(set(range(10)) - covered)}')


def _validate(document):
    _require(type(document) is dict and set(document) == _FIELDS | {'version'},
             'complete protocol fields required')
    _require(type(document['version']) is int and document['version'] == 1,
             'unsupported protocol version')
    for field in _FIELDS:
        _require(type(document[field]) is dict and bool(document[field]),
                 f'nonempty protocol {field} required')
    _require(all(type(name) is str and name.strip() for name in document['conditions']),
             'named conditions required')
    _require(all(type(name) is str and name.strip() for name in document['splits']),
             'named splits required')
    _validate_splits(document['splits'])
    _require(all(type(name) is str and name.strip() for name in document['seeds']),
             'named seeds required')
    _require(all(type(name) is str and name.strip() for name in document['allocation']),
             'named allocation required')
    _require(all(type(name) is str and name.strip() for name in document['horizons']),
             'named horizons required')
    _require(all(type(name) is str and name.strip() for name in document['analysis']),
             'named analysis settings required')
    _require(all(type(name) is str and name.strip() for name in document['outcomes']),
             'named outcome rules required')
    try:
        _encode(document)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f'protocol must contain finite JSON values: {exc}') from exc


class EvaluationProtocol:
    """An exclusive on-disk freeze with externally retained identity and digest.

    The caller declares scientific choices; this object seals and verifies them.
    It does not infer a trial allocation or authorize running an evaluation.
    """

    def __init__(self, path, *, expected_reference):
        _require(type(expected_reference) is dict and
                 set(expected_reference) == {'protocol_id', 'sha256'} and
                 all(_digest(value) for value in expected_reference.values()),
                 'external protocol identity and digest required')
        self._path = Path(path).resolve()
        self._reference = dict(expected_reference)
        self.read()

    @classmethod
    def freeze(cls, path, *, conditions, splits, seeds, allocation, horizons,
               analysis, outcomes):
        document = {'version': 1, 'conditions': conditions, 'splits': splits,
                    'seeds': seeds, 'allocation': allocation, 'horizons': horizons,
                    'analysis': analysis, 'outcomes': outcomes}
        _validate(document)
        # Detach from caller-owned mutable objects and use canonical bytes.
        document = json.loads(_encode(document))
        identity = sha256(_encode(document)).hexdigest()
        raw = _encode(dict(document, protocol_id=identity))
        path = Path(path).resolve()
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return cls(path, expected_reference={
            'protocol_id': identity, 'sha256': sha256(raw).hexdigest()})

    @property
    def reference(self):
        return dict(self._reference)

    def read(self):
        try:
            raw = self._path.read_bytes()
            _require(sha256(raw).hexdigest() == self._reference['sha256'],
                     'protocol manifest digest mismatch')
            document = json.loads(raw)
            _require(type(document) is dict, 'invalid protocol manifest')
            identity = document.pop('protocol_id')
            _validate(document)
            _require(identity == self._reference['protocol_id'] ==
                     sha256(_encode(document)).hexdigest(),
                     'protocol identity mismatch')
            _require(self._path.read_bytes() == raw,
                     'protocol changed during verification')
            return dict(document, protocol_id=identity)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            if isinstance(exc, ProtocolError):
                raise
            raise ProtocolError(f'invalid evaluation protocol: {exc}') from exc
