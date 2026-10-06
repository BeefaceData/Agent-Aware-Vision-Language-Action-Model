"""Build an auditable development-only snapshot from declared episode membership."""

import json

from memory_snapshot import MemorySnapshot, _digest, _encode, _require
from recorded_replay import TraceError


def freeze_development_snapshot(path, store, references, *, membership, compatibility,
                                retrieval):
    """Seal admitted pins and the complete selection report in snapshot metadata.

    membership declares a revision and episode_id -> development/held_out/unknown.
    compatibility declares exact policy, supervisor and settings identities from
    verified records. Episode seed/horizon are not configuration compatibility.
    Declarations are host evidence, not inferred split labels or authorization.
    """
    try:
        membership = json.loads(_encode(membership))
        compatibility = json.loads(_encode(compatibility))
        _require(type(membership) is dict and set(membership) == {'revision', 'episodes'}
                 and type(membership['revision']) is str and membership['revision'].strip(),
                 'declared development membership revision required')
        episodes = membership['episodes']
        _require(type(episodes) is dict and all(
            type(key) is str and key and value in ('development', 'held_out', 'unknown')
            for key, value in episodes.items()), 'invalid episode split membership')
        _require(type(compatibility) is dict and set(compatibility) == {
            'policy', 'supervisor', 'settings'} and all(
                type(value) is dict and value for value in compatibility.values()),
                 'explicit model and settings compatibility required')
        references = list(references)
        seen = set()
        for ref in references:
            _require(type(ref) is dict and set(ref) == {'record_id', 'sha256'} and
                     all(_digest(value) for value in ref.values()), 'invalid candidate pin')
            _require(ref['record_id'] not in seen, 'duplicate candidate identity')
            seen.add(ref['record_id'])
        included, excluded = [], []
        for ref in sorted(references, key=lambda item: item['record_id']):
            try:
                record = store.read(ref['record_id'], expected_sha256=ref['sha256'])
            except TraceError:
                excluded.append(dict(ref, reasons=['unverified_evidence']))
                continue
            reasons = []
            split = episodes.get(record['episode_id'], 'unknown')
            if split != 'development':
                reasons.append('held_out' if split == 'held_out' else 'unknown_origin')
            for key in ('policy', 'supervisor'):
                if record['models'][key] != compatibility[key]:
                    reasons.append(key + '_mismatch')
            if record['configuration']['settings'] != compatibility['settings']:
                reasons.append('settings_mismatch')
            if reasons:
                excluded.append(dict(ref, reasons=reasons))
            else:
                included.append(dict(ref, reasons=['verified_development']))
        report = {'policy': 'declared-development-exact-compatibility-v1',
                  'membership': membership, 'compatibility': compatibility,
                  'included': included, 'excluded': excluded}
        return MemorySnapshot.freeze(path, store,
            [{key: row[key] for key in ('record_id', 'sha256')} for row in included],
            metadata={'development_selection': report}, retrieval=retrieval)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise TraceError(f'invalid development snapshot declaration: {exc}') from exc
