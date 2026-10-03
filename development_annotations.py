"""Offline development annotations; never consumed as supervisor inputs."""

import argparse
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from recorded_replay import load_recorded_replay


FAMILIES = ('missed_grasp', 'lost_grasp', 'stall', 'progress', 'pause',
            'occlusion', 'unknown')
EXAMPLES = ('failed', 'successful', 'paused', 'occluded', 'productive')


class AnnotationError(ValueError):
    """Annotation does not identify valid reviewed development evidence."""


def _require(condition, message):
    if not condition:
        raise AnnotationError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _trace(directory):
    evidence = load_recorded_replay(directory).evidence()
    raw = json.dumps(evidence, sort_keys=True, allow_nan=False,
                     separators=(',', ':')).encode('utf-8')
    return evidence, sha256(raw).hexdigest()


def annotation_template(directory, *, reviewer, provenance):
    """Create an explicitly unreviewed template for a permitted development trace.

    Provenance is a human declaration, not proof of permission or split membership.
    """
    _require(_text(reviewer) and _text(provenance), 'reviewer and provenance required')
    trace, digest = _trace(directory)
    return {
        'version': 1, 'split': 'development', 'provenance': provenance,
        'source_episode_id': trace['source_episode_id'], 'trace_sha256': digest,
        'reviewer': reviewer, 'reviewed_at': None, 'status': 'draft',
        'events': [],
    }


def validate_annotation(directory, annotation):
    """Return a detached validated review; reject drafts and unresolved references.

    Inclusive sequence intervals bound onset of the labeled phenomenon, not its
    duration. Null onset requires uncertainty. Evidence may precede/follow onset.
    """
    trace, digest = _trace(directory)
    _require(isinstance(annotation, dict), 'annotation must be an object')
    expected = {'version', 'split', 'provenance', 'source_episode_id',
                'trace_sha256', 'reviewer', 'reviewed_at', 'status', 'events'}
    _require(set(annotation) == expected, 'unsupported annotation fields')
    _require(type(annotation['version']) is int and annotation['version'] == 1,
             'unsupported annotation version')
    _require(annotation['split'] == 'development', 'development split required')
    _require(annotation['source_episode_id'] == trace['source_episode_id'] and
             annotation['trace_sha256'] == digest, 'annotation trace identity mismatch')
    _require(all(_text(annotation[key]) for key in ('reviewer', 'provenance')),
             'reviewer and provenance required')
    _require(annotation['status'] == 'reviewed', 'human review is incomplete')
    try:
        reviewed = datetime.fromisoformat(annotation['reviewed_at'])
        _require(reviewed.utcoffset() is not None, 'review time needs timezone')
    except (TypeError, ValueError) as exc:
        raise AnnotationError('invalid review timestamp') from exc
    events = annotation['events']
    _require(isinstance(events, list) and bool(events), 'at least one labeled example required')
    sequences = {row['sequence'] for row in trace['observations']}
    ids = set()
    for event in events:
        _require(isinstance(event, dict) and set(event) == {
            'id', 'family', 'example', 'onset', 'uncertain', 'evidence', 'notes'},
            'unsupported event fields')
        _require(_text(event['id']) and event['id'] not in ids, 'duplicate or empty event id')
        ids.add(event['id'])
        _require(event['family'] in FAMILIES and event['example'] in EXAMPLES,
                 'unsupported family or example')
        _require(type(event['uncertain']) is bool and _text(event['notes']),
                 'explicit uncertainty and review notes required')
        _require(event['family'] != 'unknown' or event['uncertain'],
                 'unknown family requires uncertainty')
        onset = event['onset']
        if onset is None:
            _require(event['uncertain'], 'unknown onset requires uncertainty')
        else:
            _require(isinstance(onset, list) and len(onset) == 2 and
                     all(type(seq) is int and seq in sequences for seq in onset) and
                     onset[0] <= onset[1], 'invalid inclusive onset interval')
        refs = event['evidence']
        _require(isinstance(refs, list) and bool(refs), 'supporting evidence required')
        for ref in refs:
            _require(isinstance(ref, dict) and set(ref) == {'sequence', 'description'} and
                     type(ref['sequence']) is int and ref['sequence'] in sequences and
                     _text(ref['description']), 'invalid observation evidence reference')
    return deepcopy(annotation)


def load_annotation(directory, path):
    """Load strict JSON and validate against the original sealed trace."""
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result

    def invalid(value):
        raise AnnotationError(f'nonfinite JSON value: {value}')

    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'),
                           object_pairs_hook=pairs, parse_constant=invalid)
    except (OSError, ValueError) as exc:
        raise AnnotationError(f'invalid annotation JSON: {exc}') from exc
    return validate_annotation(directory, value)


def save_annotation(directory, path, annotation):
    """Validate and write a new sidecar; never overwrite an existing review."""
    value = validate_annotation(directory, annotation)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    draft = sub.add_parser('template')
    draft.add_argument('trace')
    draft.add_argument('output')
    draft.add_argument('--reviewer', required=True)
    draft.add_argument('--provenance', required=True)
    check = sub.add_parser('validate')
    check.add_argument('trace')
    check.add_argument('annotation')
    args = parser.parse_args()
    try:
        if args.command == 'template':
            value = annotation_template(args.trace, reviewer=args.reviewer,
                                        provenance=args.provenance)
            with Path(args.output).open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(value, indent=2) + '\n')
            print('Draft created; human review required.')
        else:
            value = load_annotation(args.trace, args.annotation)
            print(json.dumps({'source_episode_id': value['source_episode_id'],
                              'events': len(value['events']), 'status': 'validated'}))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
