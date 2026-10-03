"""Offline, trace-bound event metrics; never grants intervention readiness."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from development_annotations import load_annotation, validate_annotation
from recorded_replay import load_recorded_replay


FAILURE_FAMILIES = ('missed_grasp', 'lost_grasp', 'stall')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def detector_report(directory, annotation, evaluation):
    """Score declared event outputs against one reviewed development trace.

    Matching is chronological, one-to-one, same-family, earliest eligible onset
    first. Eligibility is [onset lower, onset upper + max_delay_sequences].
    Signed delay bounds are [detection - onset upper, detection - onset lower].
    Productive intervals are separately reviewed durations, never onset brackets.
    """
    review = validate_annotation(directory, annotation)
    trace = load_recorded_replay(directory).evidence()
    _require(isinstance(evaluation, dict) and set(evaluation) == {
        'version', 'source_episode_id', 'trace_sha256', 'configuration_id',
        'provenance', 'max_delay_sequences', 'events', 'productive_intervals'},
        'unsupported evaluation fields')
    _require(type(evaluation['version']) is int and evaluation['version'] == 1,
             'unsupported evaluation version')
    for key in ('source_episode_id', 'trace_sha256'):
        _require(evaluation[key] == review[key], 'evaluation trace identity mismatch')
    _require(all(_text(evaluation[key]) for key in ('configuration_id', 'provenance')),
             'configuration identity and output provenance required')
    tolerance = evaluation['max_delay_sequences']
    _require(type(tolerance) is int and tolerance >= 0, 'invalid matching tolerance')
    sequences = {row['sequence'] for row in trace['observations']}
    events = evaluation['events']
    _require(isinstance(events, list), 'events must be a list')
    ids = set()
    for event in events:
        _require(isinstance(event, dict) and set(event) == {
            'id', 'sequence', 'kind', 'family'}, 'unsupported detector event fields')
        _require(_text(event['id']) and event['id'] not in ids, 'duplicate or empty event id')
        ids.add(event['id'])
        _require(type(event['sequence']) is int and event['sequence'] in sequences,
                 'event outside source trace')
        _require(event['kind'] in ('detection', 'intervention', 'unknown', 'abstention'),
                 'invalid event kind')
        _require((event['kind'] in ('detection', 'intervention') and
                  event['family'] in FAILURE_FAMILIES) or
                 (event['kind'] in ('unknown', 'abstention') and event['family'] is None),
                 'invalid event family')
    intervals = evaluation['productive_intervals']
    _require(isinstance(intervals, list), 'productive intervals must be a list')
    productive = set()
    by_id = {event['id']: event for event in review['events']}
    for interval in intervals:
        _require(isinstance(interval, dict) and set(interval) == {
            'annotation_id', 'start', 'end', 'reviewer', 'notes'},
            'unsupported productive interval fields')
        _require(_text(interval['annotation_id']), 'productive annotation identity required')
        label = by_id.get(interval['annotation_id'])
        _require(label is not None and not label['uncertain'] and
                 label['family'] in ('progress', 'pause') and
                 label['example'] in ('productive', 'successful', 'paused'),
                 'productive interval requires a certain productive annotation')
        _require(all(_text(interval[key]) for key in ('reviewer', 'notes')),
                 'productive duration review required')
        start, end = interval['start'], interval['end']
        _require(type(start) is int and type(end) is int and
                 start in sequences and end in sequences and start <= end,
                 'invalid productive interval')
        productive.update(seq for seq in sequences if start <= seq <= end)
    labels = [event for event in review['events'] if not event['uncertain'] and
              event['family'] in FAILURE_FAMILIES and event['onset'] is not None]
    # A productive review cannot simultaneously declare a certain failure onset.
    _require(not any(any(label['onset'][0] <= seq <= label['onset'][1]
                         for seq in productive) for label in labels),
             'productive coverage conflicts with failure onset')
    families = {}
    for family in FAILURE_FAMILIES:
        expected = sorted((label for label in labels if label['family'] == family),
                          key=lambda item: (*item['onset'], item['id']))
        detections = sorted((event for event in events if event['kind'] == 'detection'
                             and event['family'] == family),
                            key=lambda item: (item['sequence'], item['id']))
        remaining = list(expected)
        matches, unmatched = [], []
        for event in detections:
            label = next((item for item in remaining if
                          item['onset'][0] <= event['sequence'] <= item['onset'][1] + tolerance),
                         None)
            if label is None:
                unmatched.append(event['id'])
                continue
            remaining.remove(label)
            matches.append({'detection_id': event['id'], 'annotation_id': label['id'],
                            'delay_sequences': [event['sequence'] - label['onset'][1],
                                                event['sequence'] - label['onset'][0]]})
        interventions = [event for event in events if event['kind'] == 'intervention'
                         and event['family'] == family]
        families[family] = {
            'labeled_events': len(expected), 'detected_events': len(detections),
            'matched_events': len(matches),
            'precision': len(matches) / len(detections) if detections else None,
            'recall': len(matches) / len(expected) if expected else None,
            'matches': matches, 'unmatched_detections': unmatched,
            'unmatched_annotations': [item['id'] for item in remaining],
            'interventions': len(interventions),
            'false_interventions': [event['id'] for event in interventions
                                    if event['sequence'] in productive],
            'interventions_outside_productive_coverage': [event['id'] for event in interventions
                                                       if event['sequence'] not in productive],
        }
    def digest(value):
        return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                 separators=(',', ':')).encode('utf-8')).hexdigest()

    return {
        'version': 1, 'source_episode_id': review['source_episode_id'],
        'trace_sha256': review['trace_sha256'], 'annotation_sha256': digest(review),
        'evaluation_sha256': digest(evaluation),
        'configuration_id': evaluation['configuration_id'],
        'matching': {'max_delay_sequences': tolerance, 'unit': 'observation_sequence',
                     'policy': 'chronological-one-to-one-earliest-onset'},
        'families': families,
        'unknown_count': sum(event['kind'] == 'unknown' for event in events),
        'abstention_count': sum(event['kind'] == 'abstention' for event in events),
        'uncertain_annotations': [event['id'] for event in review['events'] if event['uncertain']],
        'productive_sequences': len(productive),
        'false_intervention_count': sum(len(item['false_interventions']) for item in families.values()),
        'limitations': [
            'Metrics describe supplied event outputs and reviewed label coverage only.',
            'Unmatched detections are not proof of true false positives on incompletely labeled traces.',
            'No productive coverage means false interventions are unassessed, not demonstrated absent.',
            'Sequence delays are not seconds; negative lower bounds preserve onset uncertainty.',
            'This report does not establish calibrated detection or active-intervention readiness.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('annotation')
    parser.add_argument('evaluation')
    args = parser.parse_args()
    try:
        def pairs(items):
            result = {}
            for key, value in items:
                _require(key not in result, 'duplicate evaluation field')
                result[key] = value
            return result

        evaluation = json.loads(Path(args.evaluation).read_text(encoding='utf-8'),
                                object_pairs_hook=pairs)
        report = detector_report(args.trace, load_annotation(args.trace, args.annotation), evaluation)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
