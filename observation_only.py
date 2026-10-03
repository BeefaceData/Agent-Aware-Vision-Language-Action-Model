"""Offline export of diagnosis events from unchanged-policy sealed episodes."""

import argparse
from hashlib import sha256
import json

from recorded_replay import load_recorded_replay


def export_diagnosis_events(directory, *, configuration_id, provenance,
                            max_delay_sequences=0):
    """Export consecutive same-family assessment runs as single detections.

    Progress, uncertainty, absent diagnoses and skipped assessments end a run.
    Unknown and abstention are counted per assessment. Suppressed requests never
    become intervention events. Productive coverage requires separate review.
    """
    if not all(type(value) is str and value.strip()
               for value in (configuration_id, provenance)):
        raise ValueError('configuration identity and provenance required')
    if type(max_delay_sequences) is not int or max_delay_sequences < 0:
        raise ValueError('nonnegative integer matching tolerance required')
    trace = load_recorded_replay(directory).evidence()
    events = []
    previous_family = None
    families = {'suspected_missed_grasp': 'missed_grasp',
                'suspected_lost_grasp': 'lost_grasp', 'stall': 'stall'}
    for row in trace['decisions']:
        record = row['action_record']
        if record['disposition'] != 'unmodified':
            raise ValueError('observation-only export requires unchanged policy actions')
        response = record.get('supervisor_pass') or record.get('supervisor_abstention')
        diagnosis = response.get('temporal_diagnosis') if response else None
        category = diagnosis['category'] if diagnosis else None
        family = families.get(category)
        sequence = row['source_sequence']

        def emit(kind, event_family=None):
            events.append(dict(id=f'{sequence}:{kind}', sequence=sequence,
                               kind=kind, family=event_family))

        if family is not None and family != previous_family:
            emit('detection', family)
        if category == 'unknown':
            emit('unknown')
        if response and response['kind'] == 'abstain':
            emit('abstention')
        previous_family = family
    digest = sha256(json.dumps(trace, sort_keys=True, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')).hexdigest()
    return dict(version=1, source_episode_id=trace['source_episode_id'],
                trace_sha256=digest, configuration_id=configuration_id,
                provenance=provenance + '; observation-only export v1: consecutive '
                'same-family diagnoses coalesced; skipped/absent/other diagnoses '
                'end events; unknown/abstention per assessment; no interventions',
                max_delay_sequences=max_delay_sequences, events=events,
                productive_intervals=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_directory')
    parser.add_argument('--configuration-id', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--max-delay-sequences', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(export_diagnosis_events(
        args.trace_directory, configuration_id=args.configuration_id,
        provenance=args.provenance, max_delay_sequences=args.max_delay_sequences),
        indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
