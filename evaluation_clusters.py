"""Assemble matched initial-state units from a sealed evaluation schedule."""

from collections import defaultdict

from evaluation_outcomes import score_evaluation_outcomes
from evaluation_protocol import ProtocolError
from evaluation_schedule import CONDITIONS


def assemble_paired_state_clusters(protocol, rows, attempts):
    """Keep repetitions nested within their independent held-out state.

    Only a repetition with scoreable post-start outcomes in all three arms is
    eligible for paired analysis. Incomplete repetitions remain visible with
    their scheduled identities and reasons; they never become extra states.
    The outcome scorer audits every row against the frozen protocol first.
    """
    scored = score_evaluation_outcomes(protocol, rows, attempts)
    dispositions = {item['attempt_id']: item for item in scored['dispositions']}
    states = defaultdict(lambda: defaultdict(dict))
    identities = {}
    for row in rows:
        key = (row['task_id'], row['state_id'])
        identity = (row['suite'], row['selected_sha256'], row['digest_encoding'])
        if key in identities and identities[key] != identity:
            # The schedule audit above also rejects this. Keep the grouping
            # invariant explicit in case schedule validation changes.
            raise ProtocolError('conflicting selected-state identity')
        identities[key] = identity
        states[key][row['repetition']][row['condition']] = {
            'attempt_id': row['attempt_id'],
            'environment_seed': row['environment_seed'],
            **{field: dispositions[row['attempt_id']][field]
               for field in ('status', 'reason')},
        }

    clusters = []
    diagnostics = []
    for (task_id, state_id), repetitions in sorted(states.items()):
        entries = []
        for repetition, conditions in sorted(repetitions.items()):
            eligible = all(conditions[condition]['status'] in
                           ('success', 'post_start_failure')
                           for condition in CONDITIONS)
            if not eligible:
                gaps = [f"{condition}: {conditions[condition]['status']}"
                        for condition in CONDITIONS
                        if conditions[condition]['status'] not in
                        ('success', 'post_start_failure')]
                diagnostics.append(f'task {task_id} state {state_id} '
                                   f'repetition {repetition}: ' + ', '.join(gaps))
            entries.append({'repetition': repetition,
                            'environment_seed':
                            conditions[CONDITIONS[0]]['environment_seed'],
                            'paired': eligible, 'conditions': conditions})
        suite, digest, encoding = identities[(task_id, state_id)]
        clusters.append({'task_id': task_id, 'state_id': state_id,
                         'suite': suite, 'selected_sha256': digest,
                         'digest_encoding': encoding,
                         'repetitions': entries,
                         'paired_repetitions': sum(item['paired'] for item in entries)})
    return {'protocol_id': scored['protocol_id'], 'clusters': clusters,
            'independent_starting_states': len(clusters),
            'diagnostics': diagnostics}
