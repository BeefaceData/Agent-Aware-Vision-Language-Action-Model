"""Assemble matched initial-state units from a sealed evaluation schedule."""

from collections import defaultdict
from random import Random

from evaluation_outcomes import report_macro_improvement, score_evaluation_outcomes
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


def _percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return (sorted_values[lower] * (1 - fraction) +
            sorted_values[min(lower + 1, len(sorted_values) - 1)] * fraction)


def report_paired_cluster_intervals(protocol, rows, attempts):
    """Percentile intervals for equal-task differences from paired state draws.

    Each task independently draws as many initial states as it contains. A
    drawn state contributes every repetition and all three arms together.
    Incomplete pairs are rejected rather than silently changing denominators.
    The caller verifies persisted attempt artifacts before analysis.
    """
    declaration = protocol.read()
    if declaration['analysis'].get('estimator') != 'equal-task-macro' or declaration[
            'analysis'].get('interval') != 'stratified-paired-cluster-bootstrap-95':
        raise ProtocolError('unsupported sealed interval analysis')
    seed = declaration['seeds'].get('bootstrap')
    if type(seed) is not int or seed < 0:
        raise ProtocolError('nonnegative integer bootstrap seed required')

    assembled = assemble_paired_state_clusters(protocol, rows, attempts)
    if assembled['diagnostics']:
        raise ProtocolError('incomplete paired state evidence: ' +
                            assembled['diagnostics'][0])
    by_task = defaultdict(list)
    for cluster in assembled['clusters']:
        counts = {condition: [0, 0] for condition in CONDITIONS}
        for repetition in cluster['repetitions']:
            for condition in CONDITIONS:
                counts[condition][1] += 1
                counts[condition][0] += (repetition['conditions'][condition]
                                         ['status'] == 'success')
        by_task[cluster['task_id']].append(counts)
    if set(by_task) != set(range(10)) or any(not by_task[task] for task in range(10)):
        raise ProtocolError('all ten tasks need paired state clusters')

    comparisons = (('no_memory', 'baseline'), ('fixed_memory', 'baseline'),
                   ('fixed_memory', 'no_memory'))
    draws = {f'{left}_minus_{right}': [] for left, right in comparisons}
    random = Random(seed)
    for _ in range(10_000):
        task_rates = []
        for task in range(10):
            clusters = by_task[task]
            totals = {condition: [0, 0] for condition in CONDITIONS}
            for _ in clusters:
                selected = clusters[random.randrange(len(clusters))]
                for condition in CONDITIONS:
                    totals[condition][0] += selected[condition][0]
                    totals[condition][1] += selected[condition][1]
            task_rates.append({condition: totals[condition][0] /
                               totals[condition][1] for condition in CONDITIONS})
        for left, right in comparisons:
            draws[f'{left}_minus_{right}'].append(
                sum(100 * (rates[left] - rates[right])
                    for rates in task_rates) / 10)

    point = report_macro_improvement(protocol, rows, attempts)
    intervals = {}
    for name, values in draws.items():
        values.sort()
        intervals[name] = {'lower': _percentile(values, .025),
                           'upper': _percentile(values, .975)}
    return {'protocol_id': assembled['protocol_id'],
            'estimator': 'equal-task-macro',
            'interval_method': 'stratified-paired-cluster-bootstrap-95',
            'analysis_seed': seed, 'resamples': 10_000,
            'independent_starting_states': assembled['independent_starting_states'],
            'macro_differences_pp': point['macro_differences_pp'],
            'intervals_pp': intervals}
