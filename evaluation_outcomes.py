"""Account for every sealed attempt before reporting primary success."""

from dataclasses import asdict, is_dataclass

from episode_harness import EpisodeInterruption, EpisodeOutcome
from evaluation_protocol import ProtocolError
from evaluation_schedule import CONDITIONS, audit_evaluation_schedule


def score_evaluation_outcomes(protocol, rows, attempts):
    """Score completed outcomes and explicit interruptions by scheduled attempt ID.

    ``attempts`` maps IDs to EpisodeOutcome, EpisodeInterruption, or the
    completed outcome dictionaries returned by run_evaluation_schedule. The
    caller must verify retained replay evidence before passing its outcome.
    Missing and unscoreable rows remain visible and prevent a final rate.
    """
    audit_evaluation_schedule(protocol, rows)
    rules = protocol.read()['outcomes']
    if (rules.get('success') != 'evaluator-terminal-success' or
            rules.get('post_start_system_failure') != 'primary-denominator'):
        raise ProtocolError('unsupported sealed primary outcome rules')
    if type(attempts) is not dict:
        raise ProtocolError('attempt outcomes must be keyed by attempt ID')
    known = {row['attempt_id'] for row in rows}
    if set(attempts) - known:
        raise ProtocolError('outcome outside sealed schedule')

    counts = {'scheduled': len(rows), 'valid_starts': 0, 'successes': 0,
              'post_start_failures': 0, 'pre_start_exclusions': 0,
              'missing': 0, 'unscoreable': 0}
    dispositions = []
    episode_ids = set()
    for row in rows:
        identity = row['attempt_id']
        evidence = attempts.get(identity)
        if evidence is None:
            status, reason = 'missing', 'no attempt evidence'
        elif isinstance(evidence, EpisodeInterruption):
            if evidence.pre_start_failure is not None and evidence.last_observation is None:
                status, reason = 'pre_start_exclusion', evidence.stop_reason
            elif evidence.pre_start_failure is not None:
                status, reason = 'unscoreable', 'conflicting start evidence'
            elif evidence.last_observation is not None:
                status, reason = 'post_start_failure', evidence.stop_reason
            else:
                status, reason = 'unscoreable', 'start boundary unknown'
        elif isinstance(evidence, EpisodeOutcome) or type(evidence) is dict:
            outcome = asdict(evidence) if is_dataclass(evidence) else evidence
            # A returned episode outcome exists only after the initial
            # observation passed the harness start boundary.
            counts['valid_starts'] += 1
            if (type(outcome.get('success')) is not bool or
                    type(outcome.get('episode_id')) is not str or
                    not outcome['episode_id'] or
                    type(outcome.get('stop_reason')) is not str or
                    not outcome['stop_reason']):
                status, reason = 'unscoreable', 'invalid outcome fields'
            elif outcome.get('artifact_status') != 'completed':
                status, reason = 'unscoreable', 'incomplete outcome artifacts'
            elif outcome['success'] and outcome['stop_reason'] != 'success':
                status, reason = 'unscoreable', 'conflicting success evidence'
            elif not outcome['success'] and outcome['stop_reason'] == 'success':
                status, reason = 'unscoreable', 'conflicting failure evidence'
            elif outcome['success']:
                status, reason = 'success', outcome['stop_reason']
            else:
                status, reason = 'post_start_failure', outcome['stop_reason']
        else:
            status, reason = 'unscoreable', 'unsupported attempt evidence'

        episode_id = (evidence.get('episode_id') if type(evidence) is dict else
                      getattr(evidence, 'episode_id', None))
        if type(episode_id) is str and episode_id:
            if episode_id in episode_ids:
                raise ProtocolError('duplicate recorded episode identity')
            episode_ids.add(episode_id)

        if status == 'success':
            counts['successes'] += 1
        elif status == 'post_start_failure':
            counts['post_start_failures'] += 1
            if isinstance(evidence, EpisodeInterruption):
                counts['valid_starts'] += 1
        else:
            counts[{'pre_start_exclusion': 'pre_start_exclusions',
                    'missing': 'missing', 'unscoreable': 'unscoreable'}[status]] += 1
        dispositions.append({'attempt_id': identity, 'task_id': row['task_id'],
                             'condition': row['condition'], 'status': status,
                             'reason': reason})

    counts['success_rate'] = (
        counts['successes'] / counts['valid_starts']
        if counts['valid_starts'] and not counts['missing'] and
        not counts['unscoreable'] else None)
    return {'protocol_id': protocol.reference['protocol_id'],
            **counts, 'dispositions': dispositions}


def report_task_outcomes(protocol, rows, attempts):
    """Report every sealed task and condition before any aggregate analysis.

    Differences are percentage points and require complete, scoreable evidence
    for both conditions. A missing task stays in the report with a warning.
    """
    scored = score_evaluation_outcomes(protocol, rows, attempts)
    tasks = []
    warnings = []
    for task_id in range(10):
        conditions = {}
        for condition in CONDITIONS:
            dispositions = [item for item in scored['dispositions']
                            if item['task_id'] == task_id and
                            item['condition'] == condition]
            statuses = [item['status'] for item in dispositions]
            successes = statuses.count('success')
            attempts_count = successes + statuses.count('post_start_failure')
            missing = statuses.count('missing')
            unscoreable = statuses.count('unscoreable')
            pre_start_exclusions = statuses.count('pre_start_exclusion')
            rate = (successes / attempts_count if attempts_count and
                    not missing and not unscoreable else None)
            conditions[condition] = {
                'scheduled': len(dispositions), 'attempts': attempts_count,
                'successes': successes, 'missing_evidence': missing,
                'unscoreable': unscoreable,
                'pre_start_exclusions': pre_start_exclusions,
                'success_rate': rate,
            }
            if missing or unscoreable:
                warnings.append(f'task {task_id} {condition}: '
                                f'{missing} missing, {unscoreable} unscoreable')
            if not attempts_count:
                warnings.append(f'task {task_id} {condition}: no valid starts')
        if not any(item['attempts'] for item in conditions.values()):
            warnings.append(f'task {task_id}: no valid starts')
        differences = {}
        for left, right in (('no_memory', 'baseline'),
                            ('fixed_memory', 'baseline'),
                            ('fixed_memory', 'no_memory')):
            first = conditions[left]['success_rate']
            second = conditions[right]['success_rate']
            differences[f'{left}_minus_{right}'] = (
                100 * (first - second) if first is not None and
                second is not None else None)
        tasks.append({'task_id': task_id, 'conditions': conditions,
                      'differences_pp': differences})
    return {'protocol_id': scored['protocol_id'], 'tasks': tasks,
            'warnings': warnings}


def report_macro_improvement(protocol, rows, attempts):
    """Average each condition contrast across all ten LIBERO-10 tasks.

    A contrast is unavailable until every task has a scoreable rate for both
    conditions. Pooled counts are descriptive and never replace this estimator.
    """
    if protocol.read()['analysis'].get('estimator') != 'equal-task-macro':
        raise ProtocolError('equal-task-macro estimator required')
    task_report = report_task_outcomes(protocol, rows, attempts)
    tasks = task_report['tasks']
    contrasts = ('no_memory_minus_baseline', 'fixed_memory_minus_baseline',
                 'fixed_memory_minus_no_memory')
    differences = {}
    for contrast in contrasts:
        values = [task['differences_pp'][contrast] for task in tasks]
        differences[contrast] = (sum(values) / 10 if len(values) == 10 and
                                 all(value is not None for value in values)
                                 else None)
    pooled = {}
    for condition in CONDITIONS:
        entries = [task['conditions'][condition] for task in tasks]
        pooled[condition] = {
            'scheduled': sum(entry['scheduled'] for entry in entries),
            'attempts': sum(entry['attempts'] for entry in entries),
            'successes': sum(entry['successes'] for entry in entries),
            'pre_start_exclusions': sum(entry['pre_start_exclusions']
                                        for entry in entries),
            'missing_evidence': sum(entry['missing_evidence'] for entry in entries),
            'unscoreable': sum(entry['unscoreable'] for entry in entries),
        }
    return {**task_report, 'estimator': 'equal-task-macro',
            'macro_differences_pp': differences, 'pooled_counts': pooled}
