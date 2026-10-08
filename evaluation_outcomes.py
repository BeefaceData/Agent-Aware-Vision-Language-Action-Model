"""Account for every sealed attempt before reporting primary success."""

from dataclasses import asdict, is_dataclass

from episode_harness import EpisodeInterruption, EpisodeOutcome
from evaluation_protocol import ProtocolError
from evaluation_schedule import audit_evaluation_schedule


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
