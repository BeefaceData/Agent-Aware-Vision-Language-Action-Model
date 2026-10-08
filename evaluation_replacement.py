"""Plan a new attempt only for a declared pre-start initialization failure."""

from hashlib import sha256

from episode_harness import EpisodeInterruption
from evaluation_protocol import ProtocolError
from evaluation_schedule import audit_evaluation_schedule


def replacement_attempt_id(original_attempt_id, number):
    """Give each rerun a stable identity distinct from the scheduled attempt."""
    return sha256(f'{original_attempt_id}:pre-start-replacement:{number}'.encode()).hexdigest()


def next_pre_start_replacement(protocol, rows, row, history):
    """Return a replacement plan without executing or discarding any attempt.

    ``history`` is an ordered sequence of ``(attempt_id, EpisodeInterruption)``
    pairs, beginning with the scheduled attempt. The caller must retain the
    original evidence and each replacement alongside the returned plan.
    """
    audit_evaluation_schedule(protocol, rows)
    if row not in rows or type(history) not in (list, tuple) or not history:
        raise ProtocolError('scheduled row and original attempt evidence required')
    rules = protocol.read()['outcomes']
    replacement = rules.get('pre_start_replacement')
    if (rules.get('pre_start_exclusion') != 'declared-replacement-only' or
            type(replacement) is not dict or
            set(replacement) != {'eligible_stages', 'max_replacements_per_attempt'} or
            type(replacement['eligible_stages']) is not list or
            not replacement['eligible_stages'] or
            not all(type(stage) is str for stage in replacement['eligible_stages']) or
            len(set(replacement['eligible_stages'])) != len(replacement['eligible_stages']) or
            not set(replacement['eligible_stages']) <= {
                'policy_reset', 'environment_reset', 'initial_observation'} or
            type(replacement['max_replacements_per_attempt']) is not int or
            replacement['max_replacements_per_attempt'] < 0):
        raise ProtocolError('valid frozen pre-start replacement rule required')
    if len(history) - 1 >= replacement['max_replacements_per_attempt']:
        raise ProtocolError('pre-start replacement allowance exhausted')

    provenance = []
    episode_ids = set()
    for number, entry in enumerate(history):
        expected = (row['attempt_id'] if number == 0 else
                    replacement_attempt_id(row['attempt_id'], number))
        if type(entry) not in (list, tuple) or len(entry) != 2 or entry[0] != expected:
            raise ProtocolError('replacement history identity or order mismatch')
        evidence = entry[1]
        if not isinstance(evidence, EpisodeInterruption):
            raise ProtocolError('completed or unsupported attempt cannot be replaced')
        failure = evidence.pre_start_failure
        if (failure is None or evidence.last_observation is not None or
                evidence.steps != 0 or evidence.acknowledged_actions or
                evidence.failed_action is not None or
                evidence.terminal_observation is not None or
                evidence.task_status != 'unknown' or
                failure.seed != row['environment_seed'] or
                failure.stage not in replacement['eligible_stages'] or
                type(evidence.episode_id) is not str or not evidence.episode_id or
                evidence.episode_id in episode_ids):
            raise ProtocolError('attempt is not an eligible pre-start failure')
        episode_ids.add(evidence.episode_id)
        provenance.append({'attempt_id': expected,
                           'episode_id': evidence.episode_id,
                           'stop_reason': evidence.stop_reason,
                           'stage': failure.stage,
                           'environment_seed': failure.seed})

    number = len(history)
    return {'attempt_id': replacement_attempt_id(row['attempt_id'], number),
            'original_attempt_id': row['attempt_id'],
            'replaces_attempt_id': history[-1][0],
            'replacement_number': number,
            'schedule_entry': dict(row),
            'prior_attempts': provenance}
