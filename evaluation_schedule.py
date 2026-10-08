"""Deterministic paired trial schedule from a verified evaluation protocol."""

from collections import Counter, defaultdict
from hashlib import sha256
import json

from evaluation_protocol import ProtocolError


CONDITIONS = ('baseline', 'no_memory', 'fixed_memory')


def _require(value, message):
    if not value:
        raise ProtocolError(message)


def _allocation(document):
    seeds = document['seeds']
    _require(type(seeds.get('scheduling')) is int and seeds['scheduling'] >= 0,
             'nonnegative integer scheduling seed required')
    pairing = seeds.get('pairing')
    _require(type(pairing) is list and pairing and
             all(type(seed) is int and seed >= 0 for seed in pairing) and
             len(set(pairing)) == len(pairing),
             'unique nonnegative pairing seeds required')
    _require(set(document['conditions']) == set(CONDITIONS),
             'baseline, no_memory and fixed_memory conditions required')
    allocation = document['allocation']
    _require(set(allocation) == {f'task_{task}' for task in range(10)},
             'allocation required for all ten LIBERO-10 tasks')
    for task in range(10):
        entry = allocation[f'task_{task}']
        _require(type(entry) is dict and set(entry) == {'pairs'} and
                 type(entry['pairs']) is int and entry['pairs'] == len(pairing),
                 'equal declared pairs per task must match pairing seeds')
    return pairing, seeds['scheduling']


def _identity(protocol_id, task, state, repetition, seed, condition):
    value = [protocol_id, task, state['state_id'], state['selected_sha256'],
             repetition, seed, condition]
    return sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def _order_key(seed, *parts):
    return sha256(json.dumps([seed, *parts], separators=(',', ':')).encode()).digest()


def build_evaluation_schedule(protocol):
    """Return ordered attempt rows; this does not start or authorize trials.

    ``protocol`` is an EvaluationProtocol with an external reference. Its read
    verifies the seal again before the schedule is derived.
    """
    document = protocol.read()
    rows = _build(document)
    audit_evaluation_schedule(protocol, rows)
    return rows


def _build(document):
    pairing, scheduling_seed = _allocation(document)
    states = defaultdict(list)
    for state in document['splits']['held_out']:
        states[state['task_id']].append(state)
    clusters = []
    base = sorted(CONDITIONS,
                  key=lambda condition: _order_key(
                      scheduling_seed, 'condition', condition))
    for task in range(10):
        task_states = sorted(states[task], key=lambda s: s['state_id'])
        _require(task_states, f'held-out state required for task {task}')
        for repetition, seed in enumerate(pairing):
            state = task_states[repetition % len(task_states)]
            rotation = (task * len(pairing) + repetition) % 3
            order = base[rotation:] + base[:rotation]
            cluster = []
            for condition in order:
                cluster.append({
                    'attempt_id': _identity(document['protocol_id'], task, state,
                                            repetition, seed, condition),
                    'protocol_id': document['protocol_id'],
                    'suite': state['suite'], 'task_id': task,
                    'state_id': state['state_id'],
                    'selected_sha256': state['selected_sha256'],
                    'digest_encoding': state['digest_encoding'],
                    'repetition': repetition, 'environment_seed': seed,
                    'condition': condition,
                })
            clusters.append(cluster)
    clusters.sort(key=lambda cluster: _order_key(
        scheduling_seed, 'cluster', cluster[0]['task_id'],
        cluster[0]['repetition']))
    rows = [dict(row, sequence=index) for index, row in
            enumerate(row for cluster in clusters for row in cluster)]
    return rows


def audit_evaluation_schedule(protocol, rows):
    """Verify schedule membership, uniqueness, pairing and order balance."""
    document = protocol.read()
    pairing, _ = _allocation(document)
    _require(type(rows) in (list, tuple), 'schedule rows required')
    expected_count = 10 * len(pairing) * len(CONDITIONS)
    _require(len(rows) == expected_count, 'schedule attempt count mismatch')
    _require(all(type(row) is dict for row in rows), 'invalid schedule row')
    ids = [row.get('attempt_id') for row in rows]
    _require(len(set(ids)) == expected_count, 'duplicate attempt identity')
    _require([row.get('sequence') for row in rows] == list(range(expected_count)),
             'schedule sequence must be contiguous')
    held_out = {(s['task_id'], s['state_id']): s
                for s in document['splits']['held_out']}
    clusters = defaultdict(list)
    counts = Counter()
    positions = defaultdict(Counter)
    for row in rows:
        task, repetition = row.get('task_id'), row.get('repetition')
        _require(type(task) is int and 0 <= task < 10 and
                 type(repetition) is int and 0 <= repetition < len(pairing),
                 'invalid task or repetition')
        condition = row.get('condition')
        _require(condition in CONDITIONS, 'unknown condition')
        state = held_out.get((task, row.get('state_id')))
        _require(state is not None, 'state outside held-out split')
        expected = {
            'attempt_id': _identity(document['protocol_id'], task, state,
                                    repetition, pairing[repetition], condition),
            'protocol_id': document['protocol_id'], 'suite': state['suite'],
            'task_id': task, 'state_id': state['state_id'],
            'selected_sha256': state['selected_sha256'],
            'digest_encoding': state['digest_encoding'],
            'repetition': repetition, 'environment_seed': pairing[repetition],
            'condition': condition, 'sequence': row['sequence'],
        }
        _require(row == expected, 'schedule row differs from sealed declaration')
        clusters[(task, repetition)].append(row)
        counts[task] += 1
    _require(set(clusters) == {(task, rep) for task in range(10)
                               for rep in range(len(pairing))},
             'missing task/state pairing cluster')
    for (task, _), cluster in clusters.items():
        _require(len(cluster) == 3 and
                 {row['condition'] for row in cluster} == set(CONDITIONS) and
                 len({(row['state_id'], row['selected_sha256'],
                       row['environment_seed']) for row in cluster}) == 1,
                 'conditions must share one task/state and exogenous seed')
        ordered = sorted(cluster, key=lambda row: row['sequence'])
        _require([row['sequence'] for row in ordered] ==
                 list(range(ordered[0]['sequence'], ordered[0]['sequence'] + 3)),
                 'paired conditions must execute contiguously')
        for position, row in enumerate(ordered):
            positions[(task, position)][row['condition']] += 1
    _require(len(set(counts.values())) == 1, 'unequal task allocation')
    for counts_at_position in positions.values():
        spread = [counts_at_position[condition] for condition in CONDITIONS]
        _require(max(spread) - min(spread) <= 1,
                 'condition order is not counterbalanced')
    for position in range(3):
        spread = [sum(positions[(task, position)][condition]
                      for task in range(10)) for condition in CONDITIONS]
        _require(max(spread) - min(spread) <= 1,
                 'global condition order is not counterbalanced')
    _require(list(rows) == _build(document),
             'schedule order differs from recorded scheduling seed')
    return {'protocol_id': document['protocol_id'],
            'attempts': expected_count, 'pairs_per_task': len(pairing),
            'attempts_per_task': dict(sorted(counts.items())),
            'scheduling_seed': document['seeds']['scheduling']}
