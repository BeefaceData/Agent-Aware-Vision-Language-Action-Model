"""Report operational evidence for the same sealed attempts used for success."""

from dataclasses import asdict, is_dataclass
from math import isfinite

from evaluation_outcomes import score_evaluation_outcomes
from evaluation_protocol import ProtocolError
from evaluation_schedule import CONDITIONS


FIELDS = ('elapsed_seconds', 'waiting_seconds', 'executed_actions',
          'interventions', 'human_assistance', 'model_calls')


def _number(value, name, *, integer=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or (not integer and not isfinite(value)) or value < 0:
        raise ProtocolError(f'invalid {name} operational evidence')
    return value


def _external(record, episode_id):
    if type(record) is not dict or record.get('episode_id') != episode_id:
        raise ProtocolError('operational evidence episode identity mismatch')
    values = {}
    for name in ('interventions', 'human_assistance', 'model_calls'):
        value = record.get(name)
        values[name] = None if value is None else _number(value, name, integer=True)
    if record.get('model_calls') == 0 and record.get('unknown_cost_calls') not in (None, 0):
        raise ProtocolError('unknown cost calls exceed model calls')
    unknown = record.get('unknown_cost_calls')
    if unknown is not None:
        _number(unknown, 'unknown_cost_calls', integer=True)
        if values['model_calls'] is None or unknown > values['model_calls']:
            raise ProtocolError('unknown cost calls exceed model calls')
    costs = record.get('reported_cost_subtotals')
    if costs is not None:
        if type(costs) is not dict or any(
                type(currency) is not str or not currency or
                type(amount) not in (int, float) or not isfinite(amount) or amount < 0
                for currency, amount in costs.items()):
            raise ProtocolError('invalid reported cost subtotals')
        if values['model_calls'] is None:
            raise ProtocolError('cost requires model call count')
    values['reported_cost_subtotals'] = costs
    values['unknown_cost_calls'] = unknown
    values['cost_complete'] = (costs is not None and unknown == 0)
    return values


def report_operational_metrics(protocol, rows, attempts, operational):
    """Join outcome and separately retained usage by scheduled attempt ID.

    Operational records must identify the same episode. Missing fields stay
    unknown, including for started failures and incomplete artifacts. Only
    scoreable valid starts enter condition totals, exactly as in success counts.
    Callers verify persisted evidence before supplying it here.
    """
    scored = score_evaluation_outcomes(protocol, rows, attempts)
    if type(operational) is not dict or set(operational) - {
            row['attempt_id'] for row in rows}:
        raise ProtocolError('operational evidence outside sealed schedule')
    by_condition = {condition: {'attempts': 0, 'successes': 0,
                                'totals': {name: 0 for name in FIELDS},
                                'unknown_attempts': {name: 0 for name in FIELDS},
                                'reported_cost_subtotals': {},
                                'unknown_cost_attempts': 0,
                                'unknown_cost_calls': 0}
                    for condition in CONDITIONS}
    detail = []
    for row, disposition in zip(rows, scored['dispositions']):
        identity = row['attempt_id']
        evidence = attempts.get(identity)
        episode_id = (evidence.get('episode_id') if type(evidence) is dict else
                      getattr(evidence, 'episode_id', None))
        extra = (_external(operational[identity], episode_id)
                 if identity in operational else {})
        status = disposition['status']
        values = {name: None for name in FIELDS}
        if status in ('success', 'post_start_failure'):
            outcome = (asdict(evidence) if is_dataclass(evidence) else evidence)
            for name, source in (('elapsed_seconds', 'rollout_seconds'),
                                 ('waiting_seconds', 'cumulative_wait_seconds'),
                                 ('executed_actions', 'steps')):
                value = outcome.get(source)
                values[name] = (None if value is None else
                                _number(value, name,
                                        integer=name == 'executed_actions'))
            values.update({name: extra.get(name) for name in
                           ('interventions', 'human_assistance', 'model_calls')})
            group = by_condition[row['condition']]
            group['attempts'] += 1
            group['successes'] += status == 'success'
            for name, value in values.items():
                if value is None:
                    group['unknown_attempts'][name] += 1
                else:
                    group['totals'][name] += value
            if not extra.get('cost_complete'):
                group['unknown_cost_attempts'] += 1
            group['unknown_cost_calls'] += extra.get('unknown_cost_calls') or 0
            for currency, amount in (extra.get('reported_cost_subtotals') or {}).items():
                costs = group['reported_cost_subtotals']
                costs[currency] = costs.get(currency, 0) + amount
        detail.append({**disposition, 'metrics': values,
                       'reported_cost_subtotals':
                       extra.get('reported_cost_subtotals'),
                       'unknown_cost_calls': extra.get('unknown_cost_calls'),
                       'cost_complete': extra.get('cost_complete')})
    for group in by_condition.values():
        group['complete_totals'] = {
            name: group['totals'][name] if group['unknown_attempts'][name] == 0
            else None for name in FIELDS}
        group['total_cost_by_currency'] = (
            group['reported_cost_subtotals']
            if group['unknown_cost_attempts'] == 0 else None)
    return {'protocol_id': scored['protocol_id'], 'conditions': by_condition,
            'attempts': detail}
