"""Offline, development-only planning for the sealed LIBERO-10 comparison."""

from math import isfinite, sqrt
from statistics import NormalDist


def plan_trial_allocation(pilot, *, target_gain_pp=10, precision_half_width_pp=10,
                          power=0.8, cluster_correlation=0.5,
                          max_states_per_task, max_repetitions_per_state,
                          limits):
    """Estimate equal-task A/B/C allocation; never certify a future result.

    ``pilot`` has one development-only entry per task (integer keys 0..9).
    Each entry gives ``pairs``, ``c_only``, ``a_only`` and measured
    ``seconds_per_triplet``, ``calls_per_triplet``, ``cost_per_triplet``.
    Discordance refers to fixed-memory C versus baseline A on paired starts.
    The caller supplies an assumed within-state correlation because a binary
    pair count alone cannot identify it. Throughput covers all three arms.
    """
    def positive(value, name, *, zero=False):
        if type(value) not in (int, float) or not isfinite(value) or value < 0 or (not zero and value == 0):
            raise ValueError(f'{name} must be finite and positive')
        return value

    if type(pilot) is not dict or set(pilot) != set(range(10)):
        raise ValueError('development pilot requires all ten LIBERO-10 tasks')
    if type(max_states_per_task) is not int or max_states_per_task < 2:
        raise ValueError('at least two independent states per task required')
    if type(max_repetitions_per_state) is not int or max_repetitions_per_state < 1:
        raise ValueError('positive repetition cap required')
    target = positive(target_gain_pp, 'target_gain_pp') / 100
    precision = positive(precision_half_width_pp, 'precision_half_width_pp') / 100
    if target > 1 or precision > 1 or not 0 < power < 1 or not 0 <= cluster_correlation <= 1:
        raise ValueError('invalid planning probability or correlation')
    if type(limits) is not dict or set(limits) != {'seconds', 'model_calls', 'cost'}:
        raise ValueError('seconds, model_calls, and cost limits required')
    for key, value in limits.items():
        positive(value, f'{key} limit', zero=True)

    inputs = {}
    for task, row in pilot.items():
        if type(row) is not dict or set(row) != {
                'split', 'pairs', 'c_only', 'a_only', 'seconds_per_triplet',
                'calls_per_triplet', 'cost_per_triplet'} or row['split'] != 'development':
            raise ValueError(f'task {task}: development-only paired pilot required')
        n, c, a = (row[key] for key in ('pairs', 'c_only', 'a_only'))
        if any(type(x) is not int or x < 0 for x in (n, c, a)) or n == 0 or c + a > n:
            raise ValueError(f'task {task}: invalid paired discordance counts')
        for key in ('seconds_per_triplet', 'calls_per_triplet', 'cost_per_triplet'):
            positive(row[key], f'task {task} {key}', zero=True)
        observed_gain = (c - a) / n
        discordance = (c + a) / n
        # A target difference cannot exceed the discordance probability.
        # Use at least the target discordance for the planning scenario.
        scenario_discordance = max(discordance, target)
        inputs[task] = {'pilot_pairs': n, 'c_only': c, 'a_only': a,
                        'observed_gain_pp': 100 * observed_gain,
                        'discordance_rate': discordance,
                        'scenario_variance': scenario_discordance - target ** 2}

    z95 = NormalDist().inv_cdf(0.975)
    options = []
    for states in range(2, max_states_per_task + 1):
        for repetitions in range(1, max_repetitions_per_state + 1):
            pairs = states * repetitions
            inflation = 1 + (repetitions - 1) * cluster_correlation
            se = sqrt(sum(row['scenario_variance'] for row in inputs.values()) *
                      inflation / (100 * pairs))
            half_width = z95 * se
            # Normal approximation to a two-sided 95% lower endpoint > 0.
            approximate_power = NormalDist().cdf(target / se - z95) if se else 1.0
            demand = {
                'seconds': pairs * sum(row['seconds_per_triplet'] for row in pilot.values()),
                'model_calls': pairs * sum(row['calls_per_triplet'] for row in pilot.values()),
                'cost': pairs * sum(row['cost_per_triplet'] for row in pilot.values()),
            }
            precision_ok = half_width <= precision
            power_ok = approximate_power >= power
            resource_ok = all(demand[key] <= limits[key] for key in limits)
            options.append({'states_per_task': states, 'repetitions_per_state': repetitions,
                            'pairs_per_task': pairs, 'attempts': pairs * 30,
                            'estimated_half_width_pp': half_width * 100,
                            'approximate_power': approximate_power,
                            'demand': demand, 'precision_ok': precision_ok,
                            'power_ok': power_ok, 'resource_ok': resource_ok})
    eligible = [item for item in options if item['precision_ok'] and item['power_ok']]
    best = min(eligible, key=lambda item: (item['attempts'], item['demand']['seconds'],
                                          item['states_per_task'])) if eligible else None
    feasible = best is not None and best['resource_ok']
    return {
        'status': 'feasible' if feasible else 'infeasible',
        'allocation': ({f'task_{task}': {'initial_states': best['states_per_task'],
                                        'repetitions_per_state': best['repetitions_per_state'],
                                        'pairs': best['pairs_per_task']}
                        for task in range(10)} if best else None),
        'estimate': best, 'pilot': inputs, 'limits': dict(limits),
        'assumptions': {'target_gain_pp': target_gain_pp,
                        'precision_half_width_pp': precision_half_width_pp,
                        'confidence': 0.95, 'power_target': power,
                        'cluster_correlation': cluster_correlation,
                        'normal_approximation': True,
                        'independent_unit': 'initial state within task',
                        'comparison': 'fixed_memory minus baseline',
                        'arms_per_pair': 3},
        'limitations': [
            'Pilot discordance and throughput may not predict held-out outcomes.',
            'Normal power and width approximations do not guarantee bootstrap significance.',
            'This plan does not grant resource approval or select held-out states.',
        ],
    }
