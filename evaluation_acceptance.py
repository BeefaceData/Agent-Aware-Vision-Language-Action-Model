"""Apply the preregistered simulation improvement rule to sealed evidence."""

from evaluation_clusters import report_paired_cluster_intervals


def report_simulation_acceptance(protocol, rows, attempts):
    """Decide the primary C-minus-A claim in percentage points.

    The caller verifies retained attempt artifacts before supplying outcomes.
    Unsupported interval evidence is inconclusive even when the observed
    difference reaches the target.
    """
    intervals = report_paired_cluster_intervals(protocol, rows, attempts)
    contrast = 'fixed_memory_minus_baseline'
    observed = intervals['macro_differences_pp'][contrast]
    primary_interval = intervals['intervals_pp'][contrast]
    lower = primary_interval['lower'] if primary_interval is not None else None

    if intervals['status'] != 'estimated' or observed is None or lower is None:
        decision = 'inconclusive'
        reasons = list(intervals['limitations'])
        if observed is None:
            reasons.append('fixed_memory_minus_baseline: observed macro gain unavailable')
        if lower is None and not reasons:
            reasons.append('fixed_memory_minus_baseline: 95% lower endpoint unavailable')
    else:
        reasons = []
        if observed < 10:
            reasons.append('observed C-minus-A macro gain below 10 percentage points')
        if lower <= 0:
            reasons.append('95% C-minus-A lower endpoint does not exceed zero')
        decision = 'fail' if reasons else 'pass'

    return {'protocol_id': intervals['protocol_id'], 'decision': decision,
            'contrast': contrast, 'observed_gain_pp': observed,
            'minimum_observed_gain_pp': 10, 'lower_endpoint_pp': lower,
            'reasons': reasons, 'interval_report': intervals}
