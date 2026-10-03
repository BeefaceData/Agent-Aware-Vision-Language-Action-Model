"""Structural contract for observation-only temporal assessments.

References are model claims, not verified observations or correction authority.
The shared response decoder resolves references against caller-owned context.
"""

CATEGORIES = ('progress', 'suspected_missed_grasp', 'suspected_lost_grasp',
              'stall', 'unknown')


def valid_temporal_diagnosis(value, kind):
    """Accept bounded, inspectable evidence claims without evaluator fields."""
    if value is None:
        return True  # Legacy pass/abstention responses remain readable.
    if (type(value) is not dict or
            set(value) not in ({'category', 'summary', 'evidence'},
                               {'category', 'summary', 'evidence', 'conflicts'}) or
            type(value['category']) is not str or value['category'] not in CATEGORIES or
            type(value['summary']) is not str or not value['summary'].strip() or
            len(value['summary']) > 2000 or
            type(value['evidence']) is not list or len(value['evidence']) > 32):
        return False
    if (value['category'] == 'unknown') != (kind == 'abstain'):
        return False
    if value['category'] != 'unknown' and not value['evidence']:
        return False
    conflicts = value.get('conflicts', [])
    if (type(conflicts) is not list or len(conflicts) > 16 or
            any(type(conflict) is not dict or
                set(conflict) != {'case', 'evidence_indices'} or
                conflict['case'] not in ('grasp_state_conflict', 'ambiguous_object_motion') or
                type(conflict['evidence_indices']) is not list or
                not 2 <= len(conflict['evidence_indices']) <= 32 or
                any(type(index) is not int or not 0 <= index < len(value['evidence'])
                    for index in conflict['evidence_indices']) or
                len(set(conflict['evidence_indices'])) != len(conflict['evidence_indices'])
                for conflict in conflicts)):
        return False
    return all(type(item) is dict and
               set(item) == {'observation_sequence', 'source', 'description'} and
               type(item['observation_sequence']) is int and item['observation_sequence'] >= 0 and
               type(item['source']) is str and item['source'] in ('main', 'wrist', 'robot_state') and
               type(item['description']) is str and bool(item['description'].strip()) and
               len(item['description']) <= 2000 for item in value['evidence'])
