"""Check declared campaign prerequisites before any trial callback runs."""

from math import isfinite

from evaluation_protocol import ProtocolError
from evaluation_resume import run_evaluation_schedule
from evaluation_schedule import audit_evaluation_schedule
from memory_snapshot import MemorySnapshot
from recorded_replay import TraceError


def preflight_evaluation_campaign(protocol, rows, *, resolved_conditions,
                                  observed_states, snapshot_path,
                                  runtime_caps, resource_records):
    """Return an auditable blocked/ready report without constructing a model or environment.

    Inputs are independently obtained resolution and approval records. The
    caller remains responsible for authenticating their provenance.
    """
    document = protocol.read()
    audit_evaluation_schedule(protocol, rows)
    reasons = []

    def check(ok, message):
        if not ok:
            reasons.append(message)

    declared = document['conditions']
    resolved_conditions = (resolved_conditions if type(resolved_conditions) is dict
                           else {})
    for name in ('baseline', 'no_memory', 'fixed_memory'):
        entry = declared.get(name)
        resolved = resolved_conditions.get(name)
        for field in ('model_id', 'configuration_id'):
            expected = entry.get(field) if type(entry) is dict else None
            actual = resolved.get(field) if type(resolved) is dict else None
            check(type(expected) is str and bool(expected.strip()),
                  f'declare conditions.{name}.{field} in the sealed protocol')
            check(type(actual) is str and bool(actual.strip()) and actual == expected,
                  f'resolve conditions.{name}.{field} to its sealed identity')

    expected_states = document['splits']['development'] + document['splits']['held_out']
    actual_states = observed_states if type(observed_states) is list else []
    check(len(actual_states) == len(expected_states) and
          all(state in actual_states for state in expected_states) and
          all(state in expected_states for state in actual_states),
          'verify every declared development and held_out task/state digest against the installed dataset')

    fixed = declared.get('fixed_memory')
    expected_snapshot = fixed.get('memory') if type(fixed) is dict else None
    check(type(expected_snapshot) is dict and
          set(expected_snapshot) == {'snapshot_id', 'sha256'} and
          all(type(value) is str and len(value) == 64 for value in expected_snapshot.values()),
          'declare the fixed_memory snapshot identity and digest')
    if type(expected_snapshot) is dict and snapshot_path is not None:
        try:
            MemorySnapshot(snapshot_path, expected_reference=expected_snapshot).read()
        except (TraceError, OSError, ValueError, TypeError) as exc:
            reasons.append(f'verify the fixed_memory snapshot and retained records: {exc}')
    else:
        reasons.append('provide the fixed_memory snapshot path for verification')

    for field in ('primary_actions', 'supervisor_calls', 'wall_clock_seconds'):
        expected = document['horizons'].get(field)
        actual = runtime_caps.get(field) if type(runtime_caps) is dict else None
        valid = type(expected) is int and expected > 0 if field != 'wall_clock_seconds' else (
            type(expected) in (int, float) and isfinite(expected) and expected > 0)
        check(valid, f'declare a positive horizons.{field} cap')
        check(actual == expected and type(actual) is type(expected),
              f'configure runtime {field} to the sealed cap')

    records = resource_records if type(resource_records) is dict else {}
    for category in ('compute', 'api', 'data_use'):
        record = records.get(category)
        check(type(record) is dict and record.get('protocol_id') ==
              document['protocol_id'] and record.get('approved') is True and
              type(record.get('reference')) is str and bool(record['reference'].strip()),
              f'provide an approved {category} resource record for this protocol')
    for category, field, cap in (
            ('compute', 'max_seconds', 'wall_clock_seconds'),
            ('api', 'max_model_calls', 'supervisor_calls')):
        record = records.get(category)
        allowance = record.get(field) if type(record) is dict else None
        required = len(rows) * runtime_caps.get(cap, 0) if type(runtime_caps) is dict else 0
        check(type(allowance) in (int, float) and isfinite(allowance) and
              allowance >= required and required > 0,
              f'provide {category}.{field} allowance for all scheduled attempts')

    return {'status': 'blocked' if reasons else 'ready',
            'protocol_id': document['protocol_id'], 'reasons': reasons}


def start_evaluation_campaign(protocol, rows, directory, run, *,
                              resolved_conditions, observed_states,
                              snapshot_path, runtime_caps,
                              resource_records, max_new=None):
    """Preflight all gates before delegating to the sealed schedule runner."""
    report = preflight_evaluation_campaign(
        protocol, rows, resolved_conditions=resolved_conditions,
        observed_states=observed_states, snapshot_path=snapshot_path,
        runtime_caps=runtime_caps, resource_records=resource_records)
    if report['status'] != 'ready':
        raise ProtocolError('campaign preflight blocked: ' + '; '.join(report['reasons']))
    return run_evaluation_schedule(protocol, rows, directory, run, max_new=max_new)
