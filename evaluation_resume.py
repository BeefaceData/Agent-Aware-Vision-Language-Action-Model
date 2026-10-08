"""Resume a sealed schedule without reusing or silently replacing attempts."""

from dataclasses import asdict
import json
from pathlib import Path

from artifact_bundle import verify_artifact_bundle
from episode_harness import EpisodeOutcome
from evaluation_protocol import ProtocolError
from evaluation_schedule import audit_evaluation_schedule
from recorded_replay import TraceError, load_recorded_replay


def _verify_episode(directory, outcome):
    try:
        if (directory / 'bundle.json').exists():
            replay = verify_artifact_bundle(directory)
        else:
            replay = load_recorded_replay(directory / 'replay')
        replayed = replay.run()
    except (TraceError, OSError, ValueError, KeyError) as exc:
        raise ProtocolError(f'invalid retained episode evidence: {directory.name}') from exc
    if (replay.source_episode_id != outcome.get('episode_id') or
            any(getattr(replayed, key) != outcome.get(key) for key in
                ('success', 'steps', 'stop_reason', 'sum_rewards'))):
        raise ProtocolError(f'outcome differs from replay: {directory.name}')


def _record(path, row):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ProtocolError(f'incomplete or invalid attempt: {row["attempt_id"]}') from exc
    if (type(value) is not dict or value.get('version') != 1 or
            value.get('schedule_entry') != row or
            type(value.get('outcome')) is not dict or
            value['outcome'].get('artifact_status') != 'completed'):
        raise ProtocolError(f'inconsistent or incomplete attempt: {row["attempt_id"]}')
    _verify_episode(path.parent, value['outcome'])
    return value['outcome']


def run_evaluation_schedule(protocol, rows, directory, run, *, max_new=None):
    """Run missing rows in order, returning preserved and new outcomes.

    ``run(row, attempt_dir)`` executes one attempt and returns a completed
    EpisodeOutcome. A reserved directory without a complete record is blocked:
    the protocol's pre-start exclusion or post-start failure rule must be
    applied explicitly before any replacement. This interface does not decide
    that disposition or grant permission to launch live trials.
    """
    audit_evaluation_schedule(protocol, rows)
    if max_new is not None and (type(max_new) is not int or max_new < 0):
        raise ProtocolError('max_new must be a nonnegative integer')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    known = {row['attempt_id'] for row in rows}
    unexpected = {path.name for path in root.iterdir()} - known
    if unexpected:
        raise ProtocolError(f'unplanned attempt entries: {sorted(unexpected)}')

    # Audit all retained evidence before executing any new row. A failed or
    # interrupted attempt must never be mistaken for an unused identity.
    completed = {}
    for row in rows:
        attempt = root / row['attempt_id']
        if attempt.exists():
            if not attempt.is_dir():
                raise ProtocolError(f'invalid attempt destination: {row["attempt_id"]}')
            completed[row['attempt_id']] = _record(attempt / 'schedule-result.json', row)
    episode_ids = [outcome.get('episode_id') for outcome in completed.values()]
    if len(set(episode_ids)) != len(episode_ids):
        raise ProtocolError('duplicate recorded episode identity')

    new_count = 0
    for row in rows:
        identity = row['attempt_id']
        if identity in completed:
            continue
        if max_new is not None and new_count >= max_new:
            break
        attempt = root / identity
        attempt.mkdir(exist_ok=False)
        outcome = run(dict(row), attempt)
        if not isinstance(outcome, EpisodeOutcome):
            raise ProtocolError(f'run did not return an episode outcome: {identity}')
        result = asdict(outcome)
        if result.get('artifact_status') != 'completed':
            raise ProtocolError(f'incomplete episode artifacts: {identity}')
        if outcome.episode_id in episode_ids:
            raise ProtocolError(f'duplicate episode identity: {outcome.episode_id}')
        _verify_episode(attempt, result)
        # Retain the outcome exactly once. Any interruption after reservation
        # remains visible and blocks a blind replay under the same identity.
        with (attempt / 'schedule-result.json').open('x', encoding='utf-8') as stream:
            json.dump({'version': 1, 'schedule_entry': row, 'outcome': result},
                      stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
        completed[identity] = _record(attempt / 'schedule-result.json', row)
        episode_ids.append(outcome.episode_id)
        new_count += 1
    return [completed[row['attempt_id']] for row in rows
            if row['attempt_id'] in completed]
