"""Durable acknowledged-action prefixes; never a completed episode or replay."""

from hashlib import sha256
import json
import os
from pathlib import Path


def _bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _validate_action(row, header, index):
    record = row['action_record']
    proposal = f"{header['episode_id']}:{index}"
    ack = record['execution_acknowledgement']
    if not (row['kind'] == 'acknowledged_action' and
            type(row['step']) is int and row['step'] == index and
            row['source_episode_id'] == header['episode_id'] and
            type(row['source_sequence']) is int and row['source_sequence'] == index - 1 and
            record['proposal_id'] == proposal and
            record['disposition'] in ('unmodified', 'overridden') and
            record['proposed_action'] is not None and
            record['selected_action'] is not None and
            record['executed_action'] == record['selected_action'] and
            record['rejection_reason'] is None and
            isinstance(ack, dict) and ack['proposal_id'] == proposal and
            set(ack) == {'proposal_id', 'result_episode_id', 'result_sequence'} and
            (record['disposition'] != 'unmodified' or
             record['executed_action'] == record['proposed_action'])):
        raise ValueError('invalid execution acknowledgement')


class ExecutionJournal:
    """Single-writer, create-only journal. Each append returns after flush + fsync."""

    def __init__(self, path, episode_id, config):
        self._stream = Path(path).open('xb')
        self._previous = None
        self._header = dict(kind='header', version=1, episode_id=episode_id, config=config)
        self._count = 0
        try:
            self._append(self._header)
        except BaseException:
            self._stream.close()
            raise

    def _append(self, value):
        envelope = dict(previous=self._previous, record=value)
        digest = sha256(_bytes(envelope)).hexdigest()
        self._stream.write(_bytes(dict(envelope, digest=digest)) + b'\n')
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._previous = digest

    def acknowledge(self, step, source_episode_id, source_sequence, action_record):
        row = dict(kind='acknowledged_action', step=step,
                   source_episode_id=source_episode_id, source_sequence=source_sequence,
                   action_record=action_record)
        _validate_action(row, self._header, self._count + 1)
        self._append(row)
        self._count += 1

    def close(self):
        self._stream.close()


def read_execution_progress(path):
    """Read only the verified prefix, reporting the first unusable record.

    A newline commits a complete record syntactically; checksums, chaining and
    acknowledgement semantics must also pass. No repair or execution is attempted.
    Even a clean EOF cannot establish task completion or absence of later actions.
    """
    header = None
    actions = []
    previous = None
    diagnostic = None
    valid_bytes = 0
    with Path(path).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                diagnostic = f'incomplete final record at line {line_number}'
                break
            try:
                envelope = json.loads(line)
                row = envelope['record']
                expected = sha256(_bytes(dict(previous=previous, record=row))).hexdigest()
                if envelope['previous'] != previous or envelope['digest'] != expected:
                    raise ValueError('checksum or chain mismatch')
                if header is None:
                    if not (row['kind'] == 'header' and type(row['version']) is int and
                            row['version'] == 1 and isinstance(row['episode_id'], str) and
                            row['episode_id'] and isinstance(row['config'], dict)):
                        raise ValueError('invalid or unsupported journal header')
                    header = row
                else:
                    _validate_action(row, header, len(actions) + 1)
                    actions.append(row)
                previous = expected
                valid_bytes += len(line)
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                diagnostic = f'invalid record at line {line_number}: {exc}'
                break
    if header is None and diagnostic is None:
        diagnostic = 'missing journal header'
    return dict(status='execution_prefix_only', header=header,
                acknowledged_actions=actions, acknowledged_steps=len(actions),
                valid_bytes=valid_bytes, diagnostic=diagnostic,
                completion='unknown', later_execution='unknown')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='execution.jsonl from a replay directory')
    args = parser.parse_args()
    print(json.dumps(read_execution_progress(args.path), indent=2))
