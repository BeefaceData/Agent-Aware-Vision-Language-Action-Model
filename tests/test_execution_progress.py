"""Crash recovery through the recorder and public readback, without live resources."""

import json
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from episode_harness import ActionResolution, run_episode
from execution_progress import read_execution_progress
from recorded_replay import TraceRecorder, TraceError, load_recorded_replay
from replay_adapters import successful_replay


ROOT = Path(__file__).resolve().parents[1]

# os._exit bypasses exception handling, recorder.finish, atexit and stream close.
CRASH = '''
import os
from pathlib import Path
import sys
from episode_harness import run_episode
from recorded_replay import TraceRecorder
from replay_adapters import successful_replay

path = Path(sys.argv[1])
mode = sys.argv[2]
fixture = successful_replay()
trace = TraceRecorder(path, fixture.config, fixture.recorder)
def stop(step, result):
    if mode == 'tail':
        with (path / 'execution.jsonl').open('ab', buffering=0) as stream:
            stream.write(b'{"record":')
            os.fsync(stream.fileno())
    os._exit(73)
if mode == 'unacknowledged':
    fixture.environment.step = lambda action: os._exit(73)
run_episode(fixture.config, fixture.policy, fixture.environment, trace, on_step=stop)
'''


class ExecutionProgressTests(unittest.TestCase):
    def test_abrupt_process_exit_retains_only_acknowledged_progress(self):
        for mode, steps in (('acknowledged', 1), ('tail', 1), ('unacknowledged', 0)):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                path = Path(directory) / 'replay'
                child = subprocess.run([sys.executable, '-c', CRASH, str(path), mode],
                                       cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(child.returncode, 73, child.stderr)
                report = read_execution_progress(path / 'execution.jsonl')
                self.assertEqual(report['acknowledged_steps'], steps)
                self.assertEqual(report['completion'], 'unknown')
                self.assertEqual(report['later_execution'], 'unknown')
                if steps:
                    record = report['acknowledged_actions'][0]['action_record']
                    self.assertEqual(record['executed_action'], ['reach', 0.25])
                    self.assertIsNotNone(record['execution_acknowledgement'])
                if mode == 'tail':
                    self.assertIn('incomplete final record', report['diagnostic'])
                else:
                    self.assertIsNone(report['diagnostic'])
                with self.assertRaises(TraceError):
                    load_recorded_replay(path)
                cli = subprocess.run([sys.executable, 'execution_progress.py',
                                      str(path / 'execution.jsonl')], cwd=ROOT,
                                     capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(cli.stdout), report)

    def record(self, path, selector=None, recorder=None):
        fixture = successful_replay()
        trace = TraceRecorder(path, fixture.config, recorder or fixture.recorder)
        outcome = run_episode(fixture.config, fixture.policy, fixture.environment,
                              trace, action_selector=selector)
        trace.seal(outcome)
        return outcome

    def test_complete_episode_still_seals_and_replays(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'replay'
            outcome = self.record(path)
            report = read_execution_progress(path / 'execution.jsonl')
            self.assertEqual(report['acknowledged_steps'], 2)
            self.assertIsNone(report['diagnostic'])
            self.assertEqual(report['header']['episode_id'], outcome.episode_id)
            replayed = load_recorded_replay(path).run()
            self.assertEqual((replayed.success, replayed.steps), (True, 2))

    def test_rejected_proposal_is_not_an_executed_action(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'replay'
            self.record(path, lambda proposal: ActionResolution('reject', reason='unsafe'))
            self.assertEqual(read_execution_progress(path / 'execution.jsonl')
                             ['acknowledged_steps'], 0)
            self.assertEqual(load_recorded_replay(path).run().stop_reason, 'proposal_rejected')

    def test_invalid_record_stops_readback_without_losing_valid_prefix(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'replay'
            self.record(path)
            journal = path / 'execution.jsonl'
            lines = journal.read_bytes().splitlines(keepends=True)
            for tail in (b'{broken}\n', lines[2].replace(b'place', b'other'), lines[1]):
                with self.subTest(tail=tail):
                    raw = b''.join(lines[:2]) + tail + lines[2]
                    journal.write_bytes(raw)
                    report = read_execution_progress(journal)
                    self.assertEqual(report['acknowledged_steps'], 1)
                    self.assertIn('invalid record at line 3', report['diagnostic'])
                    self.assertEqual(report['valid_bytes'], sum(map(len, lines[:2])))
                    self.assertEqual(journal.read_bytes(), raw)

    def test_downstream_recorder_failure_keeps_acknowledgement(self):
        fixture = successful_replay()

        def fail(*args):
            raise OSError('video unavailable')

        fixture.recorder.record_step = fail
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'replay'
            trace = TraceRecorder(path, fixture.config, fixture.recorder)
            with self.assertRaisesRegex(OSError, 'video unavailable'):
                run_episode(fixture.config, fixture.policy, fixture.environment, trace)
            report = read_execution_progress(path / 'execution.jsonl')
            self.assertEqual(report['acknowledged_steps'], 1)
            self.assertIsNone(report['diagnostic'])
            with self.assertRaises(TraceError):
                load_recorded_replay(path)

    def test_unsupported_version_and_unfinished_header_are_explicit(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'replay'
            self.record(path)
            journal = path / 'execution.jsonl'
            header = json.loads(journal.read_bytes().splitlines()[0])
            header['record']['version'] = 99
            value = {key: header[key] for key in ('previous', 'record')}
            header['digest'] = sha256(json.dumps(value, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            for raw, diagnostic in ((json.dumps(header).encode() + b'\n', 'unsupported'),
                                    (b'', 'missing'), (b'{', 'incomplete')):
                journal.write_bytes(raw)
                report = read_execution_progress(journal)
                self.assertEqual(report['acknowledged_steps'], 0)
                self.assertIsNone(report['header'])
                self.assertIn(diagnostic, report['diagnostic'])

    def test_disk_sync_failure_stops_episode_and_prevents_sealing(self):
        fixture = successful_replay()
        original_sync = os.fsync

        def sync(fd):
            if fixture.environment.actions:
                raise OSError('disk sync failed')
            original_sync(fd)

        with TemporaryDirectory() as directory:
            path = Path(directory) / 'replay'
            trace = TraceRecorder(path, fixture.config, fixture.recorder)
            with patch('execution_progress.os.fsync', side_effect=sync):
                with self.assertRaisesRegex(OSError, 'disk sync failed'):
                    run_episode(fixture.config, fixture.policy, fixture.environment, trace)
            self.assertEqual(len(fixture.environment.actions), 1)
            with self.assertRaises(TraceError):
                load_recorded_replay(path)


if __name__ == '__main__':
    unittest.main()
