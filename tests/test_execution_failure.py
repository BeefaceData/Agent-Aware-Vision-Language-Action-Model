"""Whole attempted episodes with deterministic controller fault replay."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import ActionResolution, EpisodeConfig, run_episode
from execution_failure import ExecutionFailure
from recorded_replay import TraceRecorder, TraceError, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


class ExecutionFailureTests(unittest.TestCase):
    def replay(self, error, confirmation=True, selector=False, broken_recorder=False):
        class Environment(ReplayEnvironment):
            def step(self, action):
                if self.actions:
                    self.attempted = action
                    if error is None:
                        return None
                    raise error
                return super().step(action)

            def interrupt(self, request):
                super().interrupt(request)
                if isinstance(confirmation, BaseException):
                    raise confirmation
                return confirmation

        class Recorder(ReplayRecorder):
            def record_failure(self, *args):
                if broken_recorder:
                    raise OSError('sink unavailable')
                super().record_failure(*args)

        env = Environment(17, 'initial', [('go', ReplayStep('moving', .5, False, False, False))])
        policy = ReplayPolicy([('initial', 'go'), ('moving', 'finish')])
        recorder = Recorder()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace'
            config = EpisodeConfig(17, 3)
            trace = TraceRecorder(path, config, recorder)
            kwargs = (dict(action_selector=lambda p: ActionResolution('pass')
                if p.observation.sequence == 0 else ActionResolution('override', 'adjusted'))
                if selector else {})
            with self.assertRaises(BaseException) as caught:
                run_episode(config, policy, env, trace, **kwargs)
            if error is not None:
                self.assertIs(caught.exception, error)
            partial = caught.exception.episode_interruption
            self.assertEqual((partial.steps, partial.sum_rewards, partial.task_status),
                             (1, .5, 'unknown'))
            self.assertEqual(len(partial.acknowledged_actions), 1)
            self.assertEqual(len(policy.observations), 2)
            self.assertEqual(env.actions, ['go'])
            self.assertEqual(len(env.interruptions), 1)
            record = partial.failed_action
            self.assertEqual(record.proposed_action, 'finish')
            self.assertEqual(record.selected_action, 'adjusted' if selector else 'finish')
            self.assertIsNone(record.executed_action)
            self.assertIsNone(record.execution_acknowledgement)
            self.assertEqual(record.disposition, 'unconfirmed')
            self.assertTrue(recorder.finalized)
            self.assertFalse((path / 'manifest.json').exists())
            with self.assertRaises(TraceError):
                load_recorded_replay(path)
            if not broken_recorder:
                self.assertEqual(recorder.failures[0][3].action_record, record)
                rows = [json.loads(line) for line in (path / 'decisions.jsonl').read_text().splitlines()]
                self.assertEqual(rows[-1]['action_record'], asdict(record))
            return record

    def test_transport_certainty_is_separate_from_acknowledged_action_count(self):
        for sent in (True, False, None):
            for selector in (False, True):
                with self.subTest(sent=sent, selector=selector):
                    record = self.replay(ExecutionFailure('PRIVATE', sent=sent), selector=selector)
                    self.assertEqual(record.dispatch,
                        dict(attempted=True, sent=sent, acknowledged=False, error_type='ExecutionFailure'))
                    self.assertTrue(record.interruption['confirmed'])
                    self.assertNotIn('PRIVATE', json.dumps(asdict(record)))

    def test_timeout_and_missing_result_leave_transmission_unknown(self):
        for error in (TimeoutError('lost receipt'), None, KeyboardInterrupt()):
            with self.subTest(error=error):
                self.assertIsNone(self.replay(error).dispatch['sent'])

    def test_failed_stop_does_not_replace_execution_error(self):
        for confirmation in (False, None, TimeoutError('PRIVATE')):
            with self.subTest(confirmation=confirmation):
                record = self.replay(ExecutionFailure(sent=True), confirmation)
                self.assertFalse(record.interruption['confirmed'])

    def test_failure_evidence_survives_recorder_failure(self):
        record = self.replay(ExecutionFailure(sent=True), broken_recorder=True)
        self.assertTrue(record.interruption['confirmed'])

    def test_invalid_transport_certainty_is_rejected(self):
        with self.assertRaises(ValueError):
            ExecutionFailure(sent=1)


if __name__ == '__main__':
    unittest.main()
