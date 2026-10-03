"""Public accounting with deterministic transports; no live model calls."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from episode_harness import EpisodeConfig, run_episode
from model_usage import ModelCallJournal, ModelReply
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_provider import BoundedSupervisorProvider, ProviderRequestError
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings
from test_supervisor_vlm import PNG, pass_message, proposal, raw


class ModelUsageTests(unittest.TestCase):
    def test_success_failure_explicit_retry_and_unknown_totals(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'calls.jsonl'
            journal = ModelCallJournal(path)
            with patch('model_usage.monotonic', side_effect=[10., 12., 20., 23., 30., 34.]):
                value = journal.call(proposal(), 'fixture-provider', 'frozen-model',
                    lambda: ModelReply('response', {'input_tokens': 7, 'output_tokens': 2},
                                       .25, 'USD'))
                self.assertEqual(value, 'response')
                known = journal.summary()
                self.assertEqual(known['total_cost_by_currency'], {'USD': .25})
                self.assertEqual(known['total_usage'], {'input_tokens': 7, 'output_tokens': 2})

                def failed():
                    raise OSError('SECRET provider payload')

                with self.assertRaises(OSError):
                    journal.call(proposal(), 'fixture-provider', 'frozen-model', failed)
                failed_id = journal.summary()['calls'][-1]['call_id']
                journal.call(proposal(), 'fixture-provider', 'frozen-model',
                    lambda: ModelReply('retried', {'input_tokens': 8, 'output_tokens': 3}),
                    retry_of=failed_id)
            result = journal.summary()
            self.assertEqual(result['call_count'], 3)
            self.assertEqual(result['failed_calls'], 1)
            self.assertEqual(result['pending_calls'], 0)
            self.assertEqual(result['call_wall_seconds'], 9.)
            self.assertEqual(result['reported_usage_subtotals'],
                             {'input_tokens': 15, 'output_tokens': 5})
            self.assertEqual(result['total_usage'], {'input_tokens': None, 'output_tokens': None})
            self.assertEqual(result['unknown_usage_calls'], 1)
            self.assertEqual(result['reported_cost_subtotals'], {'USD': .25})
            self.assertIsNone(result['total_cost_by_currency'])
            self.assertEqual(result['unknown_cost_calls'], 2)
            self.assertEqual(result['calls'][-1]['retry_of'], failed_id)
            self.assertTrue(all(row['provider'] == 'fixture-provider' and
                                row['model'] == 'frozen-model' for row in result['calls']))
            self.assertNotIn('SECRET', path.read_text())
            self.assertEqual(len(path.read_text().splitlines()), 6)
            result['calls'].clear()
            self.assertEqual(journal.summary()['call_count'], 3)
            with self.assertRaises(FileExistsError):
                ModelCallJournal(path)
            with self.assertRaisesRegex(ValueError, 'another episode'):
                journal.call(proposal(episode='other'), 'p', 'm', lambda: None)
            with self.assertRaisesRegex(ValueError, 'existing journal call'):
                journal.call(proposal(), 'p', 'm', lambda: None, retry_of='missing')

    def test_invalid_usage_is_retained_as_failed_not_free(self):
        for reply in (ModelReply(None, {'tokens': -1}), ModelReply(None, {'tokens': True}),
                      ModelReply(None, cost=float('nan'), currency='USD'),
                      ModelReply(None, cost=1), ModelReply(None, currency='USD')):
            with self.subTest(reply=reply), TemporaryDirectory() as temporary:
                journal = ModelCallJournal(Path(temporary) / 'calls.jsonl')
                with self.assertRaises(ValueError):
                    journal.call(proposal(), 'p', 'm', lambda: reply)
                self.assertEqual(journal.summary()['failed_calls'], 1)
                self.assertIsNone(journal.summary()['total_cost_by_currency'])

    def test_timeout_retains_pending_call_then_late_accounting_only(self):
        release, done = Event(), Event()
        with TemporaryDirectory() as temporary:
            journal = ModelCallJournal(Path(temporary) / 'calls.jsonl')

            def provider(current, deadline, cancel):
                def operation():
                    release.wait(2)
                    return ModelReply(dict(kind='pass', episode_id=current.observation.episode_id,
                        observation_sequence=current.observation.sequence,
                        proposal_id=current.proposal_id), {'tokens': 5})
                try:
                    return journal.call(current, 'p', 'm', operation)
                finally:
                    done.set()

            bounded = BoundedSupervisorProvider(provider, .05)
            try:
                result = bounded.request(proposal())
                self.assertEqual(result.status, 'timeout')
                self.assertEqual(journal.summary()['pending_calls'], 1)
                self.assertIsNone(journal.summary()['call_wall_seconds'])
                self.assertEqual(bounded.request(proposal(1)).status, 'busy')
                self.assertEqual(journal.summary()['call_count'], 1)
            finally:
                release.set()
                self.assertTrue(done.wait(1))
            self.assertEqual(journal.summary()['pending_calls'], 0)
            self.assertEqual(journal.summary()['total_usage'], {'tokens': 5})
            self.assertIsNone(result.decision)
            self.assertEqual(result.status, 'timeout')

    def test_complete_episode_replay_and_failed_attempt_retain_usage(self):
        for broken in (False, True):
            with self.subTest(broken=broken), TemporaryDirectory() as temporary:
                config = EpisodeConfig(17, 3)
                observations = [raw(i) for i in range(3)]
                policy = ReplayPolicy(list(zip(observations, ([0], [1]))))
                environment = ReplayEnvironment(17, observations[0], [
                    ([0], ReplayStep(observations[1], 0, False, False, False)),
                    ([1], ReplayStep(observations[2], 1, True, True, False))])
                directory = Path(temporary) / 'trace'
                trace = TraceRecorder(directory, config, ReplayRecorder())
                journal = ModelCallJournal(directory / 'model_calls.jsonl')

                def transport(payload, deadline, cancel):
                    if broken:
                        raise OSError('SECRET')
                    return {**pass_message(payload, deadline, cancel),
                            'usage': {'input_tokens': 10, 'output_tokens': 3,
                                      'private': 'SECRET'}}

                provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                    VlmSettings('fixture'), lambda frame: PNG, transport,
                    call_journal=journal), 1)
                if broken:
                    with self.assertRaises(ProviderRequestError):
                        run_episode(config, policy, environment, trace, supervisor_decider=provider)
                    self.assertEqual(environment.actions, [])
                    self.assertEqual(journal.summary()['failed_calls'], 1)
                    self.assertEqual(journal.summary()['call_count'], 1)
                else:
                    outcome = run_episode(config, policy, environment, trace, supervisor_decider=provider)
                    trace.seal(outcome)
                    self.assertTrue(load_recorded_replay(directory).run().success)
                    self.assertEqual(journal.summary()['episode_id'], outcome.episode_id)
                    self.assertEqual(journal.summary()['call_count'], 2)
                    self.assertEqual(journal.summary()['total_usage'],
                                     {'input_tokens': 20, 'output_tokens': 6})
                self.assertIsNone(journal.summary()['total_cost_by_currency'])
                self.assertNotIn('SECRET', (directory / 'model_calls.jsonl').read_text())

    def test_invalid_decision_does_not_discard_returned_usage(self):
        with TemporaryDirectory() as temporary:
            journal = ModelCallJournal(Path(temporary) / 'calls.jsonl')
            provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                VlmSettings('fixture'), lambda frame: PNG,
                lambda *args: {'type': 'message', 'stop_reason': 'max_tokens',
                               'usage': {'input_tokens': 7, 'output_tokens': 3}},
                call_journal=journal), 1)
            self.assertEqual(provider.request(proposal()).status, 'error')
            self.assertEqual(journal.summary()['total_usage'],
                             {'input_tokens': 7, 'output_tokens': 3})
            self.assertEqual(journal.summary()['calls'][0]['status'], 'returned')


if __name__ == '__main__':
    unittest.main()
