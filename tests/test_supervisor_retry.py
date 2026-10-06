"""Explicit retries through complete episodes and sealed replay, without inference."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from episode_harness import EpisodeConfig, run_episode
from fallback_fixtures import MeasuredReplayEnvironment, healthy_fallback
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_provider import BoundedSupervisorProvider, ProviderRequestError
from supervisor_retry import RecoverableProviderError
from supervisor_vlm import AnthropicMessagesTransport, ChronologicalVlmAdapter, VlmSettings
from time import monotonic
from model_usage import ModelCallJournal
from test_supervisor_provider import response
from test_supervisor_vlm import PNG, proposal, pass_message


class SupervisorRetryTests(unittest.TestCase):
    def test_http_error_classification_is_explicit(self):
        for status, expected in ((429, 'rate_limited'), (503, 'unavailable'), (401, None), (500, None)):
            with self.subTest(status=status), patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'fixture'}), \
                    patch('supervisor_vlm.http.client.HTTPSConnection') as connection:
                connection.return_value.getresponse.return_value.status = status
                with self.assertRaises(RuntimeError) as caught:
                    AnthropicMessagesTransport()({}, monotonic() + 1, Event())
                self.assertEqual(getattr(caught.exception, 'error_class', None), expected)

    def test_count_exhaustion_without_fallback_records_exceptional_attempt(self):
        calls = []
        def provider(*args):
            calls.append(1)
            raise RecoverableProviderError('unavailable')
        obs = {'task': 'place'}
        config = EpisodeConfig(17, 1, max_supervisor_calls=3,
                               max_episode_seconds=2, supervisor_max_retries=1)
        env = MeasuredReplayEnvironment(17, obs, [])
        recorder = ReplayRecorder()
        with self.assertRaises(ProviderRequestError):
            run_episode(config, ReplayPolicy([(obs, [0])]), env, recorder,
                        supervisor_decider=BoundedSupervisorProvider(provider, .5))
        self.assertEqual(len(calls), 2)
        self.assertEqual(env.actions, [])
        self.assertEqual(recorder.failures[0][-1].action_record.supervisor_call_budget['attempted'], 2)

    def test_vlm_retry_reuses_history_and_journals_each_transport_attempt(self):
        payloads = []
        def transport(payload, deadline, cancel):
            payloads.append(payload)
            if len(payloads) == 1:
                raise RecoverableProviderError('unavailable')
            return pass_message(payload, deadline, cancel)
        with TemporaryDirectory() as tmp:
            journal = ModelCallJournal(Path(tmp)/'calls.jsonl')
            provider = BoundedSupervisorProvider(ChronologicalVlmAdapter(
                VlmSettings('fixture'), lambda image: PNG, transport,
                call_journal=journal), 1)
            self.assertEqual(provider.request(proposal()).error_class, 'unavailable')
            self.assertEqual(provider.request(proposal()).status, 'response')
            self.assertEqual(payloads[0], payloads[1])
            rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows if r['event'] == 'finished'],
                             ['error', 'returned'])
            self.assertEqual(provider.request(proposal(1)).status, 'response')

    def episode(self, path, provider, **settings):
        config = EpisodeConfig(17, 1, max_episode_seconds=settings.pop('seconds', 2),
            max_supervisor_calls=settings.pop('calls', 3),
            supervisor_max_retries=settings.pop('retries', 1),
            supervisor_retry_delay_seconds=settings.pop('delay', .002), **settings)
        obs = {'task': 'place', 'robot_state': {'position': [0]}}
        env = MeasuredReplayEnvironment(17, obs, [([0], ReplayStep(obs, 0, True, True, False))])
        trace = TraceRecorder(path, config, ReplayRecorder())
        outcome = run_episode(config, ReplayPolicy([(obs, [0])]), env, trace,
            supervisor_decider=BoundedSupervisorProvider(provider, .5),
            baseline_fallback=healthy_fallback())
        trace.seal(outcome)
        replay = load_recorded_replay(path)
        actual = replay.run()
        self.assertEqual(actual.supervisor_call_budget, outcome.supervisor_call_budget)
        self.assertEqual(actual.stop_reason, outcome.stop_reason)
        self.assertEqual(actual.steps, outcome.steps)
        rows = replay.evidence()['decisions']
        evidence = (rows[0]['action_record']['supervisor_call_budget'] if rows else
                    outcome.wall_clock_limit['pending_supervisor_call'])
        self.assertNotIn('PRIVATE', json.dumps(evidence))
        return outcome, evidence, env

    def test_eligible_retry_succeeds_and_retains_both_attempts(self):
        calls = []
        def provider(p, d, c):
            calls.append(p)
            if len(calls) == 1:
                raise RecoverableProviderError('unavailable')
            return response(p, d, c)
        with TemporaryDirectory() as tmp:
            outcome, evidence, env = self.episode(Path(tmp)/'trace', provider)
        self.assertTrue(outcome.success)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(env.actions), 1)
        self.assertEqual([a['status'] for a in evidence['retry_attempts']], ['error', 'response'])
        self.assertEqual(outcome.supervisor_call_budget['attempted'], 2)

    def test_retry_count_exhaustion_does_not_refund_calls(self):
        calls = []
        def provider(*args):
            calls.append(1)
            raise RecoverableProviderError('rate_limited')
        with TemporaryDirectory() as tmp:
            outcome, evidence, _ = self.episode(Path(tmp)/'trace', provider, retries=2, calls=9)
        self.assertEqual(len(calls), 3)
        self.assertEqual(evidence['attempted'], 3)
        self.assertTrue(outcome.success)  # Guarded baseline fallback.
        self.assertFalse(outcome.supervisor_call_budget['exhausted'])

    def test_retry_cannot_bypass_call_cap_or_stop_policy(self):
        def provider(*args):
            raise RecoverableProviderError('unavailable')
        for policy in ('stop', 'baseline_fallback'):
            with self.subTest(policy=policy), TemporaryDirectory() as tmp:
                outcome, evidence, env = self.episode(Path(tmp)/'trace', provider,
                    calls=1, supervisor_exhaustion_policy=policy)
                self.assertEqual(evidence['attempted'], 1)
                self.assertFalse(evidence['admitted'])
                self.assertEqual(len(env.actions), int(policy == 'baseline_fallback'))
                self.assertTrue(outcome.supervisor_call_budget['exhausted'])

    def test_unclassified_malformed_and_excluded_errors_are_not_retried(self):
        for mode in ('generic', 'malformed', 'excluded'):
            calls = []
            def provider(*args):
                calls.append(1)
                if mode == 'generic':
                    raise OSError('PRIVATE')
                if mode == 'excluded':
                    raise RecoverableProviderError('rate_limited')
                return {'invalid': 'PRIVATE'}
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                self.episode(Path(tmp)/'trace', provider,
                             supervisor_retry_errors=('unavailable',))
            self.assertEqual(len(calls), 1)

    def test_delay_consumes_episode_time_without_starting_another_call(self):
        calls = []
        def provider(*args):
            calls.append(1)
            raise RecoverableProviderError('unavailable')
        with TemporaryDirectory() as tmp:
            outcome, evidence, env = self.episode(Path(tmp)/'trace', provider, seconds=.1, delay=.3)
        self.assertEqual(outcome.stop_reason, 'wall_clock_limit')
        self.assertEqual(len(calls), 1)
        self.assertEqual(evidence['attempted'], 1)
        self.assertEqual(env.actions, [])

    def test_inflight_retry_is_invalidated_at_episode_deadline(self):
        calls, release = [], Event()
        def provider(p, d, c):
            calls.append(1)
            if len(calls) == 1:
                raise RecoverableProviderError('unavailable')
            release.wait(2)
            return response(p, d, c)
        try:
            with TemporaryDirectory() as tmp:
                outcome, evidence, env = self.episode(Path(tmp)/'trace', provider, seconds=.1)
            self.assertEqual(outcome.stop_reason, 'wall_clock_limit')
            self.assertEqual(len(calls), 2)
            self.assertEqual(evidence['retry_attempts'][-1]['status'], 'timeout')
            self.assertEqual(env.actions, [])
        finally:
            release.set()

    def test_resealed_attempt_tampering_is_rejected(self):
        calls = []
        def provider(p, d, c):
            calls.append(1)
            if len(calls) == 1:
                raise RecoverableProviderError('unavailable')
            return response(p, d, c)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace'
            self.episode(path, provider)
            original = (path/'decisions.jsonl').read_text()
            manifest = json.loads((path/'manifest.json').read_text())
            for change in ('class', 'delay', 'count', 'missing'):
                rows = [json.loads(line) for line in original.splitlines()]
                budget = rows[0]['action_record']['supervisor_call_budget']
                if change == 'class':
                    budget['retry_attempts'][0]['error_class'] = None
                elif change == 'delay':
                    budget['retry_attempts'][1]['started_at'] = budget['retry_attempts'][0]['finished_at']
                elif change == 'count':
                    budget['attempted'] = 1
                else:
                    del budget['retry_attempts']
                raw = ''.join(json.dumps(row)+'\n' for row in rows).encode()
                (path/'decisions.jsonl').write_bytes(raw)
                manifest['files']['decisions.jsonl'] = sha256(raw).hexdigest()
                (path/'manifest.json').write_text(json.dumps(manifest))
                with self.subTest(change=change), self.assertRaises(TraceError):
                    load_recorded_replay(path)

    def test_configuration_requires_explicit_bounded_policy(self):
        base = EpisodeConfig(17, 1)
        for kwargs in ({'supervisor_max_retries': True}, {'supervisor_max_retries': -1},
                       {'supervisor_max_retries': 1}, {'supervisor_retry_delay_seconds': float('nan')},
                       {'supervisor_retry_errors': ('malformed',)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(base, **kwargs)


if __name__ == '__main__':
    unittest.main()
