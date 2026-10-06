"""Supervisor allowances through public episodes and sealed replay; no inference."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from episode_harness import EpisodeConfig, run_episode
from fallback_fixtures import MeasuredReplayEnvironment, healthy_fallback
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_provider import BoundedSupervisorProvider
from test_supervisor_provider import response


class SupervisorAllowanceTests(unittest.TestCase):
    def episode(self, directory, *, limit=2, interval=3, events=(), mode='pass',
                policy='baseline_fallback', fallback=True, callback='decider'):
        config = EpisodeConfig(17, 7, interval, max_supervisor_calls=limit,
                               supervisor_exhaustion_policy=policy)
        observations = [{'task': 'place item', 'robot_state': {'position': [i]}} for i in range(8)]
        actions = [[i] for i in range(7)]
        env = MeasuredReplayEnvironment(17, observations[0], [
            (action, ReplayStep(observations[i + 1], 0, i == 6, i == 6, False))
            for i, action in enumerate(actions)])
        calls = []
        def provider(p, deadline, cancel):
            calls.append(p.observation.sequence)
            if mode == 'error':
                raise OSError('PRIVATE')
            if mode == 'malformed':
                return {'invalid': 'PRIVATE'}
            return response(p, deadline, cancel)
        recorder = ReplayRecorder()
        trace = TraceRecorder(directory, config, recorder)
        kwargs = dict(supervisor_decider=BoundedSupervisorProvider(provider, 1),
                      assessment_trigger=lambda p: p.observation.sequence in events)
        if callback != 'decider':
            kwargs = {callback: lambda *args: calls.append(len(calls))}
        outcome = run_episode(config, ReplayPolicy(list(zip(observations, actions))), env,
                              trace, baseline_fallback=healthy_fallback() if fallback else None,
                              **kwargs)
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        replayed = replay.run()
        self.assertEqual(replayed.supervisor_call_budget, outcome.supervisor_call_budget)
        self.assertEqual(replayed.stop_reason, outcome.stop_reason)
        return outcome, calls, replay.evidence()['decisions'], env

    def test_periodic_event_and_overlap_share_one_allowance(self):
        for events, expected in (((), [0, 3]), ((1, 3, 4), [0, 1])):
            with self.subTest(events=events), TemporaryDirectory() as tmp:
                outcome, calls, rows, env = self.episode(Path(tmp) / 'trace', events=events)
                self.assertTrue(outcome.success)
                self.assertEqual(calls, expected)
                self.assertEqual(outcome.supervisor_call_budget, dict(
                    limit=2, attempted=2, exhausted=True, policy='baseline_fallback'))
                refused = [r['action_record'] for r in rows if
                           (r['action_record']['supervisor_call_budget'] or {}).get('admitted') is False]
                self.assertTrue(refused)
                self.assertTrue(all(r['fallback']['selected'] == 'baseline' for r in refused))
                self.assertEqual(len(env.actions), 7)

    def test_failed_and_malformed_requests_consume_allowance(self):
        for mode in ('error', 'malformed'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                outcome, calls, rows, _ = self.episode(Path(tmp) / 'trace', mode=mode, interval=1)
                self.assertEqual(calls, [0, 1])
                self.assertEqual(outcome.supervisor_call_budget['attempted'], 2)
                self.assertNotIn('PRIVATE', json.dumps(rows))
                self.assertTrue(outcome.success)

    def test_stop_overrides_healthy_fallback_and_missing_fallback_refuses(self):
        for policy, fallback in (('stop', True), ('baseline_fallback', False)):
            with self.subTest(policy=policy), TemporaryDirectory() as tmp:
                outcome, calls, rows, env = self.episode(Path(tmp) / 'trace', limit=0,
                                                       policy=policy, fallback=fallback)
                self.assertEqual(calls, [])
                self.assertEqual(env.actions, [])
                self.assertEqual(outcome.stop_reason, 'proposal_rejected')
                self.assertTrue(outcome.supervisor_call_budget['exhausted'])
                self.assertTrue(rows[0]['action_record']['interruption']['confirmed'])

    def test_observation_callbacks_cannot_bypass_cap(self):
        for callback in ('supervisor', 'window_supervisor'):
            with self.subTest(callback=callback), TemporaryDirectory() as tmp:
                outcome, calls, _, _ = self.episode(Path(tmp) / 'trace', interval=1,
                                                   callback=callback)
                self.assertEqual(len(calls), 2)
                self.assertTrue(outcome.success)

    def test_exact_allowance_without_another_request_is_not_exhaustion(self):
        with TemporaryDirectory() as tmp:
            outcome, calls, _, _ = self.episode(Path(tmp) / 'trace', limit=3)
            self.assertEqual(calls, [0, 3, 6])
            self.assertFalse(outcome.supervisor_call_budget['exhausted'])

    def test_new_episode_resets_allowance_and_timeout_does_not_refund(self):
        release, entered = Event(), Event()
        calls = []
        def slow(p, deadline, cancel):
            calls.append(p.observation.episode_id)
            entered.set()
            release.wait(5)
            return response(p, deadline, cancel)
        provider = BoundedSupervisorProvider(slow, .02)
        obs = {'robot_state': {'position': [0]}}
        config = EpisodeConfig(17, 2, max_supervisor_calls=1,
                               supervisor_exhaustion_policy='baseline_fallback')
        try:
            for _ in range(2):
                env = MeasuredReplayEnvironment(17, obs, [
                    ([0], ReplayStep(obs, 0, False, False, False)),
                    ([0], ReplayStep(obs, 0, True, True, False))])
                outcome = run_episode(config, ReplayPolicy([(obs, [0]), (obs, [0])]),
                                      env, ReplayRecorder(), supervisor_decider=provider,
                                      baseline_fallback=healthy_fallback())
                self.assertEqual(outcome.supervisor_call_budget['attempted'], 1)
                self.assertTrue(outcome.supervisor_call_budget['exhausted'])
            self.assertTrue(entered.is_set())
            self.assertEqual(len(calls), 1)  # New episode's busy refusal also costs allowance.
        finally:
            release.set()

    def test_nonzero_stop_retains_executed_actions_and_call_count(self):
        with TemporaryDirectory() as tmp:
            outcome, calls, rows, env = self.episode(Path(tmp) / 'trace', policy='stop')
            self.assertEqual(calls, [0, 3])
            self.assertEqual(outcome.steps, 6)
            self.assertEqual(len(env.actions), 6)
            self.assertEqual(outcome.stop_reason, 'proposal_rejected')
            self.assertTrue(rows[-1]['action_record']['interruption']['confirmed'])
            self.assertEqual(outcome.supervisor_call_budget['attempted'], 2)

    def test_legacy_bundle_retains_original_configuration_evidence(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trace'
            self.episode(directory, limit=3)
            manifest_path = directory / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            del manifest['config']['max_supervisor_calls']
            del manifest['config']['supervisor_exhaustion_policy']
            del manifest['supervisor_call_budget']
            rows = [json.loads(line) for line in
                    (directory / 'decisions.jsonl').read_text().splitlines()]
            for row in rows:
                del row['action_record']['supervisor_call_budget']
            raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            (directory / 'decisions.jsonl').write_bytes(raw)
            manifest['files']['decisions.jsonl'] = sha256(raw).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            replay = load_recorded_replay(directory)
            self.assertTrue(replay.run().success)
            self.assertIsNone(replay.run().supervisor_call_budget)
            self.assertEqual(replay.evidence()['config'], manifest['config'])

    def test_invalid_configuration(self):
        for limit in (-1, True, 1.2, '2', float('inf')):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                EpisodeConfig(17, 3, max_supervisor_calls=limit)
        with self.assertRaises(ValueError):
            EpisodeConfig(17, 3, supervisor_exhaustion_policy='ignore')

    def test_resealed_budget_tampering_is_rejected(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trace'
            self.episode(directory)
            original = (directory / 'decisions.jsonl').read_text()
            manifest_path = directory / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            for changes in ({'attempted': 0}, {'admitted': False}, {'limit': 99}, {'policy': 'stop'}):
                rows = [json.loads(line) for line in original.splitlines()]
                rows[0]['action_record']['supervisor_call_budget'].update(changes)
                raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                (directory / 'decisions.jsonl').write_bytes(raw)
                manifest['files']['decisions.jsonl'] = sha256(raw).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                with self.subTest(changes=changes), self.assertRaisesRegex(TraceError, 'supervisor'):
                    load_recorded_replay(directory)


if __name__ == '__main__':
    unittest.main()
