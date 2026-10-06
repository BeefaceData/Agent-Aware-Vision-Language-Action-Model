"""Fallback admission through complete public episodes and sealed replay."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from baseline_fallback import BaselineFallback
from episode_harness import (ActionResolution, EpisodeConfig, SupervisorAbstention,
                             SupervisorResponseError, run_episode)
from fallback_fixtures import MeasuredReplayEnvironment
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayPolicy, ReplayRecorder, ReplayStep
from test_episode_timing import ControlledClock


def abstain(p):
    return SupervisorAbstention(p.observation.episode_id, p.observation.sequence,
                               p.proposal_id, 'uncertain', {'history': 'unknown'})


class FallbackTests(unittest.TestCase):
    def episode(self, directory, *, health=lambda: True, valid=lambda p: p.action == [.1],
                change=lambda p: p, delay=0, mode='abstain', configured=True):
        clock = ControlledClock()
        observations = [{'robot_state': {'position': [i]}} for i in range(3)]
        actions = [[.1], [.1]]
        config = EpisodeConfig(17, 2)
        class Policy:
            def reset(self):
                pass

            def act(self, packet):
                return deepcopy(actions[packet.sequence])

        policy = Policy()

        class Environment(MeasuredReplayEnvironment):
            def reset(self, seed, episode_id):
                return change(super().reset(seed, episode_id))

        environment = Environment(17, observations[0], [
            (a, ReplayStep(observations[i + 1], i, i == 1, i == 1, False))
            for i, a in enumerate(actions)], clock=clock)
        recorder = ReplayRecorder()
        trace = TraceRecorder(directory, config, recorder)

        def decide(p):
            clock.advance(delay)
            if mode == 'timeout':
                raise TimeoutError('PRIVATE provider details')
            if mode == 'malformed':
                raise SupervisorResponseError('PRIVATE invalid response')
            if mode == 'interrupt':
                raise KeyboardInterrupt()
            return abstain(p)

        kwargs = (dict(action_selector=lambda p: ActionResolution(
            'reject', reason='outside adjustment bounds')) if mode == 'reject' else
            dict(supervisor_decider=decide))
        outcome = run_episode(config, policy, environment, trace, clock=clock,
            baseline_fallback=BaselineFallback({'robot_state': .5}, health, valid)
                if configured else None, **kwargs)
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        self.assertEqual(replay.run().stop_reason, outcome.stop_reason)
        return outcome, replay.evidence()['decisions'], environment, recorder

    def test_healthy_abstention_rejection_and_unavailable_supervision_replay(self):
        for mode in ('abstain', 'reject', 'timeout', 'malformed'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                outcome, rows, env, _ = self.episode(Path(tmp) / 'trace', mode=mode)
                self.assertTrue(outcome.success)
                self.assertEqual(env.actions, [[.1], [.1]])
                for row in rows:
                    record = row['action_record']
                    self.assertEqual(record['fallback']['selected'], 'baseline')
                    self.assertEqual(record['proposed_action'], record['executed_action'])
                    self.assertIsNone(record['intervention_budget'])
                    self.assertNotIn('PRIVATE', json.dumps(record))

    def test_missing_stale_unverified_inputs_refuse_and_replay(self):
        cases = [dict(change=lambda p: replace(p, observation={})),
                 dict(change=lambda p: replace(p, robot_state_capture=None)),
                 dict(change=lambda p: replace(p, robot_state_capture=replace(
                     p.robot_state_capture, observation_sequence=9))),
                 dict(delay=.6), dict(configured=False)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as tmp:
                outcome, rows, env, recorder = self.episode(Path(tmp) / 'trace', **kwargs)
                self.assertEqual((outcome.steps, outcome.stop_reason), (0, 'proposal_rejected'))
                self.assertEqual(env.actions, [])
                record = rows[0]['action_record']
                self.assertEqual(record['fallback']['selected'], 'refuse')
                self.assertIn('required inputs', record['rejection_reason'])
                self.assertIsNone(record['executed_action'])
                self.assertTrue(recorder.finalized)

    def test_controller_or_proposal_check_must_affirm_validity(self):
        def broken(*args):
            raise RuntimeError('PRIVATE transport details')

        for field in ('health', 'valid'):
            for check in (lambda *args: False, lambda *args: None,
                          lambda *args: 1, broken):
                with self.subTest(field=field, check=check), TemporaryDirectory() as tmp:
                    outcome, rows, env, _ = self.episode(Path(tmp) / 'trace', **{field: check})
                    self.assertEqual(outcome.steps, 0)
                    self.assertEqual(env.actions, [])
                    self.assertEqual(rows[0]['action_record']['fallback']['selected'], 'refuse')
                    self.assertNotIn('PRIVATE', json.dumps(rows))

    def test_checks_run_for_each_current_proposal_and_receive_detached_values(self):
        seen = []

        def validate(p):
            seen.append((p.observation.sequence, p.proposal_id, deepcopy(p.action)))
            p.action[0] = 100
            return p.observation.sequence == 0

        with TemporaryDirectory() as tmp:
            outcome, rows, env, _ = self.episode(Path(tmp) / 'trace', valid=validate)
            self.assertEqual(outcome.steps, 1)
            self.assertEqual(env.actions, [[.1]])
            self.assertEqual([p[0] for p in seen], [0, 1])
            self.assertNotEqual(seen[0][1], seen[1][1])
            self.assertEqual(rows[1]['action_record']['proposed_action'], [.1])

    def test_rejected_or_unavailable_supervision_cannot_bypass_controller_fault(self):
        for mode in ('reject', 'timeout', 'malformed'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                outcome, rows, env, _ = self.episode(Path(tmp) / 'trace', mode=mode,
                                                    health=lambda: False)
                self.assertEqual(outcome.steps, 0)
                self.assertEqual(env.actions, [])
                self.assertIn('controller health', rows[0]['action_record']['rejection_reason'])

    def test_invalid_native_action_is_refused_without_dispatch(self):
        clock = ControlledClock()
        obs = {'robot_state': {'position': [0]}}
        for action in ([2.], [], [True], ['invalid']):
            with self.subTest(action=action):
                env = MeasuredReplayEnvironment(17, obs, [], clock=clock)
                recorder = ReplayRecorder()
                # This fixture's declared native action is one finite scalar in [-1, 1].
                def valid(p):
                    return (type(p.action) is list and len(p.action) == 1 and
                            type(p.action[0]) in (int, float) and -1 <= p.action[0] <= 1)
                outcome = run_episode(EpisodeConfig(17, 1), ReplayPolicy([(obs, action)]),
                    env, recorder, clock=clock, supervisor_decider=abstain,
                    baseline_fallback=BaselineFallback({'robot_state': .5}, lambda: True, valid))
                self.assertEqual(outcome.steps, 0)
                self.assertEqual(env.actions, [])
                self.assertFalse(recorder.failures[0][3].action_record.fallback['proposal_valid'])

    def test_age_is_checked_after_slow_host_checks(self):
        # Advance the same clock used by capture and dispatch inside a health check.
        clock = ControlledClock()
        obs = {'robot_state': {'position': [0]}}
        env = MeasuredReplayEnvironment(17, obs, [], clock=clock)
        recorder = ReplayRecorder()

        def slow():
            clock.advance(1)
            return True

        outcome = run_episode(EpisodeConfig(17, 1), ReplayPolicy([(obs, [.1])]),
            env, recorder, clock=clock, supervisor_decider=abstain,
            baseline_fallback=BaselineFallback({'robot_state': .5}, slow, lambda p: True))
        self.assertEqual(outcome.steps, 0)
        self.assertEqual(env.actions, [])
        self.assertEqual(outcome.cumulative_wait_seconds, 1)

    def test_interrupt_never_falls_back(self):
        with TemporaryDirectory() as tmp, self.assertRaises(KeyboardInterrupt) as error:
            self.episode(Path(tmp) / 'trace', mode='interrupt')
        self.assertEqual(error.exception.episode_interruption.steps, 0)

    def test_resealed_prerequisite_or_dispatch_tampering_is_rejected(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trace'
            self.episode(directory)
            original = (directory / 'decisions.jsonl').read_text()
            for changes in ({'controller_healthy': False}, {'proposal_valid': None},
                            {'inputs': []}, {'proposal_id': 'foreign'},
                            {'selected': 'refuse'}, {'checked_at': 102}):
                rows = [json.loads(line) for line in original.splitlines()]
                rows[0]['action_record']['fallback'].update(changes)
                raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                (directory / 'decisions.jsonl').write_bytes(raw)
                path = directory / 'manifest.json'
                manifest = json.loads(path.read_text())
                manifest['files']['decisions.jsonl'] = sha256(raw).hexdigest()
                path.write_text(json.dumps(manifest))
                with self.subTest(changes=changes), self.assertRaisesRegex(TraceError, 'fallback'):
                    load_recorded_replay(directory)


if __name__ == '__main__':
    unittest.main()
