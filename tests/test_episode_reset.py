"""Attempt isolation through public harness and production environment contracts."""

import json
import unittest
from dataclasses import asdict

from episode_harness import EpisodeConfig, ObservationRejected, run_episode
from libero_adapter import LiberoEnvironmentAdapter
from replay_adapters import ReplayRecorder, successful_replay
from run_smolvla_episode import evidence_json


class QueuedPolicy:
    def __init__(self):
        self.queue = []
        self.seen = []

    def reset(self):
        self.queue.clear()

    def act(self, packet):
        self.seen.append(packet)
        if not self.queue:
            self.queue.extend([packet.observation['command'], 'stale queued action'])
        return self.queue.pop(0)


class VectorReplay:
    num_envs = 1

    def __init__(self, terminal=True):
        self.terminal = terminal
        self.seeds = []
        self.actions = []
        self.fail_reset = False
        self.invalid_observation = False

    def reset(self, seed):
        self.seeds.append(seed)
        if self.fail_reset:
            raise TimeoutError('reset transport failed')
        self.command = f'move for seed {seed[0]}'
        return (None if self.invalid_observation else {'command': self.command}), {}

    def step(self, action):
        if action != self.command:
            raise AssertionError('action leaked from previous attempt')
        self.actions.append(action)
        return {'command': 'terminal'}, [1.0], [self.terminal], [False], {
            'is_success': [self.terminal]}


class EpisodeResetTests(unittest.TestCase):
    def test_two_complete_episodes_discard_queued_actions_and_terminal_state(self):
        raw = VectorReplay()
        environment = LiberoEnvironmentAdapter(raw)
        policy = QueuedPolicy()
        outcomes, records = [], []
        for seed in (11, 22):
            recorder = ReplayRecorder()
            outcome = run_episode(EpisodeConfig(seed, 3), policy, environment, recorder)
            outcomes.append(outcome)
            records.append(recorder)
            self.assertTrue(outcome.success)
            self.assertEqual(outcome.steps, 1)
            self.assertEqual(outcome.sum_rewards, 1.0)
            self.assertEqual(outcome.artifact_status, 'completed')
            self.assertEqual([p.sequence for p in recorder.observations], [0, 1])
            self.assertEqual({p.episode_id for p in recorder.observations},
                             {outcome.episode_id})
            with self.assertRaisesRegex(RuntimeError, 'reset is required'):
                environment.step('stale queued action')
        self.assertNotEqual(outcomes[0].episode_id, outcomes[1].episode_id)
        self.assertEqual(raw.seeds, [[11], [22]])
        self.assertEqual(raw.actions, ['move for seed 11', 'move for seed 22'])
        self.assertEqual([p.sequence for p in policy.seen], [0, 0])
        self.assertEqual(records[0].observations[0].observation['command'],
                         'move for seed 11')

    def test_reset_faults_are_pre_start_records_and_retry_gets_new_identity(self):
        for stage in ('policy_reset', 'environment_reset'):
            for error in (RuntimeError('reset failed'), TimeoutError('deadline'),
                          KeyboardInterrupt('cancelled')):
                with self.subTest(stage=stage, error=type(error).__name__):
                    fixture = successful_replay()
                    resetting = True
                    calls = []

                    class Policy:
                        def reset(self):
                            calls.append('policy_reset')
                            if resetting and stage == 'policy_reset':
                                raise error
                            fixture.policy.reset()

                        def act(self, packet):
                            return fixture.policy.act(packet)

                    class Environment:
                        def reset(self, seed, episode_id):
                            calls.append('environment_reset')
                            if resetting and stage == 'environment_reset':
                                raise error
                            return fixture.environment.reset(seed, episode_id)

                        def step(self, action):
                            return fixture.environment.step(action)

                    policy, environment = Policy(), Environment()
                    with self.assertRaises(type(error)) as caught:
                        run_episode(fixture.config, policy, environment, fixture.recorder)
                    self.assertIs(caught.exception, error)
                    partial = caught.exception.episode_interruption
                    record = json.loads(json.dumps(asdict(partial), default=evidence_json))
                    self.assertEqual(record['pre_start_failure'], {
                        'stage': stage, 'seed': fixture.config.seed,
                        'policy_reset_completed': stage == 'environment_reset',
                        'environment_reset_completed': False})
                    self.assertEqual(record['task_status'], 'unknown')
                    self.assertEqual(record['artifact_status'], 'incomplete')
                    self.assertEqual(record['steps'], 0)
                    self.assertIsNone(record['last_observation'])
                    self.assertEqual(record['acknowledged_actions'], [])
                    self.assertEqual(fixture.recorder.observations, [])
                    self.assertFalse(fixture.recorder.finalized)
                    self.assertEqual(calls, ['policy_reset'] if stage == 'policy_reset'
                                     else ['policy_reset', 'environment_reset'])
                    resetting = False
                    outcome = run_episode(fixture.config, policy, environment,
                                          fixture.recorder)
                    self.assertTrue(outcome.success)
                    self.assertNotEqual(outcome.episode_id, partial.episode_id)

    def test_old_initial_packet_cannot_start_another_episode(self):
        fixture = successful_replay()
        first = run_episode(fixture.config, fixture.policy, fixture.environment,
                            fixture.recorder)
        stale = fixture.recorder.observations[-1]

        class Environment:
            def reset(self, seed, episode_id):
                return stale

        recorder = ReplayRecorder()
        with self.assertRaises(ObservationRejected) as caught:
            run_episode(fixture.config, fixture.policy, Environment(), recorder)
        partial = caught.exception.episode_interruption
        self.assertNotEqual(partial.episode_id, first.episode_id)
        self.assertEqual(partial.pre_start_failure.stage, 'initial_observation')
        self.assertTrue(partial.pre_start_failure.environment_reset_completed)
        self.assertIsNone(partial.last_observation)
        self.assertEqual(recorder.observations, [])

    def test_failed_environment_reset_disables_stepping_until_successful_reset(self):
        for failure in ('transport', 'packet'):
            with self.subTest(failure=failure):
                raw = VectorReplay(terminal=False)
                environment = LiberoEnvironmentAdapter(raw)
                policy = QueuedPolicy()
                first = run_episode(EpisodeConfig(11, 1), policy, environment,
                                    ReplayRecorder())
                self.assertEqual(first.stop_reason, 'step_limit')
                raw.fail_reset = failure == 'transport'
                raw.invalid_observation = failure == 'packet'
                with self.assertRaises((TimeoutError, ValueError)) as caught:
                    run_episode(EpisodeConfig(22, 1), policy, environment, ReplayRecorder())
                self.assertEqual(caught.exception.episode_interruption.pre_start_failure.stage,
                                 'environment_reset')
                with self.assertRaisesRegex(RuntimeError, 'reset is required'):
                    environment.step('move for seed 11')
                self.assertEqual(raw.actions, ['move for seed 11'])
                raw.fail_reset = raw.invalid_observation = False
                last = run_episode(EpisodeConfig(22, 1), policy, environment,
                                   ReplayRecorder())
                self.assertEqual(last.steps, 1)
                self.assertEqual(raw.actions, ['move for seed 11', 'move for seed 22'])


if __name__ == '__main__':
    unittest.main()
