"""Configured horizon readback and complete baseline boundary replays."""

import unittest
from dataclasses import asdict

from episode_harness import EpisodeConfig, run_episode
from libero_adapter import read_action_horizon
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


class ConfiguredEnvironment:
    num_envs = 1

    def __init__(self, limits):
        self.limits = limits

    def call(self, name):
        if name != '_max_episode_steps':
            raise AssertionError(name)
        return self.limits


class ActionHorizonTests(unittest.TestCase):
    def test_default_and_override_readback_are_recordable(self):
        for requested in (None, 3):
            with self.subTest(requested=requested):
                horizon = read_action_horizon(ConfiguredEnvironment((3,)), requested)
                self.assertEqual(asdict(horizon), {
                    'effective': 3, 'requested_override': requested,
                    'source': ('environment_default' if requested is None
                               else 'explicit_override'),
                })

    def test_invalid_readback_and_ignored_override_are_rejected(self):
        for limits in ((), (2, 3), (0,), (-1,), (True,), (2.5,), ('3',), (None,)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                read_action_horizon(ConfiguredEnvironment(limits))
        with self.assertRaisesRegex(ValueError, 'differs from override'):
            read_action_horizon(ConfiguredEnvironment((520,)), 3)
        for requested in (0, -1, True, 2.5, '3'):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                read_action_horizon(ConfiguredEnvironment((3,)), requested)

    def test_complete_boundary_episodes_never_propose_or_execute_extra_action(self):
        for requested in (None, 3):
            for ending in ('success', 'step_limit', 'truncated'):
                with self.subTest(requested=requested, ending=ending):
                    horizon = read_action_horizon(ConfiguredEnvironment((3,)), requested)
                    policy = ReplayPolicy([(i, f'action {i}') for i in range(4)])
                    environment = ReplayEnvironment(17, 0, [
                        (f'action {i}', ReplayStep(
                            i + 1, 0.0, i == 2 and ending == 'success',
                            False, i == 2 and ending in ('success', 'truncated')))
                        for i in range(4)
                    ])
                    recorder = ReplayRecorder()
                    outcome = run_episode(EpisodeConfig(17, horizon.effective),
                                          policy, environment, recorder)
                    self.assertEqual(outcome.steps, 3)
                    self.assertEqual(outcome.stop_reason, ending)
                    self.assertEqual(outcome.success, ending == 'success')
                    self.assertEqual(environment.actions,
                                     ['action 0', 'action 1', 'action 2'])
                    self.assertEqual(len(policy.observations), 3)
                    self.assertEqual(recorder.observations[-1].observation, 3)
                    self.assertTrue(recorder.finalized)
                    self.assertEqual(outcome.artifact_status, 'completed')

    def test_invalid_harness_budget_is_rejected_before_reset(self):
        for limit in (0, -1, True, 1.5, '3', None):
            policy = ReplayPolicy([])
            environment = ReplayEnvironment(17, 0, [])
            recorder = ReplayRecorder()
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'max_steps'):
                run_episode(EpisodeConfig(17, limit), policy, environment, recorder)
            self.assertEqual(recorder.observations, [])
            self.assertEqual(environment.actions, [])


if __name__ == '__main__':
    unittest.main()
