"""Terminal evidence through the production adapter and complete episode API."""

import unittest

import numpy as np

from episode_harness import EpisodeConfig, run_episode
from libero_adapter import LiberoEnvironmentAdapter
from replay_adapters import ReplayRecorder


def observation(value, batched=True):
    shape = (1, 2, 2, 3) if batched else (2, 2, 3)
    return {'pixels': {'image': np.full(shape, value),
                       'image2': np.full(shape, value + 1)},
            'robot_state': {'position': np.full((1, 3) if batched else (3,), value)}}


class AutoResetWrapper:
    num_envs = 1

    def __init__(self, key='final_obs', success=True, truncated=False):
        self.metadata = {'autoreset_mode': 'SameStep'}
        self.key = key
        self.success = success
        self.truncated = truncated
        self.calls = 0
        self.terminal = observation(20, batched=False)
        self.info = {key: [self.terminal], '_' + key: [True],
                     'final_info': {'is_success': [success]},
                     '_final_info': [True], 'is_success': [not success]}

    def reset(self, seed):
        return observation(0), {}

    def step(self, action):
        self.calls += 1
        return observation(90), [float(self.success)], [not self.truncated], [self.truncated], self.info


class Policy:
    def reset(self):
        self.seen = []

    def act(self, packet):
        self.seen.append(packet)
        return [0] * 7


class TerminalObservationTests(unittest.TestCase):
    def test_autoreset_preserves_both_views_state_identity_and_evaluator_outcome(self):
        for key in ('final_observation', 'final_obs'):
            for success, truncated in ((True, False), (False, False), (False, True)):
                for info_style in ('mapping', 'array'):
                    with self.subTest(key=key, success=success, truncated=truncated,
                                      info_style=info_style):
                        wrapper = AutoResetWrapper(key, success, truncated)
                        if info_style == 'array':
                            wrapper.info['final_info'] = np.array(
                                [{'is_success': success}], dtype=object)
                        adapter = LiberoEnvironmentAdapter(wrapper)
                        recorder, policy = ReplayRecorder(), Policy()
                        outcome = run_episode(EpisodeConfig(0, 5), policy, adapter, recorder)
                        terminal = recorder.observations[-1]
                        self.assertEqual([p.sequence for p in recorder.observations], [0, 1])
                        self.assertEqual(terminal.episode_id, outcome.episode_id)
                        self.assertEqual(outcome.terminal_observation.episode_id, terminal.episode_id)
                        self.assertEqual(outcome.terminal_observation.sequence, terminal.sequence)
                        np.testing.assert_array_equal(terminal.observation['pixels']['image'],
                                                      observation(20)['pixels']['image'])
                        np.testing.assert_array_equal(terminal.observation['pixels']['image2'],
                                                      observation(20)['pixels']['image2'])
                        np.testing.assert_array_equal(terminal.observation['robot_state']['position'],
                                                      observation(20)['robot_state']['position'])
                        self.assertEqual(outcome.success, success)
                        self.assertEqual(outcome.sum_rewards, float(success))
                        self.assertEqual(outcome.stop_reason, 'success' if success else
                                         'truncated' if truncated else 'terminated')
                        self.assertEqual((wrapper.calls, len(policy.seen)), (1, 1))
                        self.assertTrue(recorder.finalized)
                        self.assertEqual([ref.observation_sequence for ref in terminal.frame_references], [1, 1])
                        wrapper.terminal['pixels']['image'].fill(99)
                        self.assertTrue((terminal.observation['pixels']['image'] == 20).all())
                        with self.assertRaises(RuntimeError):
                            adapter.step([0] * 7)

    def test_next_step_reset_retains_direct_terminal_return(self):
        wrapper = AutoResetWrapper(success=False)
        wrapper.metadata = {'autoreset_mode': 'NextStep'}
        wrapper.info = {'is_success': [False]}
        recorder = ReplayRecorder()
        outcome = run_episode(EpisodeConfig(0, 3), Policy(),
                              LiberoEnvironmentAdapter(wrapper), recorder)
        self.assertFalse(outcome.success)
        self.assertTrue((recorder.observations[-1].observation['pixels']['image'] == 90).all())
        self.assertEqual(outcome.terminal_observation.sequence, 1)

    def test_missing_or_masked_terminal_evidence_never_records_reset_frame(self):
        for defect in ('missing', 'masked', 'null', 'missing_info', 'masked_info'):
            with self.subTest(defect=defect):
                wrapper = AutoResetWrapper()
                if defect == 'missing':
                    del wrapper.info['final_obs']
                elif defect == 'masked':
                    wrapper.info['_final_obs'] = [False]
                elif defect == 'null':
                    wrapper.info['final_obs'] = [None]
                elif defect == 'missing_info':
                    wrapper.info['final_info'] = [None]
                else:
                    wrapper.info['_final_info'] = [False]
                recorder = ReplayRecorder()
                with self.assertRaises(ValueError):
                    run_episode(EpisodeConfig(0, 3), Policy(),
                                LiberoEnvironmentAdapter(wrapper), recorder)
                self.assertEqual(len(recorder.observations), 1)
                self.assertFalse(recorder.finalized)
                self.assertEqual(len(recorder.failures), 1)

    def test_reset_info_cannot_supply_terminal_success(self):
        wrapper = AutoResetWrapper(success=False)
        del wrapper.info['final_info']
        outcome = run_episode(EpisodeConfig(0, 3), Policy(),
                              LiberoEnvironmentAdapter(wrapper), ReplayRecorder())
        self.assertFalse(outcome.success)

    def test_missing_terminal_view_is_not_filled_from_reset(self):
        wrapper = AutoResetWrapper()
        wrapper.terminal['pixels']['image2'] = None
        recorder = ReplayRecorder()
        run_episode(EpisodeConfig(0, 3), Policy(),
                    LiberoEnvironmentAdapter(wrapper), recorder)
        terminal = recorder.observations[-1]
        self.assertIsNone(terminal.observation['pixels']['image2'])
        self.assertEqual(terminal.frame_references[1].availability, 'missing')


if __name__ == '__main__':
    unittest.main()
