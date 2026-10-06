"""Deployable observations and evaluator truth stay separate at public seams."""

import unittest
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from episode_harness import (EpisodeConfig, ObservationPacket, RobotStateCapture,
                             frame_references_for_observation, run_episode,
                             supervisor_observation)
from libero_adapter import LiberoEnvironmentAdapter
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


SENTINEL = 'PRIVATE_SIMULATOR_TRUTH'


def raw_observation():
    hidden = {'object_poses': SENTINEL, 'reward': SENTINEL,
              'is_success': SENTINEL, 'unknown_future_truth': SENTINEL}
    return {
        **hidden, 'metadata': {'nested': [hidden]}, 'task': 'place the item',
        'pixels': {'image': [[[0, 1, 2]]], 'image2': [[[3, 4, 5]]],
                   'metadata': hidden, 'object_mask': hidden},
        'robot_state': {'eef': {'pos': [0.0, 0.1, 0.2], 'metadata': hidden,
                                'object_poses': hidden},
                        'gripper': {'qpos': [0.0], 'closed': False, 'hidden': hidden},
                        'metadata': hidden, 'objects': hidden},
    }


class SupervisorObservationTests(unittest.TestCase):
    def test_nested_payload_in_typed_metadata_is_rejected(self):
        wall = datetime.now(timezone.utc)
        raw = raw_observation()
        refs = frame_references_for_observation(raw, 0, wall, 1.0)
        packet = ObservationPacket('episode', 0, wall, raw, 1.0, refs)
        for malformed in (
            replace(packet, frame_references=(replace(refs[0], image_key={'hidden': SENTINEL}),)),
            replace(packet, robot_state_capture=RobotStateCapture(0, wall, {'hidden': SENTINEL})),
        ):
            with self.subTest(packet=repr(malformed)), self.assertRaises(ValueError):
                supervisor_observation(malformed)

    def test_projection_omits_nested_truth_and_preserves_deployable_evidence(self):
        raw = raw_observation()
        packet = ObservationPacket('episode', 0, datetime.now(timezone.utc), raw, 1.0)
        visible = supervisor_observation(packet)
        self.assertNotIn(SENTINEL, repr(visible))
        self.assertEqual(visible.observation, {
            'task': 'place the item',
            'pixels': {'image': [[[0, 1, 2]]], 'image2': [[[3, 4, 5]]]},
            'robot_state': {'eef': {'pos': [0.0, 0.1, 0.2]},
                            'gripper': {'qpos': [0.0], 'closed': False}},
        })
        self.assertEqual((visible.episode_id, visible.sequence,
                          visible.captured_at, visible.captured_monotonic),
                         ('episode', 0, packet.captured_at, 1.0))
        visible.observation['robot_state']['eef']['pos'][0] = 99
        visible.observation['pixels']['image'][0][0][0] = 99
        self.assertEqual(raw['robot_state']['eef']['pos'][0], 0.0)
        self.assertEqual(raw['pixels']['image'][0][0][0], 0)

    def test_metadata_inside_measurement_slots_is_unavailable(self):
        for value in ({'values': [0], 'metadata': SENTINEL},
                      [0, {'hidden': SENTINEL}],
                      np.array([{'hidden': SENTINEL}], dtype=object)):
            with self.subTest(value=repr(value)):
                raw = {'pixels': {'image': value}, 'robot_state': {'eef': {'pos': value}}}
                packet = ObservationPacket('episode', 0, datetime.now(timezone.utc), raw, 1.0)
                visible = supervisor_observation(packet)
                self.assertNotIn(SENTINEL, repr(visible))
                self.assertIsNone(visible.observation['pixels']['image'])
                self.assertIsNone(visible.observation['robot_state']['eef']['pos'])

    def test_replay_keeps_evaluator_outcomes_and_raw_evidence_independent(self):
        for success, terminated, truncated, reason in (
            (True, False, False, 'success'),
            (False, True, False, 'terminated'),
            (False, False, True, 'truncated'),
        ):
            with self.subTest(reason=reason):
                raw = raw_observation()
                policy = ReplayPolicy(((raw, [1]), (raw, [2])))
                environment = ReplayEnvironment(17, raw, (
                    ([1], ReplayStep(raw, 0.25, False, False, False)),
                    ([2], ReplayStep(raw, 0.75, success, terminated, truncated)),
                ))
                recorder = ReplayRecorder()
                seen = []

                def observe(packet, action):
                    self.assertNotIn(SENTINEL, repr(packet))
                    seen.append((packet.sequence, action))
                    packet.observation['pixels']['image'][0][0][0] = 99

                outcome = run_episode(EpisodeConfig(17, 5), policy, environment,
                                      recorder, supervisor=observe)
                self.assertEqual(seen, [(0, [1]), (1, [2])])
                self.assertEqual(environment.actions, [[1], [2]])
                self.assertEqual((outcome.success, outcome.stop_reason, outcome.sum_rewards),
                                 (success, reason, 1.0))
                self.assertEqual(outcome.terminal_observation.sequence, 2)
                self.assertTrue(recorder.finalized)
                self.assertIn(SENTINEL, repr(recorder.observations[-1]))
                self.assertEqual(raw['pixels']['image'][0][0][0], 0)

    def test_libero_evaluator_info_does_not_reach_supervisor(self):
        class VectorEnvironment:
            num_envs = 1
            metadata = {}

            def reset(self, seed):
                raw = raw_observation()
                # The LIBERO Panda exposes two measured gripper joints.
                raw['robot_state']['gripper']['qpos'] = [[0.0, 0.0]]
                raw['pixels']['image'] = np.zeros((1, 2, 2, 3), dtype=np.uint8)
                return raw, {'metadata': SENTINEL}

            def step(self, action):
                raw, _ = self.reset(None)
                return raw, [2.0], [True], [False], {'is_success': [True], 'hidden': SENTINEL}

        class Policy:
            def reset(self):
                pass

            def act(self, packet):
                return [0] * 7

        seen = []
        outcome = run_episode(EpisodeConfig(17, 3), Policy(),
                              LiberoEnvironmentAdapter(VectorEnvironment()), ReplayRecorder(),
                              supervisor=lambda packet, action: seen.append(packet))
        self.assertEqual(len(seen), 1)
        self.assertNotIn(SENTINEL, repr(seen[0]))
        self.assertEqual(seen[0].observation['pixels']['image'], np.zeros((1, 2, 2, 3)).tolist())
        self.assertEqual([ref.camera for ref in seen[0].frame_references], ['main', 'wrist'])
        self.assertEqual((outcome.success, outcome.stop_reason, outcome.sum_rewards),
                         (True, 'success', 2.0))


if __name__ == '__main__':
    unittest.main()
