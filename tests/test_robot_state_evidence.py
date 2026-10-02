"""Arm-state association through public adapter and complete episode interfaces."""

import unittest
from datetime import datetime, timezone

from episode_harness import EpisodeConfig, run_episode
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from robot_state_evidence import (LIBERO_STATE_SPECS, StateSpec,
                                  state_fields_for_observation)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def one_arm(position, gripper):
    return {'robot_state': {
        'eef': {'pos': [position]},
        'gripper': {'qpos': [gripper]},
    }}


class StateEvidenceTests(unittest.TestCase):
    def test_one_arm_episode_exposes_return_timed_state_to_supervision(self):
        initial = one_arm([0.1, 0.2, 0.3], [0.0, 0.0])
        terminal = one_arm([0.4, 0.5, 0.6], [0.02, 0.02])
        terminal['robot_state']['eef']['quat'] = [[0.0, 0.0, 0.0, 1.0]]
        environment = ReplayEnvironment(
            7, initial, (('move', ReplayStep(terminal, 1.0, True, True, False)),),
            clock=lambda: 10.0, state_specs=LIBERO_STATE_SPECS)
        recorder = ReplayRecorder()
        inspected = []
        outcome = run_episode(
            EpisodeConfig(7, 2), ReplayPolicy(((initial, 'move'),)),
            environment, recorder,
            supervisor=lambda packet, action: inspected.append(packet.state_fields),
            clock=lambda: 10.0)

        self.assertTrue(outcome.success)
        self.assertEqual(environment.actions, ['move'])
        self.assertEqual(inspected, [recorder.observations[0].state_fields])
        for packet, expected_position in zip(
                recorder.observations, ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))):
            fields = {field.name: field for field in packet.state_fields}
            position = fields['end_effector_position']
            self.assertEqual((position.arm, position.frame, position.units,
                              position.value, position.availability),
                             ('robot0', 'world', 'm', expected_position, 'available'))
            self.assertEqual(position.captured_at, packet.captured_at)
            self.assertEqual(position.captured_monotonic, packet.captured_monotonic)
            self.assertEqual(position.time_basis, 'observation_return')
            self.assertEqual(fields['gripper_joint_position'].units, 'm')
        missing_orientation = recorder.observations[0].state_fields[1]
        self.assertEqual(missing_orientation.availability, 'missing')
        self.assertIsNone(missing_orientation.value)
        self.assertIsNone(missing_orientation.captured_at)
        self.assertEqual(recorder.observations[1].state_fields[1].value,
                         (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(recorder.observations[0].state_fields[2].value, (0.0, 0.0))

    def test_two_arm_packet_keeps_arm_and_gripper_values_separate(self):
        specs = (
            StateSpec('left', 'end_effector_position', 'left_eef', 'base', 'm', 3),
            StateSpec('left', 'gripper_joint_position', 'left_grip', 'joint', 'rad', 2),
            StateSpec('right', 'end_effector_position', 'right_eef', 'base', 'm', 3),
            StateSpec('right', 'gripper_joint_position', 'right_grip', 'joint', 'rad', 2),
        )
        raw = {'left_eef': [1, 2, 3], 'left_grip': [0, 0],
               'right_eef': [4, 5, 6]}
        packet = ReplayEnvironment(1, raw, (), clock=lambda: 4.0,
                                   state_specs=specs).reset(1, 'two-arm')
        by_arm = {(field.arm, field.name): field for field in packet.state_fields}
        self.assertEqual(by_arm['left', 'end_effector_position'].value, (1.0, 2.0, 3.0))
        self.assertEqual(by_arm['right', 'end_effector_position'].value, (4.0, 5.0, 6.0))
        self.assertEqual(by_arm['left', 'gripper_joint_position'].value, (0.0, 0.0))
        missing = by_arm['right', 'gripper_joint_position']
        self.assertEqual((missing.availability, missing.value,
                          missing.captured_at, missing.captured_monotonic),
                         ('missing', None, None, None))
        self.assertNotIn('right_grip', packet.observation)

    def test_malformed_present_measurement_is_rejected(self):
        for value in ([float('nan'), 0, 0], [0, 0], [True, 0, 0]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'invalid state measurement'):
                    state_fields_for_observation(
                        {'robot_state': {'eef': {'pos': value}}},
                        LIBERO_STATE_SPECS,
                        START, 1.0)


if __name__ == '__main__':
    unittest.main()
