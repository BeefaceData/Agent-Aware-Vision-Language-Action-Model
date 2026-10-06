"""Arm-state association through public adapter and complete episode interfaces."""

import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from episode_harness import EpisodeConfig, run_episode, supervisor_observation
from libero_adapter import LiberoEnvironmentAdapter
from recorded_replay import TraceRecorder, load_recorded_replay
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
    def test_libero_state_survives_supervision_and_sealed_replay(self):
        initial = one_arm([0.1, 0.2, 0.3], [0.0, 0.0])
        terminal = one_arm([0.4, 0.5, 0.6], [0.02, 0.02])

        class Environment:
            num_envs = 1

            def reset(self, seed):
                return initial, {}

            def step(self, action):
                return terminal, [1.0], [True], [False], {'is_success': [True]}

        with TemporaryDirectory() as directory:
            config = EpisodeConfig(7, 2)
            recorder = TraceRecorder(Path(directory) / 'trace', config, ReplayRecorder())
            seen = []
            outcome = run_episode(
                config, ReplayPolicy(((initial, [0] * 7),)),
                LiberoEnvironmentAdapter(Environment(), clock=lambda: 10.0), recorder,
                supervisor=lambda packet, action: seen.append(packet.state_fields))
            self.assertTrue(outcome.success)
            self.assertEqual(seen[0][0].value, (0.1, 0.2, 0.3))
            recorder.seal(outcome)
            replay_recorder = ReplayRecorder()
            replay = load_recorded_replay(Path(directory) / 'trace').run(replay_recorder)
            self.assertTrue(replay.success)
            self.assertEqual(replay_recorder.observations[0].state_fields, seen[0])
            final = replay_recorder.observations[-1].state_fields
            self.assertEqual(final[0].value, (0.4, 0.5, 0.6))
            self.assertEqual(final[1].availability, 'missing')
            self.assertEqual(final[2].value, (0.02, 0.02))

    def test_supervision_rejects_malformed_state_metadata(self):
        packet = ReplayEnvironment(
            1, one_arm([1, 2, 3], [0, 0]), (), clock=lambda: 4.0,
            state_specs=LIBERO_STATE_SPECS).reset(1, 'state')
        field = packet.state_fields[0]
        for malformed in (
            replace(field, value=({'hidden': 'evaluator'},)),
            replace(field, arm={'hidden': 'evaluator'}),
            replace(field, availability='missing'),
            replace(field, captured_monotonic=float('nan')),
        ):
            with self.subTest(field=malformed), self.assertRaises(ValueError):
                supervisor_observation(replace(packet, state_fields=(malformed,)))

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
