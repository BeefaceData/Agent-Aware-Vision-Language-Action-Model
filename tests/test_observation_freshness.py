"""Required-input freshness through the public packet and episode interfaces."""

import unittest
from datetime import datetime, timedelta, timezone

from episode_harness import (EpisodeConfig, ObservationPacket, RobotStateCapture,
                             ViewCapture, check_observation_freshness,
                             frame_references_for_observation, run_episode)
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
LIMITS = {'main': 0.5, 'wrist': 0.5, 'robot_state': 0.5}


def observation(main=None, wrist=None, state=None):
    pixels = {}
    if main is not None:
        pixels['image'] = [main]
    if wrist is not None:
        pixels['image2'] = [wrist]
    return {'pixels': pixels, 'robot_state': state}


class ObservationFreshnessTests(unittest.TestCase):
    def test_nonfinite_or_empty_state_measurements_are_missing(self):
        for measurement in (float('nan'), float('inf'), []):
            with self.subTest(measurement=measurement):
                raw = observation('main', 'wrist', {'eef': {'pos': [measurement]}})
                packet = ObservationPacket('episode', 0, START, raw, 10.0)
                state = check_observation_freshness(
                    packet, 10.1, {'robot_state': 0.5})[0]
                self.assertEqual(state.status, 'missing')
                self.assertFalse(state.available)
                self.assertIsNone(state.age_seconds)
                self.assertEqual(state.name, 'robot_state')

    def test_missing_and_incomplete_state_have_no_age_or_fresh_verdict(self):
        raw = observation('main', 'wrist', {'eef': {'pos': None}})
        refs = frame_references_for_observation(raw, 0, START, 10.0)
        packet = ObservationPacket('episode', 0, START, raw, 10.0, refs)

        status = {item.name: item for item in
                  check_observation_freshness(packet, 10.2, LIMITS)}

        self.assertEqual(status['robot_state'].status, 'missing')
        self.assertFalse(status['robot_state'].available)
        self.assertIsNone(status['robot_state'].age_seconds)
        self.assertIn('obtain a new observation', status['robot_state'].diagnostic)
        self.assertAlmostEqual(status['main'].age_seconds, 0.2)
        self.assertEqual(status['main'].time_basis, 'observation_return')
        self.assertEqual(status['wrist'].status, 'unverified')
        self.assertIn('timestamped input', status['wrist'].diagnostic)
        late = {item.name: item for item in
                check_observation_freshness(packet, 10.6, LIMITS)}
        self.assertEqual(late['wrist'].status, 'stale')
        self.assertEqual(late['robot_state'].status, 'missing')

    def test_exact_limit_and_invalid_metadata_have_explicit_results(self):
        raw = observation('main', 'wrist', {'eef': {'pos': [0.0]}})
        captures = {'main': ViewCapture(0, START, 10.0),
                    'wrist': ViewCapture(0, START, 10.0)}
        refs = frame_references_for_observation(raw, 0, START, 10.0,
                                                captures, max_skew_seconds=0.1)
        packet = ObservationPacket('episode', 0, START, raw, 10.0, refs,
                                   RobotStateCapture(0, START, 10.1))
        status = {item.name: item for item in
                  check_observation_freshness(packet, 10.5, LIMITS)}
        self.assertEqual(status['main'].status, 'fresh')
        self.assertEqual(status['main'].age_seconds, 0.5)
        self.assertEqual(status['robot_state'].status, 'invalid_metadata')
        self.assertIsNone(status['robot_state'].age_seconds)
        self.assertIn('repair capture metadata', status['robot_state'].diagnostic)
        with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
            check_observation_freshness(packet, 10.5, {'main': float('inf')})

    def test_complete_episode_reports_absent_camera_delayed_wrist_and_stale_state(self):
        state = {'eef': {'pos': [0.0, 0.0, 0.0]}, 'gripper': {'qpos': [0.0]}}
        initial = observation('main 0', 'wrist 0', state)
        no_main = observation(wrist='wrist 1', state=state)
        old_wrist = observation('main 2', 'wrist 1', state)
        old_state = observation('main 3', 'wrist 3', state)
        terminal = observation('main 4', 'wrist 4', state)
        delayed_cameras = {
            'main': ViewCapture(2, START + timedelta(seconds=2), 10.0),
            'wrist': ViewCapture(1, START + timedelta(seconds=1), 8.0),
        }
        initial_cameras = {'main': ViewCapture(0, START, 10.0),
                           'wrist': ViewCapture(0, START, 10.0)}
        stale_state = RobotStateCapture(1, START + timedelta(seconds=1), 8.0)
        policy = ReplayPolicy(((initial, 'advance 0'), (no_main, 'advance 1'),
                               (old_wrist, 'advance 2'), (old_state, 'finish')))
        environment = ReplayEnvironment(17, initial, (
            ('advance 0', ReplayStep(no_main, 0.0, False, False, False)),
            ('advance 1', ReplayStep(old_wrist, 0.0, False, False, False,
                                     delayed_cameras)),
            ('advance 2', ReplayStep(old_state, 0.0, False, False, False,
                                     robot_state_capture=stale_state)),
            ('finish', ReplayStep(terminal, 1.0, True, True, False)),
        ), clock=lambda: 10.0, initial_camera_captures=initial_cameras,
           initial_robot_state_capture=RobotStateCapture(0, START, 10.0),
           max_camera_skew_seconds=0.1)
        recorder = ReplayRecorder()
        reports = []

        def inspect(packet, proposed_action):
            reports.append({item.name: item for item in
                            check_observation_freshness(packet, 10.0, LIMITS)})

        outcome = run_episode(EpisodeConfig(17, 4), policy, environment, recorder,
                              supervisor=inspect, clock=lambda: 10.0)

        self.assertTrue(outcome.success)
        self.assertTrue(recorder.finalized)
        self.assertEqual(environment.actions,
                         ['advance 0', 'advance 1', 'advance 2', 'finish'])
        self.assertEqual(len(reports), 4)
        self.assertEqual({name: item.status for name, item in reports[0].items()},
                         {'main': 'fresh', 'wrist': 'fresh', 'robot_state': 'fresh'})
        self.assertEqual(reports[1]['main'].status, 'missing')
        self.assertFalse(reports[1]['main'].available)
        self.assertIsNone(reports[1]['main'].age_seconds)
        self.assertEqual(reports[2]['wrist'].status, 'stale')
        self.assertEqual(reports[2]['wrist'].age_seconds, 2.0)
        self.assertEqual(reports[2]['wrist'].observation_sequence, 1)
        self.assertEqual(reports[2]['wrist'].time_basis, 'sensor_capture')
        self.assertIn('exceeds 0.500s', reports[2]['wrist'].diagnostic)
        self.assertEqual(reports[3]['robot_state'].status, 'stale')
        self.assertEqual(reports[3]['robot_state'].age_seconds, 2.0)
        self.assertEqual(reports[3]['robot_state'].observation_sequence, 1)
        self.assertEqual(reports[3]['robot_state'].time_basis, 'sensor_capture')
        self.assertIn('obtain a new capture', reports[3]['robot_state'].diagnostic)


if __name__ == '__main__':
    unittest.main()
