"""Public temporal history contract and complete-episode integration."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from episode_harness import (ActionResolution, EpisodeConfig, ObservationPacket,
                             frame_references_for_observation, run_episode)
from observation_window import (ObservationWindowBuilder, SequenceInterval,
                                WindowAction, WindowSettings)
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


def raw(sequence, wrist=True):
    return {'task': 'place the item', 'pixels': {
        'image': [[[sequence]]], **({'image2': [[[sequence + 10]]]} if wrist else {})},
        'robot_state': {'eef': {'pos': [sequence, 0, 0]}},
        'evaluator': {'success': 'PRIVATE_TRUTH'}}


def packet(sequence, wrist=True):
    observation = raw(sequence, wrist)
    wall = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=sequence)
    return ObservationPacket('episode', sequence, wall, observation, float(sequence),
                             frame_references_for_observation(
                                 observation, sequence, wall, float(sequence)))


class ObservationWindowTests(unittest.TestCase):
    def test_first_step_and_independent_snapshot(self):
        builder = ObservationWindowBuilder('episode', 'place the item')
        source = packet(0)
        builder.append(source)
        source.observation['pixels']['image'][0][0][0] = 99
        window = builder.snapshot()
        self.assertEqual(window.task, 'place the item')
        self.assertEqual(window.ordering, 'observation_sequence_ascending')
        self.assertEqual([p.sequence for p in window.observations], [0])
        self.assertEqual(window.actions, ())
        self.assertEqual(window.missing_intervals, ())
        self.assertIsNone(window.omitted_prefix)
        self.assertNotIn('PRIVATE_TRUTH', repr(window))
        self.assertEqual(window.observations[0].observation['pixels']['image'], [[[0]]])
        window.observations[0].observation['pixels']['image'][0][0][0] = 88
        self.assertEqual(builder.snapshot().observations[0].observation['pixels']['image'], [[[0]]])

    def test_full_window_truncation_gaps_and_missing_view(self):
        builder = ObservationWindowBuilder('episode', 'place the item', WindowSettings(3, 1))
        for sequence in (0, 1, 3, 4, 6):
            builder.append(packet(sequence, wrist=sequence != 4))
            if sequence == 3:
                self.assertIsNone(builder.snapshot().omitted_prefix)
                self.assertEqual(len(builder.snapshot().observations), 3)
        window = builder.snapshot()
        self.assertEqual([p.sequence for p in window.observations], [3, 4, 6])
        self.assertEqual(window.omitted_prefix, SequenceInterval(0, 2))
        self.assertEqual(window.missing_intervals, (SequenceInterval(5, 5),))
        missing = window.observations[1]
        self.assertNotIn('image2', missing.observation['pixels'])
        self.assertEqual(missing.frame_references[1].availability, 'missing')
        self.assertEqual(missing.frame_references[1].captured_at, None)
        self.assertEqual(window.observations[0].captured_monotonic, 3.0)

    def test_rejected_order_identity_time_and_task_leave_history_unchanged(self):
        builder = ObservationWindowBuilder('episode', 'place the item', WindowSettings(2, 0))
        builder.append(packet(2))
        for invalid in (packet(1), packet(2), replace(packet(3), episode_id='other'),
                        replace(packet(3), captured_monotonic=1),
                        replace(packet(3), captured_at=packet(0).captured_at),
                        replace(packet(3), observation={'task': 'new instruction'})):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                builder.append(invalid)
            self.assertEqual([p.sequence for p in builder.snapshot().observations], [2])
        builder.append(replace(packet(3), observation={'pixels': {'image': [[[3]]]}}))
        self.assertEqual(builder.snapshot().observations[-1].observation['task'], 'place the item')

    def test_action_bound_is_independent_and_actions_are_detached(self):
        for limit in (0, 1, 3):
            with self.subTest(limit=limit):
                builder = ObservationWindowBuilder('episode', 'place the item', WindowSettings(1, limit))
                for sequence in range(4):
                    builder.append(packet(sequence))
                    action = WindowAction(sequence, f'p{sequence}', [sequence], [-sequence], sequence + .5)
                    builder.record_action(action)
                    action.proposed_action[0] = 100
                window = builder.snapshot()
                self.assertEqual(len(window.observations), 1)
                self.assertEqual([a.source_sequence for a in window.actions], list(range(4 - limit, 4)))
                self.assertEqual(window.omitted_action_count, 4 - limit)
                for action in window.actions:
                    self.assertEqual(action.proposed_action, [action.source_sequence])
                    self.assertEqual(action.executed_action, [-action.source_sequence])
                if limit:
                    window.actions[-1].executed_action[0] = 100
                    self.assertEqual(builder.snapshot().actions[-1].executed_action, [-3])

    def test_settings_and_invalid_actions_fail_explicitly(self):
        for observations, actions in ((0, 1), (True, 1), (2, -1), (2, 1.5)):
            with self.assertRaises(ValueError):
                WindowSettings(observations, actions)
        builder = ObservationWindowBuilder('episode', 'place the item')
        with self.assertRaises(ValueError):
            builder.snapshot()
        builder.append(packet(1))
        for action in (WindowAction(0, 'p', [0], [0], 2),
                       WindowAction(1, 'p', [0], [0], .5),
                       WindowAction(1, 'p', [0], [0], float('nan'))):
            with self.assertRaises(ValueError):
                builder.record_action(action)
        builder.record_action(WindowAction(1, 'p', [0], [0], 2))
        with self.assertRaises(ValueError):
            builder.record_action(WindowAction(1, 'p', [0], [0], 2))

    def test_complete_episodes_bound_history_and_preserve_execution_and_evaluation(self):
        for success, terminated, truncated, reason in (
            (True, True, False, 'success'), (False, True, False, 'terminated'),
            (False, False, True, 'truncated'), (False, False, False, 'step_limit'),
        ):
            with self.subTest(reason=reason):
                observations = [raw(i, wrist=i != 1) for i in range(5)]
                policy = ReplayPolicy([(observations[i], [i]) for i in range(4)])
                environment = ReplayEnvironment(17, observations[0], [
                    ([10 + i], ReplayStep(observations[i + 1], .25,
                                          success and i == 3, terminated and i == 3,
                                          truncated and i == 3)) for i in range(4)])
                recorder = ReplayRecorder()
                seen = []

                def observe(window, proposal):
                    seen.append(window)
                    sequence = window.observations[-1].sequence
                    self.assertNotIn('PRIVATE_TRUTH', repr(window))
                    self.assertEqual(window.task, 'place the item')
                    self.assertEqual(proposal, [sequence])
                    self.assertEqual([p.sequence for p in window.observations],
                                     list(range(max(0, sequence - 1), sequence + 1)))
                    self.assertEqual(len(window.actions), min(sequence, 1))
                    if sequence:
                        self.assertEqual(window.actions[-1].proposed_action, [sequence - 1])
                        self.assertEqual(window.actions[-1].executed_action, [sequence + 9])
                    proposal[0] = 999
                    window.observations[-1].observation['pixels']['image'][0][0][0] = 999

                # Reuse adapters to verify history and action counts reset per attempt.
                for attempt in range(2):
                    outcome = run_episode(EpisodeConfig(17, 4), policy, environment, recorder,
                        window_supervisor=observe, window_settings=WindowSettings(2, 1),
                        action_selector=lambda proposal: ActionResolution('override', [proposal.action[0] + 10]))
                    self.assertEqual((outcome.success, outcome.stop_reason, outcome.steps), (success, reason, 4))
                    self.assertEqual(outcome.sum_rewards, 1.0)
                    self.assertTrue(recorder.finalized)
                    self.assertEqual(environment.actions, [[10], [11], [12], [13]])
                    self.assertEqual(observations[0]['pixels']['image'], [[[0]]])
                    self.assertEqual(seen[-1].omitted_prefix, SequenceInterval(0, 1))
                    self.assertEqual(seen[-1].omitted_action_count, 2)
                self.assertNotEqual(seen[0].episode_id, seen[4].episode_id)

    def test_temporal_supervisor_failure_records_wait_without_execution(self):
        now = [10.0]
        observation = raw(0)
        policy = ReplayPolicy([(observation, [0])])
        environment = ReplayEnvironment(17, observation, [], clock=lambda: now[0])
        recorder = ReplayRecorder()

        def fail(window, action):
            now[0] += 2.0
            raise TimeoutError('supervisor unavailable')

        with self.assertRaises(TimeoutError):
            run_episode(EpisodeConfig(17, 1), policy, environment, recorder,
                        window_supervisor=fail, clock=lambda: now[0])
        self.assertEqual(environment.actions, [])
        self.assertEqual(recorder.failures[0][3].stage, 'supervisor')
        self.assertEqual(recorder.failures[0][3].timing.cumulative_wait_seconds, 2.0)
        self.assertTrue(recorder.finalized)


if __name__ == '__main__':
    unittest.main()
