"""Public gripper evidence and assessment-only complete sealed replay."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import (ActionProposal, EpisodeConfig, ObservationPacket,
                             RobotStateCapture, SupervisorPass, run_episode)
from gripper_transitions import GripperTransitionTrigger
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


WALL = datetime(2026, 1, 1, tzinfo=timezone.utc)


def proposal(index, closed, episode='episode', capture=True):
    wall = WALL + timedelta(seconds=index)
    return ActionProposal(f'{episode}:{index + 1}', ObservationPacket(
        episode, index, wall, {'robot_state': {'gripper': {'closed': closed}}},
        float(index), robot_state_capture=(RobotStateCapture(index, wall, float(index))
                                          if capture else None)), [1])


class GripperTests(unittest.TestCase):
    def test_transitions_retain_identity_times_and_uncertainty(self):
        trigger = GripperTransitionTrigger()
        self.assertFalse(trigger(proposal(0, False)))
        self.assertTrue(trigger(proposal(1, True)))
        evidence = trigger.evidence
        self.assertEqual(evidence.transition, 'open_to_closed')
        self.assertEqual([s.proposal_id for s in evidence.samples], ['episode:1', 'episode:2'])
        self.assertEqual([s.observation_sequence for s in evidence.samples], [0, 1])
        self.assertEqual([s.captured_monotonic for s in evidence.samples], [0., 1.])
        self.assertEqual(evidence.samples[1].captured_at, WALL + timedelta(seconds=1))
        self.assertEqual(evidence.samples[1].robot_state_capture.captured_monotonic, 1.)
        self.assertIn('gripper_state_alone_does_not_establish_object_grasp_or_loss',
                      evidence.limitations)
        self.assertFalse(trigger(proposal(2, True)))
        self.assertIsNone(trigger.evidence.transition)
        self.assertTrue(trigger(proposal(3, False)))
        self.assertEqual(trigger.evidence.transition, 'closed_to_open')
        self.assertEqual(evidence.transition, 'open_to_closed')

    def test_unavailable_invalid_and_unverified_state(self):
        for value in (None, [], [1], 'closed', .5, float('nan'), float('inf'), {}):
            with self.subTest(value=value):
                trigger = GripperTransitionTrigger()
                trigger(proposal(0, False))
                self.assertFalse(trigger(proposal(1, value, capture=False)))
                self.assertIn('gripper_closed_state_unavailable_or_invalid',
                              trigger.evidence.limitations)
                self.assertIn('gripper_sensor_capture_time_unverified',
                              trigger.evidence.limitations)
                self.assertIsNone(trigger.evidence.samples[-1].closed)
                self.assertFalse(trigger(proposal(2, True)))
                self.assertTrue(trigger(proposal(3, False)))

    def test_episode_and_sequence_boundaries(self):
        for next_sample in (proposal(2, True), proposal(0, True),
                            proposal(1, True, episode='next')):
            trigger = GripperTransitionTrigger()
            trigger(proposal(0, False))
            self.assertFalse(trigger(next_sample))
            self.assertIn('no_contiguous_previous_gripper_state', trigger.evidence.limitations)

    def test_no_inference_from_commands_positions_or_evaluator(self):
        trigger = GripperTransitionTrigger()
        for index in range(3):
            p = proposal(index, None)
            p.observation.observation['robot_state']['gripper'] = {'qpos': [index]}
            p.observation.observation['success'] = True
            self.assertFalse(trigger(p))
            self.assertIsNone(trigger.evidence.samples[-1].closed)

    def test_missed_normal_and_unavailable_complete_sealed_replays(self):
        # Identical gripper observations cannot distinguish a normal grasp from
        # a missed grasp. Independent terminal outcomes remain evaluator-only.
        for success, states, expected in ((False, [0, 1, 1, 0, 0], [0, 1, 3]),
                                          (True, [0, 1, 1, 0, 0], [0, 1, 3]),
                                          (False, [None] * 5, [0])):
            with self.subTest(success=success, states=states), TemporaryDirectory() as temporary:
                observations = [{'task': 'pick item', 'robot_state': {
                    'gripper': {} if state is None else {'closed': state}}} for state in states]
                actions = [[1]] * 4
                config = EpisodeConfig(17, 4, 10)
                environment = ReplayEnvironment(17, observations[0], [
                    (action, ReplayStep(observations[i + 1], 0, success if i == 3 else False,
                                       i == 3, False)) for i, action in enumerate(actions)])
                trigger = GripperTransitionTrigger()
                evidence, assessed = [], []

                def inspect(p):
                    emit = trigger(p)
                    evidence.append(trigger.evidence)
                    return emit

                def decide(p):
                    assessed.append(p.observation.sequence)
                    return SupervisorPass(p.observation.episode_id,
                                          p.observation.sequence, p.proposal_id)

                directory = Path(temporary) / 'trace'
                recorder = TraceRecorder(directory, config, ReplayRecorder())
                outcome = run_episode(config, ReplayPolicy(list(zip(observations, actions))),
                    environment, recorder, supervisor_decider=decide, assessment_trigger=inspect)
                self.assertEqual(outcome.success, success)
                self.assertEqual(assessed, expected)
                self.assertEqual(environment.actions, actions)
                self.assertEqual(len(evidence), 4)
                recorder.seal(outcome)
                replay = load_recorded_replay(directory)
                self.assertEqual(replay.run().success, success)
                self.assertEqual([i for i, row in enumerate(replay.evidence()['decisions'])
                                  if row['action_record']['supervisor_pass']], expected)


if __name__ == '__main__':
    unittest.main()
