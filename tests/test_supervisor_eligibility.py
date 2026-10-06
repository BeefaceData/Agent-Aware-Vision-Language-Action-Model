"""Public sensing eligibility, restoration, and sealed episode replay."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from baseline_fallback import BaselineFallback

from episode_harness import (ActionProposal, EpisodeConfig, ObservationPacket,
                             RobotStateCapture, SupervisorAbstention, SupervisorPass,
                             ViewCapture, frame_references_for_observation, run_episode)
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_eligibility import ObservationEligibleSupervisor


WALL = datetime(2026, 1, 1, tzinfo=timezone.utc)


def raw(wrist=True):
    return {'task': 'place item', 'pixels': {'image': [[[0]]],
            **({'image2': [[[1]]]} if wrist else {})},
            'robot_state': {'position': [0]}, 'private_evaluator': 'hidden'}


def packet(index=0, wrist=True):
    observation = raw(wrist)
    captures = {name: ViewCapture(index, WALL, 10.) for name in ('main', 'wrist')}
    if not wrist:
        del captures['wrist']
    refs = frame_references_for_observation(observation, index, WALL, 10., captures, .1)
    return ObservationPacket('episode', index, WALL, observation, 10., refs,
                             RobotStateCapture(index, WALL, 10.))


def passed(proposal):
    return SupervisorPass(proposal.observation.episode_id, proposal.observation.sequence,
                          proposal.proposal_id)


class EligibilityTests(unittest.TestCase):
    def test_required_camera_loss_stale_state_and_restoration(self):
        calls = []

        def supervisor(proposal):
            calls.append(proposal)
            return passed(proposal)

        gate = ObservationEligibleSupervisor(supervisor,
            {'main': .5, 'wrist': .5, 'robot_state': .5}, clock=lambda: 10.2)
        missing = packet(wrist=False)
        stale = replace(packet(1), robot_state_capture=RobotStateCapture(
            0, WALL - timedelta(seconds=1), 9.))
        for index, (observation, name, status) in enumerate((
                (missing, 'wrist', 'missing'), (stale, 'robot_state', 'stale'))):
            result = gate(ActionProposal(str(index), observation, [0]))
            self.assertIsInstance(result, SupervisorAbstention)
            self.assertEqual(result.evidence_availability[name], status)
            self.assertIn(name, result.reason)
            self.assertIn(status, result.reason)
            self.assertEqual(result.proposal_id, str(index))
        self.assertEqual(calls, [])
        self.assertIsInstance(gate(ActionProposal('2', packet(2), [0])), SupervisorPass)
        self.assertEqual(len(calls), 1)
        self.assertNotIn('private_evaluator', calls[0].observation.observation)

    def test_optional_loss_and_caller_limits_are_detached(self):
        limits = {'main': .5}
        gate = ObservationEligibleSupervisor(passed, limits, clock=lambda: 10.2)
        limits['wrist'] = .5
        self.assertIsInstance(gate(ActionProposal('0', packet(wrist=False), [0])), SupervisorPass)
        limits['main'] = 100
        stale = replace(packet(), frame_references=tuple(
            replace(ref, captured_monotonic=9.) for ref in packet().frame_references))
        self.assertIsInstance(gate(ActionProposal('1', stale, [0])), SupervisorAbstention)

    def test_unverified_and_future_sensor_capture_abstain(self):
        gate = ObservationEligibleSupervisor(passed, {'robot_state': .5}, clock=lambda: 10.2)
        for capture in (None, RobotStateCapture(0, WALL, 11.)):
            result = gate(ActionProposal('0', replace(packet(), robot_state_capture=capture), [0]))
            self.assertEqual(result.evidence_availability['robot_state'], 'unknown')
            self.assertIsInstance(result, SupervisorAbstention)

    def test_invalid_configuration(self):
        for limits in ({}, {'unknown': 1}, {'main': True}, {'main': -1},
                       {'main': float('inf')}, {'main': float('nan')}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                ObservationEligibleSupervisor(passed, limits)

    def test_complete_episode_retains_abstentions_then_restores_assessment(self):
        observations = [raw(False), raw(), raw(), raw()]
        actions = [[0], [1], [2]]
        config = EpisodeConfig(17, 3)
        policy = ReplayPolicy(list(zip(observations, actions)))
        captures = lambda sequence: {name: ViewCapture(sequence, WALL, 10.)
                                    for name in ('main', 'wrist')}
        environment = ReplayEnvironment(17, observations[0], [
            (actions[0], ReplayStep(observations[1], 0, False, False, False,
                camera_captures=captures(1), robot_state_capture=RobotStateCapture(
                    0, WALL - timedelta(seconds=1), 9.))),
            (actions[1], ReplayStep(observations[2], 0, False, False, False,
                camera_captures=captures(2), robot_state_capture=RobotStateCapture(2, WALL, 10.))),
            (actions[2], ReplayStep(observations[3], 1, True, True, False))],
            clock=lambda: 10., max_camera_skew_seconds=.1,
            initial_camera_captures={'main': ViewCapture(0, WALL, 10.)},
            initial_robot_state_capture=RobotStateCapture(0, WALL, 10.))
        assessed = []

        def supervisor(proposal):
            assessed.append(proposal.observation.sequence)
            return passed(proposal)

        gate = ObservationEligibleSupervisor(supervisor,
            {'main': .5, 'wrist': .5, 'robot_state': .5}, clock=lambda: 10.2)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trace'
            recorder = TraceRecorder(directory, config, ReplayRecorder())
            outcome = run_episode(config, policy, environment, recorder, supervisor_decider=gate,
                                  clock=lambda: 10.2,
                                  baseline_fallback=BaselineFallback(
                                      {'main': .5}, lambda: True, lambda p: True))
            recorder.seal(outcome)
            replay = load_recorded_replay(directory)
            self.assertTrue(replay.run().success)
            rows = replay.evidence()['decisions']
            for index, name, status in ((0, 'wrist', 'missing'), (1, 'robot_state', 'stale')):
                evidence = rows[index]['action_record']['supervisor_abstention']
                self.assertEqual(evidence['evidence_availability'][name], status)
            self.assertIsNotNone(rows[2]['action_record']['supervisor_pass'])
        self.assertEqual(assessed, [2])
        self.assertEqual(environment.actions, actions)


if __name__ == '__main__':
    unittest.main()
