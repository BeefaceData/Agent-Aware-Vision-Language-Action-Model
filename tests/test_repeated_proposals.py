"""Public repetition signals, productive motion, and complete sealed episodes."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import ActionProposal, EpisodeConfig, ObservationPacket, SupervisorPass, run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from repeated_proposals import RepeatedProposalTrigger, RepetitionSettings


def proposal(sequence, action=None, position=0, episode='test'):
    state = {} if position is None else {'position': [position]}
    return ActionProposal(f'{episode}:{sequence + 1}', ObservationPacket(
        episode, sequence, datetime.now(timezone.utc),
        {'robot_state': state, 'private_evaluator': True}, float(sequence)),
        [1] if action is None else action)


class RepetitionTests(unittest.TestCase):
    def test_boundaries_and_inspectable_evidence(self):
        for action_delta, position_delta, expected in (
                (0.25, 0.5, True), (0.251, 0.5, False), (0.25, 0.501, False)):
            trigger = RepeatedProposalTrigger(RepetitionSettings(3, 0.25, 0.5))
            self.assertFalse(trigger(proposal(0)))
            self.assertFalse(trigger(proposal(1)))
            self.assertEqual(trigger(proposal(2, [1 + action_delta], position_delta)), expected)
            if expected:
                signal = trigger.signal
                self.assertEqual(signal.proposal_ids, ('test:1', 'test:2', 'test:3'))
                self.assertEqual(signal.observation_sequences, (0, 1, 2))
                self.assertEqual(signal.positions, ((0,), (0,), (0.5,)))
                self.assertEqual(signal.settings, RepetitionSettings(3, 0.25, 0.5))

    def test_latch_rearm_and_episode_isolation(self):
        trigger = RepeatedProposalTrigger(RepetitionSettings(3))
        values = [1, 1, 1, 1, 2, 2, 2, 2]
        self.assertEqual([i for i, value in enumerate(values)
                          if trigger(proposal(i, [value]))], [2, 6])
        self.assertIsNone(trigger.signal)
        self.assertEqual([i for i in range(4)
                          if trigger(proposal(i, episode='next'))], [2])

    def test_missing_invalid_and_gapped_evidence_break_window(self):
        for middle in (proposal(1, position=None), proposal(1, [float('nan')]),
                       proposal(1, [True]), proposal(1, [1, 2]), proposal(2)):
            trigger = RepeatedProposalTrigger(RepetitionSettings(3))
            self.assertFalse(trigger(proposal(0)))
            self.assertFalse(trigger(middle))
            self.assertFalse(trigger(proposal(4)))
            self.assertFalse(trigger(proposal(5)))
            self.assertTrue(trigger(proposal(6)))

    def test_productive_repetition_and_stalled_complete_replay(self):
        for moving, expected in ((True, [0]), (False, [0, 2])):
            with self.subTest(moving=moving), TemporaryDirectory() as temporary:
                observations = [{'robot_state': {'eef': {'pos': [i if moving else 0]}},
                                 'task': 'move item'} for i in range(7)]
                actions = [[1]] * 6
                config = EpisodeConfig(17, 6, 10)
                policy = ReplayPolicy(list(zip(observations, actions)))
                environment = ReplayEnvironment(17, observations[0], [
                    (action, ReplayStep(observations[i + 1], 0, i == 5, i == 5, False))
                    for i, action in enumerate(actions)])
                trigger = RepeatedProposalTrigger(RepetitionSettings(3))
                signals, assessed = [], []

                def inspect(p):
                    emitted = trigger(p)
                    if emitted:
                        signals.append(trigger.signal)
                    return emitted

                def decide(p):
                    assessed.append(p.observation.sequence)
                    return SupervisorPass(p.observation.episode_id,
                                          p.observation.sequence, p.proposal_id)

                directory = Path(temporary) / 'trace'
                recorder = TraceRecorder(directory, config, ReplayRecorder())
                outcome = run_episode(config, policy, environment, recorder,
                                      supervisor_decider=decide, assessment_trigger=inspect)
                self.assertTrue(outcome.success)
                self.assertEqual(assessed, expected)
                self.assertEqual(environment.actions, actions)
                self.assertEqual(len(signals), 0 if moving else 1)
                recorder.seal(outcome)
                replay = load_recorded_replay(directory)
                self.assertTrue(replay.run().success)
                self.assertEqual([i for i, row in enumerate(replay.evidence()['decisions'])
                                  if row['action_record']['supervisor_pass']], expected)

    def test_configuration(self):
        for kwargs in ({'window': True}, {'window': 1}, {'window': 2.5},
                       {'action_tolerance': -1}, {'position_tolerance': float('inf')},
                       {'action_tolerance': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RepetitionSettings(**kwargs)


if __name__ == '__main__':
    unittest.main()
