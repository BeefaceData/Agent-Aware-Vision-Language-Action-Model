"""Public displacement evidence and assessment-only sealed episode checks."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from end_effector_progress import EndEffectorProgressTrigger, ProgressSettings
from episode_harness import ActionProposal, EpisodeConfig, ObservationPacket, SupervisorPass, run_episode, supervisor_observation
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


def observation(position, frame='world', units='m'):
    return {'robot_state': {'eef': {'pos': position, 'frame': frame, 'position_units': units,
                                  'private_object_pose': [99]}}, 'success': True}


def proposal(i, position=(0, 0, 0), frame='world', units='m', episode='test', time=None):
    return ActionProposal(f'{episode}:{i}', ObservationPacket(episode, i,
        datetime.now(timezone.utc), observation(position, frame, units),
        float(i) if time is None else time), [1])


def trigger():
    return EndEffectorProgressTrigger(ProgressSettings('world', 'm', 3, 1, 3, .5))


class ProgressTests(unittest.TestCase):
    def test_distance_boundaries_and_return_motion(self):
        for positions, expected in (([(0, 0, 0), (.3, .4, 0), (0, 0, 0)], True),
                                    ([(0, 0, 0), (.3, .401, 0), (0, 0, 0)], False)):
            t = trigger()
            results = [t(proposal(i, p)) for i, p in enumerate(positions)]
            self.assertEqual(results, [False, False, expected])
            self.assertEqual(t.evidence.status, 'low_progress' if expected else 'motion_observed')
            self.assertEqual(t.evidence.interval_seconds, 2)
            self.assertEqual([s.proposal_id for s in t.evidence.samples], ['test:0', 'test:1', 'test:2'])
            self.assertEqual(t.evidence.settings.frame, 'world')
            if expected:
                self.assertEqual(t.evidence.displacement, .5)
            self.assertFalse(t(proposal(3)))

    def test_invalid_missing_or_incompatible_evidence_is_unknown(self):
        bad = [proposal(1, p) for p in (None, [], [0], [True, 0, 0], [float('nan'), 0, 0])]
        bad += [proposal(1, frame=f) for f in (None, '', 'camera')]
        bad += [proposal(1, units='cm'), proposal(1, time=float('nan'))]
        bad += [replace(proposal(1), observation=replace(proposal(1).observation, captured_monotonic=None))]
        for p in bad:
            t = trigger()
            t(proposal(0))
            self.assertFalse(t(p))
            self.assertEqual(t.evidence.status, 'unknown')
            self.assertIsNone(t.evidence.displacement)
            self.assertFalse(t(proposal(2)))
            self.assertFalse(t(proposal(3)))
            self.assertTrue(t(proposal(4)))

    def test_interval_sequence_episode_and_rearm(self):
        for times in ([0, .1, .2], [0, 1, 4], [0, 0, 1], [0, -1, 1]):
            t = trigger()
            self.assertFalse(any(t(proposal(i, time=v)) for i, v in enumerate(times)))
            self.assertEqual(t.evidence.status, 'unknown')
        for p in (proposal(4), proposal(1, episode='next')):
            t = trigger()
            t(proposal(0))
            self.assertFalse(t(p))
            self.assertEqual(t.evidence.status, 'unknown')
        t = trigger()
        self.assertEqual([i for i, x in enumerate([0, 0, 0, 0, 2, 2, 2])
                          if t(proposal(i, (x, 0, 0)))], [2, 6])

    def test_sanitized_frame_contract(self):
        p = supervisor_observation(proposal(0).observation)
        self.assertEqual(p.observation, {'robot_state': {'eef': {
            'pos': [0, 0, 0], 'frame': 'world', 'position_units': 'm'}}})

    def test_configuration(self):
        for change in ({'frame': ''}, {'position_units': None}, {'window': True},
                       {'window': 1}, {'min_interval_seconds': 0},
                       {'max_interval_seconds': .01}, {'displacement_tolerance': -1},
                       {'displacement_tolerance': float('inf')}):
            with self.assertRaises(ValueError):
                ProgressSettings(**(dict(frame='world', position_units='m') | change))

    def test_stall_pause_motion_and_unknown_complete_replay(self):
        # A failed stall candidate and a successful intentional pause produce
        # the same motion signal. Supervisor assessment preserves policy actions.
        for success, moving, frame, expected in ((False, False, 'world', [0, 2]),
                (True, False, 'world', [0, 2]), (True, True, 'world', [0]),
                (False, False, None, [0])):
            with self.subTest(success=success, moving=moving, frame=frame), TemporaryDirectory() as temp:
                observations = [observation([i if moving else 0, 0, 0], frame) for i in range(5)]
                actions = [[1]] * 4
                config = EpisodeConfig(17, 4, 10)
                env = ReplayEnvironment(17, observations[0], [
                    (a, ReplayStep(observations[i + 1], 0, success if i == 3 else False,
                                   i == 3, False)) for i, a in enumerate(actions)])
                t = EndEffectorProgressTrigger(ProgressSettings('world', 'm', 3, 1e-12, 60, .5))
                assessed = []
                def decide(p):
                    assessed.append(p.observation.sequence)
                    return SupervisorPass(p.observation.episode_id, p.observation.sequence, p.proposal_id)
                directory = Path(temp) / 'trace'
                recorder = TraceRecorder(directory, config, ReplayRecorder())
                outcome = run_episode(config, ReplayPolicy(list(zip(observations, actions))),
                    env, recorder, supervisor_decider=decide, assessment_trigger=t)
                self.assertEqual(outcome.success, success)
                self.assertEqual(assessed, expected)
                self.assertEqual(env.actions, actions)
                recorder.seal(outcome)
                replay = load_recorded_replay(directory)
                self.assertEqual(replay.run().success, success)
                self.assertEqual([i for i, row in enumerate(replay.evidence()['decisions'])
                                  if row['action_record']['supervisor_pass']], expected)


if __name__ == '__main__':
    unittest.main()
