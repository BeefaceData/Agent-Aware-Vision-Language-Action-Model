"""Public visual evidence and assessment scheduling on synthetic camera scenes."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import (ActionProposal, EpisodeConfig, ObservationPacket,
    SupervisorPass, frame_references_for_observation, run_episode)
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from visual_change import VisualChangeSettings, VisualChangeTrigger


def observation(value, wrist=None):
    return {'pixels': {'image': [[value] * 3 for _ in range(3)] if value is not None else None,
                       'image2': [[wrist]] if wrist is not None else None},
            'success': True, 'private_object_pose': [99]}


def proposal(i, value=0, episode='test', time=None, wrist=None):
    obs = observation(value, wrist)
    wall = datetime.now(timezone.utc)
    timestamp = float(i) if time is None else time
    refs = frame_references_for_observation(obs, i, wall, timestamp)
    return ActionProposal(f'{episode}:{i}', ObservationPacket(episode, i, wall, obs,
                          timestamp, refs), [1])


def trigger(**changes):
    return VisualChangeTrigger(VisualChangeSettings(**(dict(
        pixel_max=100, low_change=.1, high_change=.8) | changes)))


class VisualChangeTests(unittest.TestCase):
    def test_thresholds_view_interval_and_sample_identity(self):
        for value, expected in ((0, 'low_change'), (10, 'low_change'),
                                (11, 'intermediate_change'), (79, 'intermediate_change'),
                                (80, 'high_change'), (100, 'high_change')):
            t = trigger()
            self.assertFalse(t(proposal(0)))
            self.assertEqual(t(proposal(1, value)), expected != 'intermediate_change')
            evidence = t.evidence
            self.assertEqual(evidence.status, expected)
            self.assertAlmostEqual(evidence.mean_absolute_change, value / 100)
            self.assertEqual(evidence.interval_seconds, 1)
            self.assertEqual(evidence.settings.view, 'main')
            self.assertEqual([s.proposal_id for s in evidence.samples], ['test:0', 'test:1'])
            self.assertIn('camera_motion_lighting_and_occlusion_are_not_disambiguated', evidence.limitations)
        t = trigger(view='wrist')
        t(proposal(0, 0, wrist=50))
        self.assertTrue(t(proposal(1, 100, wrist=50)))
        self.assertEqual(t.evidence.mean_absolute_change, 0)

    def test_once_per_regime_rearm_and_episode_isolation(self):
        t = trigger()
        self.assertEqual([i for i, value in enumerate([0, 0, 0, 50, 50, 50, 0, 100, 0])
                          if t(proposal(i, value))], [1, 4, 7])
        self.assertFalse(t(proposal(0, episode='next')))
        self.assertTrue(t(proposal(1, episode='next')))

    def test_absent_invalid_stale_and_incompatible_frames_are_unknown(self):
        bad = [proposal(1, value) for value in (None, -1, 101, True, float('nan'))]
        p = proposal(1)
        bad += [replace(p, observation=replace(p.observation, frame_references=())),
                replace(p, observation=replace(p.observation, frame_references=tuple(
                    replace(r, observation_sequence=0) for r in p.observation.frame_references))),
                replace(p, observation=replace(p.observation, frame_references=tuple(
                    replace(r, synchronization='unsynchronized') for r in p.observation.frame_references))),
                replace(p, observation=replace(p.observation, frame_references=tuple(
                    replace(r, time_basis='camera_capture') for r in p.observation.frame_references))),
                replace(p, observation=replace(p.observation, observation={'pixels': {'image': [[0]]}})),
                proposal(3), proposal(1, time=0), proposal(1, time=.05), proposal(1, time=3)]
        for candidate in bad:
            t = trigger()
            t(proposal(0))
            self.assertFalse(t(candidate))
            self.assertEqual(t.evidence.status, 'unknown')
            self.assertIsNone(t.evidence.mean_absolute_change)
        t = trigger()
        self.assertFalse(t(proposal(0)))
        self.assertFalse(t(proposal(1, None)))
        self.assertFalse(t(proposal(2)))
        self.assertTrue(t(proposal(3)))

    def test_rgb_grid_is_bounded_and_detached(self):
        t = trigger(grid_size=2)
        p = proposal(0)
        pixels = [[[0, 50, 100] for _ in range(20)] for _ in range(20)]
        p.observation.observation['pixels']['image'] = pixels
        t(p)
        evidence = t.evidence
        self.assertEqual(evidence.samples[0].shape, (20, 20, 3))
        self.assertEqual(len(evidence.samples[0].values), 12)
        pixels[0][0][0] = 100
        self.assertEqual(evidence.samples[0].values[0], 0)

    def test_camera_shift_and_occlusion_remain_ambiguous_change(self):
        for before, after in (([[0, 100], [0, 100]], [[100, 0], [100, 0]]),
                              ([[100, 100], [100, 100]], [[0, 0], [0, 0]])):
            t = trigger()
            for i, frame in enumerate((before, after)):
                p = proposal(i)
                p.observation.observation['pixels']['image'] = frame
                result = t(p)
            self.assertTrue(result)
            self.assertEqual(t.evidence.status, 'high_change')
            self.assertEqual(t.evidence.mean_absolute_change, 1)
            self.assertIn('pixel_change_does_not_establish_task_progress_or_failure',
                          t.evidence.limitations)

    def test_configuration_and_interval_boundaries(self):
        for changes in ({'view': 'other'}, {'pixel_max': 0}, {'pixel_max': True},
                        {'grid_size': 65}, {'grid_size': True}, {'low_change': -.1},
                        {'high_change': .1}, {'high_change': 1.1},
                        {'min_interval_seconds': 0}, {'max_interval_seconds': .01},
                        {'low_change': float('nan')}):
            with self.assertRaises(ValueError):
                trigger(**changes)
        for interval in (.1, 2):
            t = trigger()
            t(proposal(0))
            self.assertTrue(t(proposal(1, time=interval)))

    def test_complete_sealed_replays_keep_execution_and_periodic_assessment(self):
        # Camera motion/occlusion can look like a large change; a stationary
        # scene can be either successful or unsuccessful. Neither is a verdict.
        for values, success, expected in (([0, 0, 0, 0], False, [0, 1, 3]),
                ([0, 0, 0, 0], True, [0, 1, 3]),
                ([0, 20, 40, 60], True, [0, 3]),
                ([0, 100, 0, 100], False, [0, 1, 3]),
                ([None] * 4, False, [0, 3])):
            with self.subTest(values=values, success=success), TemporaryDirectory() as temp:
                observations = [observation(v) for v in values + [values[-1]]]
                actions = [[1]] * 4
                config = EpisodeConfig(17, 4, 3)
                ticks = iter(range(100))
                env = ReplayEnvironment(17, observations[0], [
                    (a, ReplayStep(observations[i + 1], 0, success if i == 3 else False,
                                   i == 3, False)) for i, a in enumerate(actions)],
                    clock=lambda: float(next(ticks)))
                t = trigger()
                assessed, evidence = [], []
                def assess(p):
                    assessed.append(p.observation.sequence)
                    return SupervisorPass(p.observation.episode_id, p.observation.sequence, p.proposal_id)
                def event(p):
                    result = t(p)
                    evidence.append(t.evidence)
                    return result
                directory = Path(temp) / 'trace'
                recorder = TraceRecorder(directory, config, ReplayRecorder())
                outcome = run_episode(config, ReplayPolicy(list(zip(observations, actions))),
                    env, recorder, supervisor_decider=assess, assessment_trigger=event,
                    clock=lambda: 100.)
                self.assertEqual(outcome.success, success)
                self.assertEqual(assessed, expected)
                self.assertEqual(env.actions, actions)
                self.assertEqual(len(evidence), 4)
                recorder.seal(outcome)
                replay = load_recorded_replay(directory)
                self.assertEqual(replay.run().success, success)
                self.assertEqual([i for i, row in enumerate(replay.evidence()['decisions'])
                                  if row['action_record']['supervisor_pass']], expected)


if __name__ == '__main__':
    unittest.main()
