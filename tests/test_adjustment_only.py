"""Adjustment-only diagnostic episodes through public executors and sealed replay."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from baseline_fallback import BaselineFallback
from episode_harness import ActionResolution, EpisodeConfig, RobotStateCapture, ViewCapture, run_episode
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
import test_reopen_retreat as recoveries
import test_single_action_adjustment as adjustments
from test_supervisor_vlm import raw


class AdjustmentOnlyTests(unittest.TestCase):
    def episode(self, directory, *, mode='adjustment_only', fallback=True,
                request_changes=None, cap=2, horizon=4, healthy=True):
        recovery = recoveries.ReopenRetreatTests()
        recovery.setUp()
        adjustment = adjustments.SingleActionAdjustmentTests()
        adjustment.setUp()
        action = adjustment.action
        adjusted = [action[0] + .01 / .05, action[1] - .02 / .05,
                    action[2] + .005 / .05, *action[3:]]
        combined = mode == 'combined'
        commands = ([recovery.opening, recovery.retreat] if combined else [action])
        adjustment_sequence = len(commands)
        commands += [adjusted, action]
        wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        environment = ReplayEnvironment(17, raw(0), tuple(
            (command, ReplayStep(raw(i), int(i == len(commands)), i == len(commands),
                i == len(commands), False, {'main': ViewCapture(i, wall, 10.)},
                RobotStateCapture(i, wall, 10.)))
            for i, command in enumerate(commands, 1)), clock=lambda: 10.,
            initial_camera_captures={'main': ViewCapture(0, wall, 10.)},
            max_camera_skew_seconds=.1,
            initial_robot_state_capture=RobotStateCapture(0, wall, 10.))
        policy = ReplayPolicy(tuple((raw(i), action)
            for i in (0, adjustment_sequence, adjustment_sequence + 1)))
        executor = recovery.executor()

        def select(proposal):
            if proposal.observation.sequence == 0:
                return recovery.resolve(executor, proposal, now=10.)
            if proposal.observation.sequence == adjustment_sequence:
                request = adjustment.request(proposal) | (request_changes or {})
                return adjustment.executor.resolve(proposal, request)
            return ActionResolution('pass')

        config = EpisodeConfig(17, horizon, max_interventions=cap, correction_mode=mode,
            max_supervisor_calls=4, max_episode_seconds=20.,
            correction_timeout_seconds=1., correction_max_age_seconds=1.)
        # Permit baseline fallback for the disallowed recovery, but retain an
        # explicit rejection if the later adjustment is invalid or over budget.
        guard = BaselineFallback({'robot_state': .5}, lambda: healthy,
            lambda p: p.action == action and p.observation.sequence == 0)
        trace = TraceRecorder(directory, config, ReplayRecorder())
        outcome = run_episode(config, policy, environment, trace, action_selector=select,
            recovery_observer=recovery.observe, baseline_fallback=guard if fallback else None,
            clock=lambda: 10.)
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        self.assertEqual(replay.report()['replay_outcome']['steps'], outcome.steps)
        return outcome, replay, environment, commands

    def test_full_episode_only_bounded_adjustment_changes_baseline(self):
        with TemporaryDirectory() as tmp:
            outcome, replay, env, commands = self.episode(Path(tmp) / 'adjustment')
            combined, both, _, _ = self.episode(Path(tmp) / 'combined', mode='combined')
            manifest = json.loads((Path(tmp) / 'adjustment' / 'manifest.json').read_text())
        self.assertTrue(outcome.success)
        self.assertTrue(combined.success)
        self.assertEqual((outcome.steps, combined.steps), (3, 4))
        self.assertEqual(env.actions, commands)
        self.assertEqual(manifest['config']['correction_mode'], 'adjustment_only')
        self.assertEqual(replay.evidence()['config'] | {'correction_mode': 'combined'},
                         both.evidence()['config'])
        rows = [r['action_record'] for r in replay.evidence()['decisions']]
        self.assertEqual([r['disposition'] for r in rows],
                         ['unmodified', 'overridden', 'unmodified'])
        self.assertTrue(all(r.get('recovery') is None for r in rows))
        self.assertIn('recovery disallowed in adjustment_only mode', rows[0]['fallback']['cause'])
        self.assertEqual(rows[1]['intervention_budget']['interventions'], 1)
        self.assertEqual(rows[0]['proposed_action'], rows[2]['executed_action'])
        self.assertEqual(rows[1]['executed_action'][3:], rows[1]['proposed_action'][3:])
        report = replay.report()
        self.assertEqual(report['correction_mode'], 'adjustment_only')
        self.assertTrue(report['optional_ablation'])
        self.assertFalse(report['primary_acceptance_evidence'])

    def test_disallowed_recovery_stops_without_explicit_fallback(self):
        with TemporaryDirectory() as tmp:
            outcome, replay, env, _ = self.episode(Path(tmp) / 'trace', fallback=False)
        self.assertEqual((outcome.steps, outcome.stop_reason), (0, 'proposal_rejected'))
        self.assertEqual(env.actions, [])
        self.assertEqual(replay.evidence()['decisions'][0]['action_record']['rejection_reason'],
                         'recovery disallowed in adjustment_only mode')

    def test_disallowed_recovery_cannot_bypass_fallback_health_check(self):
        with TemporaryDirectory() as tmp:
            outcome, replay, env, _ = self.episode(Path(tmp) / 'trace', healthy=False)
        self.assertEqual(outcome.steps, 0)
        self.assertEqual(env.actions, [])
        self.assertEqual(replay.evidence()['decisions'][0]['action_record']['fallback']['selected'],
                         'refuse')

    def test_adjustment_validation_remains_enforced(self):
        cases = ({'frame': 'camera'}, {'proposal_id': 'stale'},
                 {'residual': {'translation_x': .04, 'translation_y': 0, 'translation_z': 0}},
                 {'residual': {'translation_x': float('nan'), 'translation_y': 0, 'translation_z': 0}})
        for changes in cases:
            with self.subTest(changes=changes), TemporaryDirectory() as tmp:
                outcome, replay, env, commands = self.episode(Path(tmp) / 'trace',
                    request_changes=changes)
                self.assertEqual((outcome.steps, outcome.stop_reason), (1, 'proposal_rejected'))
                self.assertEqual(env.actions, commands[:1])
                self.assertTrue(replay.evidence()['decisions'][-1]['action_record']['rejection_reason'])

    def test_intervention_cap_and_action_horizon_remain_enforced(self):
        for kwargs, steps in (({'cap': 0}, 1), ({'horizon': 2}, 2)):
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as tmp:
                outcome, _, env, commands = self.episode(Path(tmp) / 'trace', **kwargs)
                self.assertEqual(outcome.steps, steps)
                self.assertFalse(outcome.success)
                self.assertEqual(env.actions, commands[:steps])

    def test_replay_rejects_recovery_mislabeled_as_adjustment_only(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trace'
            self.episode(directory, mode='combined')
            path = directory / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['config']['correction_mode'] = 'adjustment_only'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TraceError, 'recovery disallowed'):
                load_recorded_replay(directory)


if __name__ == '__main__':
    unittest.main()
