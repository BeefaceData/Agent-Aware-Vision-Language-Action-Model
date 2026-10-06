"""Recovery-only ablation through executors, dispatch and sealed replay."""

from dataclasses import replace
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


class RecoveryOnlyTests(unittest.TestCase):
    def setUp(self):
        self.recovery = recoveries.ReopenRetreatTests()
        self.recovery.setUp()
        self.adjustment = adjustments.SingleActionAdjustmentTests()
        self.adjustment.setUp()

    def episode(self, directory, *, mode='recovery_only', fallback=True,
                invalid_recovery=False, cap=2, abort=False):
        action = self.adjustment.action
        adjusted = [action[0] + .01 / .05, action[1] - .02 / .05,
                    action[2] + .005 / .05, *action[3:]]
        commands = [action if mode == 'recovery_only' else adjusted,
                    self.recovery.opening, self.recovery.retreat, action]
        wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        environment = ReplayEnvironment(17, raw(0), tuple(
            (command, ReplayStep(raw(i), int(i == 4), i == 4, i == 4, False,
                {'main': ViewCapture(i, wall, 10.)}, RobotStateCapture(i, wall, 10.)))
            for i, command in enumerate(commands, 1)), clock=lambda: 10.,
            initial_camera_captures={'main': ViewCapture(0, wall, 10.)},
            max_camera_skew_seconds=.1,
            initial_robot_state_capture=RobotStateCapture(0, wall, 10.))
        policy = ReplayPolicy(tuple((raw(i), action) for i in (0, 1, 3)))
        executor = self.recovery.executor()

        def select(proposal):
            if proposal.observation.sequence == 0:
                return self.adjustment.executor.resolve(proposal, self.adjustment.request(proposal))
            if proposal.observation.sequence == 1:
                from test_recovery_eligibility import scene_for
                scene = scene_for(proposal)
                if invalid_recovery:
                    scene = replace(scene, possible_held_payload=True)
                return self.recovery.resolve(executor, proposal, scene=scene, now=10.)
            return ActionResolution('pass')

        def observe(plan, index, packet):
            assessment = self.recovery.observe(plan, index, packet)
            return replace(assessment, clearance_unverified=True) if abort else assessment

        config = EpisodeConfig(17, 4, max_interventions=cap, correction_mode=mode,
            max_supervisor_calls=4, max_episode_seconds=20.,
            correction_timeout_seconds=1., correction_max_age_seconds=1.)
        # Rejection of a recovery remains fail-closed in this fixture; only the
        # explicitly disallowed adjustment may take the guarded baseline path.
        guard = BaselineFallback({'robot_state': .5}, lambda: True,
            lambda p: p.action == action and not (invalid_recovery and p.observation.sequence == 1))
        trace = TraceRecorder(directory, config, ReplayRecorder())
        outcome = run_episode(config, policy, environment, trace, action_selector=select,
            recovery_observer=observe, baseline_fallback=guard if fallback else None,
            clock=lambda: 10.)
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        self.assertEqual(replay.report()['replay_outcome']['steps'], outcome.steps)
        return outcome, replay, environment

    def test_same_requests_only_recovery_changes_baseline_in_diagnostic_mode(self):
        with TemporaryDirectory() as tmp:
            outcome, replay, env = self.episode(Path(tmp) / 'recovery')
            # A fresh executor avoids intentionally consumed proposal identities.
            self.setUp()
            combined, both, _ = self.episode(Path(tmp) / 'combined', mode='combined')
            manifest = json.loads((Path(tmp) / 'recovery' / 'manifest.json').read_text())
        self.assertTrue(outcome.success)
        self.assertTrue(combined.success)
        self.assertEqual(outcome.steps, 4)
        self.assertEqual(manifest['config']['correction_mode'], 'recovery_only')
        config = replay.evidence()['config']
        self.assertEqual(config | {'correction_mode': 'combined'}, both.evidence()['config'])
        rows = [r['action_record'] for r in replay.evidence()['decisions']]
        self.assertEqual([r['disposition'] for r in rows],
                         ['unmodified', 'overridden', 'overridden', 'unmodified'])
        self.assertIn('adjustment disallowed', rows[0]['fallback']['cause'])
        self.assertEqual(env.actions, [self.adjustment.action, self.recovery.opening,
                                      self.recovery.retreat, self.adjustment.action])
        self.assertEqual(rows[2]['intervention_budget']['interventions'], 1)
        self.assertEqual(replay.report()['correction_mode'], 'recovery_only')
        self.assertTrue(replay.report()['optional_ablation'])
        self.assertFalse(replay.report()['primary_acceptance_evidence'])

    def test_disallowed_adjustment_stops_without_explicit_fallback(self):
        with TemporaryDirectory() as tmp:
            outcome, replay, env = self.episode(Path(tmp) / 'trace', fallback=False)
        self.assertEqual((outcome.steps, outcome.stop_reason), (0, 'proposal_rejected'))
        self.assertEqual(env.actions, [])
        record = replay.evidence()['decisions'][0]['action_record']
        self.assertEqual(record['rejection_reason'], 'adjustment disallowed in recovery_only mode')

    def test_recovery_scene_validation_and_intervention_cap_remain_enforced(self):
        for kwargs in ({'invalid_recovery': True}, {'cap': 0}):
            with self.subTest(kwargs=kwargs), TemporaryDirectory() as tmp:
                self.setUp()
                outcome, replay, env = self.episode(Path(tmp) / 'trace', **kwargs)
                self.assertEqual((outcome.steps, outcome.stop_reason), (1, 'proposal_rejected'))
                self.assertEqual(env.actions, [self.adjustment.action])
                self.assertIsNotNone(replay.evidence()['decisions'][-1]['action_record']['rejection_reason'])

    def test_recovery_abort_still_stops_the_episode(self):
        with TemporaryDirectory() as tmp:
            outcome, replay, _ = self.episode(Path(tmp) / 'trace', abort=True)
        self.assertEqual((outcome.steps, outcome.stop_reason), (2, 'recovery_aborted'))
        self.assertEqual(replay.evidence()['decisions'][-1]['action_record']
                         ['recovery']['check']['reason'], 'clearance_unverified')

    def test_replay_rejects_an_adjustment_mislabeled_as_recovery_only(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'trace'
            self.episode(directory, mode='combined')
            path = directory / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['config']['correction_mode'] = 'recovery_only'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TraceError, 'adjustment disallowed'):
                load_recorded_replay(directory)

    def test_invalid_mode_fails_before_startup(self):
        for mode in ('adjustment_only', 'unknown', None, True):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'correction mode'):
                EpisodeConfig(17, 4, correction_mode=mode)


if __name__ == '__main__':
    unittest.main()
