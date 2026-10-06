"""Recovery followed by adjustment through public executors and sealed replay."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import ActionResolution, EpisodeConfig, RobotStateCapture, ViewCapture, run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from test_recovery_eligibility import request_for
import test_reopen_retreat as recoveries
import test_single_action_adjustment as adjustments
from test_supervisor_vlm import raw


class CombinedCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.recovery = recoveries.ReopenRetreatTests()
        self.recovery.setUp()
        self.adjustment = adjustments.SingleActionAdjustmentTests()
        self.adjustment.setUp()

    def run_scenario(self, directory, *, baseline=False, horizon=4, failure=None):
        action = self.adjustment.action
        adjusted = [action[0] + .01 / .05, action[1] - .02 / .05,
                    action[2] + .005 / .05, *action[3:]]
        commands = ([action] * 4 if baseline else
                    [self.recovery.opening, self.recovery.retreat, adjusted, action])
        wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        turns = tuple((command, ReplayStep(raw(i), int(i == 4), i == 4, i == 4, False,
            {'main': ViewCapture(i, wall, 10.)}, RobotStateCapture(i, wall, 10.)))
            for i, command in enumerate(commands, 1))
        environment = ReplayEnvironment(17, raw(0), turns, clock=lambda: 10.,
            initial_camera_captures={'main': ViewCapture(0, wall, 10.)},
            max_camera_skew_seconds=.1,
            initial_robot_state_capture=RobotStateCapture(0, wall, 10.))

        class Policy(ReplayPolicy):
            def __init__(self):
                super().__init__(tuple((raw(i), action) for i in
                                      (range(4) if baseline else (0, 2, 3))))
                self.resumes = []

            def resume(self, packet):
                super().resume(packet)
                self.resumes.append(packet.sequence)

        policy = Policy()
        requests, selections = [], []
        executor = self.recovery.executor()

        def select(proposal):
            sequence = proposal.observation.sequence
            selections.append(sequence)
            if baseline or sequence == 3:
                return ActionResolution('pass')
            if sequence == 0:
                request = request_for(proposal)
                resolution = self.recovery.resolve(executor, proposal, request, now=10.)
                self.assertEqual(resolution.kind, 'recovery')
            else:
                request = self.adjustment.request(proposal)
                if failure == 'stale_adjustment':
                    request['proposal_id'] = requests[0]['proposal_id']
                elif failure == 'unsupported_frame':
                    request['frame'] = 'camera'
                resolution = self.adjustment.executor.resolve(proposal, request)
                self.assertEqual(resolution.kind, 'reject' if failure else 'override')
            requests.append(request)
            return resolution

        def observe(plan, index, packet):
            assessment = self.recovery.observe(plan, index, packet)
            return (replace(assessment, clearance_unverified=True)
                    if failure == 'clearance' else assessment)

        config = EpisodeConfig(17, horizon, max_interventions=2,
                               recovery_attempt_limits=(('reopen_and_retreat', 1),))
        trace = TraceRecorder(directory, config, ReplayRecorder())
        outcome = run_episode(config, policy, environment, trace,
            action_selector=select, recovery_observer=observe, clock=lambda: 10.)
        trace.seal(outcome)
        replay = load_recorded_replay(directory)
        replayed = replay.run()
        self.assertEqual((replayed.success, replayed.steps, replayed.stop_reason),
                         (outcome.success, outcome.steps, outcome.stop_reason))
        rows = replay.evidence()['decisions']
        self.assertEqual(outcome.steps, len(environment.actions))
        self.assertEqual([r['action_record']['executed_action'] for r in rows
                          if r['action_record']['executed_action'] is not None],
                         [list(command) for command in environment.actions])
        return outcome, rows, policy, selections, requests

    def test_recovery_then_adjustment_resumes_fresh_and_preserves_baseline(self):
        with TemporaryDirectory() as tmp:
            outcome, rows, policy, selections, requests = self.run_scenario(Path(tmp) / 'combined')
            baseline, base_rows, _, _, _ = self.run_scenario(Path(tmp) / 'baseline', baseline=True)
        self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason), (True, 4, 'success'))
        self.assertEqual((baseline.steps, baseline.stop_reason), (4, 'success'))
        self.assertEqual(selections, [0, 2, 3])
        self.assertEqual([p.sequence for p in policy.observations], [0, 2, 3])
        self.assertEqual(policy.resumes, [2, 3])
        self.assertEqual([r['kind'] for r in requests], ['recovery', 'adjustment'])
        self.assertNotEqual(requests[0]['decision_id'], requests[1]['decision_id'])
        self.assertNotEqual(requests[0]['proposal_id'], requests[1]['proposal_id'])
        records = [row['action_record'] for row in rows]
        self.assertEqual([r['disposition'] for r in records], ['overridden'] * 3 + ['unmodified'])
        self.assertEqual([r['recovery']['check']['status'] for r in records[:2]],
                         ['continuing', 'completed'])
        retained_request = records[0]['recovery']['sequence']['request']
        self.assertEqual(retained_request | {'parameters': dict(retained_request['parameters'])}, requests[0])
        self.assertEqual(records[0]['recovery']['sequence'], records[1]['recovery']['sequence'])
        self.assertEqual(records[0]['proposal_id'], requests[0]['proposal_id'])
        self.assertEqual(records[2]['proposal_id'], requests[1]['proposal_id'])
        self.assertEqual(rows[2]['source_sequence'], requests[1]['observation_sequence'])
        self.assertEqual([r['intervention_budget']['interventions'] for r in records[:3]], [1, 1, 2])
        self.assertEqual(records[2]['intervention_budget']['recovery_attempts'], {'reopen_and_retreat': 1})
        self.assertIsNone(records[2]['recovery'])
        self.assertEqual(records[2]['executed_action'][3:], self.adjustment.action[3:])
        self.assertEqual(records[3]['executed_action'], self.adjustment.action)
        self.assertEqual([r['proposed_action'] for r in records], [self.adjustment.action] * 4)
        self.assertTrue(all(r['action_record']['disposition'] == 'unmodified' for r in base_rows))

    def test_both_modes_charge_the_same_action_horizon(self):
        for horizon in (2, 3):
            with self.subTest(horizon=horizon), TemporaryDirectory() as tmp:
                outcome, rows, policy, selections, _ = self.run_scenario(Path(tmp) / 'trace', horizon=horizon)
                self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason),
                                 (False, horizon, 'step_limit'))
                self.assertEqual(len(rows), horizon)
                self.assertEqual(selections, [0] if horizon == 2 else [0, 2])
                self.assertEqual(policy.resumes, [] if horizon == 2 else [2])

    def test_later_invalid_adjustment_keeps_recovery_and_rejection_evidence(self):
        for failure, reason in (('stale_adjustment', 'current proposal'),
                                ('unsupported_frame', 'frame')):
            with self.subTest(failure=failure), TemporaryDirectory() as tmp:
                outcome, rows, policy, selections, _ = self.run_scenario(Path(tmp) / 'trace', failure=failure)
                self.assertEqual((outcome.steps, outcome.stop_reason), (2, 'proposal_rejected'))
                self.assertEqual(selections, [0, 2])
                self.assertEqual(policy.resumes, [2])
                self.assertEqual(rows[1]['action_record']['recovery']['check']['status'], 'completed')
                rejected = rows[-1]['action_record']
                self.assertIn(reason, rejected['rejection_reason'])
                self.assertEqual(rejected['disposition'], 'rejected')
                self.assertIsNone(rejected['executed_action'])
                self.assertTrue(rejected['interruption']['confirmed'])

    def test_recovery_abort_prevents_the_later_adjustment(self):
        with TemporaryDirectory() as tmp:
            outcome, rows, policy, selections, requests = self.run_scenario(Path(tmp) / 'trace', failure='clearance')
        self.assertEqual((outcome.steps, outcome.stop_reason), (1, 'recovery_aborted'))
        self.assertEqual(selections, [0])
        self.assertEqual(policy.resumes, [])
        self.assertEqual([r['kind'] for r in requests], ['recovery'])
        record = rows[-1]['action_record']
        self.assertEqual(record['recovery']['check']['reason'], 'clearance_unverified')
        self.assertEqual(record['intervention_budget']['interventions'], 1)
        self.assertEqual(record['recovery']['check']['path'], 'stop_episode')


if __name__ == '__main__':
    unittest.main()
