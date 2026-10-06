"""Exclusive correction modes through public boundaries and sealed replay."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import (ActionResolution, EpisodeConfig, RobotStateCapture,
                             SupervisorResponseError, ViewCapture, run_episode)
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
import test_reopen_retreat as recovery_fixture
import test_single_action_adjustment as adjustment_fixture
import test_supervisor_response as response_fixture
from test_supervisor_vlm import raw


class CorrectionModeTests(unittest.TestCase):
    def setUp(self):
        self.recovery = recovery_fixture.ReopenRetreatTests()
        self.recovery.setUp()
        self.adjustment = adjustment_fixture.SingleActionAdjustmentTests()
        self.adjustment.setUp()
        self.responses = response_fixture.SupervisorResponseTests()
        self.responses.setUp()

    def test_each_decision_decodes_to_exactly_one_mode(self):
        fixture = self.responses
        for response in fixture.responses:
            with self.subTest(kind=response['kind']):
                result = fixture.decoder.decode(response, fixture.proposal)
                self.assertEqual(result.kind, response['kind'])

    def test_cross_mode_fields_reject_even_when_empty_or_zero(self):
        fixture = self.responses
        for response in fixture.responses:
            fields = ({'residual': {}, 'scope': 'single_action'}
                      if response['kind'] == 'recovery' else
                      {'tool_name': '', 'parameters': {}, 'evidence': []})
            for field, value in fields.items():
                with self.subTest(kind=response['kind'], field=field):
                    with self.assertRaisesRegex(SupervisorResponseError, 'conflicting'):
                        fixture.decoder.decode(response | {field: value}, fixture.proposal)
        combined = fixture.responses[2] | fixture.responses[3]
        for kind in ('pass', 'abstain', 'recovery', 'adjustment'):
            with self.assertRaisesRegex(SupervisorResponseError, 'conflicting'):
                fixture.decoder.decode(combined | {'kind': kind}, fixture.proposal)

    def test_conflicting_provider_response_never_dispatches(self):
        fixture = self.responses
        config, policy, environment, recorder = fixture.fixture()
        response = fixture.responses[2] | {'residual': {'translation_x': 0}}
        with self.assertRaisesRegex(SupervisorResponseError, 'conflicting'):
            run_episode(config, policy, environment, recorder,
                supervisor_decider=lambda p: fixture.decoder.decode(fixture.bind(response, p), p))
        self.assertEqual(environment.actions, [])
        self.assertTrue(recorder.finalized)
        self.assertEqual(recorder.failures[0][3].stage, 'supervisor')

    def test_conflicting_host_resolutions_never_dispatch(self):
        for mode in ('pass', 'override', 'reject', 'recovery'):
            with self.subTest(mode=mode):
                config, policy, environment, select, calls = self.recovery.episode()
                recorder = ReplayRecorder()

                def conflicting(proposal):
                    plan = select(proposal).recovery
                    return ActionResolution(mode, action=[0.3] * 7,
                        reason='rejected' if mode == 'reject' else None, recovery=plan)

                with self.assertRaisesRegex(ValueError, 'resolution'):
                    run_episode(config, policy, environment, recorder,
                        action_selector=conflicting, recovery_observer=self.recovery.observe,
                        clock=lambda: 10.)
                self.assertEqual(environment.actions, [])
                self.assertTrue(recorder.finalized)
                self.assertEqual(recorder.failures[0][3].stage, 'selection')

    def test_adjustment_then_recovery_has_no_residual_carryover_in_sealed_replay(self):
        wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        baseline = self.adjustment.action
        adjusted = [0.1 + 0.01 / 0.05, 0.2 - 0.02 / 0.05,
                    0.3 + 0.005 / 0.05, *baseline[3:]]
        commands = [adjusted, self.recovery.opening, self.recovery.retreat, baseline]
        steps = tuple((action, ReplayStep(raw(i), int(i == 4), i == 4, i == 4, False,
            {'main': ViewCapture(i, wall, 10.)}, RobotStateCapture(i, wall, 10.)))
            for i, action in enumerate(commands, 1))
        environment = ReplayEnvironment(17, raw(0), steps, clock=lambda: 10.,
            initial_camera_captures={'main': ViewCapture(0, wall, 10.)},
            max_camera_skew_seconds=0.1,
            initial_robot_state_capture=RobotStateCapture(0, wall, 10.))
        policy = ReplayPolicy(tuple((raw(i), baseline) for i in (0, 1, 3)))
        executor = self.recovery.executor()
        selections = []

        def select(proposal):
            sequence = proposal.observation.sequence
            selections.append(sequence)
            if sequence == 0:
                return self.adjustment.executor.resolve(proposal, self.adjustment.request(proposal))
            if sequence == 1:
                return self.recovery.resolve(executor, proposal, now=10.)
            return ActionResolution('pass')

        config = EpisodeConfig(17, 4)
        with TemporaryDirectory() as tmp:
            trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
            outcome = run_episode(config, policy, environment, trace, action_selector=select,
                recovery_observer=self.recovery.observe, clock=lambda: 10.)
            trace.seal(outcome)
            self.assertTrue(outcome.success)
            self.assertEqual(outcome.steps, 4)
            self.assertEqual(selections, [0, 1, 3])
            self.assertEqual(environment.actions, commands)
            replay = load_recorded_replay(trace.directory)
            replayed = replay.run()
            self.assertEqual((replayed.success, replayed.steps), (True, 4))
            rows = [row['action_record'] for row in replay.evidence()['decisions']]
            self.assertEqual([row['executed_action'] for row in rows],
                             [list(action) for action in commands])
            self.assertIsNone(rows[0]['recovery'])
            self.assertEqual([row['recovery']['action_index'] for row in rows[1:3]], [0, 1])
            self.assertEqual(rows[1]['recovery']['sequence'], rows[2]['recovery']['sequence'])
            self.assertEqual(rows[-1]['disposition'], 'unmodified')
            self.assertEqual([row['proposed_action'] for row in rows], [baseline] * 4)


if __name__ == '__main__':
    unittest.main()
