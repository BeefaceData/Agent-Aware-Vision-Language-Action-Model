"""Synthetic execution contracts, not robot safety or measured recovery efficacy."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from correction_validation import CorrectionValidator
from episode_harness import (ActionResolution, EpisodeConfig, ObservationRejected,
                             RobotStateCapture, ViewCapture, run_episode)
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from recovery_eligibility import ReopenRetreatEligibility
from recovery_registry import RecoveryRegistry
from recovery_monitor import RecoveryAssessment
from reopen_retreat import ReopenRetreatControl, ReopenRetreatExecutor
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from translation_conversion import LiberoTranslationConverter
from test_recovery_eligibility import fresh_proposal, request_for, scene_for
from test_recovery_registry import fixture_tool
from test_supervisor_vlm import raw
from test_translation_conversion import evidence_fixture


class ReopenRetreatTests(unittest.TestCase):
    def setUp(self):
        report = evidence_fixture()
        evidence = json.dumps(report).encode()
        self.converter = LiberoTranslationConverter(evidence,
            expected_sha256=hashlib.sha256(evidence).hexdigest(),
            runtime_controller=report['controller'] | {'position_limits': None},
            target='panda_arm', maximum_metres=0.03)
        self.control = ReopenRetreatControl('synthetic_left_world_retreat_v1',
                                          'panda_arm', (0, 0, 1), -1)
        self.current = replace(fresh_proposal(), action=[0.1] * 7)
        self.opening = (0.,) * 6 + (-1,)
        self.retreat = (0., 0., 0.02 / 0.05, 0., 0., 0., -1)

    def executor(self, tool=None, control=None):
        tool = tool or fixture_tool()
        gate = ReopenRetreatEligibility(CorrectionValidator(
            RecoveryRegistry((tool,)), tool.required_capabilities), self.control.envelope_id, 0.5)
        return ReopenRetreatExecutor(gate, self.converter, control or self.control)

    def resolve(self, executor=None, proposal=None, request=None, scene=None, now=2.1):
        current = proposal or self.current
        return (executor or self.executor()).resolve(current, request or request_for(current),
            scene=scene or scene_for(current), now_monotonic=now)

    def test_bounded_sequence_uses_replacement_commands_and_retains_request(self):
        before = deepcopy(self.current)
        result = self.resolve()
        self.assertEqual(result.kind, 'recovery')
        self.assertEqual(result.recovery.actions, (self.opening, self.retreat))
        self.assertEqual(result.recovery.request['proposal_id'], self.current.proposal_id)
        self.assertEqual(result.recovery.action_limit, 3)
        self.assertEqual(result.recovery.controller_evidence_sha256, self.converter.evidence_sha256)
        self.assertEqual(self.current, before)
        for direction in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)):
            result = self.resolve(executor=self.executor(control=replace(self.control, direction_world=direction)))
            self.assertEqual(result.recovery.actions[1][:3], tuple(d * 0.02 / 0.05 for d in direction))

    def test_limits_invalid_requests_scene_and_duplicate_consumption_reject(self):
        executor = self.executor()
        self.assertEqual(self.resolve(executor).kind, 'recovery')
        self.assertIn('consumed', self.resolve(executor).reason)
        self.assertIn('two actions', self.resolve(self.executor(replace(fixture_tool(), action_limit=1))).reason)
        for change in ({'parameters': {'retreat_m': 0.031}}, {'parameters': {'retreat_m': -0.01}},
                       {'code': 'execute_me()'}, {'proposal_id': 'old'}):
            self.assertEqual(self.resolve(request=request_for(self.current) | change).kind, 'reject')
        self.assertEqual(self.resolve(scene=replace(scene_for(self.current), possible_held_payload=True)).kind, 'reject')
        self.assertEqual(self.resolve(now=3).kind, 'reject')
        for distance in (0, 0.03):
            result = self.resolve(request=request_for(self.current) | {'parameters': {'retreat_m': distance}},
                                  scene=replace(scene_for(self.current), clear_retreat_m=distance))
            self.assertEqual(result.kind, 'recovery')
            self.assertAlmostEqual(result.recovery.actions[1][2], distance / 0.05)

    def test_host_geometry_gripper_target_and_envelope_are_explicit(self):
        for changes in ({'direction_world': (0, 0, 0)}, {'direction_world': (0, 0, 2)},
                        {'direction_world': (True, 0, 0)}, {'direction_world': (float('nan'), 0, 1)},
                        {'gripper_open_native': 1.01}, {'gripper_open_native': True},
                        {'gripper_open_native': float('inf')}):
            with self.assertRaises(ValueError):
                replace(self.control, **changes)
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.executor(control=replace(self.control, envelope_id='other'))
        self.assertEqual(self.resolve(self.executor(control=replace(self.control, target='both_arms'))).kind, 'reject')

    def observe(self, plan, index, packet):
        return RecoveryAssessment(packet.episode_id, packet.sequence, plan.envelope_id,
                                  True, index == 1, False)

    def episode(self, *, horizon=3, terminal_first=False, invalid=False, fail=False, policy=None,
                tool=None):
        wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        class Environment(ReplayEnvironment):
            def step(inner, action):
                if fail and len(inner.actions) == 1:
                    raise RuntimeError('controller failure')
                result = super().step(action)
                if invalid:
                    return replace(result, observation=replace(result.observation, sequence=0))
                return result
        environment = Environment(17, raw(0), (
            (self.opening, ReplayStep(raw(1), 0, terminal_first, terminal_first, False,
                {"main": ViewCapture(1, wall, 10.)}, RobotStateCapture(1, wall, 10.))),
            (self.retreat, ReplayStep(raw(2), 0, False, False, False,
                {"main": ViewCapture(2, wall, 10.)}, RobotStateCapture(2, wall, 10.))),
            ([0.2] * 7, ReplayStep(raw(3), 1, True, True, False))), clock=lambda: 10.,
            initial_camera_captures={'main': ViewCapture(0, wall, 10.)},
            max_camera_skew_seconds=0.1,
            initial_robot_state_capture=RobotStateCapture(0, wall, 10.))
        class Policy(ReplayPolicy):
            def __init__(inner):
                super().__init__(((raw(0), [0.1] * 7), (raw(2), [0.2] * 7)))
                inner.resumes = []
            def resume(inner, packet):
                inner.resumes.append(packet.sequence)
                super().resume(packet)
        policy = policy or Policy()
        executor = self.executor(tool)
        calls = []
        def select(current):
            calls.append(current.observation.sequence)
            if current.observation.sequence == 0:
                return self.resolve(executor, current, now=10.)
            return ActionResolution('pass')
        return EpisodeConfig(17, horizon), policy, environment, select, calls

    def test_full_episode_records_every_action_and_resumes_once_on_fresh_state(self):
        config, policy, environment, select, calls = self.episode()
        with TemporaryDirectory() as tmp:
            trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
            outcome = run_episode(config, policy, environment, trace,
                                  recovery_observer=self.observe, action_selector=select, clock=lambda: 10.)
            trace.seal(outcome)
            self.assertTrue(outcome.success)
            self.assertEqual(outcome.steps, 3)
            self.assertEqual(calls, [0, 2])
            self.assertEqual([p.sequence for p in policy.observations], [0, 2])
            self.assertEqual(policy.resumes, [2])
            replay = load_recorded_replay(trace.directory)
            self.assertTrue(replay.run().success)
            rows = [r['action_record'] for r in replay.evidence()['decisions']]
            self.assertEqual([r['executed_action'] for r in rows],
                             [list(self.opening), list(self.retreat), [0.2] * 7])
            self.assertEqual([r['recovery']['action_index'] for r in rows[:2]], [0, 1])
            self.assertEqual(rows[0]['recovery']['sequence'], rows[1]['recovery']['sequence'])
            self.assertIsNone(rows[2]['recovery'])
            self.assertEqual([r['recovery']['check']['status'] for r in rows[:2]],
                             ['continuing', 'completed'])

    def test_local_failure_stops_with_sealed_partial_execution_and_no_resume(self):
        cases = (
            ('opening', 1, 'gripper_open_confirmed'),
            ('retreat', 2, 'retreat_target_reached'),
            ('limit', 2, 'action_limit_reached'),
            ('clearance', 1, 'clearance_unverified'),
            ('unknown', 1, 'clearance_unverified'),
            ('missing', 1, 'missing_assessment'),
            ('foreign', 1, 'assessment_identity_mismatch'),
            ('error', 1, 'missing_assessment'),
        )
        for mode, count, reason in cases:
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                tool = replace(fixture_tool(), action_limit=2 if mode == 'limit' else 3)
                config, policy, environment, select, calls = self.episode(tool=tool)
                seen = []
                def observe(plan, index, packet):
                    seen.append(packet)
                    self.assertNotIn('private_evaluator', packet.observation)
                    self.assertNotIn('private', packet.observation['robot_state'])
                    assessment = self.observe(plan, index, packet)
                    if mode == 'opening':
                        return replace(assessment, gripper_open_confirmed=False)
                    if mode in ('retreat', 'limit'):
                        return replace(assessment, retreat_target_reached=False)
                    if mode in ('clearance', 'unknown'):
                        return replace(assessment, clearance_unverified=True if mode == 'clearance' else None)
                    if mode == 'missing':
                        return None
                    if mode == 'foreign':
                        return replace(assessment, observation_sequence=0)
                    raise RuntimeError('local assessor failed')
                trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
                outcome = run_episode(config, policy, environment, trace,
                    action_selector=select, recovery_observer=observe, clock=lambda: 10.)
                trace.seal(outcome)
                self.assertEqual((outcome.stop_reason, outcome.steps, outcome.success),
                                 ('recovery_aborted', count, False))
                self.assertEqual(len(environment.actions), count)
                self.assertEqual(len(seen), count)
                self.assertEqual(calls, [0])
                self.assertEqual(policy.resumes, [])
                self.assertEqual(len(policy.observations), 1)
                replay = load_recorded_replay(trace.directory)
                self.assertEqual(replay.run().stop_reason, 'recovery_aborted')
                check = replay.evidence()['decisions'][-1]['action_record']['recovery']['check']
                self.assertEqual((check['reason'], check['executed_actions'], check['path']),
                                 (reason, count, 'stop_episode'))
                self.assertEqual({item['name'] for item in check['evidence']},
                                 {'main', 'robot_state'})

    def test_missing_or_stale_post_action_sensor_aborts_before_retreat(self):
        for mode in ('missing', 'stale'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                config, policy, environment, select, _ = self.episode()
                step = environment.step
                def changed(action):
                    result = step(action)
                    packet = result.observation
                    if mode == 'missing':
                        packet = replace(packet, observation=packet.observation | {'robot_state': {}})
                    else:
                        packet = replace(packet, robot_state_capture=replace(
                            packet.robot_state_capture, captured_monotonic=9.))
                    return replace(result, observation=packet)
                environment.step = changed
                trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
                outcome = run_episode(config, policy, environment, trace, action_selector=select,
                                      recovery_observer=self.observe, clock=lambda: 10.)
                trace.seal(outcome)
                self.assertEqual(outcome.steps, 1)
                self.assertEqual(policy.resumes, [])
                replay = load_recorded_replay(trace.directory)
                self.assertEqual(replay.run().stop_reason, 'recovery_aborted')
                check = replay.evidence()['decisions'][0]['action_record']['recovery']['check']
                self.assertEqual(check['reason'], 'stale_observation')

    def test_missing_assessor_and_unknown_conditions_fail_before_actuation(self):
        config, policy, environment, select, _ = self.episode()
        with self.assertRaisesRegex(ValueError, 'local observation assessor'):
            run_episode(config, policy, environment, ReplayRecorder(),
                        action_selector=select, clock=lambda: 10.)
        self.assertEqual(environment.actions, [])
        for change in ({'completion_conditions': ('made_up',)},
                       {'abort_conditions': ('made_up',)}):
            self.assertEqual(self.resolve(self.executor(replace(fixture_tool(), **change))).kind,
                             'reject')

    def test_horizon_preflight_rejects_without_partial_execution(self):
        config, policy, environment, select, _ = self.episode(horizon=1)
        with TemporaryDirectory() as tmp:
            trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
            outcome = run_episode(config, policy, environment, trace,
                                  recovery_observer=self.observe, action_selector=select, clock=lambda: 10.)
            trace.seal(outcome)
            self.assertEqual(outcome.stop_reason, 'proposal_rejected')
            self.assertEqual(environment.actions, [])
            replay = load_recorded_replay(trace.directory)
            self.assertEqual(replay.run().stop_reason, 'proposal_rejected')
            self.assertIn('horizon', replay.evidence()['decisions'][0]['action_record']['rejection_reason'])

    def test_terminal_and_exact_horizon_do_not_resume_or_execute_extra_actions(self):
        for options, steps in (({'terminal_first': True}, 1), ({'horizon': 2}, 2)):
            config, policy, environment, select, _ = self.episode(**options)
            with TemporaryDirectory() as tmp:
                trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
                outcome = run_episode(config, policy, environment, trace,
                                      recovery_observer=self.observe, action_selector=select, clock=lambda: 10.)
                trace.seal(outcome)
                self.assertEqual(outcome.steps, steps)
                self.assertEqual(policy.resumes, [])
                self.assertEqual(len(policy.observations), 1)
                self.assertEqual(load_recorded_replay(trace.directory).run().steps, steps)
                self.assertEqual(outcome.success, options.get('terminal_first', False))

    def test_terminal_task_success_does_not_turn_local_abort_into_completion(self):
        config, policy, environment, select, _ = self.episode(terminal_first=True)
        with TemporaryDirectory() as tmp:
            trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
            outcome = run_episode(config, policy, environment, trace, action_selector=select,
                                  recovery_observer=lambda *args: None, clock=lambda: 10.)
            trace.seal(outcome)
            self.assertEqual((outcome.success, outcome.steps, outcome.stop_reason),
                             (True, 1, 'success'))
            self.assertEqual(policy.resumes, [])
            replay = load_recorded_replay(trace.directory)
            self.assertTrue(replay.run().success)
            check = replay.evidence()['decisions'][0]['action_record']['recovery']['check']
            self.assertEqual(check['status'], 'aborted')

    def test_invalid_observation_or_controller_failure_stops_pending_sequence(self):
        for options, error in (({'invalid': True}, ObservationRejected), ({'fail': True}, RuntimeError)):
            config, policy, environment, select, _ = self.episode(**options)
            recorder = ReplayRecorder()
            with self.assertRaises(error) as caught:
                run_episode(config, policy, environment, recorder,
                            recovery_observer=self.observe, action_selector=select, clock=lambda: 10.)
            self.assertEqual(len(environment.actions), 1)
            self.assertEqual(policy.resumes, [])
            self.assertEqual(len(policy.observations), 1)
            self.assertEqual(caught.exception.episode_interruption.steps, 1)
            self.assertEqual(recorder.steps[0][3].action_record.recovery['action_index'], 0)
            if options.get('fail'):
                record = recorder.failures[0][3].action_record
                self.assertEqual(record.recovery['action_index'], 1)
                self.assertEqual(record.selected_action, self.retreat)
                self.assertIsNone(record.executed_action)
                self.assertEqual(record.recovery['check']['reason'], 'controller_failure')
                self.assertEqual(record.recovery['check']['executed_actions'], 1)
            else:
                self.assertEqual(recorder.steps[0][3].action_record.recovery['check']['reason'],
                                 'stale_observation')

    def test_temporal_history_keeps_recovery_observations_without_extra_assessment(self):
        config, policy, environment, select, _ = self.episode()
        windows = []
        outcome = run_episode(config, policy, environment, ReplayRecorder(),
            recovery_observer=self.observe, action_selector=select, window_supervisor=lambda window, action: windows.append(window),
            clock=lambda: 10.)
        self.assertTrue(outcome.success)
        self.assertEqual(len(windows), 2)
        self.assertEqual([p.sequence for p in windows[-1].observations], [0, 1, 2])
        self.assertEqual(len(windows[-1].actions), 2)

    def test_unsupported_resume_and_foreign_plan_fail_before_execution(self):
        for mode in ('resume', 'foreign'):
            config, policy, environment, select, _ = self.episode()
            if mode == 'resume':
                policy.resume = None
            def choose(current):
                result = select(current)
                if mode == 'foreign':
                    return replace(result, recovery=replace(result.recovery,
                        request=result.recovery.request | {'proposal_id': 'foreign'}))
                return result
            with self.assertRaises((ValueError, NotImplementedError)):
                run_episode(config, policy, environment, ReplayRecorder(),
                            recovery_observer=self.observe, action_selector=choose, clock=lambda: 10.)
            self.assertEqual(environment.actions, [])

    def test_replay_rejects_corrupt_recovery_provenance_even_with_matching_checksum(self):
        for damage in ('missing', 'order', 'source', 'command', 'limit', 'check', 'evidence'):
            with self.subTest(damage=damage), TemporaryDirectory() as tmp:
                config, policy, environment, select, _ = self.episode()
                trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())
                outcome = run_episode(config, policy, environment, trace,
                                      recovery_observer=self.observe, action_selector=select, clock=lambda: 10.)
                trace.seal(outcome)
                path = trace.directory / 'decisions.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                record = rows[1]['action_record']
                if damage == 'missing':
                    record['recovery'] = None
                elif damage == 'order':
                    record['recovery']['action_index'] = 0
                elif damage == 'source':
                    record['recovery']['sequence']['request']['proposal_id'] = 'foreign'
                elif damage == 'command':
                    record['recovery']['sequence']['actions'][1][2] = 0.1
                elif damage == 'check':
                    record['recovery']['check']['status'] = 'continuing'
                elif damage == 'evidence':
                    record['recovery']['check']['assessment']['retreat_target_reached'] = False
                else:
                    record['recovery']['sequence']['action_limit'] = 1
                data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                path.write_bytes(data)
                manifest_path = trace.directory / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['files']['decisions.jsonl'] = hashlib.sha256(data).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(TraceError):
                    load_recorded_replay(trace.directory)


if __name__ == '__main__':
    unittest.main()
