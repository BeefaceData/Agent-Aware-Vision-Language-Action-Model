"""Memory advice cannot override current execution contracts in sealed replay."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import unittest

from decision_memory import DecisionMemory
from episode_harness import EpisodeConfig, RobotStateCapture, ViewCapture, run_episode
from memory_correction import MemoryCorrectionExecutor
from recorded_replay import TraceError, TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
import test_memory_compatibility as memories
import test_single_action_adjustment as adjustments
import test_reopen_retreat as recoveries
from test_recovery_eligibility import fresh_proposal, request_for, scene_for
from test_supervisor_vlm import raw


class MemoryCorrectionTests(unittest.TestCase):
    def setUp(self):
        fixture = memories.MemoryCompatibilityTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.ref = fixture.retain('history', success=True)
        self.store = fixture.store
        self.selection = DecisionMemory(self.store, [self.ref], lambda proposal: dict(
            task=memories.TASK, robot_capabilities=asdict(fixture.single),
            progress_context=memories.PROGRESS, failure_category='unknown',
            max_entries=1, max_summary_bytes=10000, max_context_bytes=100000))
        self.adjustment = adjustments.SingleActionAdjustmentTests()
        self.adjustment.setUp()

    def episode(self, name, *, changes=None, action=None, geometry=True, cap=2,
                old_request=False, stale=False):
        action = action or self.adjustment.action
        observations = [raw(i) | {'task': memories.TASK['instruction']} for i in range(3)]
        captured = []

        def request(proposal, memory):
            self.assertNotIn('privileged_state', proposal.observation.observation)
            captured.append(deepcopy(memory))
            self.assertEqual(memory['selected'][0]['record_id'], self.ref['record_id'])
            # Advice is detached: a callback cannot alter retained provenance.
            memory['selected'].clear()
            if proposal.observation.sequence == 0:
                return None
            if old_request:
                return self.store.read(self.ref['record_id'],
                    expected_sha256=self.ref['sha256'])['request']
            return self.adjustment.request(proposal) | (changes or {})

        def scene_check(proposal, resolution):
            self.assertNotIn('privileged_state', proposal.observation.observation)
            self.assertEqual(proposal.observation.sequence, 1)
            return geometry

        gate = MemoryCorrectionExecutor(self.selection, self.adjustment.executor,
            request, scene_check, required_sources={'main': .5, 'robot_state': .5})
        wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        adjusted = [action[0] + .01 / .05, action[1] - .02 / .05,
                    action[2] + .005 / .05, *action[3:]]
        environment = ReplayEnvironment(17, observations[0], [
            (action, ReplayStep(observations[1], 0, False, False, False,
                {'main': ViewCapture(1, wall, 10.)}, RobotStateCapture(1, wall, 10.))),
            (adjusted, ReplayStep(observations[2], 1, True, True, False))],
            clock=lambda: 10., max_camera_skew_seconds=.1,
            initial_camera_captures={'main': ViewCapture(0, wall, 10.)},
            initial_robot_state_capture=RobotStateCapture(0, wall, 10.))
        config = EpisodeConfig(17, 3, max_interventions=cap)
        trace = TraceRecorder(self.root / name, config, ReplayRecorder())
        outcome = run_episode(config, ReplayPolicy(list(zip(observations, [action, action]))),
            environment, trace, action_selector=lambda p: gate.resolve(p,
                now_monotonic=11. if stale and p.observation.sequence else 10.), clock=lambda: 10.)
        trace.seal(outcome)
        replay = load_recorded_replay(trace.directory)
        again = replay.run()
        self.assertEqual((again.success, again.steps, again.stop_reason),
                         (outcome.success, outcome.steps, outcome.stop_reason))
        rows = replay.evidence()['decisions']
        for row in rows:
            self.assertEqual(row['action_record']['correction_memory']['selected'][0]['record_id'],
                             self.ref['record_id'])
        self.assertEqual(rows[0]['action_record']['correction_memory'],
                         json.loads(json.dumps(captured[0])))
        return outcome, environment.actions, rows[-1]['action_record'], trace.directory

    def test_relevant_memory_cannot_authorize_current_invalid_composed_action(self):
        outcome, actions, record, _ = self.episode('final-bounds', action=[.9, 0, 0, 0, 0, 0, -1])
        self.assertEqual((outcome.steps, outcome.stop_reason), (1, 'proposal_rejected'))
        self.assertEqual(actions, [[.9, 0, 0, 0, 0, 0, -1]])
        self.assertEqual(record['rejection_reason'], 'adjusted action outside native action bounds')
        self.assertIsNone(record['executed_action'])

    def test_old_identity_frames_units_and_bounds_are_rejected_in_complete_replays(self):
        cases = [({'frame': 'camera'}, 'frame'), ({'target': 'other_arm'}, 'target'),
                 ({'proposal_id': 'old'}, 'current proposal'),
                 ({'units': dict(translation_x='cm', translation_y='m', translation_z='m')}, 'units'),
                 ({'residual': dict(translation_x=.04, translation_y=0, translation_z=0)}, 'bounds')]
        for index, (changes, reason) in enumerate(cases):
            with self.subTest(changes=changes):
                self.adjustment.setUp()
                outcome, actions, record, _ = self.episode(str(index), changes=changes)
                self.assertEqual((outcome.steps, len(actions)), (1, 1))
                self.assertIn(reason, record['rejection_reason'])
        outcome, actions, record, _ = self.episode('old', old_request=True)
        self.assertEqual(len(actions), 1)
        self.assertIn('current proposal', record['rejection_reason'])

    def test_unknown_geometry_stale_evidence_and_budget_still_reject(self):
        for name, options, reason in [
                ('geometry', {'geometry': None}, 'geometry'),
                ('nonboolean', {'geometry': 1}, 'geometry'),
                ('stale', {'stale': True}, 'fresh current scene'),
                ('budget', {'cap': 0}, 'intervention')]:
            with self.subTest(name=name):
                self.adjustment.setUp()
                outcome, actions, record, _ = self.episode(name, **options)
                self.assertEqual((outcome.steps, len(actions)), (1, 1))
                self.assertIn(reason, record['rejection_reason'])

    def test_valid_current_adjustment_executes_and_keeps_sealed_advice(self):
        outcome, actions, record, _ = self.episode('valid')
        self.assertTrue(outcome.success)
        self.assertEqual(len(actions), 2)
        self.assertEqual(record['disposition'], 'overridden')
        self.assertEqual(actions[1][3:], actions[0][3:])

    def test_recovery_history_cannot_override_current_clearance_or_envelope(self):
        fixture = recoveries.ReopenRetreatTests()
        fixture.setUp()
        current = replace(fresh_proposal(), action=[.1] * 7)
        for scene, reason in [(replace(scene_for(current), clear_retreat_m=.001), 'clearance'),
                              (replace(scene_for(current), envelope_id='old'), 'envelope')]:
            gate = MemoryCorrectionExecutor(self.selection, fixture.executor(),
                lambda p, memory: request_for(p), lambda p, r: True,
                required_sources={'main': .5, 'robot_state': .5})
            result = gate.resolve(current, now_monotonic=2.1, scene=scene)
            self.assertEqual(result.kind, 'reject')
            self.assertIn(reason, result.reason)
            self.assertEqual(result.memory_context['selected'][0]['record_id'], self.ref['record_id'])

    def test_complete_recovery_keeps_initiating_memory_and_resumes_from_fresh_state(self):
        fixture = recoveries.ReopenRetreatTests()
        fixture.setUp()
        config, policy, environment, _, _ = fixture.episode()
        gate = MemoryCorrectionExecutor(self.selection, fixture.executor(),
            lambda p, memory: request_for(p) if p.observation.sequence == 0 else None,
            lambda p, r: True, required_sources={'main': .5, 'robot_state': .5})
        trace = TraceRecorder(self.root / 'recovery', config, ReplayRecorder())
        outcome = run_episode(config, policy, environment, trace,
            action_selector=lambda p: gate.resolve(p, scene=scene_for(p), now_monotonic=10.),
            recovery_observer=fixture.observe, clock=lambda: 10.)
        trace.seal(outcome)
        replay = load_recorded_replay(trace.directory)
        self.assertTrue(outcome.success)
        self.assertTrue(replay.run().success)
        self.assertEqual(policy.resumes, [2])
        rows = [row['action_record'] for row in replay.evidence()['decisions']]
        self.assertEqual(rows[0]['correction_memory']['selected'][0]['record_id'], self.ref['record_id'])
        self.assertEqual(rows[0]['recovery']['sequence'], rows[1]['recovery']['sequence'])
        self.assertEqual([list(action) for action in environment.actions],
                         [list(fixture.opening), list(fixture.retreat), [.2] * 7])

    def test_corrupt_history_fails_before_request_or_geometry_callbacks(self):
        (self.root / 'history/context.json').unlink()

        def forbidden(*args):
            self.fail('corrupt memory reached a correction callback')

        gate = MemoryCorrectionExecutor(self.selection, self.adjustment.executor,
            forbidden, forbidden, required_sources={'main': .5})
        with self.assertRaises(TraceError):
            gate.resolve(fresh_proposal(), now_monotonic=2.1)

    def test_replay_rejects_altered_memory_even_after_rehash(self):
        _, _, _, directory = self.episode('tamper', geometry=False)
        path = directory / 'decisions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['action_record']['correction_memory']['selected'].clear()
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['decisions.jsonl'] = sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(TraceError, 'invalid correction memory'):
            load_recorded_replay(directory)


if __name__ == '__main__':
    unittest.main()
