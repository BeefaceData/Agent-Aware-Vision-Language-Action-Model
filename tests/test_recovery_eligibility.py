"""Synthetic public eligibility checks; no measured perception or robot safety."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from correction_validation import CorrectionValidator
from episode_harness import (ActionResolution, EpisodeConfig, RobotStateCapture,
                             run_episode)
from observation_window import ObservationWindowBuilder
from recorded_replay import TraceRecorder, load_recorded_replay
from recovery_eligibility import RecoverySceneAssessment, ReopenRetreatEligibility
from recovery_registry import RecoveryRegistry
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_recovery import RecoveryEvidenceReference
from test_recovery_registry import fixture_tool
from test_supervisor_vlm import proposal, raw


def scene_for(current):
    return RecoverySceneAssessment(
        current.observation.episode_id, current.observation.sequence,
        current.proposal_id, 'synthetic_left_world_retreat_v1',
        'suspected_missed_grasp', False, True, 0.02,
        tuple(RecoveryEvidenceReference(current.observation.sequence, name)
              for name in ('main', 'robot_state')))


def request_for(current):
    return dict(kind='recovery', episode_id=current.observation.episode_id,
                observation_sequence=current.observation.sequence,
                proposal_id=current.proposal_id, decision_id='recovery-check',
                tool_name='reopen_and_retreat', parameters={'retreat_m': 0.02},
                evidence=[dict(observation_sequence=current.observation.sequence,
                               source=name) for name in ('main', 'robot_state')])


def fresh_proposal(index=2):
    current = proposal(index)
    packet = current.observation
    return replace(current, observation=replace(packet,
        frame_references=tuple(replace(ref, time_basis='camera_capture')
                               for ref in packet.frame_references),
        robot_state_capture=RobotStateCapture(index, packet.captured_at, float(index))))


class RecoveryEligibilityTests(unittest.TestCase):
    def setUp(self):
        tool = fixture_tool()
        self.gate = ReopenRetreatEligibility(
            CorrectionValidator(RecoveryRegistry((tool,)), tool.required_capabilities),
            'synthetic_left_world_retreat_v1', 0.5)
        self.current = fresh_proposal()
        self.scene = scene_for(self.current)
        self.request = request_for(self.current)

    def check(self, **kwargs):
        arguments = dict(scene=self.scene, now_monotonic=2.1)
        arguments.update(kwargs)
        return self.gate.check(self.request, self.current, **arguments)

    def test_eligible_missed_grasp_at_inclusive_clearance_leaves_proposal_unchanged(self):
        before = deepcopy(self.current)
        resolved = self.check()
        self.assertEqual(resolved.request.parameters, (('retreat_m', 0.02),))
        self.assertEqual(resolved.tool.name, 'reopen_and_retreat')
        self.assertEqual(self.current, before)
        self.assertEqual(self.gate.check(resolved.request, self.current,
            scene=self.scene, now_monotonic=2.5), resolved)

    def test_uncertain_holding_and_clearance_fail_closed(self):
        for field, values, reason in (
                ('possible_held_payload', (True, None, 0, 'false'), 'held payload'),
                ('gripper_sweep_clear', (False, None, 1, 'true'), 'gripper clearance'),
                ('clear_retreat_m', (None, -1, 0.019, True, '0.1', float('nan'),
                                     float('inf'), 10 ** 400), 'retreat clearance'),
                ('diagnosis', ('unknown', 'stall', 'suspected_lost_grasp'), 'missed-grasp'),
                ('envelope_id', ('unreviewed', None), 'envelope')):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, reason):
                    self.check(scene=replace(self.scene, **{field: value}))

    def test_scene_is_bound_to_current_proposal_and_declared_sources(self):
        for changes in ({'episode_id': 'other'}, {'proposal_id': 'old'},
                        {'observation_sequence': 1}, {'observation_sequence': True},
                        {'evidence': (RecoveryEvidenceReference(2, 'main'),)},
                        {'evidence': (RecoveryEvidenceReference(2, 'private_evaluator'),)},
                        {'evidence': self.scene.evidence * 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check(scene=replace(self.scene, **changes))
        for evidence in ((), None, ('main',)):
            with self.assertRaises(ValueError):
                replace(self.scene, evidence=evidence)

    def test_missing_stale_unknown_capture_and_invalid_source_metadata_reject(self):
        packet = self.current.observation
        variants = (
            replace(packet, observation={'task': 'place item'}),
            replace(packet, robot_state_capture=None),
            replace(packet, frame_references=tuple(replace(ref, time_basis='observation_return')
                                                   for ref in packet.frame_references)),
            replace(packet, frame_references=tuple(replace(ref, observation_sequence=1)
                                                   for ref in packet.frame_references)),
            replace(packet, observation=raw(2) | {'robot_state': {'position': []}}),
        )
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.gate.check(self.request, replace(self.current, observation=variant),
                                scene=self.scene, now_monotonic=2.1)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.check(now_monotonic=2.501)

    def test_request_references_resolve_only_in_active_window(self):
        history = ObservationWindowBuilder('episode', 'place item')
        history.append(fresh_proposal(1).observation)
        history.append(self.current.observation)
        self.request['evidence'] = [dict(observation_sequence=1, source='main')]
        self.assertEqual(self.check(window=history.snapshot()).tool.name, 'reopen_and_retreat')
        with self.assertRaisesRegex(ValueError, 'outside active window'):
            self.check()
        for sequence in (0, 3):
            self.request['evidence'][0]['observation_sequence'] = sequence
            with self.assertRaises(ValueError):
                self.check(window=history.snapshot())
        self.request['evidence'][0]['observation_sequence'] = 1
        foreign = replace(history.snapshot(), observations=(proposal(1, 'foreign').observation,
                                                            self.current.observation))
        with self.assertRaisesRegex(ValueError, 'invalid active'):
            self.check(window=foreign)
        with self.assertRaisesRegex(ValueError, 'current declared'):
            self.check(window=history.snapshot(), scene=replace(self.scene, evidence=(
                RecoveryEvidenceReference(1, 'main'), RecoveryEvidenceReference(2, 'robot_state'))))

    def test_executor_revalidates_bounds_and_rejects_model_scene_overrides(self):
        for change in ({'parameters': {'retreat_m': 0.031}}, {'possible_held_payload': False},
                       {'envelope_id': self.scene.envelope_id}, {'clear_retreat_m': 1},
                       {'tool_name': 'other'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.gate.check(self.request | change, self.current,
                                scene=self.scene, now_monotonic=2.1)
        for value in (-1, True, float('inf'), float('nan'), 10 ** 400):
            with self.assertRaises(ValueError):
                replace(self.gate, max_age_seconds=value)

    def test_complete_rejection_replay_retains_reason_without_recovery_actions(self):
        with TemporaryDirectory() as tmp:
            initial, next_observation = raw(0), raw(1)
            config = EpisodeConfig(17, 3)
            environment = ReplayEnvironment(17, initial, (
                ([0], ReplayStep(next_observation, 0, False, False, False)),))
            trace = TraceRecorder(Path(tmp) / 'episode', config, ReplayRecorder())

            def select(current):
                if current.observation.sequence == 0:
                    return ActionResolution('pass')
                try:
                    self.gate.check(request_for(current), current,
                        scene=scene_for(current), now_monotonic=current.observation.captured_monotonic)
                except ValueError as exc:
                    return ActionResolution('reject', reason=str(exc))
                self.fail('unverified sensor timing permitted recovery')

            result = run_episode(config, ReplayPolicy(((initial, [0]), (next_observation, [1]))),
                                 environment, trace, action_selector=select)
            trace.seal(result)
            self.assertEqual(environment.actions, [[0]])
            self.assertEqual(result.stop_reason, 'proposal_rejected')
            replay = load_recorded_replay(trace.directory)
            self.assertEqual(replay.run().stop_reason, 'proposal_rejected')
            record = replay.evidence()['decisions'][-1]['action_record']
            self.assertIsNone(record['executed_action'])
            self.assertIn('unverified', record['rejection_reason'])


if __name__ == '__main__':
    unittest.main()
