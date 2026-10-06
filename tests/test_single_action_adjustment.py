"""Synthetic one-action execution contracts, including sealed episode replay."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import ActionProposal, EpisodeConfig, ObservationPacket, run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from single_action_adjustment import SingleActionAdjustment
from translation_conversion import LiberoTranslationConverter
from test_translation_conversion import evidence_fixture


class SingleActionAdjustmentTests(unittest.TestCase):
    def setUp(self):
        report = evidence_fixture()
        evidence = json.dumps(report).encode()
        self.converter = LiberoTranslationConverter(
            evidence, expected_sha256=hashlib.sha256(evidence).hexdigest(),
            runtime_controller=report['controller'] | {'position_limits': None},
            target='panda_arm', maximum_metres=0.03)
        self.executor = SingleActionAdjustment(self.converter)
        self.action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1]
        self.proposal = ActionProposal('p', ObservationPacket(
            'e', 0, datetime.now(timezone.utc), {}), self.action)

    def request(self, proposal):
        return dict(kind='adjustment', scope='single_action',
                    episode_id=proposal.observation.episode_id,
                    observation_sequence=proposal.observation.sequence,
                    proposal_id=proposal.proposal_id, decision_id='d',
                    target='panda_arm', frame='world',
                    units={name: 'm' for name in ('translation_x', 'translation_y', 'translation_z')},
                    residual=dict(translation_x=0.01, translation_y=-0.02, translation_z=0.005))

    def test_adjusts_only_translation_without_mutating_proposal(self):
        before = deepcopy(self.proposal)
        resolution = self.executor.resolve(self.proposal, self.request(self.proposal))
        self.assertEqual(resolution.kind, 'override')
        for value, expected in zip(resolution.action[:3], [0.3, -0.2, 0.4]):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(resolution.action[3:], self.action[3:])
        self.assertEqual(self.proposal, before)
        resolution.action[0] = 100
        self.assertEqual(self.proposal, before)

    def test_rejects_stale_or_unsupported_requests_and_duplicate_use(self):
        request = self.request(self.proposal)
        for changes in ({'episode_id': 'old'}, {'proposal_id': 'old'},
                        {'observation_sequence': 1}, {'frame': 'camera'},
                        {'target': 'both_arms'}, {'residual': {'rotation_x': 0.01}}):
            with self.subTest(changes=changes):
                self.assertEqual(self.executor.resolve(self.proposal, request | changes).kind, 'reject')
        self.assertEqual(self.executor.resolve(self.proposal, request).kind, 'override')
        self.assertEqual(self.executor.resolve(self.proposal, request).kind, 'reject')
        next_proposal = replace(self.proposal, proposal_id='next',
                                observation=replace(self.proposal.observation, sequence=1))
        self.assertEqual(self.executor.resolve(next_proposal, request).kind, 'reject')
        self.assertEqual(self.executor.resolve(next_proposal, None).kind, 'pass')

    def test_revalidates_typed_requests_and_rejects_invalid_final_actions(self):
        request = self.request(self.proposal)
        checked = self.converter.convert(request, self.proposal).request
        self.assertEqual(self.executor.resolve(self.proposal, replace(checked, frame='tool')).kind, 'reject')
        for value in (0.9, 1.1, float('nan'), True):
            action = self.action.copy()
            action[0] = value
            with self.subTest(value=value):
                self.assertEqual(self.executor.resolve(replace(self.proposal, action=action), checked).kind, 'reject')
        self.assertEqual(self.executor.resolve(self.proposal, checked).kind, 'override')

    def test_inclusive_native_boundaries(self):
        for sign in (-1, 1):
            proposal = replace(self.proposal, action=[sign * 0.5, 0, 0, 0, 0, 0, sign])
            request = self.request(proposal)
            request['residual'] = dict(translation_x=sign * 0.025, translation_y=0, translation_z=0)
            result = SingleActionAdjustment(self.converter).resolve(proposal, request)
            self.assertEqual(result.kind, 'override')
            self.assertEqual(result.action, [sign, 0, 0, 0, 0, 0, sign])

    def test_complete_sealed_episode_has_one_override_and_no_carryover(self):
        with TemporaryDirectory() as tmp:
            initial, current, final = {'task': 'place'}, {'task': 'place', 'time': 1}, {'task': 'done'}
            adjusted = [0.1 + 0.01 / 0.05, 0.2 - 0.02 / 0.05,
                        0.3 + 0.005 / 0.05, 0.4, 0.5, 0.6, -1]
            policy = ReplayPolicy(((initial, self.action), (current, self.action)))
            environment = ReplayEnvironment(17, initial, (
                (adjusted, ReplayStep(current, 0, False, False, False)),
                (self.action, ReplayStep(final, 1, True, True, False))))
            config = EpisodeConfig(17, 3)
            directory = Path(tmp) / 'episode'
            recorder = TraceRecorder(directory, config, ReplayRecorder())

            def select(proposal):
                request = self.request(proposal) if proposal.observation.sequence == 0 else None
                return self.executor.resolve(proposal, request)

            outcome = run_episode(config, policy, environment, recorder, action_selector=select)
            recorder.seal(outcome)
            self.assertTrue(outcome.success)
            self.assertEqual(environment.actions, [adjusted, self.action])
            replay = load_recorded_replay(directory)
            self.assertTrue(replay.run().success)
            records = [row['action_record'] for row in replay.evidence()['decisions']]
            self.assertEqual([r['disposition'] for r in records], ['overridden', 'unmodified'])
            self.assertEqual([r['proposed_action'] for r in records], [self.action, self.action])
            self.assertEqual([r['executed_action'] for r in records], [adjusted, self.action])
            self.assertNotEqual(records[0]['proposal_id'], records[1]['proposal_id'])

    def test_complete_rejected_episode_never_dispatches_out_of_range_adjustment(self):
        with TemporaryDirectory() as tmp:
            action = [0.9, 0, 0, 0, 0, 0, -1]
            initial = {'task': 'place'}
            environment = ReplayEnvironment(17, initial, ())
            config = EpisodeConfig(17, 2)
            directory = Path(tmp) / 'episode'
            recorder = TraceRecorder(directory, config, ReplayRecorder())
            result = run_episode(config, ReplayPolicy(((initial, action),)), environment, recorder,
                                 action_selector=lambda p: self.executor.resolve(p, self.request(p)))
            recorder.seal(result)
            self.assertEqual(result.stop_reason, 'proposal_rejected')
            self.assertEqual(environment.actions, [])
            self.assertEqual(load_recorded_replay(directory).run().stop_reason, 'proposal_rejected')


if __name__ == '__main__':
    unittest.main()
