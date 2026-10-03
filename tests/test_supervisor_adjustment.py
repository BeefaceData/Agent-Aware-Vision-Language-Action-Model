"""Recorded numerical requests remain single-action data without authority."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from episode_harness import ActionProposal, EpisodeConfig, ObservationPacket, run_episode
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder
from supervisor_adjustment import AdjustmentComponent, AdjustmentRequestDecoder


class AdjustmentRequestTests(unittest.TestCase):
    def setUp(self):
        self.response = json.loads((Path(__file__).parent / 'fixtures' /
                                    'adjustment_response.json').read_text())
        self.proposal = ActionProposal('fixture-proposal', ObservationPacket(
            'fixture-episode', 2, datetime.now(timezone.utc), {'task': 'place item'}), [0.1])
        self.components = [AdjustmentComponent('translation_x', 'm', -0.03, 0.03),
                           AdjustmentComponent('rotation_z', 'rad', -0.05, 0.05)]
        self.decoder = AdjustmentRequestDecoder('left_arm', 'world', self.components)

    def test_fixture_records_detached_single_action_request(self):
        before = deepcopy(self.proposal)
        request = self.decoder.decode(self.response, self.proposal)
        record = json.loads(json.dumps(asdict(request)))
        self.assertEqual(record, self.response | {
            'units': [['translation_x', 'm'], ['rotation_z', 'rad']],
            'residual': [['translation_x', 0.01], ['rotation_z', -0.02]]})
        self.response['residual']['translation_x'] = 100
        self.response['units']['translation_x'] = 'changed'
        self.components.clear()
        self.assertEqual(request.residual, (('translation_x', 0.01), ('rotation_z', -0.02)))
        self.assertEqual(request.units[0], ('translation_x', 'm'))
        self.assertEqual(self.proposal, before)
        with self.assertRaises(FrozenInstanceError):
            request.target = 'right_arm'

    def test_bounds_are_inclusive_and_values_are_finite_numbers(self):
        for value in (-0.03, 0, 0.03):
            self.response['residual']['translation_x'] = value
            self.assertEqual(self.decoder.decode(self.response, self.proposal).residual[0][1], value)
        for value in (-0.031, 0.031, True, None, '0.01', [], {},
                      float('nan'), float('inf'), -float('inf'), 10 ** 400):
            with self.subTest(value=value):
                self.response['residual']['translation_x'] = value
                with self.assertRaises(ValueError):
                    self.decoder.decode(self.response, self.proposal)

    def test_foreign_stale_or_persistent_requests_fail(self):
        for change in ({'episode_id': 'foreign'}, {'proposal_id': 'next-action'},
                       {'observation_sequence': 1}, {'observation_sequence': True},
                       {'observation_sequence': -1}, {'decision_id': ''},
                       {'decision_id': None}, {'scope': 'chunk'}, {'scope': 'persistent'},
                       {'kind': 'recovery'}, {'kind': 'pass'}, {'target': 'right_arm'},
                       {'frame': 'camera'}, {'tool_name': 'retreat'}, {'code': 'exec'},
                       {'instruction': 'change task'}, {'maximum': 100},
                       {'residual': {'translation_x': 0.01}},
                       {'residual': [0.01, -0.02]},
                       {'residual': {'translation_x': 0.01, 'rotation_z': 0, 'extra': 0}},
                       {'units': {'translation_x': 'cm', 'rotation_z': 'rad'}}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.decoder.decode(self.response | change, self.proposal)
        for field in self.response:
            incomplete = self.response.copy()
            del incomplete[field]
            with self.assertRaises(ValueError):
                self.decoder.decode(incomplete, self.proposal)
        with self.assertRaises(ValueError):
            self.decoder.decode([], self.proposal)

    def test_trusted_configuration_and_coordinated_group(self):
        for name, unit, minimum, maximum in (
                ('', 'm', 0, 1), ('x', '', 0, 1), ('x', 'm', True, 1),
                ('x', 'm', 0, float('inf')), ('x', 'm', 2, 1)):
            with self.assertRaises(ValueError):
                AdjustmentComponent(name, unit, minimum, maximum)
        for target, frame, components in (('', 'world', self.components),
                                         ('arms', '', self.components),
                                         ('arms', 'world', []),
                                         ('arms', 'world', self.components * 2),
                                         ('arms', 'world', ['x'])):
            with self.assertRaises(ValueError):
                AdjustmentRequestDecoder(target, frame, components)
        decoder = AdjustmentRequestDecoder('coupled_arms', 'world', self.components)
        request = decoder.decode(self.response | {'target': 'coupled_arms'}, self.proposal)
        self.assertEqual(request.target, 'coupled_arms')
        # A group label records configured semantics; it does not prove atomic control.

    def test_request_cannot_authorize_execution_in_current_harness(self):
        observation = {'task': 'place item'}
        policy = ReplayPolicy(((observation, [0.1]),))
        environment = ReplayEnvironment(17, observation, ())
        recorder = ReplayRecorder()

        def decide(proposal):
            response = self.response | {
                'episode_id': proposal.observation.episode_id,
                'observation_sequence': proposal.observation.sequence,
                'proposal_id': proposal.proposal_id}
            return self.decoder.decode(response, proposal)

        with self.assertRaisesRegex(ValueError, 'current proposal'):
            run_episode(EpisodeConfig(17, 1), policy, environment, recorder,
                        supervisor_decider=decide)
        self.assertEqual(environment.actions, [])
        self.assertEqual(recorder.failures[0][3].stage, 'supervisor')
        self.assertTrue(recorder.finalized)


if __name__ == '__main__':
    unittest.main()
