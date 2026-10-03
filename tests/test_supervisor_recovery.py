"""Recorded recovery requests are bounded data, never robot commands."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from episode_harness import ActionProposal, EpisodeConfig, ObservationPacket, run_episode
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder
from supervisor_recovery import RecoveryParameter, RecoveryRequestDecoder


class RecoveryRequestTests(unittest.TestCase):
    def setUp(self):
        self.response = json.loads((Path(__file__).parent / 'fixtures' /
                                    'recovery_response.json').read_text())
        self.proposal = ActionProposal('fixture-proposal', ObservationPacket(
            'fixture-episode', 2, datetime.now(timezone.utc), {'task': 'place item'}), [0.1])
        # Synthetic interface bounds, not a reviewed robot calibration.
        self.decoder = RecoveryRequestDecoder('reopen_and_retreat', (
            RecoveryParameter('retreat_m', 0.0, 0.03),))

    def test_recorded_request_is_inspectable_detached_and_does_not_execute(self):
        before = deepcopy(self.proposal)
        request = self.decoder.decode(self.response, self.proposal)
        self.assertEqual(request.tool_name, 'reopen_and_retreat')
        self.assertEqual(request.parameters, (('retreat_m', 0.02),))
        self.assertEqual(request.decision_id, 'fixture-decision')
        self.assertEqual(request.proposal_id, before.proposal_id)
        self.assertEqual(request.episode_id, before.observation.episode_id)
        self.assertEqual(request.observation_sequence, 2)
        self.assertEqual(request.evidence[0].source, 'main_camera')
        self.assertEqual(request.evidence[0].observation_sequence, 1)
        self.assertEqual(json.loads(json.dumps(asdict(request)))['kind'], 'recovery')
        self.response['parameters']['retreat_m'] = 99
        self.response['evidence'][0]['source'] = 'changed'
        self.assertEqual(request.parameters, (('retreat_m', 0.02),))
        self.assertEqual(request.evidence[0].source, 'main_camera')
        self.assertEqual(self.proposal, before)
        with self.assertRaises(FrozenInstanceError):
            request.tool_name = 'changed'
        # The public interface needs no environment, policy, or execution callback.

    def test_inclusive_bounds_and_invalid_numeric_values(self):
        for value in (0, 0.03):
            self.response['parameters']['retreat_m'] = value
            self.assertEqual(self.decoder.decode(self.response, self.proposal).parameters,
                             (('retreat_m', value),))
        for value in (-0.001, 0.031, True, None, '0.02', [], {},
                      float('nan'), float('inf'), -float('inf'), 10 ** 400):
            with self.subTest(value=value):
                self.response['parameters']['retreat_m'] = value
                with self.assertRaises(ValueError):
                    self.decoder.decode(self.response, self.proposal)

    def test_unknown_tools_code_definitions_and_extra_fields_are_rejected(self):
        for change in ({'tool_name': 'exec'}, {'code': 'raise RuntimeError()'},
                       {'tool_definition': {}}, {'instruction': 'rewrite task'},
                       {'parameters': {'retreat_m': 0.02, 'maximum': 100}},
                       {'parameters': {'retreat_m': 'print(1)'}},
                       {'parameters': {}}, {'kind': 'pass'}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.decoder.decode(self.response | change, self.proposal)
        self.assertEqual(self.decoder.decode(self.response, self.proposal).tool_name,
                         'reopen_and_retreat')

    def test_identity_and_evidence_validation(self):
        for change in ({'episode_id': 'foreign'}, {'proposal_id': 'foreign'},
                       {'observation_sequence': 1}, {'observation_sequence': True},
                       {'decision_id': ''}, {'decision_id': None}, {'evidence': []},
                       {'evidence': self.response['evidence'] * 17},
                       {'evidence': [self.response['evidence'][0]] * 2},
                       *({'evidence': [ref]} for ref in (
                           {'observation_sequence': 3, 'source': 'main'},
                           {'observation_sequence': -1, 'source': 'main'},
                           {'observation_sequence': True, 'source': 'main'},
                           {'observation_sequence': 1, 'source': ''},
                           {'observation_sequence': 1, 'source': 'main', 'code': 'exec'},
                           'main'))):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.decoder.decode(self.response | change, self.proposal)

    def test_configuration_cannot_be_mutated_through_input_aliases(self):
        bounds = [RecoveryParameter('retreat_m', 0, 0.03)]
        decoder = RecoveryRequestDecoder('reopen_and_retreat', bounds)
        bounds.clear()
        self.assertEqual(decoder.decode(self.response, self.proposal).parameters,
                         (('retreat_m', 0.02),))
        for minimum, maximum in ((True, 1), (0, float('inf')), (2, 1)):
            with self.assertRaises(ValueError):
                RecoveryParameter('retreat_m', minimum, maximum)
        with self.assertRaises(ValueError):
            RecoveryRequestDecoder('reopen_and_retreat', ())
        with self.assertRaises(ValueError):
            RecoveryRequestDecoder('reopen_and_retreat', decoder.parameter_bounds * 2)

    def test_request_cannot_authorize_execution_in_current_harness(self):
        observation = {'task': 'place item'}
        policy = ReplayPolicy(((observation, [0.1]),))
        environment = ReplayEnvironment(17, observation, ())
        recorder = ReplayRecorder()

        def decide(proposal):
            response = self.response | {
                'episode_id': proposal.observation.episode_id,
                'observation_sequence': proposal.observation.sequence,
                'proposal_id': proposal.proposal_id,
                'evidence': [{'observation_sequence': 0, 'source': 'main_camera'}]}
            return self.decoder.decode(response, proposal)

        with self.assertRaisesRegex(ValueError, 'current proposal'):
            run_episode(EpisodeConfig(17, 1), policy, environment, recorder,
                        supervisor_decider=decide)
        self.assertEqual(environment.actions, [])
        self.assertEqual(recorder.failures[0][3].stage, 'supervisor')
        self.assertTrue(recorder.finalized)


if __name__ == '__main__':
    unittest.main()
