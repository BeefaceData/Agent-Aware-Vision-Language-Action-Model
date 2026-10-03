"""Malformed provider data is rejected and remains diagnosable in traces."""

from datetime import datetime, timezone
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from episode_harness import (ActionProposal, EpisodeConfig, ObservationPacket,
                             SupervisorResponseError, run_episode)
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_adjustment import AdjustmentComponent, AdjustmentRequestDecoder
from supervisor_recovery import RecoveryParameter, RecoveryRequestDecoder
from supervisor_response import SupervisorResponseDecoder


class SupervisorResponseTests(unittest.TestCase):
    def setUp(self):
        self.proposal = ActionProposal('fixture-proposal', ObservationPacket(
            'fixture-episode', 2, datetime.now(timezone.utc), {'task': 'place item'}), [0.1])
        identity = {'episode_id': 'fixture-episode', 'observation_sequence': 2,
                    'proposal_id': 'fixture-proposal'}
        self.responses = [identity | {'kind': 'pass'}, identity | {
            'kind': 'abstain', 'diagnosis': 'unknown', 'reason': 'wrist occluded',
            'evidence_availability': {'main': 'available', 'wrist': 'missing'}}]
        for name in ('recovery', 'adjustment'):
            self.responses.append(json.loads((Path(__file__).parent / 'fixtures' /
                                              f'{name}_response.json').read_text()))
        self.decoder = SupervisorResponseDecoder(
            RecoveryRequestDecoder('reopen_and_retreat', (RecoveryParameter('retreat_m', 0, .03),)),
            AdjustmentRequestDecoder('left_arm', 'world', (
                AdjustmentComponent('translation_x', 'm', -.03, .03),
                AdjustmentComponent('rotation_z', 'rad', -.05, .05))))

    def test_all_supported_variants_and_detached_evidence(self):
        for response in self.responses:
            self.assertEqual(self.decoder.decode(response, self.proposal).kind, response['kind'])
        result = self.decoder.decode(self.responses[1], self.proposal)
        self.responses[1]['evidence_availability']['wrist'] = 'available'
        self.assertEqual(result.evidence_availability['wrist'], 'missing')
        for response in self.responses[2:]:
            with self.assertRaisesRegex(SupervisorResponseError, 'not configured'):
                SupervisorResponseDecoder().decode(response, self.proposal)

    def invalid_cases(self):
        for value in (None, [], 'exec(payload)', 3):
            yield value, 'must be an object'
        for kind in ('execute', '', None, [], {}, True, float('nan')):
            yield self.responses[0] | {'kind': kind}, 'unknown supervisor decision type'
        for response in self.responses:
            for field in response:
                incomplete = response.copy()
                del incomplete[field]
                yield incomplete, 'decision type' if field == 'kind' else 'fields'
            for extra in ('code', 'tool_code', 'instruction', 'action', 'maximum', 'extra'):
                yield response | {extra: 'PRIVATE_PAYLOAD'}, 'fields'
            for changes in ({'episode_id': 'foreign'}, {'proposal_id': 'obsolete'},
                            {'observation_sequence': 1}, {'observation_sequence': True},
                            {'observation_sequence': float('inf')}):
                yield response | changes, 'current proposal'
        for value in (float('nan'), float('inf'), -float('inf'), 10 ** 400, True, '0.01'):
            yield self.responses[2] | {'parameters': {'retreat_m': value}}, 'bounds'
            yield self.responses[3] | {'residual': {
                'translation_x': value, 'rotation_z': 0}}, 'bounds'
        yield self.responses[2] | {'parameters': {'retreat_m': .01, 'code': 'exec'}}, 'fields'
        yield self.responses[2] | {'tool_name': 'exec'}, 'unknown recovery tool'
        yield self.responses[2] | {'evidence': [{
            'observation_sequence': 2, 'source': 'main', 'code': 'exec'}]}, 'evidence reference'
        yield self.responses[3] | {'units': {'translation_x': 'm', 'code': 'exec'}}, 'units'
        yield self.responses[1] | {'diagnosis': 'certain'}, 'diagnosis'
        yield self.responses[1] | {'evidence_availability': {'main': {'code': 'exec'}}}, 'availability'
        yield self.responses[1] | {'reason': float('nan')}, 'diagnosis'

    def test_invalid_categories_have_actionable_reasons(self):
        for response, reason in self.invalid_cases():
            with self.subTest(response=response):
                with self.assertRaisesRegex(SupervisorResponseError, reason) as error:
                    self.decoder.decode(response, self.proposal)
                self.assertNotIn('PRIVATE_PAYLOAD', str(error.exception))

    def fixture(self):
        observations = [{'task': 'place item', 'robot_state': {'position': [i]}}
                        for i in range(3)]
        actions = [[.1], [.2]]
        return (EpisodeConfig(17, 3), ReplayPolicy(tuple(zip(observations, actions))),
                ReplayEnvironment(17, observations[0], (
                    (actions[0], ReplayStep(observations[1], 0, False, False, False)),
                    (actions[1], ReplayStep(observations[2], 1, True, True, False)))),
                ReplayRecorder())

    def bind(self, response, proposal):
        return response | {'episode_id': proposal.observation.episode_id,
                           'observation_sequence': proposal.observation.sequence,
                           'proposal_id': proposal.proposal_id}

    def test_complete_episode_pass_and_abstention_survive_sealed_replay(self):
        config, policy, environment, recorder = self.fixture()
        responses = iter(self.responses[:2])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trace'
            trace = TraceRecorder(directory, config, recorder)
            outcome = run_episode(config, policy, environment, trace,
                                  supervisor_decider=lambda p: self.decoder.decode(
                                      self.bind(next(responses), p), p))
            trace.seal(outcome)
            replay = load_recorded_replay(directory)
            self.assertTrue(replay.run().success)
            rows = replay.evidence()['decisions']
            self.assertEqual(rows[0]['action_record']['supervisor_pass']['kind'], 'pass')
            self.assertEqual(rows[1]['action_record']['supervisor_abstention']['kind'], 'abstain')
        self.assertEqual(environment.actions, [[.1], [.2]])

    def test_rejection_after_pass_retains_reason_without_executing_correction(self):
        invalid = [({'kind': 'execute'}, 'unknown supervisor decision type'),
                   (self.responses[0] | {'tool_code': 'PRIVATE_PAYLOAD'}, 'fields'),
                   (self.responses[2] | {'parameters': {'retreat_m': float('nan')}}, 'bounds'),
                   (self.responses[3] | {'extra': 1}, 'fields')]
        for response, reason in invalid:
            with self.subTest(reason=reason), TemporaryDirectory() as temporary:
                config, policy, environment, recorder = self.fixture()
                directory = Path(temporary) / 'trace'
                trace = TraceRecorder(directory, config, recorder)
                responses = iter((self.responses[0], response))
                with self.assertRaisesRegex(SupervisorResponseError, reason):
                    run_episode(config, policy, environment, trace,
                                supervisor_decider=lambda p: self.decoder.decode(
                                    self.bind(next(responses), p), p))
                self.assertEqual(environment.actions, [[.1]])
                self.assertTrue(recorder.finalized)
                failure = recorder.failures[0][3]
                self.assertEqual(failure.stage, 'supervisor')
                record = failure.action_record
                self.assertEqual(record.disposition, 'rejected')
                self.assertIn(reason, record.rejection_reason)
                self.assertIsNone(record.selected_action)
                self.assertIsNone(record.executed_action)
                self.assertIsNone(record.execution_acknowledgement)
                rows = [json.loads(line) for line in
                        (directory / 'decisions.jsonl').read_text().splitlines()]
                self.assertEqual(rows[-1]['action_record']['rejection_reason'], record.rejection_reason)
                self.assertNotIn('PRIVATE_PAYLOAD', (directory / 'decisions.jsonl').read_text())
                self.assertFalse((directory / 'manifest.json').exists())

    def test_decoded_corrections_remain_without_execution_authority(self):
        for response in self.responses[2:]:
            with self.subTest(kind=response['kind']):
                config, policy, environment, recorder = self.fixture()
                # Current observation is zero, so use a structurally valid reference.
                if response['kind'] == 'recovery':
                    response = response | {'evidence': [
                        {'observation_sequence': 0, 'source': 'main_camera'}]}
                with self.assertRaisesRegex(ValueError, 'current proposal'):
                    run_episode(config, policy, environment, recorder,
                                supervisor_decider=lambda p: self.decoder.decode(
                                    self.bind(response, p), p))
                self.assertEqual(environment.actions, [])
                self.assertTrue(recorder.finalized)

    def test_previous_episode_responses_are_rejected_and_logged(self):
        for template in self.responses:
            if template['kind'] == 'recovery':
                template = template | {'evidence': [
                    {'observation_sequence': 0, 'source': 'main_camera'}]}
            for typed in (False, True) if template['kind'] in ('pass', 'abstain') else (False,):
                with self.subTest(kind=template['kind'], typed=typed):
                    config, policy, environment, recorder = self.fixture()
                    previous = []

                    def capture(proposal):
                        bound = self.bind(template, proposal)
                        decoded = self.decoder.decode(bound, proposal)
                        previous.append(decoded if typed else bound)
                        return self.decoder.decode(self.bind(self.responses[0], proposal), proposal)

                    # Finish a real attempt, then deliver its response after reset.
                    self.assertTrue(run_episode(config, policy, environment, recorder,
                                                supervisor_decider=capture).success)
                    stale = previous[0]
                    self.assert_identity_rejection(
                        lambda proposal: stale if typed else self.decoder.decode(stale, proposal),
                        executed=[])

    def test_superseded_response_identities_are_rejected_independently(self):
        for template in self.responses:
            if template['kind'] == 'recovery':
                template = template | {'evidence': [
                    {'observation_sequence': 0, 'source': 'main_camera'}]}
            for field in ('episode_id', 'observation_sequence', 'proposal_id', 'all'):
                for typed in (False, True) if template['kind'] in ('pass', 'abstain') else (False,):
                    with self.subTest(kind=template['kind'], field=field, typed=typed):
                        previous = []

                        def decide(proposal):
                            current = self.bind(template, proposal)
                            decoded = self.decoder.decode(current, proposal)
                            if not previous:
                                previous.append(current)
                                return self.decoder.decode(
                                    self.bind(self.responses[0], proposal), proposal)
                            old = previous[0]
                            changes = (old if field == 'all' else
                                       {field: 'previous-episode'} if field == 'episode_id' else
                                       {field: old[field]})
                            if typed:
                                return replace(decoded,
                                               **{key: value for key, value in changes.items()
                                                  if key in ('episode_id', 'observation_sequence',
                                                             'proposal_id')})
                            return self.decoder.decode(current | changes, proposal)

                        self.assert_identity_rejection(decide, executed=[[.1]])

    def assert_identity_rejection(self, decide, executed):
        config, policy, environment, recorder = self.fixture()
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trace'
            trace = TraceRecorder(directory, config, recorder)
            with self.assertRaisesRegex(SupervisorResponseError, 'current proposal'):
                run_episode(config, policy, environment, trace, supervisor_decider=decide)
            self.assertEqual(environment.actions, executed)
            self.assertTrue(recorder.finalized)
            failure = recorder.failures[0][3]
            self.assertEqual(failure.stage, 'supervisor')
            record = failure.action_record
            self.assertEqual(record.disposition, 'rejected')
            self.assertIn('current proposal', record.rejection_reason)
            self.assertIsNone(record.selected_action)
            self.assertIsNone(record.executed_action)
            self.assertIsNone(record.execution_acknowledgement)
            rows = [json.loads(line) for line in
                    (directory / 'decisions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), len(executed) + 1)
            self.assertEqual(rows[-1]['action_record']['rejection_reason'], record.rejection_reason)
            self.assertEqual(rows[-1]['action_record']['disposition'], 'rejected')
            self.assertFalse((directory / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
