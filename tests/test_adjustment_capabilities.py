"""Synthetic capability declarations; no physical or coupled execution claims."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from correction_validation import CorrectionValidator
from episode_harness import ActionProposal, EpisodeConfig, ObservationPacket, run_episode
from recorded_replay import TraceRecorder, load_recorded_replay
from recovery_registry import RecoveryRegistry
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder
from single_action_adjustment import SingleActionAdjustment
from supervisor_adjustment import AdjustmentComponent, AdjustmentRequestDecoder
from translation_conversion import LiberoTranslationConverter
from test_translation_conversion import evidence_fixture


class AdjustmentCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.translation = AdjustmentRequestDecoder('panda_arm', 'world', tuple(
            AdjustmentComponent('translation_' + axis, 'm', -0.03, 0.03)
            for axis in 'xyz'))
        self.validator = CorrectionValidator(RecoveryRegistry(()), adjustment=self.translation)
        self.proposal = ActionProposal('p', ObservationPacket(
            'e', 0, datetime.now(timezone.utc), {}), [0] * 7)

    def request(self, contract, proposal=None):
        proposal = proposal or self.proposal
        return dict(kind='adjustment', scope='single_action',
                    episode_id=proposal.observation.episode_id,
                    observation_sequence=proposal.observation.sequence,
                    proposal_id=proposal.proposal_id, decision_id='d',
                    target=contract.target, frame=contract.frame,
                    units={c.name: c.unit for c in contract.components},
                    residual={c.name: c.maximum for c in contract.components})

    def executor(self):
        report = evidence_fixture()
        evidence = json.dumps(report).encode()
        return SingleActionAdjustment(LiberoTranslationConverter(
            evidence, expected_sha256=hashlib.sha256(evidence).hexdigest(),
            runtime_controller=report['controller'] | {'position_limits': None},
            target='panda_arm', maximum_metres=0.03))

    def test_translation_rejects_extra_rotation_gripper_and_coupling_even_zero(self):
        for name, unit in (('rotation_x', 'rad'), ('rotation_y', 'rad'),
                           ('rotation_z', 'rad'), ('gripper', 'm'),
                           ('left_translation_x', 'm'), ('right_translation_x', 'm')):
            broader = AdjustmentRequestDecoder('panda_arm', 'world',
                self.translation.components + (AdjustmentComponent(name, unit, -0.01, 0.01),))
            for value in (0, 0.01):
                response = self.request(broader)
                response['residual'][name] = value
                for request in (response, broader.decode(response, self.proposal)):
                    with self.subTest(name=name, value=value, request=type(request)), \
                            self.assertRaisesRegex(ValueError, 'unsupported adjustment components: ' + name):
                        self.validator.validate(request, self.proposal)

    def test_translation_rejects_group_target_and_response_capability_overrides(self):
        response = self.request(self.translation)
        for target in ('both_arms', 'left_arm', 'right_arm'):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'target or coordinated group'):
                self.validator.validate(response | {'target': target}, self.proposal)
        for field in ('capabilities', 'components', 'minimum', 'maximum'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'response fields'):
                self.validator.validate(response | {field: ['rotation_z']}, self.proposal)

    def test_separately_declared_rotation_gripper_and_two_arm_contracts_validate(self):
        contracts = (
            AdjustmentRequestDecoder('wrist', 'tool', (
                AdjustmentComponent('rotation_z', 'rad', -0.02, 0.02),)),
            AdjustmentRequestDecoder('gripper', 'tool', (
                AdjustmentComponent('aperture', 'm', -0.001, 0.001),)),
            AdjustmentRequestDecoder('coordinated_pair', 'world', (
                AdjustmentComponent('left_translation_x', 'm', -0.01, 0.01),
                AdjustmentComponent('right_translation_x', 'm', -0.02, 0.02),
                AdjustmentComponent('relative_rotation_z', 'rad', -0.03, 0.03))),
        )
        before = deepcopy(self.proposal)
        for contract in contracts:
            validator = CorrectionValidator(RecoveryRegistry(()), adjustment=contract)
            for boundary in ('minimum', 'maximum'):
                response = self.request(contract)
                response['residual'] = {c.name: getattr(c, boundary) for c in contract.components}
                checked = validator.validate(response, self.proposal)
                self.assertEqual(checked.target, contract.target)
                self.assertEqual(dict(checked.residual), response['residual'])
                self.assertEqual(validator.validate(checked, self.proposal), checked)
                # Validation of a different host contract cannot widen this executor.
                self.assertEqual(self.executor().resolve(self.proposal, checked).kind, 'reject')
        self.assertEqual(self.proposal, before)

    def test_group_contract_enforces_each_component_units_bounds_and_complete_pair(self):
        contract = AdjustmentRequestDecoder('paired_arms', 'world', (
            AdjustmentComponent('left_translation_x', 'm', -0.01, 0.01),
            AdjustmentComponent('right_translation_x', 'm', -0.02, 0.02)))
        validator = CorrectionValidator(RecoveryRegistry(()), adjustment=contract)
        response = self.request(contract)
        for name in response['residual']:
            for value in (True, float('nan'), float('inf'), 0.021, -0.021):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validator.validate(response | {'residual': response['residual'] | {name: value}}, self.proposal)
            with self.assertRaisesRegex(ValueError, 'units'):
                validator.validate(response | {'units': response['units'] | {name: 'cm'}}, self.proposal)
            with self.assertRaisesRegex(ValueError, 'missing required'):
                validator.validate(response | {'residual': {name: 0}}, self.proposal)
        for change in ({'target': 'left_arm'}, {'frame': 'camera'}, {'proposal_id': 'old'}):
            with self.assertRaises(ValueError):
                validator.validate(response | change, self.proposal)

    def test_unsupported_components_in_either_map_and_malformed_names_fail_explicitly(self):
        response = self.request(self.translation)
        for field in ('units', 'residual'):
            with self.assertRaisesRegex(ValueError, 'unsupported adjustment components: rotation_z'):
                self.validator.validate(response | {field: response[field] | {'rotation_z': 0}}, self.proposal)
            for name in (None, 3, ''):
                with self.assertRaisesRegex(ValueError, 'invalid adjustment component maps'):
                    self.validator.validate(response | {field: response[field] | {name: 0}}, self.proposal)

    def test_complete_sealed_replays_preserve_rejections_without_dispatch(self):
        for component, unit in (('rotation_z', 'rad'), ('gripper', 'm'), ('right_translation_x', 'm')):
            with self.subTest(component=component), TemporaryDirectory() as tmp:
                initial, action = {'task': 'place'}, [0] * 7
                environment = ReplayEnvironment(17, initial, ())
                config = EpisodeConfig(17, 2)
                directory = Path(tmp) / 'episode'
                recorder = TraceRecorder(directory, config, ReplayRecorder())
                executor = self.executor()

                def select(proposal):
                    response = self.request(self.translation, proposal)
                    response['units'][component] = unit
                    response['residual'][component] = 0
                    return executor.resolve(proposal, response)

                result = run_episode(config, ReplayPolicy(((initial, action),)),
                                     environment, recorder, action_selector=select)
                recorder.seal(result)
                self.assertEqual(result.stop_reason, 'proposal_rejected')
                self.assertEqual(result.steps, 0)
                self.assertEqual(environment.actions, [])
                replay = load_recorded_replay(directory)
                self.assertEqual(replay.run().stop_reason, 'proposal_rejected')
                record = replay.evidence()['decisions'][0]['action_record']
                self.assertEqual(record['rejection_reason'], 'unsupported adjustment components: ' + component)
                self.assertEqual(record['disposition'], 'rejected')
                self.assertEqual(record['proposed_action'], action)
                self.assertIsNone(record['executed_action'])


if __name__ == '__main__':
    unittest.main()
