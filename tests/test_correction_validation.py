"""Public executor parameter boundary; synthetic limits, no robot operation."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from correction_validation import CorrectionValidator
from episode_harness import (ActionProposal, ActionResolution, EpisodeConfig,
                             ObservationPacket, run_episode)
from recorded_replay import TraceRecorder, load_recorded_replay
from recovery_registry import RecoveryRegistry
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep
from supervisor_adjustment import AdjustmentComponent, AdjustmentRequestDecoder
from supervisor_recovery import RecoveryParameter, RecoveryRequestDecoder
from test_recovery_registry import fixture_tool


class CorrectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.tool = fixture_tool()
        self.adjustment = AdjustmentRequestDecoder('left_arm', 'world', (
            AdjustmentComponent('translation_x', 'm', -0.03, 0.03),
            AdjustmentComponent('rotation_z', 'rad', -0.05, 0.05)))
        self.validator = CorrectionValidator(RecoveryRegistry((self.tool,)),
                                             self.tool.required_capabilities,
                                             self.adjustment)
        self.responses = {kind: json.loads((Path(__file__).parent / 'fixtures' /
                          f'{kind}_response.json').read_text())
                          for kind in ('recovery', 'adjustment')}
        self.proposal = ActionProposal('fixture-proposal', ObservationPacket(
            'fixture-episode', 2, datetime.now(timezone.utc), {'task': 'place item'}), [0.1])

    def test_inclusive_boundaries_for_both_wire_and_decoded_requests(self):
        before = deepcopy(self.proposal)
        for kind, field, values, decoder in (
                ('recovery', 'parameters', ({'retreat_m': 0}, {'retreat_m': 0.03}),
                 RecoveryRequestDecoder(self.tool.name, self.tool.parameter_bounds)),
                ('adjustment', 'residual', (
                    {'translation_x': -0.03, 'rotation_z': -0.05},
                    {'translation_x': 0.03, 'rotation_z': 0.05}), self.adjustment)):
            for value in values:
                response = self.responses[kind] | {field: value}
                expected = decoder.decode(response, self.proposal)
                for request in (response, expected):
                    checked = self.validator.validate(request, self.proposal)
                    self.assertEqual(checked.request if kind == 'recovery' else checked, expected)
        self.assertEqual(self.proposal, before)

    def test_nonfinite_malformed_and_out_of_range_parameters_fail(self):
        for kind, field, name in (('recovery', 'parameters', 'retreat_m'),
                                  ('adjustment', 'residual', 'translation_x')):
            for value in (True, None, '0.01', [], {}, float('nan'), float('inf'),
                          -float('inf'), 10 ** 400, -0.031, 0.031):
                response = deepcopy(self.responses[kind])
                response[field][name] = value
                with self.subTest(kind=kind, value=value), self.assertRaises(ValueError):
                    self.validator.validate(response, self.proposal)
            for parameters in ([], {}, {name: 0, 'extra': 0}):
                with self.subTest(kind=kind, parameters=parameters), self.assertRaises(ValueError):
                    self.validator.validate(self.responses[kind] | {field: parameters}, self.proposal)

    def test_executor_limits_override_permissive_upstream_decoders(self):
        permissive_recovery = RecoveryRequestDecoder(self.tool.name, (
            RecoveryParameter('retreat_m', 0, 100),))
        permissive_adjustment = replace(self.adjustment, components=(
            AdjustmentComponent('translation_x', 'm', -100, 100),
            self.adjustment.components[1]))
        for kind, field, name, decoder in (
                ('recovery', 'parameters', 'retreat_m', permissive_recovery),
                ('adjustment', 'residual', 'translation_x', permissive_adjustment)):
            response = deepcopy(self.responses[kind])
            response[field][name] = 10
            request = decoder.decode(response, self.proposal)
            with self.assertRaisesRegex(ValueError, 'bounds'):
                self.validator.validate(request, self.proposal)

    def test_target_frame_units_and_request_authority_cannot_change_contract(self):
        for change in ({'target': 'right_arm'}, {'target': 'coupled_arms'},
                       {'target': ['left_arm']}, {'frame': 'camera'},
                       {'units': {'translation_x': 'cm', 'rotation_z': 'rad'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validator.validate(self.responses['adjustment'] | change, self.proposal)
        for kind, response in self.responses.items():
            for change in ({'maximum': 100}, {'parameter_bounds': []},
                           {'components': []}, {'action_limit': 100},
                           {'capabilities': list(self.tool.required_capabilities)},
                           {'implementation': 'arbitrary_code'}, {'episode_id': 'foreign'},
                           {'proposal_id': 'stale'}, {'observation_sequence': 1}):
                with self.subTest(kind=kind, change=change), self.assertRaises(ValueError):
                    self.validator.validate(response | change, self.proposal)
        with self.assertRaises(ValueError):
            self.validator.validate(self.responses['recovery'] | {'target': 'right_arm'}, self.proposal)

    def test_manually_constructed_typed_requests_receive_full_validation(self):
        adjustment = self.adjustment.decode(self.responses['adjustment'], self.proposal)
        recovery = self.validator.validate(self.responses['recovery'], self.proposal).request
        for request in (
                replace(adjustment, residual=adjustment.residual * 2),
                replace(adjustment, residual=(('translation_x', 0),)),
                replace(adjustment, residual=(('translation_x', float('nan')), ('rotation_z', 0))),
                replace(adjustment, residual=None),
                replace(adjustment, units=adjustment.units * 2),
                replace(adjustment, target='right_arm'),
                replace(recovery, parameters=recovery.parameters * 2),
                replace(recovery, parameters=(([], 0),)),
                replace(recovery, parameters=(('retreat_m', True),)),
                replace(recovery, evidence=('not_evidence',)),
                replace(recovery, evidence=None),
                replace(recovery, proposal_id='previous')):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.validator.validate(request, self.proposal)

    def test_configuration_and_returned_data_are_detached_and_immutable(self):
        capabilities = list(self.tool.required_capabilities)
        validator = replace(self.validator, capabilities=capabilities)
        capabilities.clear()
        checked = validator.validate(self.responses['recovery'], self.proposal)
        adjustment = validator.validate(self.responses['adjustment'], self.proposal)
        self.responses['recovery']['parameters']['retreat_m'] = 100
        self.responses['adjustment']['residual']['translation_x'] = 100
        self.assertEqual(checked.request.parameters, (('retreat_m', 0.02),))
        self.assertEqual(adjustment.residual[0], ('translation_x', 0.01))
        with self.assertRaises(FrozenInstanceError):
            validator.adjustment = None
        with self.assertRaises(FrozenInstanceError):
            checked.tool.parameter_bounds = ()
        with self.assertRaises(FrozenInstanceError):
            adjustment.target = 'right_arm'

    def test_disabled_modes_incompatible_capabilities_and_invalid_configuration(self):
        disabled = CorrectionValidator(RecoveryRegistry(()))
        for response in self.responses.values():
            with self.assertRaises(ValueError):
                disabled.validate(response, self.proposal)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            replace(self.validator, capabilities=()).validate(self.responses['recovery'], self.proposal)
        for changes in ({'recovery': {}}, {'adjustment': {}}, {'capabilities': 'left_arm'},
                        {'capabilities': [True]}, {'capabilities': ['a', 'a']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.validator, **changes)
        for request in (None, [], {}, {'kind': []}, {'kind': 'pass'}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.validator.validate(request, self.proposal)

    def test_complete_replays_reject_invalid_corrections_before_execution(self):
        for kind in self.responses:
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                initial, current = {'task': 'place item'}, {'task': 'place item', 'time': 1}
                config = EpisodeConfig(17, 3)
                policy = ReplayPolicy(((initial, [0.1]), (current, [0.2])))
                environment = ReplayEnvironment(17, initial, (
                    ([0.1], ReplayStep(current, 0, False, False, False)),))
                directory = Path(tmp) / 'episode'
                trace = TraceRecorder(directory, config, ReplayRecorder())

                def select(proposal):
                    if proposal.observation.sequence == 0:
                        return ActionResolution('pass')
                    response = deepcopy(self.responses[kind])
                    response.update(episode_id=proposal.observation.episode_id,
                                    observation_sequence=proposal.observation.sequence,
                                    proposal_id=proposal.proposal_id)
                    if kind == 'recovery':
                        response['parameters']['retreat_m'] = 100
                        response['evidence'] = [dict(observation_sequence=1, source='robot_state')]
                    else:
                        response['target'] = 'right_arm'
                    try:
                        self.validator.validate(response, proposal)
                    except ValueError as exc:
                        return ActionResolution('reject', reason=str(exc))
                    self.fail('invalid correction passed executor validation')

                result = run_episode(config, policy, environment, trace, action_selector=select)
                trace.seal(result)
                self.assertEqual(environment.actions, [[0.1]])
                self.assertEqual(result.stop_reason, 'proposal_rejected')
                self.assertEqual(result.steps, 1)
                replay = load_recorded_replay(directory)
                self.assertEqual(replay.run().stop_reason, 'proposal_rejected')
                self.assertEqual(replay.evidence()['decisions'][-1]['action_record']['disposition'],
                                 'rejected')


if __name__ == '__main__':
    unittest.main()
