"""Synthetic public conversion checks; no simulator or correction execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import hashlib
import json
import unittest

from episode_harness import ActionProposal, ObservationPacket
from translation_conversion import LiberoTranslationConverter


def evidence_fixture():
    controller = dict(name='OSC_POSE', robot='OnTheGroundPanda',
                      reference_frame='world', translation_unit='m',
                      input_min=[-1.0] * 6, input_max=[1.0] * 6,
                      output_min=[-0.05] * 3 + [-0.5] * 3,
                      output_max=[0.05] * 3 + [0.5] * 3,
                      metres_per_native_translation_unit=[0.05] * 3,
                      native_translation_indices=[0, 1, 2],
                      native_action_dimension=7, use_delta=True, impedance_mode='fixed')
    probes = []
    for axis in range(3):
        for sign in (-1, 1):
            native, delta = [0.0] * 6, [0.0] * 3
            native[axis], delta[axis] = sign * 0.2, sign * 0.01
            probes.append(dict(native_pose_command=native, goal_delta_world_m=delta,
                               recovered_native_translation=native[:3]))
    return dict(schema_version=1, controller=controller, verification=dict(
        passed=True, components=['translation_x', 'translation_y', 'translation_z'], probes=probes))


class TranslationConversionTests(unittest.TestCase):
    def setUp(self):
        self.report = evidence_fixture()
        self.runtime = deepcopy(self.report['controller']) | {'position_limits': None}
        self.proposal = ActionProposal('p', ObservationPacket(
            'e', 0, datetime.now(timezone.utc), {}), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1])
        self.response = dict(kind='adjustment', scope='single_action', episode_id='e',
                             observation_sequence=0, proposal_id='p', decision_id='d',
                             target='panda_arm', frame='world',
                             units={name: 'm' for name in ('translation_x', 'translation_y', 'translation_z')},
                             residual=dict(translation_x=0.01, translation_y=-0.02, translation_z=0.005))

    def converter(self, **overrides):
        evidence = json.dumps(self.report).encode()
        config = dict(expected_sha256=hashlib.sha256(evidence).hexdigest(),
                      runtime_controller=self.runtime, target='panda_arm', maximum_metres=0.03)
        return LiberoTranslationConverter(evidence, **(config | overrides))

    def test_metres_divide_by_scale_and_only_fill_translation_slots(self):
        before = deepcopy(self.proposal)
        result = self.converter().convert(self.response, self.proposal)
        for actual, expected in zip(result.native_residual, (0.2, -0.4, 0.1, 0, 0, 0, 0)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result.request.residual[0], ('translation_x', 0.01))
        self.assertEqual(self.proposal, before)

    def test_both_directions_each_axis_zero_and_inclusive_bounds(self):
        converter = self.converter()
        for axis, name in enumerate(self.response['residual']):
            for metres in (-0.03, -0.01, 0, 0.01, 0.03):
                residual = dict.fromkeys(self.response['residual'], 0)
                residual[name] = metres
                result = converter.convert(self.response | {'residual': residual}, self.proposal)
                self.assertAlmostEqual(result.native_residual[axis], metres * 20)
                self.assertEqual(sum(v != 0 for v in result.native_residual), int(metres != 0))

    def test_unknown_frames_units_targets_components_and_stale_requests_fail(self):
        converter = self.converter()
        for change in ({'frame': 'camera'}, {'frame': 'tool'}, {'target': 'both_arms'},
                       {'units': dict.fromkeys(self.response['units'], 'normalized')},
                       {'units': dict.fromkeys(self.response['units'], 'cm')},
                       {'residual': {'rotation_x': 0.01}}, {'proposal_id': 'old'},
                       {'episode_id': 'other'}, {'observation_sequence': 1},
                       {'scale': 1}, {'maximum_metres': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                converter.convert(self.response | change, self.proposal)

    def test_numeric_and_proposal_validation(self):
        converter = self.converter()
        for value in (0.031, -0.031, 0.2, True, '0.01', None, float('nan'), float('inf'), 10 ** 400):
            response = deepcopy(self.response)
            response['residual']['translation_x'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                converter.convert(response, self.proposal)
        for action in ([0] * 6, [0] * 8, [True] * 7, [float('nan')] * 7, None):
            with self.assertRaises(ValueError):
                converter.convert(self.response, replace(self.proposal, action=action))
        for maximum in (0, -0.01, 0.051, True, float('inf')):
            with self.assertRaises(ValueError):
                self.converter(maximum_metres=maximum)

    def test_evidence_must_match_hash_and_verify_all_signed_probes(self):
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.converter(expected_sha256='0' * 64)
        for mutation in ('failed', 'missing', 'duplicate', 'wrong_direction', 'bad_number', 'schema'):
            self.report = evidence_fixture()
            verification = self.report['verification']
            if mutation == 'failed':
                verification['passed'] = False
            elif mutation == 'missing':
                verification['probes'].pop()
            elif mutation == 'duplicate':
                verification['probes'][0] = verification['probes'][1]
            elif mutation == 'wrong_direction':
                verification['probes'][0]['goal_delta_world_m'][0] = 0.01
            elif mutation == 'bad_number':
                verification['probes'][0]['goal_delta_world_m'][0] = True
            else:
                self.report['schema_version'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.converter()

    def test_report_and_runtime_must_have_the_verified_mapping(self):
        changes = dict(reference_frame='camera', translation_unit='cm',
                       robot='OtherRobot', use_delta=False, impedance_mode='variable',
                       input_max=[2.0] * 6, output_max=[0.1] * 6,
                       metres_per_native_translation_unit=[0.1] * 3,
                       native_translation_indices=[2, 1, 0], native_action_dimension=14)
        for location in ('report', 'runtime'):
            for key, value in changes.items():
                self.report = evidence_fixture()
                self.runtime = deepcopy(self.report['controller']) | {'position_limits': None}
                target = self.report['controller'] if location == 'report' else self.runtime
                target[key] = value
                with self.subTest(location=location, key=key), self.assertRaises(ValueError):
                    self.converter()
        self.report = evidence_fixture()
        for runtime in (self.report['controller'], self.report['controller'] | {'position_limits': [0, 1]}):
            with self.assertRaisesRegex(ValueError, 'geometry'):
                self.converter(runtime_controller=runtime)

    def test_typed_requests_revalidate_and_configuration_is_detached(self):
        converter = self.converter()
        result = converter.convert(self.response, self.proposal)
        self.runtime['output_max'][0] = 100
        self.response['residual']['translation_x'] = 100
        self.assertEqual(converter.convert(result.request, self.proposal), result)
        for request in (replace(result.request, residual=result.request.residual * 2),
                        replace(result.request, frame='camera'),
                        replace(result.request, residual=(('translation_x', 1), ('translation_y', 0), ('translation_z', 0)))):
            with self.assertRaises(ValueError):
                converter.convert(request, self.proposal)
        with self.assertRaises(FrozenInstanceError):
            result.native_residual = ()
        with self.assertRaises(FrozenInstanceError):
            converter.evidence_sha256 = 'changed'


if __name__ == '__main__':
    unittest.main()
