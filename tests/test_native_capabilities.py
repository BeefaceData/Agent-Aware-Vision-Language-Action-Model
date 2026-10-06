"""Public native-interface contracts; synthetic execution is not robot evidence."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from action_capabilities import (ActionCapabilities, ActionComponent,
                                 libero_native_capabilities, validate_action_pair)
from episode_harness import EpisodeConfig, run_episode
from libero_adapter import read_native_capabilities
from policy_adapter import ResetOnResumePolicyAdapter
from recorded_replay import TraceRecorder, load_recorded_replay
from replay_adapters import ReplayEnvironment, ReplayPolicy, ReplayRecorder, ReplayStep


def two_arm():
    return ActionCapabilities(tuple(
        ActionComponent(name, arm, 'paired', 'joint_position', 'rad', 'joint', -2., 2.)
        for arm in ('left', 'right') for name in ('shoulder', 'elbow')),
        50., ('policy_action', 'coordinated_joint_position'))


def fixture(capabilities):
    action = tuple(0.1 for _ in capabilities.components)
    policy = ReplayPolicy((('start', action), ('middle', action)))
    environment = ReplayEnvironment(17, 'start', (
        (action, ReplayStep('middle', 0., False, False, False)),
        (action, ReplayStep('done', 1., True, True, False))))
    policy.action_capabilities = capabilities
    environment.action_capabilities = capabilities
    return policy, environment, ReplayRecorder()


def vector_environment():
    robot = SimpleNamespace(robot_model=type('OnTheGroundPanda', (), {})())
    robot.controller = SimpleNamespace(name='OSC_POSE', impedance_mode='fixed',
        use_delta=True, position_limits=None, orientation_limits=None,
        input_min=np.full(6, -1.), input_max=np.full(6, 1.),
        output_min=np.array([-0.05] * 3 + [-0.5] * 3),
        output_max=np.array([0.05] * 3 + [0.5] * 3))
    wrapper = SimpleNamespace(control_mode='relative',
        _env=SimpleNamespace(robots=[robot], env=SimpleNamespace(control_freq=20)))
    return SimpleNamespace(num_envs=1, envs=[SimpleNamespace(unwrapped=wrapper)],
        single_action_space=SimpleNamespace(shape=(7,), low=np.full(7, -1.), high=np.full(7, 1.)))


class NativeCapabilityTests(unittest.TestCase):
    def test_single_and_two_arm_complete_episodes_seal_and_replay(self):
        single = replace(libero_native_capabilities(), layout='flat')
        for capabilities in (single, two_arm()):
            with self.subTest(dimension=len(capabilities.components)), TemporaryDirectory() as tmp:
                policy, environment, recorder = fixture(capabilities)
                config = EpisodeConfig(17, 3)
                trace = TraceRecorder(Path(tmp) / 'trace', config, recorder)
                outcome = run_episode(config, policy, environment, trace)
                trace.seal(outcome)
                replayed = load_recorded_replay(Path(tmp) / 'trace').run()
                self.assertTrue(outcome.success)
                self.assertEqual((replayed.success, replayed.steps), (True, 2))
                self.assertEqual(len(environment.actions[0]), len(capabilities.components))
                self.assertEqual(environment.actions[0], environment.actions[1])

    def test_mismatch_rejected_before_reset_inference_or_recording(self):
        base = two_arm()
        variants = [replace(base, control_frequency_hz=25.),
                    replace(base, layout='single-vector-batch'),
                    replace(base, components=base.components[:-1]),
                    replace(base, operations=('policy_action',))]
        for field, value in (('arm', 'other'), ('group', 'independent'),
                             ('representation', 'joint_delta'), ('unit', 'degrees'),
                             ('frame', 'world'), ('scale', 0.5), ('name', 'wrist'),
                             ('minimum', -1.), ('maximum', 1.)):
            variants.append(replace(base, components=(replace(base.components[0],
                **{field: value}),) + base.components[1:]))
        for supported in variants + [None, {}]:
            with self.subTest(supported=supported):
                policy, environment, recorder = fixture(base)
                environment.action_capabilities = supported
                def forbidden(*args):
                    self.fail('incompatible pair started execution')
                policy.reset = environment.reset = policy.act = forbidden
                with self.assertRaises(ValueError):
                    run_episode(EpisodeConfig(17, 3), policy, environment, recorder)
                self.assertEqual(environment.actions, [])
                self.assertFalse(recorder.finalized)

    def test_environment_can_support_wider_ranges_and_extra_operations(self):
        policy, environment, _ = fixture(two_arm())
        environment.action_capabilities = replace(two_arm(),
            components=tuple(replace(c, minimum=-3., maximum=3.) for c in two_arm().components),
            operations=two_arm().operations + ('stop',))
        validate_action_pair(policy, environment)

    def test_one_sided_declaration_and_malformed_metadata_fail(self):
        with self.assertRaises(ValueError):
            validate_action_pair(SimpleNamespace(), SimpleNamespace(action_capabilities=two_arm()))
        for fields in ({'control_frequency_hz': float('nan')},
                       {'control_frequency_hz': True}, {'components': ()},
                       {'components': (two_arm().components[0],) * 2},
                       {'operations': ()}, {'operations': ('step', 'step')}, {'layout': ''}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(two_arm(), **fields)
        for fields in ({'scale': 0}, {'maximum': float('inf')},
                       {'minimum': 2}, {'frame': ''}, {'unit': None}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(two_arm().components[0], **fields)

    def test_policy_resume_preserves_native_contract(self):
        policy = ResetOnResumePolicyAdapter(lambda: None, lambda packet: None,
                                            action_capabilities=two_arm())
        policy.reset()
        policy.resume(SimpleNamespace(episode_id='e', sequence=1))
        validate_action_pair(policy, SimpleNamespace(action_capabilities=two_arm()))

    def test_libero_readback_checks_actual_controller_and_transport(self):
        env = vector_environment()
        declaration = read_native_capabilities(env)
        self.assertEqual(declaration, libero_native_capabilities())
        self.assertEqual(declaration.control_frequency_hz, 20.)
        self.assertEqual(declaration.components[0].scale, 0.05)
        self.assertEqual(declaration.operations, ('policy_action',))
        changes = (
            lambda e: setattr(e.envs[0].unwrapped, 'control_mode', 'absolute'),
            lambda e: setattr(e.envs[0].unwrapped._env.env, 'control_freq', 30),
            lambda e: setattr(e.single_action_space, 'shape', (14,)),
            lambda e: setattr(e.single_action_space, 'high', np.full(7, 2.)),
            lambda e: setattr(e.envs[0].unwrapped._env.robots[0].controller,
                              'output_max', np.full(6, 1.)),
            lambda e: setattr(e.envs[0].unwrapped._env, 'robots', []),
        )
        for change in changes:
            with self.subTest(change=change):
                env = vector_environment()
                change(env)
                with self.assertRaises(ValueError):
                    read_native_capabilities(env)


if __name__ == '__main__':
    unittest.main()
