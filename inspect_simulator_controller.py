"""Capture controller evidence without loading a policy or stepping an episode."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import platform
import sys


def capture():
    os.environ.setdefault('MUJOCO_GL', 'egl')
    versions = {name: version(name) for name in
                ('lerobot', 'hf-libero', 'robosuite', 'mujoco', 'numpy')}
    expected = {'lerobot': '0.4.3', 'hf-libero': '0.1.4', 'robosuite': '1.4.0'}
    if any(versions[name] != value for name, value in expected.items()):
        raise RuntimeError(f'Unverified package versions: {versions}')

    # Keep LIBERO's initial interactive dataset-path setup out of this probe.
    # This is an environment-local configuration, not the user's global config.
    config_dir = Path(os.environ.setdefault(
        'LIBERO_CONFIG_PATH', str(Path(sys.prefix) / 'libero-config')))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / 'config.yaml'
    if not config_path.exists():
        root = distribution('hf-libero').locate_file('libero/libero')
        paths = {'benchmark_root': str(root),
                 'bddl_files': str(root / 'bddl_files'),
                 'init_states': str(root / 'init_files'),
                 'datasets': str(root.parent / 'datasets'),
                 'assets': str(root / 'assets')}
        with config_path.open('x', encoding='utf-8') as output:
            json.dump(paths, output)  # JSON is valid YAML.

    import numpy as np
    from libero.libero.envs.env_wrapper import ControlEnv

    # LeRobot's OffScreenRenderEnv inherits this controller setup unchanged.
    # Disable cameras/rendering: only the actual robot/controller state is used.
    task_file = ('LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_'
                 'tomato_sauce_in_the_basket.bddl')
    task_path = distribution('hf-libero').locate_file(
        'libero/libero/bddl_files/libero_10/' + task_file)
    env = ControlEnv(bddl_file_name=str(task_path), use_camera_obs=False,
                     has_renderer=False, has_offscreen_renderer=False)
    try:
        env.seed(0)
        env.reset()
        robot = env.robots[0]
        controller = robot.controller
        if (controller.name != 'OSC_POSE' or not controller.use_delta or
                controller.impedance_mode != 'fixed'):
            raise RuntimeError('Expected fixed-impedance, relative OSC_POSE')
        if controller.position_limits is not None:
            raise RuntimeError('Position clipping needs separate calibration')

        # Check world-space goals against MuJoCo's actual end-effector site.
        site_world = np.array(controller.sim.data.site_xpos[
            controller.sim.model.site_name2id(controller.eef_name)], copy=True)
        np.testing.assert_allclose(controller.ee_pos, site_world, atol=1e-12)
        scale = ((controller.output_max - controller.output_min) /
                 (controller.input_max - controller.input_min))
        if not np.all(np.isfinite(scale)) or not np.all(scale[:3] > 0):
            raise RuntimeError('Invalid controller scaling')
        zero = controller.scale_action(np.zeros(6))
        np.testing.assert_allclose(zero, np.zeros(6), atol=1e-12)

        probes = []
        for axis in range(3):
            for native_value in (-0.2, 0.2):
                action = np.zeros(6)
                action[axis] = native_value
                controller.set_goal(action)
                displacement = np.array(controller.goal_pos) - site_world
                expected_delta = action[:3] * scale[:3]
                np.testing.assert_allclose(displacement, expected_delta,
                                           rtol=0, atol=1e-10)
                recovered = displacement / scale[:3]
                np.testing.assert_allclose(recovered, action[:3], atol=1e-9)
                probes.append({'native_pose_command': action.tolist(),
                               'goal_delta_world_m': displacement.tolist(),
                               'recovered_native_translation': recovered.tolist()})

        sources = {}
        for package, relative in (
            ('lerobot', 'lerobot/envs/libero.py'),
            ('hf-libero', 'libero/libero/envs/env_wrapper.py'),
            ('robosuite', 'robosuite/controllers/config/osc_pose.json'),
            ('robosuite', 'robosuite/controllers/osc.py'),
            ('robosuite', 'robosuite/controllers/base_controller.py'),
            ('robosuite', 'robosuite/utils/control_utils.py'),
        ):
            path = distribution(package).locate_file(relative)
            sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources['libero/libero/bddl_files/libero_10/' + task_file] = (
            hashlib.sha256(task_path.read_bytes()).hexdigest())

        return {
            'schema_version': 1,
            'captured_at': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'executable': sys.executable,
            'platform': platform.platform(), 'versions': versions,
            'scope': 'New local simulator controller verification; not historical baseline reconstruction',
            'environment': {'suite': 'libero_10', 'task_id': 0, 'seed': 0,
                            'bddl_file': task_file,
                            'control_mode': 'relative', 'episode_steps': 0,
                            'policy_loaded': False, 'rendering': False},
            'controller': {
                'name': controller.name, 'robot': type(robot.robot_model).__name__,
                'class': type(controller).__module__ + '.' + type(controller).__name__,
                'reference_frame': 'world', 'translation_unit': 'm',
                'input_min': controller.input_min.tolist(),
                'input_max': controller.input_max.tolist(),
                'output_min': controller.output_min.tolist(),
                'output_max': controller.output_max.tolist(),
                'metres_per_native_translation_unit': scale[:3].tolist(),
                'native_translation_indices': [0, 1, 2],
                'native_action_dimension': int(env.env.action_dim),
                'use_delta': bool(controller.use_delta),
                'impedance_mode': controller.impedance_mode,
                'eef_position_world_m': site_world.tolist(),
                'eef_orientation_world': controller.ee_ori_mat.tolist(),
            },
            'verification': {'passed': True, 'probes': probes,
                             'components': ['translation_x', 'translation_y', 'translation_z'],
                             'meaning': 'Command-to-goal mapping, not achieved physical displacement'},
            'installed_source_sha256': sources,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new path to preserve evidence')
    evidence = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(evidence, output, indent=2, allow_nan=False)
        output.write('\n')
    print(f'Controller checks passed; evidence: {args.output}')


if __name__ == '__main__':
    main()
