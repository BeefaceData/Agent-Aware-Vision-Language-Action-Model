#!/usr/bin/env python3
"""Run ONE SmolVLA/LIBERO episode with an explicit observation/action loop.

Target: Python 3.10, lerobot==0.4.3, hf-libero==0.1.4.
Verified against the published lerobot-0.4.3 wheel source:
  lerobot/envs/libero.py: create_libero_envs, task_ids in gym_kwargs
  lerobot/scripts/lerobot_eval.py: rollout processing order
  lerobot/envs/factory.py: make_env_pre_post_processors
  lerobot/processor/env_processor.py: LiberoProcessorStep
  lerobot/envs/utils.py: preprocess_observation, add_envs_task
Sources (release tag): https://github.com/huggingface/lerobot/tree/v0.4.3/src/lerobot
Checkpoint: https://huggingface.co/HuggingFaceVLA/smolvla_libero

This script is a custom adaptation, not an official LeRobot example.

Example:
  python run_smolvla_episode.py --suite libero_10 --task-id 0 --seed 0
Video FPS changes playback only, not simulation control frequency.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', default='libero_10', choices=[
        'libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90'])
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', default='HuggingFaceVLA/smolvla_libero')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Default: suite horizon (520 for libero_10).')
    parser.add_argument('--video-fps', type=int, default=20,
                        help='Playback FPS; use 80 to match wrapper metadata.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='New directory; existing directories are rejected.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task_id < 0 or args.video_fps <= 0 or (
        args.max_steps is not None and args.max_steps <= 0
    ):
        raise ValueError('Task ID must be nonnegative; FPS and max steps must be positive.')
    if version('lerobot') != '0.4.3':
        raise RuntimeError('This script targets lerobot==0.4.3; activate smolvla_libero.')

    # Must precede simulator imports. Respect an explicitly selected backend.
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.random_utils import set_seed

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this interpreter. Check the active environment.')
    set_seed(args.seed)
    out = args.output_dir or Path('outputs') / (
        f'smolvla_{args.suite}_task{args.task_id}_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    out.mkdir(parents=True, exist_ok=False)
    video_path = out / 'episode.mp4'
    summary = {
        'status': 'initializing', 'suite': args.suite, 'task_id': args.task_id,
        'seed': args.seed, 'initial_state_index': 0, 'policy': args.policy,
        'device': args.device, 'video_fps': args.video_fps,
        'render_backend': os.environ['MUJOCO_GL'],
        'versions': {p: version(p) for p in ['lerobot', 'torch', 'gymnasium', 'hf-libero']},
        'video_path': str(video_path.resolve()), 'steps': 0, 'success': False,
    }
    env = writer = None
    start = perf_counter()
    try:
        print('Loading policy...', flush=True)
        # Device override also controls initial checkpoint loading.
        policy_config = PreTrainedConfig.from_pretrained(args.policy)
        policy_config.device = args.device
        policy = SmolVLAPolicy.from_pretrained(args.policy, config=policy_config)
        policy.to(args.device).eval()
        policy.requires_grad_(False)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, args.policy,
            preprocessor_overrides={'device_processor': {'device': args.device}},
        )
        cfg = LiberoConfig(task=args.suite)
        env_pre, env_post = make_env_pre_post_processors(cfg, policy.config)
        # The Python factory supports this filter in 0.4.3, unlike env.task_ids CLI.
        selected = create_libero_envs(
            task=args.suite, n_envs=1,
            gym_kwargs={**cfg.gym_kwargs, 'task_ids': [args.task_id]},
            env_cls=gym.vector.SyncVectorEnv,
            camera_name=cfg.camera_name, init_states=True,
            control_mode=cfg.control_mode, episode_length=args.max_steps,
        )
        env = selected[args.suite][args.task_id]
        limit = int(env.call('_max_episode_steps')[0])
        instruction = env.call('task_description')[0]
        summary.update(task=instruction, max_steps=limit, status='running')
        print(f'Task {args.task_id}: {instruction}', flush=True)
        print(f'Max steps: {limit}; output: {out.resolve()}', flush=True)

        policy.reset()
        observation, _ = env.reset(seed=[args.seed])
        writer = imageio.get_writer(str(video_path), fps=args.video_fps,
                                   codec='libx264', macro_block_size=1)

        def record_frame(obs):
            # Use the returned observation, not env.render(): the 0.4.3 wrapper
            # internally resets on success. This preserves the terminal frame.
            frame = obs['pixels']['image'][0]
            writer.append_data(np.ascontiguousarray(frame[::-1, ::-1]))

        record_frame(observation)
        reward_sum = 0.0
        rollout_start = perf_counter()
        with (out / 'steps.jsonl').open('w', encoding='utf-8') as log:
            for step in range(1, limit + 1):
                # Same processing order as LeRobot 0.4.3 rollout().
                batch = preprocess_observation(observation)
                batch = add_envs_task(env, batch)
                batch = env_pre(batch)
                batch = preprocessor(batch)
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = postprocessor(action)
                action = env_post({'action': action})['action']
                action_numpy = action.detach().cpu().numpy()
                if action_numpy.shape != (1, 7) or not np.isfinite(action_numpy).all():
                    raise RuntimeError(f'Invalid action: {action_numpy!r}')

                # Future observer/intervention hook belongs here.
                observation, reward, terminated, truncated, info = env.step(action_numpy)
                record_frame(observation)
                success_info = info.get('final_info', info)
                success = bool(np.asarray(success_info.get('is_success', [False])).reshape(-1)[0])
                terminated_flag = bool(terminated[0])
                truncated_flag = bool(truncated[0])
                reward_value = float(reward[0])
                reward_sum += reward_value
                row = {'step': step, 'action': action_numpy[0].tolist(),
                       'reward': reward_value, 'success': success,
                       'terminated': terminated_flag, 'truncated': truncated_flag}
                log.write(json.dumps(row) + '\n')
                log.flush()
                summary.update(steps=step, success=success, sum_rewards=reward_sum)
                if step % 25 == 0 or success or terminated_flag or truncated_flag:
                    print(f'Step {step}/{limit} | reward={reward_value} | success={success}', flush=True)
                if success or terminated_flag or truncated_flag:
                    summary['stop_reason'] = ('success' if success else
                                              'terminated' if terminated_flag else 'truncated')
                    break
            else:
                summary['stop_reason'] = 'step_limit'
        summary.update(status='completed', rollout_seconds=perf_counter() - rollout_start)
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                       error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['total_seconds'] = perf_counter() - start
        # Save diagnostics even if simulation or encoding fails.
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                (out / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
                print('Result:', out.resolve() / 'result.json', flush=True)
    print(f"Finished: success={summary['success']}, steps={summary['steps']}", flush=True)
    print('Video:', video_path.resolve(), flush=True)


if __name__ == '__main__':
    main()
