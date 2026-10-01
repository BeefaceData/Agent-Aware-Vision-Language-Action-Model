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
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import monotonic

from episode_harness import EpisodeConfig, ObservationPacket, StepResult, run_episode


def parse_args(argv=None):
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
    return parser.parse_args(argv)


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
    env = recorder = None
    start = monotonic()
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

        class BaselinePolicy:
            def reset(self):
                policy.reset()

            def act(self, packet):
                # Same processing order as LeRobot 0.4.3 rollout().
                batch = preprocess_observation(packet.observation)
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
                return action_numpy

        class BaselineEnvironment:
            def reset(self, seed, episode_id):
                self.episode_id = episode_id
                self.sequence = 0
                observation, _ = env.reset(seed=[seed])
                captured_monotonic = monotonic()
                captured_at = datetime.now(timezone.utc)
                return ObservationPacket(episode_id, self.sequence,
                                         captured_at, observation,
                                         captured_monotonic)

            def step(self, action):
                observation, reward, terminated, truncated, info = env.step(action)
                captured_monotonic = monotonic()
                captured_at = datetime.now(timezone.utc)
                self.sequence += 1
                success_info = info.get('final_info', info)
                return StepResult(
                    observation=ObservationPacket(
                        self.episode_id, self.sequence,
                        captured_at, observation, captured_monotonic),
                    reward=float(reward[0]),
                    success=bool(np.asarray(success_info.get('is_success', [False])).reshape(-1)[0]),
                    terminated=bool(terminated[0]),
                    truncated=bool(truncated[0]),
                )

        class BaselineRecorder:
            def __init__(self):
                self.writer = None
                self.log = None
                self.cumulative_wait_seconds = 0.0

            def record_frame(self, packet):
                # The wrapper may reset internally on success; returned pixels
                # preserve the terminal observation instead of env.render().
                frame = packet.observation['pixels']['image'][0]
                self.writer.append_data(np.ascontiguousarray(frame[::-1, ::-1]))

            def begin(self, packet):
                self.writer = imageio.get_writer(str(video_path), fps=args.video_fps,
                                                 codec='libx264', macro_block_size=1)
                self.record_frame(packet)
                self.log = (out / 'steps.jsonl').open('w', encoding='utf-8')

            def record_step(self, step, source, action, result, ingestion):
                if ingestion.accepted:
                    self.record_frame(result.observation)

                def identity(packet):
                    return {'episode_id': packet.episode_id,
                            'sequence': packet.sequence,
                            'captured_at': packet.captured_at.isoformat(),
                            'captured_monotonic': packet.captured_monotonic}

                timing = asdict(result.timing)
                timing.update(observation_age_seconds=result.timing.observation_age_seconds,
                              decision_latency_seconds=result.timing.decision_latency_seconds,
                              execution_seconds=result.timing.execution_seconds)
                self.cumulative_wait_seconds = result.timing.cumulative_wait_seconds

                row = {'step': step, 'action': action[0].tolist(),
                       'source_observation': identity(source),
                       'result_observation': (identity(result.observation)
                                              if isinstance(result.observation,
                                                            ObservationPacket) else None),
                       'observation_ingestion': ingestion.code,
                       'timing': timing,
                       'reward': result.reward, 'success': result.success,
                       'terminated': result.terminated, 'truncated': result.truncated}
                self.log.write(json.dumps(row) + '\n')
                self.log.flush()

            def record_failure(self, step, source, action, failure):
                timing = asdict(failure.timing)
                timing.update(
                    observation_age_seconds=failure.timing.observation_age_seconds,
                    decision_latency_seconds=failure.timing.decision_latency_seconds)
                self.cumulative_wait_seconds = failure.timing.cumulative_wait_seconds
                row = {
                    'attempted_step': step, 'completed_steps': step - 1,
                    'status': 'failed',
                    'failure_stage': failure.stage, 'error_type': failure.error_type,
                    'proposed_action': action[0].tolist(),
                    'source_observation': {
                        'episode_id': source.episode_id,
                        'sequence': source.sequence,
                        'captured_at': source.captured_at.isoformat(),
                        'captured_monotonic': source.captured_monotonic,
                    },
                    'result_observation': None, 'timing': timing,
                }
                self.log.write(json.dumps(row) + '\n')
                self.log.flush()

            def finish(self):
                self.close()
                return {'video_path': str(video_path.resolve()),
                        'steps_path': str((out / 'steps.jsonl').resolve())}

            def close(self):
                try:
                    if self.log is not None:
                        self.log.close()
                        self.log = None
                finally:
                    if self.writer is not None:
                        self.writer.close()
                        self.writer = None

        recorder = BaselineRecorder()
        reward_total = 0.0

        def show_progress(step, result):
            nonlocal reward_total
            reward_total += result.reward
            summary.update(steps=step, success=result.success, sum_rewards=reward_total)
            if step % 25 == 0 or result.success or result.terminated or result.truncated:
                print(f'Step {step}/{limit} | reward={result.reward} | success={result.success}',
                      flush=True)

        outcome = run_episode(EpisodeConfig(args.seed, limit), BaselinePolicy(),
                              BaselineEnvironment(), recorder, show_progress)
        summary.update(status='completed', steps=outcome.steps,
                       episode_id=outcome.episode_id,
                       success=outcome.success, sum_rewards=outcome.sum_rewards,
                       stop_reason=outcome.stop_reason,
                       rollout_seconds=outcome.rollout_seconds,
                       cumulative_wait_seconds=outcome.cumulative_wait_seconds)
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                       error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['total_seconds'] = monotonic() - start
        if recorder is not None:
            summary['cumulative_wait_seconds'] = recorder.cumulative_wait_seconds
        # Save diagnostics even if simulation or encoding fails.
        try:
            if recorder is not None:
                recorder.close()
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
