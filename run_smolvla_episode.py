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
from pathlib import Path
from time import monotonic

from baseline_environment import capture_environment
from attempt_identity import AttemptIdentityRecorder, reserve_attempt
from camera_evidence import CameraEvidenceRecorder
from artifact_finalization import ArtifactResources
from libero_adapter import LiberoEnvironmentAdapter, read_action_horizon
from libero_initial_state import select_initial_state
from episode_harness import EpisodeConfig, exception_stop_reason, run_episode
from recorded_replay import TraceRecorder
from policy_adapter import ResetOnResumePolicyAdapter
from pinned_policy import POLICY_REPOSITORY, POLICY_REVISION, resolve_policy_assets


def evidence_json(value):
    """Encode retained timestamps and native array evidence without repr fallback."""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported evidence type: {type(value).__name__}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', default='libero_10', choices=[
        'libero_10', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90'])
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--initial-state-id', type=int, default=0,
                        help='Explicit index in the task initial-state catalog (default: 0).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', default=POLICY_REPOSITORY,
                        choices=[POLICY_REPOSITORY],
                        help='Frozen baseline repository; revision is enforced by the asset lock.')
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
    if args.task_id < 0 or args.initial_state_id < 0 or args.video_fps <= 0 or (
        args.max_steps is not None and args.max_steps <= 0
    ):
        raise ValueError('Task and initial-state IDs must be nonnegative; '
                         'FPS and max steps must be positive.')
    # Must precede simulator imports. Respect an explicitly selected backend.
    os.environ.setdefault('MUJOCO_GL', 'egl')
    out = args.output_dir or Path('outputs') / (
        f'smolvla_{args.suite}_task{args.task_id}_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    reserve_attempt(out)
    environment = capture_environment(out / 'environment.json', args.device)
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)
    video_path = out / 'episode.mp4'
    wrist_video_path = out / 'episode_wrist.mp4'
    frames_path = out / 'frames.jsonl'
    summary = {
        'status': 'initializing', 'suite': args.suite, 'task_id': args.task_id,
        'seed': args.seed, 'initial_state_index': args.initial_state_id, 'policy': args.policy,
        'policy_revision': POLICY_REVISION, 'policy_assets': None,
        'device': args.device, 'video_fps': args.video_fps,
        'render_backend': os.environ['MUJOCO_GL'],
        'environment_manifest': 'environment.json',
        'versions': {row['name']: row['version'] for row in environment['dependencies']},
        'video_path': None, 'wrist_video_path': None, 'frames_path': None,
        'steps': 0, 'success': False,
        'requested_max_steps': args.max_steps,
        'artifact_status': 'incomplete', 'artifact_diagnostics': [],
        'task_status': 'unknown',
    }
    env = recorder = adapter = identity_recorder = None
    start = monotonic()
    try:
        selection = select_initial_state(args.suite, args.task_id, args.initial_state_id)
        summary['initial_state'] = selection.identity()
        print('Loading policy...', flush=True)
        assets = resolve_policy_assets(args.policy)
        summary['policy_assets'] = assets.identity()
        # Resolve every asset before creating/resetting the environment.
        preflight_path = out / 'policy-assets.json'
        preflight_path.write_text(json.dumps(summary['policy_assets'], indent=2),
                                  encoding='utf-8')
        policy, preprocessor, postprocessor = assets.load(args.device)
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
        horizon = read_action_horizon(env, args.max_steps)
        limit = horizon.effective
        summary['action_horizon'] = asdict(horizon)
        instruction = env.call('task_description')[0]
        summary.update(task=instruction, max_steps=limit, status='running')
        print(f'Task {args.task_id}: {instruction}', flush=True)
        print(f'Max steps: {limit}; output: {out.resolve()}', flush=True)

        class BaselinePolicy(ResetOnResumePolicyAdapter):
            def __init__(self):
                super().__init__(policy.reset, self.infer)

            def infer(self, packet):
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

        class BaselineRecorder:
            def __init__(self):
                self.camera_evidence = None
                self.log = None
                self.cumulative_wait_seconds = 0.0
                self.resources = ArtifactResources()

            def record_frame(self, packet):
                # The adapter selects terminal evidence before recording.
                # Never replace a missing view.
                self.camera_evidence.record(packet)

            def begin(self, packet):
                # Retain verified readback before the first policy proposal.
                (out / 'initial-state.json').write_text(
                    json.dumps(adapter.initial_state_evidence, indent=2), encoding='utf-8')
                self.camera_evidence = CameraEvidenceRecorder(
                    video_path, wrist_video_path, frames_path,
                    lambda path: imageio.get_writer(
                        str(path), fps=args.video_fps, codec='libx264',
                        macro_block_size=1),
                    lambda pixels: np.ascontiguousarray(pixels[0][::-1, ::-1]))
                self.resources.add('camera evidence', self.camera_evidence.close)
                self.log = (out / 'steps.jsonl').open('w', encoding='utf-8')
                self.resources.add('steps log', self.log.close)
                self.record_frame(packet)

            @staticmethod
            def action_evidence(record):
                def native(value):
                    return value[0].tolist() if value is not None else None

                return {
                    'proposal_id': record.proposal_id,
                    'proposed_action': native(record.proposed_action),
                    'selected_action': native(record.selected_action),
                    'executed_action': native(record.executed_action),
                    'execution_acknowledgement': (
                        asdict(record.execution_acknowledgement)
                        if record.execution_acknowledgement is not None else None),
                    'action_disposition': record.disposition,
                    'rejection_reason': record.rejection_reason,
                }

            def record_step(self, step, source, action, result, ingestion):
                if ingestion.accepted:
                    self.record_frame(result.observation)

                def identity(packet):
                    return {'episode_id': packet.episode_id,
                            'sequence': packet.sequence,
                            'captured_at': packet.captured_at.isoformat(),
                            'captured_monotonic': packet.captured_monotonic,
                            'cameras': {ref.camera: ref.synchronization
                                        for ref in packet.frame_references}}

                timing = asdict(result.timing)
                timing.update(observation_age_seconds=result.timing.observation_age_seconds,
                              decision_latency_seconds=result.timing.decision_latency_seconds,
                              execution_seconds=result.timing.execution_seconds)
                self.cumulative_wait_seconds = result.timing.cumulative_wait_seconds

                row = {'step': step, 'action': action[0].tolist(),
                       **self.action_evidence(result.action_record),
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
                    'status': ('rejected' if failure.action_record.disposition == 'rejected'
                               else 'failed'),
                    'failure_stage': failure.stage, 'error_type': failure.error_type,
                    **self.action_evidence(failure.action_record),
                    'source_observation': {
                        'episode_id': source.episode_id,
                        'sequence': source.sequence,
                        'captured_at': source.captured_at.isoformat(),
                        'captured_monotonic': source.captured_monotonic,
                        'cameras': {ref.camera: ref.synchronization
                                    for ref in source.frame_references},
                    },
                    'result_observation': None, 'timing': timing,
                }
                self.log.write(json.dumps(row) + '\n')
                self.log.flush()

            def finish(self):
                self.close()
                artifacts = {'steps_path': str((out / 'steps.jsonl').resolve()),
                             **self.camera_evidence.artifacts}
                return artifacts

            def close(self):
                self.resources.close()

        recorder = BaselineRecorder()
        reward_total = 0.0

        def show_progress(step, result):
            nonlocal reward_total
            reward_total += result.reward
            summary.update(steps=step, success=result.success, sum_rewards=reward_total)
            if step % 25 == 0 or result.success or result.terminated or result.truncated:
                print(f'Step {step}/{limit} | reward={result.reward} | success={result.success}',
                      flush=True)

        episode_config = EpisodeConfig(args.seed, limit)
        adapter = LiberoEnvironmentAdapter(env, initial_state=selection)
        identity_recorder = AttemptIdentityRecorder(out, episode_config, lambda: {
            'task': {'suite': args.suite, 'task_id': args.task_id,
                     'instruction': instruction},
            'initial_state': dict(adapter.initial_state_evidence),
            'seeds': {'global': args.seed, 'environment_reset': args.seed},
            'policy_assets': summary['policy_assets'],
            'environment': environment,
            'settings': {'device': args.device, 'video_fps': args.video_fps,
                         'render_backend': os.environ['MUJOCO_GL'],
                         'action_horizon': asdict(horizon),
                         'environment_config': {
                             'task': args.suite, 'n_envs': 1,
                             'gym_kwargs': {**cfg.gym_kwargs, 'task_ids': [args.task_id]},
                             'env_cls': 'gymnasium.vector.SyncVectorEnv',
                             'camera_name': cfg.camera_name, 'init_states': True,
                             'control_mode': cfg.control_mode,
                             'episode_length': args.max_steps},
                         'processor_overrides': {
                             'device': args.device,
                             'tokenizer': summary['policy_assets']['assets']['backbone']},
                         'mode': 'baseline'},
        }, recorder)
        trace_recorder = TraceRecorder(out / 'replay', episode_config, identity_recorder)
        outcome = run_episode(episode_config, BaselinePolicy(),
                              adapter, trace_recorder, show_progress)
        summary.update(status=('completed' if outcome.artifact_status == 'completed'
                               else 'error'), steps=outcome.steps,
                       artifact_status=outcome.artifact_status,
                       artifact_diagnostics=list(outcome.artifact_diagnostics),
                       episode_id=outcome.episode_id,
                       success=outcome.success, sum_rewards=outcome.sum_rewards,
                       stop_reason=outcome.stop_reason, task_status=outcome.task_status,
                       terminal_observation=(asdict(outcome.terminal_observation)
                           if outcome.terminal_observation is not None else None),
                       rollout_seconds=outcome.rollout_seconds,
                       cumulative_wait_seconds=outcome.cumulative_wait_seconds,
                       video_path=outcome.artifacts.get('video_path'),
                       wrist_video_path=outcome.artifacts.get('wrist_video_path'),
                       frames_path=outcome.artifacts.get('frames_path'))
        if outcome.artifact_status == 'completed':
            try:
                summary['replay_manifest'] = trace_recorder.seal(outcome)
            except Exception as exc:
                summary.update(status='error', artifact_status='incomplete',
                               artifact_diagnostics=[f'{type(exc).__name__}: {exc}'])
                raise
    except BaseException as exc:
        summary.update(status='interrupted' if not isinstance(exc, Exception) else 'error',
                       error=f'{type(exc).__name__}: {exc}',
                       error_type=type(exc).__name__)
        # Sealing/cleanup errors cannot replace an already returned episode reason.
        if 'stop_reason' not in summary:
            summary.update(stop_reason=exception_stop_reason(exc))
        partial = getattr(exc, 'episode_interruption', None)
        if partial is not None:
            summary.update(episode_id=partial.episode_id, steps=partial.steps,
                           sum_rewards=partial.sum_rewards,
                           partial_evidence=asdict(partial),
                           stop_reason=partial.stop_reason, task_status=partial.task_status,
                           artifact_status=partial.artifact_status,
                           success=partial.task_status == 'success')
            summary['artifact_diagnostics'].extend(partial.artifact_diagnostics)
            if partial.pre_start_failure is not None:
                summary['status'] = 'pre_start_failure'
        raise
    finally:
        if identity_recorder is not None:
            summary.update(identity_recorder.reference)
        if adapter is not None:
            summary['initial_state'] = dict(adapter.initial_state_evidence)
        summary['total_seconds'] = monotonic() - start
        if recorder is not None:
            summary['cumulative_wait_seconds'] = recorder.cumulative_wait_seconds
        # Save diagnostics even if simulation or encoding fails.
        try:
            if recorder is not None:
                try:
                    recorder.close()
                except BaseException as exc:
                    diagnostic = f'{type(exc).__name__}: {exc}'
                    if diagnostic not in summary['artifact_diagnostics']:
                        summary['artifact_diagnostics'].append(diagnostic)
                    summary['artifact_status'] = 'incomplete'
                    if summary['status'] == 'completed':
                        summary['status'] = 'error'
            for key, path in (('video_path', video_path),
                              ('wrist_video_path', wrist_video_path),
                              ('frames_path', frames_path)):
                if path.exists():
                    summary[key] = str(path.resolve())
        finally:
            try:
                if env is not None:
                    try:
                        env.close()
                    except BaseException as cleanup_error:
                        summary['artifact_diagnostics'].append(
                            f'{type(cleanup_error).__name__}: {cleanup_error}')
                        summary['artifact_status'] = 'incomplete'
                        if summary['status'] == 'completed':
                            summary['status'] = 'error'
            finally:
                (out / 'result.json').write_text(json.dumps(summary, indent=2, default=evidence_json), encoding='utf-8')
                print('Result:', out.resolve() / 'result.json', flush=True)
    if summary['artifact_status'] != 'completed':
        raise RuntimeError('Episode artifacts incomplete; see result.json diagnostics')
    print(f"Finished: success={summary['success']}, steps={summary['steps']}", flush=True)
    print('Videos:', summary['video_path'], summary['wrist_video_path'], flush=True)


if __name__ == '__main__':
    main()
