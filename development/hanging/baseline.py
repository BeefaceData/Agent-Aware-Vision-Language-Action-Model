"""Frozen-policy rollout, with no planner, perception service or analytic recovery."""

import json
from pathlib import Path
import time

import numpy as np

from harness.environment.base import vector14
from harness.jsonl import append_jsonl


def validate_options(seed, max_actions, chunk_size, time_limit_minutes):
    for name, value, lower, upper in (("seed", seed, 0, 2**31-1),
            ("max_actions", max_actions, 1, 900), ("chunk_size", chunk_size, 1, 24),
            ("time_limit_minutes", time_limit_minutes, 1, 75)):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"invalid_{name}")


def rollout(environment, vla, store, video, output, *, seed, max_actions,
            chunk_size, deadline, clock=time.monotonic):
    """Dependencies are injected so action budgets and failure handling run on CPU."""
    output = Path(output)
    summary = dict(mode="frozen_policy_baseline", seed=seed, policy_actions=0,
                   vla_chunks=0, native_success=False, outcome="action_limit",
                   planner_calls=0, analytic_ticks=0, execution_uncertain=False,
                   error_type=None)
    started = clock()
    try:
        observation = environment.reset(seed)
        store.save(observation)
        video.append(observation, "reset", 0, 0)
        vla.reset_history()
        while summary["policy_actions"] < max_actions:
            if environment.native_success:
                summary.update(native_success=True, outcome="native_success")
                break
            if clock() >= deadline:
                summary["outcome"] = "time_limit"
                break
            requested = min(chunk_size, max_actions-summary["policy_actions"])
            summary["vla_chunks"] += 1
            actions = np.asarray(vla.infer(observation, requested))
            if (actions.ndim != 2 or actions.shape[1:] != (14,)
                    or not 1 <= len(actions) <= requested
                    or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all()):
                raise ValueError("invalid_policy_chunk")
            # Validate every action before executing any of this chunk.
            actions = [vector14(action) for action in actions]
            for action in actions:
                if clock() >= deadline:
                    summary["outcome"] = "time_limit"
                    break
                index = summary["policy_actions"] + 1
                append_jsonl(output / "actions.jsonl", dict(event="action_requested",
                    index=index, sequence=observation.sequence, action=list(action)))
                summary["execution_uncertain"] = True
                observation = environment.step(action)  # Never retry a failed step.
                summary["execution_uncertain"] = False
                summary["policy_actions"] = index
                append_jsonl(output / "actions.jsonl", dict(event="action_confirmed",
                    index=index, sequence=observation.sequence,
                    command_state=list(observation.command_state)))
                store.save(observation)
                video.append(observation, "frozen_policy", index, 0)
                if environment.native_success:
                    summary.update(native_success=True, outcome="native_success")
                    break
            if summary["outcome"] != "action_limit":
                break
        # Success on the last permitted action must still be recorded.
        if environment.native_success:
            summary.update(native_success=True, outcome="native_success")
    except KeyboardInterrupt:
        summary["outcome"] = "interrupted"
    except Exception as exc:
        summary.update(outcome="execution_error", error_type=type(exc).__name__)
    finally:
        try:
            video.close()
        except Exception as exc:
            summary["video_error_type"] = type(exc).__name__
        summary["elapsed_seconds"] = clock() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(robotwin_root, output, *, seed, max_actions, chunk_size, time_limit_minutes,
        perception_camera=""):
    validate_options(seed, max_actions, chunk_size, time_limit_minutes)
    from harness.environment.perception_cameras import parse_camera_spec, describe
    from harness.environment.layout import CAMERAS

    camera = parse_camera_spec(perception_camera) if perception_camera else None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(mode="frozen_policy_baseline", seed=seed, max_actions=max_actions,
        chunk_size=chunk_size, time_limit_minutes=time_limit_minutes,
        policy="robotwin-policy/PolicyWorker", planner_started=False,
        rgbd_saved=True, native_success_is_not_verified_hanging=True,
        perception_camera=None if camera is None else describe(camera),
        policy_cameras=list(CAMERAS))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from development.simulator.local_simulator import LocalSimulator
    from development.demos.episode_video import EpisodeVideo
    from harness.environment.local import LocalEnvironmentClient
    from harness.perception.observations import ObservationStore
    from harness.vla.modal_client import ModalVLAClient

    simulator = None
    video = None
    started = time.monotonic()
    try:
        simulator = LocalSimulator(robotwin_root, rgbd_enabled=True,
                                   perception_cameras=() if camera is None else (camera,))
        video = EpisodeVideo(output, cameras=(*CAMERAS, camera.name) if camera else CAMERAS)
        return rollout(LocalEnvironmentClient(simulator, render_once_per_step=True),
            ModalVLAClient(), ObservationStore(output / "observations"), video, output,
            seed=seed, max_actions=max_actions, chunk_size=chunk_size,
            deadline=started + time_limit_minutes*60)
    except Exception as exc:
        summary = dict(mode="frozen_policy_baseline", outcome="initialization_error",
                       error_type=type(exc).__name__, seed=seed, policy_actions=0)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        if video is not None:
            video.close()
        if simulator is not None:
            simulator.close()
