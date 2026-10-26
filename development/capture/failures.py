"""Capture frozen Modal-policy failures for offline recovery development only."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from uuid import uuid4

import numpy as np

from harness.jsonl import append_jsonl
from harness.environment.base import copy_images, vector14
from development.checkpoints.bank import CheckpointBank, sha256, write_json_atomic
from development.checkpoints.provenance import local_provenance
from development.checkpoints.schemas import CAMERAS, FailureCheckpoint, RecordedFrame


class CaptureRecorder:
    """Transparent simulator transport wrapper; records only confirmed responses.

    Raw simulator observations never go into the public trace. Success and error
    information live in the separate development evaluation artifact.
    """

    def __init__(self, bank, run_id, episode_id, seed, sim_client, *, history=40, image_interval=24):
        if history < 1 or image_interval < 1:
            raise ValueError("history and image_interval must be positive")
        self.bank, self.run_id, self.episode_id = bank, run_id, episode_id
        self.seed, self.sim_client = seed, sim_client
        self.directory = bank.resolve_artifact(f"runs/{run_id}/{episode_id}")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.directory / "actions.jsonl"
        self.states, self.actions = deque(maxlen=history), deque(maxlen=history)
        self.image_interval = image_interval
        self.step = None
        self.last_images = None
        self.success = None
        self.pending_action = None
        self.finalized = False

    def __call__(self, host, port, path, payload):
        if self.finalized:
            raise ValueError("capture already finalized")
        if path == "/reset":
            if self.step is not None or payload != {"seed": self.seed}:
                raise ValueError("capture requires exactly one matching seeded reset")
            action = None
            step = 0
        elif path == "/step" and self.step is not None:
            action = vector14(payload["action"])
            self.pending_action = action
            step = self.step + 1
        else:
            raise ValueError("capture requires reset followed by action steps")
        observation, timing = self.sim_client(host, port, path, payload)
        state = vector14(observation["state"])
        images = copy_images({name: observation[name] for name in CAMERAS})
        frame = RecordedFrame(step=step, action=action, command_state=state)
        append_jsonl(self.trace_path, frame.model_dump(mode="json"))
        self.step, self.last_images = step, images
        self.states.append(state)
        if action is not None:
            self.actions.append(action)
        self.success = bool(observation["success"])
        self.pending_action = None
        if step % self.image_interval == 0:
            self.save_images()
        return observation, timing

    def save_images(self):
        path = self.directory / f"images_{self.step:06d}.npz"
        if not path.exists():
            with path.open("xb") as stream:
                np.savez_compressed(stream, **self.last_images)
        return path

    def finish(self, *, reason, provenance, error_type=None):
        if self.finalized:
            raise ValueError("capture already finalized")
        self.finalized = True
        evaluation_path = self.directory / "evaluation.json"
        write_json_atomic(evaluation_path, {
            "schema_version": 1, "episode_id": self.episode_id, "seed": self.seed,
            "reason": reason, "success_at_last_confirmed_frame": self.success,
            "confirmed_steps": self.step, "error_type": error_type,
            "unconfirmed_action": self.pending_action,
            "checkpoint_boundary": "last confirmed frame; an errored request may have changed the live simulator",
        })
        provenance_path = self.directory / "provenance.json"
        write_json_atomic(provenance_path, provenance)
        if self.step is None or self.success:
            return None  # No observed failure state, or a successful rollout.
        image_path = self.save_images()
        relative = lambda path: path.relative_to(self.bank.path).as_posix()
        assets = [image_path, self.trace_path, evaluation_path, provenance_path]
        checkpoint = FailureCheckpoint(
            checkpoint_id=f"{self.episode_id}_step_{self.step:06d}", run_id=self.run_id,
            episode_id=self.episode_id, seed=self.seed, step=self.step,
            images={name: relative(image_path) for name in CAMERAS}, state=self.states[-1],
            recent_states=tuple(self.states), recent_actions=tuple(self.actions),
            trace_path=relative(self.trace_path), evaluation_path=relative(evaluation_path),
            provenance_path=relative(provenance_path),
            artifact_hashes={relative(path): sha256(path) for path in assets},
            notes="Unclassified failure candidate at last confirmed frame; inspect and label before skill development.",
        )
        return self.bank.add(checkpoint)


def capture_episode(worker, bank, *, seed, max_steps, chunk_size, sim_host="127.0.0.1",
                    sim_port=8110, sim_client=None, history=40, image_interval=24,
                    video=False, policy_identity=None, run_id=None):
    import run_inference_chunked as runner

    if seed < 0 or min(max_steps, chunk_size) < 1:
        raise ValueError("seed must be nonnegative; steps/chunk size must be positive")
    run_id = run_id or f"capture_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:12]}"
    episode_id = f"{run_id}_seed_{seed}_{uuid4().hex[:8]}"
    from harness.environment.robotwin import call_sim

    recorder = CaptureRecorder(bank, run_id, episode_id, seed, sim_client or call_sim,
                               history=history, image_interval=image_interval)
    provenance = {
        "schema_version": 1, "local": local_provenance(),
        "policy": {"identity": policy_identity or "injected worker", "remote_weights_attested": False},
        "simulator": {"host": sim_host, "port": sim_port, "task_assumed": "hanging_mug",
                      "config_assumed": "demo_clean", "server_identity_attested": False},
        "capture": {"max_steps": max_steps, "chunk_size": chunk_size, "history": history,
                    "image_interval": image_interval, "video": video},
        "state_semantics": "arm_drive_targets_and_gripper_commands",
        "reconstruction": "seeded_reset_and_full_confirmed_action_prefix",
    }
    error_type = None
    try:
        success = runner.run_episode(
            worker, seed, max_steps, chunk_size,
            recorder.directory / "rollout.mp4" if video else None,
            sim_host, sim_port, recorder.directory / "timing", sim_client=recorder,
        )
        reason = "success" if success else "horizon"
    except Exception as exc:
        # Do not persist arbitrary transport exceptions (which may contain secrets).
        reason, error_type = "error", type(exc).__name__
    checkpoint = recorder.finish(reason=reason, provenance=provenance, error_type=error_type)
    return {"episode_id": episode_id, "reason": reason, "error_type": error_type,
            "confirmed_steps": recorder.step,
            "checkpoint_id": checkpoint.checkpoint_id if checkpoint else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/checkpoints")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--history", type=int, default=40)
    parser.add_argument("--image-interval", type=int, default=24)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--sim-host", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=8110)
    parser.add_argument("--modal-app", default="robotwin-policy")
    parser.add_argument("--modal-class", default="PolicyWorker")
    args = parser.parse_args()
    if args.seed < 0 or min(args.episodes, args.steps, args.chunk_size, args.history, args.image_interval) < 1:
        parser.error("seed must be nonnegative and counts must be positive")
    from harness.vla.modal_client import ModalPolicyClient

    bank = CheckpointBank(args.output)
    worker = ModalPolicyClient(args.modal_app, args.modal_class)
    results = []
    for seed in range(args.seed, args.seed + args.episodes):
        result = capture_episode(
            worker, bank, seed=seed, max_steps=args.steps, chunk_size=args.chunk_size,
            sim_host=args.sim_host, sim_port=args.sim_port, history=args.history,
            image_interval=args.image_interval, video=not args.no_video,
            policy_identity={"app": args.modal_app, "class": args.modal_class},
        )
        results.append(result)
        if result["error_type"]:
            break  # Transport errors can leave the simulator in an unknown state.
    print(json.dumps(results, indent=2))
    if any(result["error_type"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
