#!/usr/bin/env python
"""Run a trained policy on the RoboTwin `hanging_mug` task and record what it does.

The simulator and the policy need incompatible versions of numpy, so they run as
two processes that talk over HTTP on localhost:

    python run_inference.py sim                          # terminal 1 (sim env)
    python run_inference.py policy --weights ./weights   # terminal 2 (policy env)

`supervise()` near the bottom is the hook to build on: it sees every observation
and the action the policy chose, and whatever it returns is what the robot does.
"""

import argparse
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")
ACTION_DIM = 14
HOST, PORT = "127.0.0.1", 8110


# ──────────────────────────── simulator side ────────────────────────────


def build_task(robotwin_root: str, task_name: str):
    """Import a RoboTwin task. Its modules read ./assets by relative path, so the
    checkout has to be the working directory as well as importable."""
    root = os.path.abspath(robotwin_root)
    if not os.path.isdir(os.path.join(root, "envs")):
        sys.exit(f"no RoboTwin checkout at {root}")
    sys.path.insert(0, root)
    os.chdir(root)

    import importlib

    import yaml
    from envs import CONFIGS_PATH

    with open(os.path.join(CONFIGS_PATH, "demo_clean.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)

    robot = embodiments[args.get("embodiment", ["aloha-agilex"])[0]]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot
    args["dual_arm_embodied"] = True
    for side in ("left", "right"):
        with open(os.path.join(args[f"{side}_robot_file"], "config.yml"), encoding="utf-8") as f:
            args[f"{side}_embodiment_config"] = yaml.safe_load(f)

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        cameras = yaml.safe_load(f)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = cameras[head]["h"], cameras[head]["w"]
    args["render_freq"] = 0
    args["task_name"] = task_name
    args["task_config"] = "demo_clean"

    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)(), args


def observation_of(task, blank, success):
    raw = task.get_obs()
    cameras = raw.get("observation", {})
    out = {}
    for camera in CAMERAS:
        image = (cameras.get(camera) or {}).get("rgb")
        if image is None:
            out[camera] = blank
            continue
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        out[camera] = np.ascontiguousarray(image[..., :3])

    vector = (raw.get("joint_action") or {}).get("vector")
    joints = np.zeros(ACTION_DIM, dtype=np.float32)
    if vector is not None:
        flat = np.asarray(vector, dtype=np.float32).ravel()
        joints = flat[:ACTION_DIM] if flat.size >= ACTION_DIM else joints
    out["state"] = joints
    out["success"] = np.bool_(success)
    return out


def serve_simulator(robotwin_root: str, task_name: str) -> None:
    import torch

    task, setup = build_task(robotwin_root, task_name)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}")
            if self.path == "/reset":
                with torch.enable_grad():
                    task.setup_demo(**dict(setup, seed=int(body.get("seed", 0)), is_test=True))
                out = observation_of(task, blank, False)
            else:
                with torch.enable_grad():
                    task.take_action(np.asarray(body["action"], dtype=np.float64))
                success = bool(getattr(task, "eval_success", False)) or bool(task.check_success())
                out = observation_of(task, blank, success)

            buffer = io.BytesIO()
            np.savez(buffer, **out)
            payload = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    print(f"simulator ready on http://{HOST}:{PORT} ({task_name})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


# ────────────────────────────── policy side ──────────────────────────────


def call(path: str, payload: dict) -> dict:
    import requests

    reply = requests.post(f"http://{HOST}:{PORT}{path}", json=payload, timeout=600)
    reply.raise_for_status()
    with np.load(io.BytesIO(reply.content)) as data:
        return {key: data[key] for key in data.files}


def load_policy(weights: str, device: str):
    """Load the shared checkpoint and the normalisation it was trained with."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(weights)
    config.pretrained_path = weights
    config.device = device

    policy = get_policy_class(config.type).from_pretrained(weights, config=config)
    policy.to(device)
    policy.eval()
    policy.reset()
    device_override = {"device": device}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=weights,
        preprocessor_overrides={
            "device_processor": device_override,
        },
        postprocessor_overrides={
            "device_processor": device_override,
        },
    )
    return policy, preprocessor, postprocessor


def supervise(step: int, images: dict, state, action):
    """Your agent goes here.

    Called once per control step, before the action reaches the robot.

    images   {"head_camera": HxWx3 uint8, "left_camera": ..., "right_camera": ...}
    state    14 joint positions, left arm then right arm
    action   14 joint targets the policy chose

    Return the action to execute. Return it unchanged to let the policy drive,
    or return something else to take over. Nothing here is load-bearing: delete
    it and write your own.
    """
    return action


def run_episode(weights: str, seed: int, steps: int, device: str, video: Path | None):
    import torch

    policy, preprocessor, postprocessor = load_policy(weights, device)
    observation = call("/reset", {"seed": seed})
    frames, success = [], False

    for step in range(steps):
        images = {camera: observation[camera] for camera in CAMERAS}
        frames.append(images["head_camera"].copy())

        batch = {
            "observation.images.cam_high": images["head_camera"],
            "observation.images.cam_left_wrist": images["left_camera"],
            "observation.images.cam_right_wrist": images["right_camera"],
            "observation.state": observation["state"],
        }
        batch = {
            k: torch.as_tensor(v).to(device).unsqueeze(0)
            for k, v in batch.items()
        }
        for key in list(batch):
            if "images" in key:
                batch[key] = batch[key].permute(0, 3, 1, 2).float() / 255.0
        batch["task"] = ["hang the mug on the rack"]

        with torch.no_grad():
            action = postprocessor(policy.select_action(preprocessor(batch)))
        action = action.squeeze(0).cpu().numpy()

        action = supervise(step, images, observation["state"], action)

        observation = call("/step", {"action": np.asarray(action, dtype=float).tolist()})
        if bool(observation["success"]):
            success = True
            print(f"success at step {step}", flush=True)
            break

    if video is not None and frames:
        import imageio.v3 as iio

        video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video, np.stack(frames), fps=30, codec="libx264")
        print(f"wrote {video}", flush=True)

    return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["sim", "policy"])
    parser.add_argument("--robotwin-root", default="./RoboTwin")
    parser.add_argument("--task", default="hanging_mug")
    parser.add_argument("--weights", default="./weights")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-dir", default="videos")
    args = parser.parse_args()

    if args.role == "sim":
        serve_simulator(args.robotwin_root, args.task)
        return

    wins = 0
    for episode in range(args.episodes):
        seed = args.seed + episode
        video = Path(args.video_dir) / f"episode_{seed}.mp4"
        if run_episode(args.weights, seed, args.steps, args.device, video):
            wins += 1
        print(f"episode {episode + 1}/{args.episodes} (seed {seed}): {wins} successes so far", flush=True)

    print(f"\nsuccess rate: {100 * wins / args.episodes:.0f}% ({wins}/{args.episodes})")


if __name__ == "__main__":
    main()
