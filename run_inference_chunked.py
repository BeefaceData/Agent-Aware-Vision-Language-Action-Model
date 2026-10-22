#!/usr/bin/env python
"""Run RoboTwin locally with Modal action-chunk inference and latency profiling.

Terminal 1:
    conda activate robotwin
    python run_inference_chunked.py sim --robotwin-root ./RoboTwin

Terminal 2:
    conda activate robotwin
    python run_inference_chunked.py policy \
        --episodes 1 --steps 100 --chunk-size 8 --no-video

The policy is queried once per chunk rather than once per simulator step.

IMPORTANT:
Action chunking is open-loop within each chunk: the chunk is predicted from the
observation at the start of that chunk. RoboTwin observations are still updated
after every action, and `supervise()` is still called at every action step.
"""

import argparse
import csv
import io
import json
import os
import statistics
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")
ACTION_DIM = 14
HOST, PORT = "127.0.0.1", 8110


# ─────────────────────────── simulator side ────────────────────────────


def build_task(robotwin_root: str, task_name: str):
    root = os.path.abspath(robotwin_root)
    if not os.path.isdir(os.path.join(root, "envs")):
        sys.exit(f"no RoboTwin checkout at {root}")

    sys.path.insert(0, root)
    os.chdir(root)

    import importlib
    import yaml
    from envs import CONFIGS_PATH

    with open(
        os.path.join(CONFIGS_PATH, "demo_clean.yml"),
        encoding="utf-8",
    ) as f:
        args = yaml.safe_load(f)

    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"),
        encoding="utf-8",
    ) as f:
        embodiments = yaml.safe_load(f)

    robot = embodiments[
        args.get("embodiment", ["aloha-agilex"])[0]
    ]["file_path"]

    args["left_robot_file"] = robot
    args["right_robot_file"] = robot
    args["dual_arm_embodied"] = True

    for side in ("left", "right"):
        with open(
            os.path.join(
                args[f"{side}_robot_file"],
                "config.yml",
            ),
            encoding="utf-8",
        ) as f:
            args[f"{side}_embodiment_config"] = (
                yaml.safe_load(f)
            )

    with open(
        os.path.join(CONFIGS_PATH, "_camera_config.yml"),
        encoding="utf-8",
    ) as f:
        cameras = yaml.safe_load(f)

    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head]["h"]
    args["head_camera_w"] = cameras[head]["w"]
    args["render_freq"] = 0
    args["task_name"] = task_name
    args["task_config"] = "demo_clean"

    module = importlib.import_module(
        f"envs.{task_name}"
    )

    return getattr(module, task_name)(), args


def observation_of(task, blank, success):
    raw = task.get_obs()
    cameras = raw.get("observation", {})
    out = {}

    for camera in CAMERAS:
        camera_data = cameras.get(camera) or {}
        image = camera_data.get("rgb")

        if image is None:
            out[camera] = blank
            continue

        image = np.asarray(
            image,
            dtype=np.uint8,
        )

        if image.ndim == 2:
            image = np.stack(
                [image] * 3,
                axis=-1,
            )

        out[camera] = np.ascontiguousarray(
            image[..., :3]
        )

        # Optional perception-only RGB-D channel. The Modal Neotix client still
        # packs only the three RGB keys and the unchanged 14-DoF state.
        depth = camera_data.get("depth")
        intrinsic = camera_data.get("intrinsic_cv")
        extrinsic = camera_data.get("extrinsic_cv")
        if depth is not None:
            if intrinsic is None or extrinsic is None:
                raise ValueError("rgbd_camera_calibration_missing")
            depth = np.asarray(depth)
            if depth.shape != image.shape[:2]:
                raise ValueError("rgbd_depth_rgb_shape_mismatch")
            out[f"{camera}__depth_mm"] = np.ascontiguousarray(depth, dtype=np.float32)
            out[f"{camera}__intrinsic_cv"] = np.asarray(intrinsic, dtype=np.float64)
            out[f"{camera}__extrinsic_cv"] = np.asarray(extrinsic, dtype=np.float64)

    vector = (
        raw.get("joint_action") or {}
    ).get("vector")

    joints = np.zeros(
        ACTION_DIM,
        dtype=np.float32,
    )

    if vector is not None:
        flat = np.asarray(
            vector,
            dtype=np.float32,
        ).ravel()

        if flat.size >= ACTION_DIM:
            joints = flat[:ACTION_DIM]

    out["state"] = joints
    out["success"] = np.bool_(success)

    return out


def serve_simulator(
    robotwin_root: str,
    task_name: str,
    host: str,
    port: int,
):
    import torch

    task, setup = build_task(
        robotwin_root,
        task_name,
    )

    blank = np.zeros(
        (240, 320, 3),
        dtype=np.uint8,
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(
            self,
            format,
            *args,
        ):
            pass

        def do_POST(self):
            n = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            body = json.loads(
                self.rfile.read(n) or b"{}"
            )

            action_ms = 0.0
            obs_ms = 0.0

            if self.path == "/reset":
                t0 = time.perf_counter()

                with torch.enable_grad():
                    task.setup_demo(
                        **dict(
                            setup,
                            seed=int(
                                body.get(
                                    "seed",
                                    0,
                                )
                            ),
                            is_test=True,
                        )
                    )

                action_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                t0 = time.perf_counter()
                out = observation_of(
                    task,
                    blank,
                    False,
                )
                obs_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

            elif self.path == "/step":
                t0 = time.perf_counter()

                with torch.enable_grad():
                    task.take_action(
                        np.asarray(
                            body["action"],
                            dtype=np.float64,
                        )
                    )

                success = bool(
                    getattr(
                        task,
                        "eval_success",
                        False,
                    )
                ) or bool(
                    task.check_success()
                )

                action_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                t0 = time.perf_counter()
                out = observation_of(
                    task,
                    blank,
                    success,
                )
                obs_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

            else:
                self.send_error(404)
                return

            t0 = time.perf_counter()
            buffer = io.BytesIO()
            np.savez(buffer, **out)
            payload = buffer.getvalue()
            encode_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/octet-stream",
            )
            self.send_header(
                "Content-Length",
                str(len(payload)),
            )
            self.send_header(
                "X-Sim-Action-Ms",
                f"{action_ms:.6f}",
            )
            self.send_header(
                "X-Sim-Observation-Ms",
                f"{obs_ms:.6f}",
            )
            self.send_header(
                "X-Sim-Encode-Ms",
                f"{encode_ms:.6f}",
            )
            self.end_headers()
            self.wfile.write(payload)

    print(
        f"simulator ready on "
        f"http://{host}:{port} "
        f"({task_name})",
        flush=True,
    )

    ThreadingHTTPServer(
        (host, port),
        Handler,
    ).serve_forever()


# ───────────────────────── communication helpers ───────────────────────


def call_sim(
    host: str,
    port: int,
    path: str,
    payload: dict,
):
    import requests

    t0 = time.perf_counter()

    reply = requests.post(
        f"http://{host}:{port}{path}",
        json=payload,
        timeout=600,
    )

    http_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    reply.raise_for_status()

    with np.load(
        io.BytesIO(reply.content)
    ) as data:
        observation = {
            key: data[key]
            for key in data.files
        }

    timing = {
        "sim_http_ms": http_ms,
        "sim_action_ms": float(
            reply.headers.get(
                "X-Sim-Action-Ms",
                "nan",
            )
        ),
        "sim_observation_ms": float(
            reply.headers.get(
                "X-Sim-Observation-Ms",
                "nan",
            )
        ),
        "sim_encode_ms": float(
            reply.headers.get(
                "X-Sim-Encode-Ms",
                "nan",
            )
        ),
    }

    return observation, timing


def pack_observation(images, state):
    t0 = time.perf_counter()

    buffer = io.BytesIO()

    np.savez_compressed(
        buffer,
        head_camera=np.asarray(
            images["head_camera"],
            dtype=np.uint8,
        ),
        left_camera=np.asarray(
            images["left_camera"],
            dtype=np.uint8,
        ),
        right_camera=np.asarray(
            images["right_camera"],
            dtype=np.uint8,
        ),
        state=np.asarray(
            state,
            dtype=np.float32,
        ),
    )

    payload = buffer.getvalue()

    return (
        payload,
        (time.perf_counter() - t0) * 1000.0,
        len(payload) / 1024.0,
    )


class ModalPolicyClient:
    def __init__(
        self,
        app_name: str,
        class_name: str,
    ):
        import modal

        cls = modal.Cls.from_name(
            app_name,
            class_name,
        )

        self.worker = cls()

    def reset(self):
        t0 = time.perf_counter()
        info = self.worker.reset_episode.remote()

        return info, (
            time.perf_counter() - t0
        ) * 1000.0

    def act_chunk(
        self,
        images,
        state,
        chunk_size,
    ):
        (
            packed,
            pack_ms,
            payload_kb,
        ) = pack_observation(
            images,
            state,
        )

        t0 = time.perf_counter()

        result = self.worker.act_chunk.remote(
            packed,
            requested_actions=chunk_size,
        )

        rpc_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        actions = np.asarray(
            result["actions"],
            dtype=np.float32,
        )

        if (
            actions.ndim != 2
            or actions.shape[1] != ACTION_DIM
        ):
            raise RuntimeError(
                "Modal returned invalid chunk shape "
                f"{actions.shape}; expected [T, {ACTION_DIM}]"
            )

        modal_compute_ms = float(
            result.get(
                "modal_compute_ms",
                float("nan"),
            )
        )

        timing = {
            "pack_ms": pack_ms,
            "payload_kb": payload_kb,
            "modal_rpc_ms": rpc_ms,
            "modal_compute_ms": modal_compute_ms,
            "modal_overhead_ms": (
                rpc_ms - modal_compute_ms
                if np.isfinite(modal_compute_ms)
                else float("nan")
            ),
            "cloud_preprocess_ms": float(
                result.get(
                    "preprocess_ms",
                    float("nan"),
                )
            ),
            "cloud_inference_ms": float(
                result.get(
                    "inference_ms",
                    float("nan"),
                )
            ),
            "cloud_postprocess_ms": float(
                result.get(
                    "postprocess_ms",
                    float("nan"),
                )
            ),
            "returned_actions": int(
                result["num_actions"]
            ),
            "chunk_limit": int(
                result["chunk_limit"]
            ),
        }

        return actions, timing


def supervise(
    step: int,
    images: dict,
    state,
    action,
    *,
    harness=None,
    chunk_id=0,
    chunk_pos=0,
    chunk_size=None,
):
    """Agentic-supervision hook.

    Still called every physical simulator step, even though the base policy is
    queried only once per action chunk.
    """
    if harness is not None:
        return harness.before_action(
            step, images, state, action, chunk_id=chunk_id, chunk_pos=chunk_pos,
            chunk_size=chunk_size,
        )
    return action


# ───────────────────────────── diagnostics ─────────────────────────────


def mean(rows, key):
    vals = [
        float(row[key])
        for row in rows
        if np.isfinite(float(row[key]))
    ]
    return statistics.mean(vals) if vals else float("nan")


def print_summary(
    step_rows,
    chunk_rows,
):
    print("\n" + "=" * 82)
    print("ACTION-CHUNK LATENCY SUMMARY")
    print("=" * 82)

    if chunk_rows:
        avg_chunk = mean(
            chunk_rows,
            "actual_chunk_size",
        )

        avg_rpc = mean(
            chunk_rows,
            "modal_rpc_ms",
        )

        avg_compute = mean(
            chunk_rows,
            "modal_compute_ms",
        )

        avg_overhead = mean(
            chunk_rows,
            "modal_overhead_ms",
        )

        avg_inference = mean(
            chunk_rows,
            "cloud_inference_ms",
        )

        print(
            f"Average actions per cloud call : "
            f"{avg_chunk:.2f}"
        )
        print(
            f"Modal RPC per chunk            : "
            f"{avg_rpc:.1f} ms"
        )
        print(
            f"Cloud compute per chunk        : "
            f"{avg_compute:.1f} ms"
        )
        print(
            f"  policy inference             : "
            f"{avg_inference:.1f} ms"
        )
        print(
            f"Network/RPC overhead estimate  : "
            f"{avg_overhead:.1f} ms"
        )

        if avg_chunk > 0:
            print(
                f"Amortized Modal RPC / action   : "
                f"{avg_rpc / avg_chunk:.1f} ms"
            )

    if step_rows:
        avg_sim = mean(
            step_rows,
            "sim_http_ms",
        )
        avg_action = mean(
            step_rows,
            "sim_action_ms",
        )
        avg_obs = mean(
            step_rows,
            "sim_observation_ms",
        )
        avg_effective = mean(
            step_rows,
            "effective_step_ms",
        )

        print("-" * 82)
        print(
            f"Simulator HTTP / action        : "
            f"{avg_sim:.1f} ms"
        )
        print(
            f"  RoboTwin action              : "
            f"{avg_action:.1f} ms"
        )
        print(
            f"  Camera/observation           : "
            f"{avg_obs:.1f} ms"
        )
        print(
            f"Effective avg control step     : "
            f"{avg_effective:.1f} ms"
        )

        hz = (
            1000.0 / avg_effective
            if avg_effective > 0
            else float("nan")
        )

        est_minutes = (
            avg_effective * 1200.0
            / 60000.0
        )

        print(
            f"Effective control frequency    : "
            f"{hz:.2f} steps/s"
        )
        print(
            f"Estimated 1200-step runtime    : "
            f"{est_minutes:.1f} minutes"
        )

    print("=" * 82)


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"wrote {path}",
        flush=True,
    )


# ───────────────────────────── episode loop ────────────────────────────


def run_episode(
    worker,
    seed,
    max_steps,
    chunk_size,
    video,
    sim_host,
    sim_port,
    timing_dir,
    harness=None,
    sim_client=None,
):
    # Optional imports and all harness policy are outside the baseline path.
    from contextlib import nullcontext

    lifecycle = (
        harness.episode(seed=seed, max_steps=max_steps, chunk_size=chunk_size)
        if harness is not None else nullcontext(None)
    )
    with lifecycle as episode_harness:
        return _run_episode(
            worker, seed, max_steps, chunk_size, video, sim_host, sim_port,
            timing_dir, episode_harness, sim_client,
        )


def _run_episode(
    worker,
    seed,
    max_steps,
    chunk_size,
    video,
    sim_host,
    sim_port,
    timing_dir,
    harness=None,
    sim_client=None,
):
    simulator_call = call_sim if sim_client is None else sim_client
    info, reset_rpc_ms = worker.reset()

    print(
        f"Modal reset: {reset_rpc_ms:.1f} ms | "
        f"model chunk limit: "
        f"{info.get('chunk_limit', 'unknown')}",
        flush=True,
    )

    observation, _ = simulator_call(
        sim_host,
        sim_port,
        "/reset",
        {"seed": seed},
    )

    frames = []
    step_rows = []
    chunk_rows = []

    success = False
    global_step = 0
    chunk_id = 0
    episode_start = time.perf_counter()

    while global_step < max_steps:
        images = {
            camera: observation[camera]
            for camera in CAMERAS
        }

        actions, chunk_timing = (
            worker.act_chunk(
                images,
                observation["state"],
                chunk_size,
            )
        )

        # Never execute past the episode horizon.
        remaining = (
            max_steps - global_step
        )

        actions = actions[:remaining]

        actual_chunk_size = len(actions)

        if actual_chunk_size == 0:
            raise RuntimeError(
                "Modal returned an empty action chunk"
            )

        amortized_rpc_ms = (
            chunk_timing["modal_rpc_ms"]
            / actual_chunk_size
        )

        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "start_step": global_step,
                "requested_chunk_size": chunk_size,
                "actual_chunk_size": actual_chunk_size,
                **chunk_timing,
                "amortized_rpc_ms": amortized_rpc_ms,
            }
        )

        print(
            f"[chunk {chunk_id:03d}] "
            f"steps {global_step}-"
            f"{global_step + actual_chunk_size - 1} | "
            f"actions={actual_chunk_size} | "
            f"RPC={chunk_timing['modal_rpc_ms']:.1f} ms | "
            f"cloud={chunk_timing['modal_compute_ms']:.1f} ms | "
            f"inference={chunk_timing['cloud_inference_ms']:.1f} ms | "
            f"overhead≈{chunk_timing['modal_overhead_ms']:.1f} ms | "
            f"amortized={amortized_rpc_ms:.1f} ms/action",
            flush=True,
        )

        for chunk_pos, base_action in enumerate(
            actions
        ):
            step_start = time.perf_counter()

            # Current observation, not the stale chunk-start observation, is
            # passed to the supervisor at every physical control step.
            current_images = {
                camera: observation[camera]
                for camera in CAMERAS
            }

            if video is not None:
                frames.append(
                    current_images[
                        "head_camera"
                    ].copy()
                )

            t0 = time.perf_counter()

            hook_options = (
                {"harness": harness, "chunk_id": chunk_id, "chunk_pos": chunk_pos,
                 "chunk_size": actual_chunk_size}
                if harness is not None else {}
            )
            action = supervise(
                global_step,
                current_images,
                observation["state"],
                base_action,
                **hook_options,
            )

            supervise_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            if harness is not None and harness.aborted:
                # Discard this unexecuted proposal and the rest of its chunk.
                break

            observation, sim_timing = simulator_call(
                sim_host,
                sim_port,
                "/step",
                {
                    "action": np.asarray(
                        action,
                        dtype=float,
                    ).tolist()
                },
            )

            if harness is not None:
                harness.after_action(global_step, action, observation["state"])

            local_step_ms = (
                time.perf_counter()
                - step_start
            ) * 1000.0

            # Amortized cloud latency gives the meaningful per-action cost.
            effective_step_ms = (
                local_step_ms
                + amortized_rpc_ms
            )

            step_rows.append(
                {
                    "step": global_step,
                    "chunk_id": chunk_id,
                    "chunk_pos": chunk_pos,
                    "chunk_size": actual_chunk_size,
                    "amortized_modal_rpc_ms": amortized_rpc_ms,
                    "supervise_ms": supervise_ms,
                    "sim_http_ms": sim_timing["sim_http_ms"],
                    "sim_action_ms": sim_timing["sim_action_ms"],
                    "sim_observation_ms": sim_timing["sim_observation_ms"],
                    "sim_encode_ms": sim_timing["sim_encode_ms"],
                    "local_step_ms": local_step_ms,
                    "effective_step_ms": effective_step_ms,
                }
            )

            global_step += 1

            if bool(
                observation["success"]
            ):
                success = True
                print(
                    f"success at step {global_step - 1}",
                    flush=True,
                )
                break

            if global_step >= max_steps:
                break

        if success or (harness is not None and harness.aborted):
            break

        chunk_id += 1

    if harness is not None and not harness.aborted:
        harness.complete(success=success)

    episode_wall_s = (
        time.perf_counter()
        - episode_start
    )

    print_summary(
        step_rows,
        chunk_rows,
    )

    timing_dir = Path(timing_dir)

    write_csv(
        timing_dir
        / f"episode_{seed}_steps.csv",
        step_rows,
    )

    write_csv(
        timing_dir
        / f"episode_{seed}_chunks.csv",
        chunk_rows,
    )

    if video is not None and frames:
        import imageio.v3 as iio

        video.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        iio.imwrite(
            video,
            np.stack(frames),
            fps=30,
            codec="libx264",
        )

        print(
            f"wrote {video}",
            flush=True,
        )

    print(
        f"episode wall time: "
        f"{episode_wall_s:.2f}s",
        flush=True,
    )

    return success


# ───────────────────────────────── CLI ─────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "role",
        choices=["sim", "policy"],
    )

    parser.add_argument(
        "--robotwin-root",
        default="./RoboTwin",
    )

    parser.add_argument(
        "--task",
        default="hanging_mug",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help=(
            "Number of actions executed per Modal policy call. "
            "Start with 4 or 8; increase only after checking task behavior."
        ),
    )

    parser.add_argument(
        "--video-dir",
        default="videos",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
    )

    parser.add_argument(
        "--timing-dir",
        default="timings_chunked",
    )

    parser.add_argument(
        "--sim-host",
        default=HOST,
    )

    parser.add_argument(
        "--sim-port",
        type=int,
        default=PORT,
    )

    parser.add_argument(
        "--modal-app",
        default="robotwin-policy",
    )

    parser.add_argument(
        "--modal-class",
        default="PolicyWorker",
    )

    parser.add_argument(
        "--harness-config",
        help="Optional YAML for disabled or PASS-only supervision (policy role only)",
    )

    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error(
            "--chunk-size must be >= 1"
        )

    if args.role == "sim":
        if args.harness_config:
            parser.error("--harness-config belongs to the policy role")
        serve_simulator(
            args.robotwin_root,
            args.task,
            args.sim_host,
            args.sim_port,
        )
        return

    harness = None
    if args.harness_config:
        try:
            from harness.config import load_config
            from harness.orchestrator import PassThroughHarness

            config = load_config(args.harness_config)
        except ImportError:
            parser.error("install requirements-harness.txt to use --harness-config")
        except (OSError, ValueError):
            parser.error("invalid or unreadable harness config; check the documented schema")
        if args.task != config.task:
            parser.error("--task must match the harness task (hanging_mug)")
        if config.harness.enabled:
            try:
                harness = PassThroughHarness(config)
            except (OSError, ValueError):
                parser.error("invalid skill registry or unwritable harness output paths")
            print(f"PASS-only harness run: {harness.run_id} | traces: {harness.trace_dir}")

    worker = ModalPolicyClient(
        args.modal_app,
        args.modal_class,
    )

    wins = 0

    for episode in range(
        args.episodes
    ):
        seed = args.seed + episode

        video = (
            None
            if args.no_video
            else Path(args.video_dir)
            / f"episode_{seed}.mp4"
        )

        won = run_episode(
            worker=worker,
            seed=seed,
            max_steps=args.steps,
            chunk_size=args.chunk_size,
            video=video,
            sim_host=args.sim_host,
            sim_port=args.sim_port,
            timing_dir=args.timing_dir,
            harness=harness,
        )

        wins += int(won)

        print(
            f"episode {episode + 1}/"
            f"{args.episodes}: "
            f"{wins} successes so far",
            flush=True,
        )

    print(
        f"\nsuccess rate: "
        f"{100 * wins / args.episodes:.0f}% "
        f"({wins}/{args.episodes})"
    )


if __name__ == "__main__":
    main()
