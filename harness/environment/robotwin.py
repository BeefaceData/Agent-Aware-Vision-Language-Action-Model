"""RoboTwin task construction, observation payloads and the simulator HTTP client.

Shared by the frozen baseline runner (``run_inference_chunked.py``) and the
supervised runtime. Simulator imports happen inside ``build_task`` so importing
this module never requires a RoboTwin checkout.
"""

import io
import os
import sys
import time

import numpy as np

from .layout import ACTION_DIM, CAMERAS


def build_task(robotwin_root: str, task_name: str):
    """Instantiate a RoboTwin task with the demo_clean configuration.

    RoboTwin resolves assets relative to its checkout, so this changes the
    process working directory. Resolve all other paths before calling it.
    """
    root = os.path.abspath(robotwin_root)
    if not os.path.isdir(os.path.join(root, "envs")):
        sys.exit(f"no RoboTwin checkout at {root}")

    sys.path.insert(0, root)
    os.chdir(root)

    import importlib
    import yaml
    from envs import CONFIGS_PATH

    def load(*parts):
        with open(os.path.join(*parts), encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    args = load(CONFIGS_PATH, "demo_clean.yml")
    embodiments = load(CONFIGS_PATH, "_embodiment_config.yml")
    robot = embodiments[args.get("embodiment", ["aloha-agilex"])[0]]["file_path"]
    args["left_robot_file"] = robot
    args["right_robot_file"] = robot
    args["dual_arm_embodied"] = True
    for side in ("left", "right"):
        args[f"{side}_embodiment_config"] = load(args[f"{side}_robot_file"], "config.yml")

    cameras = load(CONFIGS_PATH, "_camera_config.yml")
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head]["h"]
    args["head_camera_w"] = cameras[head]["w"]
    args["render_freq"] = 0
    args["task_name"] = task_name
    args["task_config"] = "demo_clean"

    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)(), args


def observation_of(task, blank, success, perception_cameras=None):
    """Return the public RGB (+optional RGB-D) payload, 14-DoF state and success flag.

    ``perception_cameras`` maps names to fixed cameras added by
    ``perception_cameras.add_fixed_camera``; they are rendered with depth after
    RoboTwin's own cameras.
    """
    raw = task.get_obs()
    cameras = raw.get("observation", {})
    out = {}

    for camera in CAMERAS:
        camera_data = cameras.get(camera) or {}
        image = camera_data.get("rgb")
        if image is None:
            out[camera] = blank
            continue
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        out[camera] = np.ascontiguousarray(image[..., :3])

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

    if perception_cameras:
        from .perception_cameras import fixed_camera_payload
        for name, camera in perception_cameras.items():
            out.update(fixed_camera_payload(name, camera))

    vector = (raw.get("joint_action") or {}).get("vector")
    joints = np.zeros(ACTION_DIM, dtype=np.float32)
    if vector is not None:
        flat = np.asarray(vector, dtype=np.float32).ravel()
        if flat.size >= ACTION_DIM:
            joints = flat[:ACTION_DIM]

    out["state"] = joints
    out["success"] = np.bool_(success)
    return out


def call_sim(host: str, port: int, path: str, payload: dict):
    """POST one request to the simulator server; return (observation, timing_ms)."""
    import requests

    started = time.perf_counter()
    reply = requests.post(f"http://{host}:{port}{path}", json=payload, timeout=600)
    http_ms = (time.perf_counter() - started) * 1000.0
    reply.raise_for_status()

    with np.load(io.BytesIO(reply.content)) as data:
        observation = {key: data[key] for key in data.files}

    def header_ms(name):
        return float(reply.headers.get(name, "nan"))

    timing = {
        "sim_http_ms": http_ms,
        "sim_action_ms": header_ms("X-Sim-Action-Ms"),
        "sim_observation_ms": header_ms("X-Sim-Observation-Ms"),
        "sim_encode_ms": header_ms("X-Sim-Encode-Ms"),
    }
    return observation, timing
