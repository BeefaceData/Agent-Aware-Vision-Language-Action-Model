"""Capture public RGB-D frames with offline simulator labels for perception evaluation.

Runs the frozen policy alone (no planner, no SAM 3, no Opus) through the same
local simulator and environment client as ``run_skills.py``, so frames cover
reset, grasp, handoff and hang views at no planner cost:

    python -m development.capture.labeled_frames --seed 3 --output data/perception/frames/seed_3
    python -m development.capture.labeled_frames --seed 3 --output data/perception/frames_cam/seed_3 \
      --perception-camera default        # or "x,y,z/tx,ty,tz/fovy" in metres and degrees

Output folder:
  observations/observation_XXXXXXXX.npz  public observations (same format as episode runs)
  labels.jsonl                           PRIVILEGED simulator truth, one line per saved frame
  manifest.json                          seed, settings, RoboTwin commit, camera lens hashes

``labels.jsonl`` holds simulator-only facts (object poses, contacts). It is for
offline evaluators under ``development/`` only and must never be given to the
runtime, the planner or a public perception provider.
"""

import argparse
from functools import partial
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from harness.environment.perception_cameras import describe, parse_camera_spec
from harness.jsonl import append_jsonl
from harness.perception.rgbd import intrinsic_sha256
from development.simulator.rack_geometry import mug_contacts, pose_matrix

CAPTURE_EVERY_ACTIONS = 16
# RoboTwin's own evaluation step limit for hanging_mug (task_config/_eval_step_limit.yml).
MAX_POLICY_ACTIONS = 900
# Matches the full-task runtime config's VLA chunk size.
POLICY_CHUNK_SIZE = 16
# State thresholds for the derived label; they describe the simulator truth,
# not the public perception thresholds being evaluated.
LIFTED_HEIGHT_M = 0.03
TOPPLED_TILT_DEG = 45.0
# Far below the tabletop: the mug fell to the floor (seed 4 came to rest about
# 0.70 m down), where no camera sees it.
OFF_TABLE_DROP_M = 0.20
PRIVILEGED_FILE = "labels.jsonl"


def pose7(pose):
    """Return a sapien-style pose as ``[x, y, z, qw, qx, qy, qz]`` floats."""
    return [float(value) for value in (*pose.p, *pose.q)]


def tilt_deg(initial_pose, current_pose):
    """Angle (degrees) between the mug's reset-time up direction and world up now."""
    initial = pose_matrix(initial_pose)[:3, :3]
    current = pose_matrix(current_pose)[:3, :3]
    up_in_mug_frame = initial.T @ np.array([0.0, 0.0, 1.0])
    up_now = current @ up_in_mug_frame
    return math.degrees(math.acos(float(np.clip(up_now[2], -1.0, 1.0))))


def mug_state(label):
    """Name the mug's situation from simulator truth, for grouping evaluation results.

    ``staged`` means back on the table after an earlier lift (the handoff point).
    ``toppled`` means tilted past ``TOPPLED_TILT_DEG`` with no gripper touching it
    (fallen over); a mug the policy rotates in its grip, as it does for hanging,
    is labelled by its grip instead. ``off_table`` means it fell more than
    ``OFF_TABLE_DROP_M`` below its reset height, off the table.
    """
    contacts = label["contacts"]
    left, right, table = contacts["left_gripper"], contacts["right_gripper"], contacts["table"]
    if label["native_success"]:
        return "hung"
    if label["mug"].get("height_change_m", 0.0) <= -OFF_TABLE_DROP_M:
        return "off_table"
    if label["mug"]["tilt_deg"] > TOPPLED_TILT_DEG and not (left or right):
        return "toppled"
    if contacts["rack"]:
        return "touching_rack"
    if left and right:
        return "held_both"
    if (left or right) and table:
        return "gripped_on_table"
    if left:
        return "held_left"
    if right:
        return "held_right"
    if table:
        return "staged" if label["lifted_before"] else "resting"
    return "unsupported"


def simulator_label(task, *, contact_names, initial_mug_pose, lifted_before, sequence, policy_actions):
    """Read privileged simulator truth for one frame (offline evaluation only)."""
    mug_pose = pose7(task.mug.get_pose())
    rack_name = task.rack.actor.name
    robot = task.robot
    label = {
        "sequence": sequence,
        "policy_actions": policy_actions,
        "mug": {"pose": mug_pose,
                "functional_point": [float(v) for v in task.mug.get_functional_point(0)],
                "tilt_deg": tilt_deg(initial_mug_pose, mug_pose),
                "height_change_m": mug_pose[2] - initial_mug_pose[2]},
        "rack": {"pose": pose7(task.rack.get_pose()),
                 "functional_point": [float(v) for v in task.rack.get_functional_point(0)]},
        "staging_target": [float(v) for v in task.middle_pos[:3]],
        "contacts": {"left_gripper": any(name.startswith("fl_") for name in contact_names),
                     "right_gripper": any(name.startswith("fr_") for name in contact_names),
                     "table": "table" in contact_names,
                     "rack": rack_name in contact_names,
                     "names": sorted(contact_names)},
        "grippers": {"left": float(robot.get_left_gripper_val()),
                     "right": float(robot.get_right_gripper_val())},
        "native_success": bool(task.check_success()),
        "lifted_before": lifted_before,
    }
    label["state"] = mug_state(label)
    return label


def roll_out(environment, task, vla, store, labels_path, *, seed, every, max_actions, chunk_size,
             contacts=mug_contacts):
    """Reset once, run the policy and save a frame plus label every ``every`` actions.

    Returns ``(frames_saved, actions_executed, stop_reason)``.
    """
    observation = environment.reset(seed)
    initial_mug_pose = pose7(task.mug.get_pose())
    lifted_before = False
    saved = []

    def update_lift():
        # Checked after every action so a lift between two saved frames still counts.
        nonlocal lifted_before
        lifted_before = lifted_before or pose7(task.mug.get_pose())[2] - initial_mug_pose[2] >= LIFTED_HEIGHT_M

    def record(observation, actions):
        label = simulator_label(task, contact_names=contacts(task), initial_mug_pose=initial_mug_pose,
                                lifted_before=lifted_before, sequence=observation.sequence,
                                policy_actions=actions)
        reference = store.save(observation)
        append_jsonl(labels_path, {**label, "observation": reference["path"]})
        saved.append(observation.sequence)

    record(observation, 0)
    vla.reset_history()
    actions, stop_reason = 0, "action_limit"
    while actions < max_actions and stop_reason == "action_limit":
        chunk = np.asarray(vla.infer(observation, min(chunk_size, max_actions - actions)))
        if chunk.ndim != 2 or len(chunk) == 0:
            stop_reason = "policy_returned_no_actions"
            break
        for action in chunk:
            observation = environment.step(action)
            actions += 1
            update_lift()
            if environment.native_success:
                stop_reason = "native_success"
                break
            if actions % every == 0:
                record(observation, actions)
    if saved[-1] != observation.sequence:
        record(observation, actions)  # always keep the final state
    return len(saved), actions, stop_reason


def robotwin_commit(robotwin_root):
    try:
        return subprocess.run(["git", "-C", str(robotwin_root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def camera_summary(store_directory):
    """Lens hash and image size per camera from the first saved frame."""
    first = sorted(Path(store_directory).glob("observation_*.npz"))[0]
    from harness.perception.replay import load_public_observation

    observation = load_public_observation(first)
    return {name: {"image_hw": list(np.asarray(observation.images[name]).shape[:2]),
                   "depth": name in (observation.camera_frames or {}),
                   "intrinsic_sha256": (intrinsic_sha256(observation.camera_frames[name])
                                        if name in (observation.camera_frames or {}) else None)}
            for name in sorted(observation.images)}


def capture(robotwin_root, output, seed, *, every=CAPTURE_EVERY_ACTIONS, max_actions=MAX_POLICY_ACTIONS,
            chunk_size=POLICY_CHUNK_SIZE, perception_camera=None):
    """Run one labelled rollout into the new folder ``output`` and return its manifest.

    ``perception_camera``: an optional ``FixedCamera`` rendered into every frame
    for perception only; the policy still receives its three cameras.
    """
    from development.simulator.local_simulator import LocalSimulator
    from harness.environment.local import LocalEnvironmentClient
    from harness.perception.observations import ObservationStore
    from harness.vla.modal_client import ModalVLAClient

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = robotwin_commit(robotwin_root)
    simulator = LocalSimulator(robotwin_root, rgbd_enabled=True,
                               diagnostics=partial(append_jsonl, output / "simulator_timing.jsonl"),
                               perception_cameras=() if perception_camera is None else (perception_camera,))
    try:
        environment = LocalEnvironmentClient(simulator)
        store = ObservationStore(output / "observations")
        frames, actions, stop_reason = roll_out(
            environment, simulator.task, ModalVLAClient(), store, output / PRIVILEGED_FILE,
            seed=seed, every=every, max_actions=max_actions, chunk_size=chunk_size)
    finally:
        simulator.close()
    manifest = {"schema_version": 1, "mode": "labeled_frames", "task": "hanging_mug", "seed": seed,
                "policy": "frozen Neotix policy on Modal (robotwin-policy)", "planner_started": False,
                "capture_every_actions": every, "max_policy_actions": max_actions,
                "policy_chunk_size": chunk_size, "frames_saved": frames, "policy_actions": actions,
                "stop_reason": stop_reason, "robotwin_commit": commit,
                "cameras": camera_summary(output / "observations"),
                "perception_camera": None if perception_camera is None else describe(perception_camera),
                "public_files": ["observations/"], "privileged_files": [PRIVILEGED_FILE],
                "privileged_note": "labels.jsonl is simulator truth for offline evaluation only"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing directories are never overwritten")
    parser.add_argument("--robotwin-root", type=Path, default=root / "RoboTwin")
    parser.add_argument("--every", type=int, default=CAPTURE_EVERY_ACTIONS, help="save a frame every N actions")
    parser.add_argument("--max-actions", type=int, default=MAX_POLICY_ACTIONS)
    parser.add_argument("--chunk-size", type=int, default=POLICY_CHUNK_SIZE)
    parser.add_argument("--perception-camera", metavar="SPEC",
                        help="add a perception-only camera: 'default' or 'x,y,z/tx,ty,tz/fovy' "
                             "(write --perception-camera=SPEC if it starts with '-')")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.every < 1 or args.max_actions < 1 or not 1 <= args.chunk_size <= 64:
        parser.error("seed must be >= 0, --every and --max-actions >= 1, --chunk-size 1-64")
    try:
        camera = None if args.perception_camera is None else parse_camera_spec(args.perception_camera)
    except ValueError as error:
        parser.error(str(error))
    # RoboTwin changes the working directory during start-up, so resolve paths first.
    manifest = capture(args.robotwin_root.resolve(), args.output.resolve(), args.seed, every=args.every,
                       max_actions=args.max_actions, chunk_size=args.chunk_size, perception_camera=camera)
    print(json.dumps({key: manifest[key] for key in ("seed", "frames_saved", "policy_actions", "stop_reason")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
