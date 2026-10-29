"""Compare SAM 3 mug prompts and score thresholds on captured frames; development only.

The runtime asks SAM 3 for "mug" and keeps scores of at least 0.40. It misses
mugs a gripper has turned on their side. This probe asks the deployed worker's
``detect_mug_prompt_variants`` for several prompts with scores down to 0.05,
once per frame and camera, for the frames where the mug is held, gripped or
tilted, and stores the answers beside the frames (Modal GPU, one job at a time):

    python -m development.replays.sam3_mug_probe --frames data/perception/frames_cam/seed_1 --query

Then, offline and as often as needed, it grounds every returned mask with the
frame's depth and checks it against where the simulator says the mug is:

    python -m development.replays.sam3_mug_probe \\
      --frames data/perception/frames_cam/seed_0 --frames data/perception/frames_cam/seed_1 \\
      --runtime-report data/perception/reports/with_camera/frames.jsonl

For each prompt and threshold it reports how many frames the runtime missed it
would find, and how many frame-camera answers would ground something else.
Labels are read only to score; nothing here reaches runtime perception.
"""

import argparse
import base64
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from development.capture.labeled_frames import PRIVILEGED_FILE, mug_state
from development.replays.sam3_cache import encode_png
from harness.perception.replay import load_public_observation

PROBE_FILE = "sam3_mug_probe.json"
PROBE_STATES = ("gripped_on_table", "held_left", "held_right", "held_both", "touching_rack", "hung",
                "toppled")
THRESHOLDS = (0.10, 0.20, 0.30, 0.40)
# A mask grounds the mug when the median of its visible surface lies this close
# to the mug's true origin. The runtime "mug" answers at >= 0.40 sit at a median
# 0.06 m (the origin is the mug's base centre, the visible surface its side).
HIT_RADIUS_M = 0.10


def probe_labels(frames_dir):
    labels = [json.loads(line) for line in (Path(frames_dir) / PRIVILEGED_FILE).read_text().splitlines()]
    return sorted((label for label in labels if mug_state(label) in PROBE_STATES),
                  key=lambda label: label["sequence"])


def query(frames_dir, worker, *, log=print):
    """Ask ``worker`` once per probed frame and camera; return the probe document."""
    frames = {}
    labels = probe_labels(frames_dir)
    for index, label in enumerate(labels, start=1):
        observation = load_public_observation(Path(frames_dir) / label["observation"])
        frames[str(label["sequence"])] = {
            camera: [{"prompt": variant["prompt"],
                      "detections": [{"confidence": item["confidence"],
                                      "mask_png": base64.b64encode(item["mask_png"]).decode("ascii")}
                                     for item in variant["detections"]]}
                     for variant in worker.detect_mug_prompt_variants.remote(
                         encode_png(image), observation_sequence=label["sequence"], camera=camera)["variants"]]
            for camera, image in observation.images.items()}
        log(f"  {index}/{len(labels)} frames")
    return {"schema_version": 1, "frames": frames}


def mask_median_point(mask, frame):
    """World median of the mask's valid-depth pixels, or None when there are none."""
    ys, xs = np.nonzero(mask)
    depth = frame.depth_m[ys, xs].astype(np.float64)
    valid = depth > 0
    if not valid.any():
        return None
    pixels = np.stack((xs[valid], ys[valid], np.ones(int(valid.sum()))), axis=1).astype(np.float64)
    camera_xyz = (pixels @ np.linalg.inv(frame.intrinsic).T) * depth[valid, None]
    world = (camera_xyz - frame.world_to_camera[:3, 3]) @ frame.world_to_camera[:3, :3]
    return np.median(world, axis=0)


def grounded_answers(frames_dir):
    """Yield (sequence, state, camera, prompt, confidence, distance_m) for every probed mask."""
    from PIL import Image

    probe = json.loads((Path(frames_dir) / PROBE_FILE).read_text())
    for label in probe_labels(frames_dir):
        answers = probe["frames"].get(str(label["sequence"]))
        if answers is None:
            continue
        observation = load_public_observation(Path(frames_dir) / label["observation"])
        truth = np.asarray(label["mug"]["pose"][:3], dtype=float)
        for camera, variants in answers.items():
            frame = observation.camera_frames[camera]
            for variant in variants:
                for item in variant["detections"]:
                    with Image.open(BytesIO(base64.b64decode(item["mask_png"]))) as opened:
                        mask = np.asarray(opened) > 0
                    point = mask_median_point(mask, frame)
                    distance = None if point is None else float(np.linalg.norm(point - truth))
                    yield (label["sequence"], mug_state(label), camera, variant["prompt"],
                           item["confidence"], distance)


def score(answers, missed):
    """Per prompt and threshold: missed frames found, all frames found, wrong frame-camera answers.

    ``answers``: (capture, sequence, state, camera, prompt, confidence, distance_m).
    ``missed``: the (capture, sequence) pairs the runtime did not ground.
    Each frame-camera pair is judged by its best mask at or above the threshold.
    """
    best = {}
    for capture, sequence, _state, camera, prompt, confidence, distance in answers:
        for threshold in THRESHOLDS:
            if confidence < threshold:
                continue
            key = (prompt, threshold, capture, sequence, camera)
            if key not in best or confidence > best[key][0]:
                best[key] = (confidence, distance is not None and distance <= HIT_RADIUS_M)
    found, wrong = defaultdict(set), defaultdict(int)
    for (prompt, threshold, capture, sequence, _camera), (_confidence, hit) in best.items():
        if hit:
            found[(prompt, threshold)].add((capture, sequence))
        else:
            wrong[(prompt, threshold)] += 1
    return {key: {"found_missed": len(found[key] & missed), "found": len(found[key]), "wrong": wrong[key]}
            for key in sorted(set(found) | set(wrong))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, action="append", required=True,
                        help="labeled_frames capture folder; repeatable")
    parser.add_argument("--query", action="store_true", help="query the deployed SAM 3 worker (Modal GPU)")
    parser.add_argument("--runtime-report", type=Path,
                        help="perception_evaluation frames.jsonl marking the frames the runtime missed")
    args = parser.parse_args(argv)
    if args.query:
        import modal

        worker = modal.Cls.from_name("robotwin-perception-sam3", "Sam3PerceptionWorker")()
        for folder in args.frames:
            target = folder / PROBE_FILE
            if target.exists():
                print(f"{target} exists; skipping")
                continue
            print(f"Probing SAM 3 mug prompts for {folder} ...")
            target.write_text(json.dumps(query(folder, worker)) + "\n")
        return 0
    missed = set()
    if args.runtime_report:
        for line in args.runtime_report.read_text().splitlines():
            record = json.loads(line)
            if record["state"] in PROBE_STATES and record["mug_status"] != "grounded":
                missed.add((record["capture"], record["sequence"]))
    answers, probed = [], set()
    for folder in args.frames:
        for answer in grounded_answers(folder):
            answers.append((folder.name, *answer))
            probed.add((folder.name, answer[0]))
    reference = [item[6] for item in answers
                 if item[4] == "mug" and item[5] >= 0.40 and item[6] is not None]
    print(f"{len(probed)} probed frames, {len(missed & probed)} missed by the runtime")
    if reference:
        print("'mug' >= 0.40 distance to the true mug (m): median "
              f"{np.median(reference):.3f}, p90 {np.percentile(reference, 90):.3f}")
    print("| Prompt | Threshold | Runtime misses found | Frames found | Wrong answers |")
    print("|---|---|---|---|---|")
    for (prompt, threshold), row in score(answers, missed & probed).items():
        print(f"| {prompt} | {threshold:.2f} | {row['found_missed']} | {row['found']} | {row['wrong']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
