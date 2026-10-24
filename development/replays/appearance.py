"""Replay reviewed RGB-D mug masks into public appearance descriptors.

This is an offline measurement tool: it starts no simulator, model, planner,
or robot motion. The input masks must be reviewed annotations linked to their
source observation hashes.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from harness.perception.grounding import RGBDPerceptionProvider
from harness.perception.replay import RecordedDetectionAdapter, load_public_observation


MAX_ANNOTATION_BYTES = 2_000_000


def replay(observations_dir, detections_path):
    observations_dir, detections_path = Path(observations_dir), Path(detections_path)
    raw = detections_path.read_bytes()
    if len(raw) > MAX_ANNOTATION_BYTES:
        raise ValueError("appearance_annotations_too_large")
    annotations = json.loads(raw)
    detector = RecordedDetectionAdapter(annotations)
    frames = annotations.get("frames", [])
    if not frames:
        raise ValueError("appearance_annotations_empty")

    provider = RGBDPerceptionProvider(detector)
    rows, observation_hashes = [], {}
    previous_sequence = -1
    for frame in frames:
        sequence = frame["sequence"]
        if sequence <= previous_sequence:
            raise ValueError("appearance_sequences_not_increasing")
        previous_sequence = sequence
        path = observations_dir / f"observation_{sequence:08d}.npz"
        if not path.is_file():
            raise ValueError("appearance_observation_missing")
        digest = sha256(path.read_bytes()).hexdigest()
        declared = annotations.get("source_sha256", {}).get(path.name)
        if declared is not None and declared != digest:
            raise ValueError("appearance_observation_hash_mismatch")
        observation = load_public_observation(path)
        if observation.sequence != sequence:
            raise ValueError("appearance_observation_sequence_mismatch")
        summary = provider.planner_summary(observation)
        # Keep only the public descriptors. World positions, calibration, and
        # sampled points stay inside the perception provider.
        rows.append({"sequence": sequence, "source_observation": path.name,
                     "visual_context": summary})
        observation_hashes[path.name] = digest

    return {"schema_version": 1, "mode": "offline_mug_appearance_replay",
        "robot_motion": False, "simulator_started": False,
        "detector_model_executed": False,
        "annotation_sha256": sha256(raw).hexdigest(),
        "observation_hashes": observation_hashes, "frames": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True,
                        help="reviewed mask annotation JSON with optional source hashes")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = replay(args.observations, args.detections)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"frames": len(report["frames"]),
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
