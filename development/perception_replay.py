"""Replay reviewed detector masks over saved public observations; no simulator.

Example (CPU only):
  python -m development.perception_replay --observations DIR \
    --detections masks.json --output NEW_REPORT.json

RGB-only observation archives are recorded as RGB-D unavailable. This utility
does not run a learned detector, call a planner, or execute robot motion.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from harness.perception.anchors import AnchorRegistry
from harness.perception.grounding import RGBDPerceptionProvider
from harness.perception.replay import RecordedDetectionAdapter, load_public_observation


def replay(observations_dir, detections_path):
    observations_dir = Path(observations_dir).resolve()
    detections_path = Path(detections_path).resolve()
    if not observations_dir.is_dir() or not detections_path.is_file():
        raise ValueError("replay_input_missing")
    archives = sorted(observations_dir.glob("observation_*.npz"))
    if not archives or len(archives) > 10000:
        raise ValueError("invalid_observation_archive_count")
    detector = RecordedDetectionAdapter(detections_path)
    provider = RGBDPerceptionProvider(detector, AnchorRegistry())
    rows = []
    previous_sequence = -1
    input_hashes = {}
    for path in archives:
        observation = load_public_observation(path)
        if observation.sequence <= previous_sequence:
            raise ValueError("observation_sequences_not_increasing")
        previous_sequence = observation.sequence
        input_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        result = provider.observe(observation)
        rows.append({
            "sequence": observation.sequence,
            "rgbd_available": bool(observation.camera_frames),
            "status": result.status,
            "objects": {name: {
                "position_m": item.position,
                "confidence": item.confidence,
                "camera": item.camera,
                "valid_depth_fraction": item.valid_depth_fraction,
            } for name, item in result.objects.items()},
            "evidence": [provider.predicate_evidence(name, observation)
                         for name in ("mug_visible", "rack_visible", "mug_near_rack",
                                      "mug_lifted", "release_ready")],
        })
    detector_hash = sha256(detections_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "mode": "offline_perception_replay",
        "robot_motion": False,
        "simulator_started": False,
        "detector_model_executed": False,
        "detector_annotation_sha256": detector_hash,
        "observation_hashes": input_hashes,
        "frames": rows,
        "summary": {
            "observation_count": len(rows),
            "rgbd_frame_count": sum(row["rgbd_available"] for row in rows),
            "grounded_mug_frames": sum(row["status"]["mug"] == "grounded" for row in rows),
            "grounded_rack_frames": sum(row["status"]["rack"] == "grounded" for row in rows),
            "release_ready_frames": sum(
                any(item["name"] == "release_ready" and item["value"] is True
                    for item in row["evidence"]) for row in rows),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True,
                        help="reviewed version-1 mask annotation JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="new report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        try:
            report = replay(args.observations, args.detections)
        except Exception as exc:
            report = {"schema_version": 1, "mode": "offline_perception_replay",
                      "robot_motion": False, "simulator_started": False,
                      "error_type": type(exc).__name__, "error": str(exc)}
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report.get("summary", report), indent=2))
    return 1 if "error_type" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
