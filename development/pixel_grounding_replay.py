"""Replay reviewed planner pixels through saved public RGB-D observations.

CPU-only: no simulator, detector, planner, or robot motion is used. Annotation
JSON schema: {"schema_version": 1, "points": [{"sequence": 7,
"camera": "head_camera", "uv": [0.5, 0.5]}]}.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from harness.perception.replay import load_public_observation
from harness.perception.preparation import PlannerPixelAnchorPreparer
from harness.perception.anchors import AnchorRegistry
from harness.skills.schemas import GroundingPixel, SkillCommand


def replay(observations_dir, annotations_path):
    observations_dir = Path(observations_dir).resolve()
    annotations_path = Path(annotations_path).resolve()
    if not observations_dir.is_dir() or not annotations_path.is_file():
        raise ValueError("replay_input_missing")
    try:
        annotations = json.loads(annotations_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise ValueError("invalid_pixel_annotations") from None
    if (not isinstance(annotations, dict) or annotations.get("schema_version") != 1
            or not isinstance(annotations.get("points"), list)
            or not annotations["points"] or len(annotations["points"]) > 10000):
        raise ValueError("invalid_pixel_annotations")

    by_sequence = {}
    for path in observations_dir.glob("observation_*.npz"):
        try:
            sequence = int(path.stem.removeprefix("observation_"))
        except ValueError:
            continue
        if sequence in by_sequence:
            raise ValueError("duplicate_observation_sequence")
        by_sequence[sequence] = path

    rows, hashes, seen = [], {}, set()
    for item in annotations["points"]:
        if not isinstance(item, dict) or type(item.get("sequence")) is not int:
            raise ValueError("invalid_pixel_annotation_point")
        sequence = item["sequence"]
        if sequence < 0 or sequence in seen:
            raise ValueError("duplicate_or_invalid_pixel_annotation_sequence")
        seen.add(sequence)
        pixel = GroundingPixel(camera=item.get("camera"), uv=item.get("uv"))
        path = by_sequence.get(sequence)
        if path is None:
            raise ValueError("annotated_observation_missing")
        observation = load_public_observation(path)
        frame = (observation.camera_frames or {}).get(pixel.camera)
        if frame is None:
            raise ValueError("rgbd_camera_frame_unavailable")
        height, width = frame.depth_m.shape
        u, v = pixel.uv[0] * (width - 1), pixel.uv[1] * (height - 1)
        point = frame.unproject(u, v, radius=1)
        local_depth_gap = frame.local_depth_gap(
            u, v, search_radius=PlannerPixelAnchorPreparer.DEPTH_SEARCH_RADIUS_PX)
        x, y = int(round(u)), int(round(v))
        roi = frame.depth_m[max(0, y-1):min(height, y+2),
                            max(0, x-1):min(width, x+2)]
        valid_depth = roi[(roi > 0) & np.isfinite(roi)]
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        rows.append({"sequence": sequence, "camera": pixel.camera,
                     "uv": list(pixel.uv), "sampled_depth_m": float(np.median(valid_depth)),
                     "world_xyz_m": list(point),
                     "local_depth_gap_m": local_depth_gap,
                     "foreground_supported": local_depth_gap <=
                        PlannerPixelAnchorPreparer.MAX_LOCAL_DEPTH_GAP_M})

    return {"schema_version": 1, "mode": "offline_pixel_grounding_replay",
            "robot_motion": False, "simulator_started": False,
            "planner_or_detector_executed": False,
            "foreground_check": {
                "search_radius_px": PlannerPixelAnchorPreparer.DEPTH_SEARCH_RADIUS_PX,
                "max_depth_gap_m": PlannerPixelAnchorPreparer.MAX_LOCAL_DEPTH_GAP_M},
            "annotation_sha256": sha256(annotations_path.read_bytes()).hexdigest(),
            "observation_hashes": hashes, "points": rows,
            "summary": {"annotated_points": len(rows),
                        "grounded_points": len(rows),
                        "foreground_supported_points": sum(row["foreground_supported"]
                                                            for row in rows)}}


def replay_reselection(observations_dir, scenarios_path):
    """Exercise the real preparation guard over an ordered pixel reselection.

    Scenario JSON: ``{"schema_version":1,"scenarios":[{"name":"...",
    "sequence":0,"arm":"left","measured_tcp_pose":[x,y,z,qx,qy,qz,qw],
    "pixels":[{"camera":"head_camera","uv":[u,v]}, ...]}]}``.
    Up to three pixels are accepted per scenario: initial selection plus two retries.
    The real anchor preparer runs, but there is no planner, simulator, skill
    executor, or robot motion.
    """
    observations_dir = Path(observations_dir).resolve()
    scenarios_path = Path(scenarios_path).resolve()
    if not observations_dir.is_dir() or not scenarios_path.is_file():
        raise ValueError("replay_input_missing")
    try:
        data = json.loads(scenarios_path.read_text())
    except (OSError, json.JSONDecodeError):
        raise ValueError("invalid_reselection_scenarios") from None
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or not isinstance(data.get("scenarios"), list)
            or not data["scenarios"] or len(data["scenarios"]) > 100):
        raise ValueError("invalid_reselection_scenarios")
    rows, hashes = [], {}
    for scenario in data["scenarios"]:
        if (not isinstance(scenario, dict) or type(scenario.get("sequence")) is not int
                or scenario.get("arm") not in ("left", "right")
                or not isinstance(scenario.get("name"), str)
                or not isinstance(scenario.get("pixels"), list)
                or not scenario["pixels"] or len(scenario["pixels"]) > 3):
            raise ValueError("invalid_reselection_scenario")
        sequence = scenario["sequence"]
        path = observations_dir / f"observation_{sequence:08d}.npz"
        if not path.is_file():
            raise ValueError("annotated_observation_missing")
        observation = load_public_observation(path)
        if len(scenario["measured_tcp_pose"]) != 7:
            raise ValueError("invalid_reselection_tcp_pose")
        environment = SimpleNamespace(arm_state=lambda _arm, pose=scenario["measured_tcp_pose"]:
                                     {"pose": pose})
        anchors = AnchorRegistry()
        prepare = PlannerPixelAnchorPreparer(environment, anchors)
        selections = []
        for index, point in enumerate(scenario["pixels"]):
            if not isinstance(point, dict):
                raise ValueError("invalid_reselection_pixel")
            pixel = GroundingPixel(camera=point.get("camera"), uv=point.get("uv"))
            command = SkillCommand.model_validate({"action": "move_to", "arm": scenario["arm"],
                "parameters": {"target": "mug_pregrasp", "grounding": pixel.model_dump()}})
            try:
                anchor = prepare(command, observation)
                outcome = "accepted"
                row = {"selection": index + 1, "outcome": outcome,
                       "anchor_pose": list(anchor.pose), "anchor_source": anchor.source}
            except ValueError as exc:
                if str(exc) != "grounding_pixel_behind_local_surface":
                    raise
                outcome = "rejected_depth_inconsistent"
                row = {"selection": index + 1, "outcome": outcome}
            selections.append(row)
            if outcome == "accepted":
                break
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        rows.append({"name": scenario["name"], "sequence": sequence,
                     "selections": selections,
                     "reselection_budget_exhausted": len(selections) == 3 and
                         all(item["outcome"] != "accepted" for item in selections)})
    return {"schema_version": 1, "mode": "offline_pixel_reselection_replay",
            "robot_motion": False, "simulator_started": False,
            "planner_or_detector_executed": False, "skill_executor_executed": False,
            "reselection_budget": 2,
            "foreground_check": {"search_radius_px": PlannerPixelAnchorPreparer.DEPTH_SEARCH_RADIUS_PX,
                                  "max_depth_gap_m": PlannerPixelAnchorPreparer.MAX_LOCAL_DEPTH_GAP_M},
            "scenario_sha256": sha256(scenarios_path.read_bytes()).hexdigest(),
            "observation_hashes": hashes, "scenarios": rows,
            "summary": {"scenarios": len(rows),
                        "accepted_after_reselection": sum(any(item["outcome"] == "accepted"
                            for item in row["selections"][1:]) for row in rows),
                        "budget_exhausted": sum(row["reselection_budget_exhausted"] for row in rows)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--annotations", type=Path)
    inputs.add_argument("--reselection-scenarios", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="new report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = (replay_reselection(args.observations, args.reselection_scenarios)
                  if args.reselection_scenarios else replay(args.observations, args.annotations))
    except Exception as exc:
        report = {"schema_version": 1, "mode": "offline_pixel_reselection_replay"
                      if args.reselection_scenarios else "offline_pixel_grounding_replay",
                  "robot_motion": False, "simulator_started": False,
                  "error_type": type(exc).__name__, "error": str(exc)}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report.get("summary", report), indent=2))
    return 1 if "error_type" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
