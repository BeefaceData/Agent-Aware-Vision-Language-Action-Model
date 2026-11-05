"""Score public RGB-D perception against offline simulator labels; no simulator, no Modal.

Replays captured frames (``development.capture.labeled_frames``) through the
runtime perception provider, with SAM 3 answers served from each capture's
``sam3_cache.json`` (``development.replays.sam3_cache``), then compares the
public evidence with the privileged labels:

    python -m development.replays.perception_evaluation \\
      --frames data/perception/frames/seed_0 --frames data/perception/frames/seed_1 \\
      --output data/perception/reports/baseline

Pass ``--ignore-camera perception_camera`` to score the same frames as if that
camera did not exist, for a with/without comparison.

Reports detection rate and 3D error per mug state, mug pose accuracy (axis,
base position, upright/toppled calls), and for each predicate a confusion table
with precision, recall, unknown rate and false-true count.
Labels never reach the provider; they are only read for scoring and to tell
the detector which phase the runtime planner would be in.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import statistics

import numpy as np

from harness.perception.replay import load_public_observation
from development.capture.labeled_frames import LIFTED_HEIGHT_M, PRIVILEGED_FILE, TOPPLED_TILT_DEG, mug_state
from development.replays.sam3_cache import CACHE_FILE, CachedSam3Worker
from development.simulator.rack_geometry import pose_matrix
from harness.environment.layout import PERCEPTION_CAMERAS

EVALUATED_PREDICATES = ("mug_lifted", "release_ready", "mug_staged_on_table", "right_mug_lifted")
# The full-task configs ground the rack from the right wrist camera.
RACK_GROUNDING_CAMERA = "right_camera"
# RoboTwin mug meshes stand along their +y axis; hanging_mug.py spawns them
# rotated so that +y points up. The mesh origin is at base height but is the
# centre of a box that includes the handle (towards -z), so the body's axis
# sits 1.6-2.6 cm away from it along +z.
MUG_AXIS_IN_MODEL = np.array((0.0, 1.0, 0.0))
# Per hanging-task model (ids 0-9), measured from 039_mug meshes with scale
# applied: handle functional point in the model frame, and body axis offset
# along model z.
MUG_MODELS = {
    0: ((0.0, 0.0718, -0.0480), 0.0221),
    1: ((0.0, 0.0584, -0.0390), 0.0179),
    2: ((0.0, 0.0649, -0.0444), 0.0205),
    3: ((0.0, 0.0423, -0.0300), 0.0200),
    4: ((0.0, 0.0504, -0.0389), 0.0206),
    5: ((0.0, 0.0663, -0.0552), 0.0162),
    6: ((0.0, 0.0619, -0.0377), 0.0188),
    7: ((0.0, 0.0428, -0.0347), 0.0195),
    8: ((0.0, 0.0489, -0.0402), 0.0174),
    9: ((0.0, 0.0715, -0.0350), 0.0261),
}
# A confident pose further off than this would send the arm to the wrong place.
POSE_WRONG_AXIS_DEG = 20.0
POSE_WRONG_BASE_M = 0.03


def lifted_so_far(label):
    return label["lifted_before"] or label["mug"]["height_change_m"] >= LIFTED_HEIGHT_M


def detector_phase(label):
    """The phase the runtime planner would report, approximated from simulator truth.

    It only chooses which wrist view the detector tries when the head camera
    misses the mug; it never enters the evidence being scored.
    """
    if not lifted_so_far(label):
        return "grasp"
    if label["state"] in ("held_right", "touching_rack", "hung"):
        return "hang"
    return "handoff"


def right_gripper_lifting(label):
    """The right gripper alone holds the mug off the table in this frame."""
    contacts = label["contacts"]
    return (contacts["right_gripper"] and not contacts["left_gripper"] and not contacts["table"]
            and label["mug"]["height_change_m"] >= LIFTED_HEIGHT_M)


def truth_predicates(label, *, right_lifted_before=False):
    """What each public predicate should answer, by its documented meaning.

    - mug_lifted: the mug has been lifted at some point (the public signal latches).
    - release_ready: the mug is upright and supported by the table, so opening a
      gripper would not drop it.
    - mug_staged_on_table: release_ready after an earlier lift (the handoff point).
    - right_mug_lifted: the right gripper alone has held the mug off the table at
      some point. It is the handoff milestone that moves the planner to HANG for
      the rest of the episode, so the public signal latches like mug_lifted.
      ``right_lifted_before`` says whether an earlier frame showed it.
    """
    contacts = label["contacts"]
    upright = label["mug"]["tilt_deg"] <= TOPPLED_TILT_DEG
    supported = contacts["table"] and upright
    return {"mug_lifted": lifted_so_far(label),
            "release_ready": supported,
            "mug_staged_on_table": supported and label["lifted_before"],
            "right_mug_lifted": right_lifted_before or right_gripper_lifting(label)}


def mug_error(grounded, label):
    """Horizontal and vertical offset of the grounded mug point from the true base centre.

    The grounded point is the median of visible surface points, so it sits
    about half a mug height above the base and is pulled towards the camera.
    """
    estimate = np.asarray(grounded.position, dtype=float)
    truth = true_mug_base(label)
    return float(np.linalg.norm(estimate[:2] - truth[:2])), float(estimate[2] - truth[2])


def true_mug_axis(label):
    """Unit vector from the mug's base towards its opening, from the label pose."""
    return pose_matrix(label["mug"]["pose"])[:3, :3] @ MUG_AXIS_IN_MODEL


def mug_model_id(label):
    """The episode's mug model: recorded by newer captures, else matched by its handle point."""
    if "model_id" in label["mug"]:
        return label["mug"]["model_id"]
    transform = pose_matrix(label["mug"]["pose"])
    handle = transform[:3, :3].T @ (np.asarray(label["mug"]["functional_point"][:3]) - transform[:3, 3])
    distances = {model: float(np.linalg.norm(handle - point)) for model, (point, _axis) in MUG_MODELS.items()}
    model = min(distances, key=distances.get)
    if distances[model] > 0.002:
        raise ValueError("unknown_mug_model")
    return model


def true_mug_base(label):
    """Centre of the mug body's base: the label origin moved onto the body axis."""
    transform = pose_matrix(label["mug"]["pose"])
    body_offset = MUG_MODELS[mug_model_id(label)][1]
    return transform[:3, 3] + transform[:3, :3] @ np.array((0.0, 0.0, body_offset))


def tilt_from_up_deg(axis):
    return float(np.degrees(np.arccos(min(1.0, abs(float(axis[2]))))))


def pose_errors(pose, label):
    """Compare a fitted mug pose with the label; None fields where the pose has no answer."""
    truth_axis = true_mug_axis(label)
    errors = {"pose_tilt_deg": None, "true_tilt_deg": tilt_from_up_deg(truth_axis),
              "axis_error_deg": None, "base_error_m": None, "base_xy_error_m": None, "base_z_error_m": None}
    if pose is None:
        return errors
    axis = np.asarray(pose.axis, dtype=float)
    # Without a known base the fitted axis has no direction, so compare lines.
    alignment = axis @ truth_axis if pose.base_known else abs(axis @ truth_axis)
    errors.update(pose_tilt_deg=pose.tilt_deg(),
                  axis_error_deg=float(np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))))
    if pose.base_known:
        offset = np.asarray(pose.position) - true_mug_base(label)
        errors.update(base_error_m=float(np.linalg.norm(offset)), base_xy_error_m=float(np.linalg.norm(offset[:2])),
                      base_z_error_m=float(offset[2]))
    return errors


def without_cameras(observation, cameras):
    """The same public observation with the named perception-only cameras removed."""
    if not cameras:
        return observation
    frames = observation.camera_frames
    return replace(observation,
                   images={name: image for name, image in observation.images.items() if name not in cameras},
                   camera_frames=None if frames is None else
                   {name: frame for name, frame in frames.items() if name not in cameras} or None)


def evaluate_capture(frames_dir, provider, *, ignore_cameras=()):
    """Replay one capture in sequence order; return one record per labelled frame."""
    frames_dir = Path(frames_dir)
    labels = sorted((json.loads(line) for line in (frames_dir / PRIVILEGED_FILE).read_text().splitlines()),
                    key=lambda item: item["sequence"])
    # Recompute the derived state from the stored facts, so captures made under
    # an earlier rule are scored with the current one.
    labels = [{**label, "state": mug_state(label)} for label in labels]
    records = []
    right_lifted_before = False
    for label in labels:
        observation = without_cameras(load_public_observation(frames_dir / label["observation"]),
                                      ignore_cameras)
        phase = detector_phase(label)
        provider.set_task_phase(phase)
        result = provider.observe(observation)
        mug = result.objects.get("mug")
        record = {"capture": frames_dir.name, "sequence": label["sequence"],
                  "policy_actions": label["policy_actions"], "state": label["state"], "phase": phase,
                  "mug_status": result.status.get("mug"), "mug_camera": None if mug is None else mug.camera,
                  "mug_xy_error_m": None, "mug_z_offset_m": None, "predicates": {}}
        if mug is not None:
            record["mug_xy_error_m"], record["mug_z_offset_m"] = mug_error(mug, label)
        # Only trusted poses are answers; an untrusted one can only veto a release.
        trusted = result.mug_pose if result.mug_pose is not None and result.mug_pose.trusted else None
        record["pose"] = {"pose_reason": result.mug_pose_reason, **pose_errors(trusted, label)}
        truth = truth_predicates(label, right_lifted_before=right_lifted_before)
        right_lifted_before = truth["right_mug_lifted"]
        for name in EVALUATED_PREDICATES:
            evidence = provider.predicate_evidence(name, observation)
            record["predicates"][name] = {"public": evidence["value"], "reason": evidence["reason"],
                                          "truth": truth[name]}
        records.append(record)
    return records


def _rate(numerator, denominator):
    return None if denominator == 0 else round(numerator / denominator, 3)


def _distribution(values):
    if not values:
        return None
    return {"median": round(statistics.median(values), 4), "mean": round(statistics.fmean(values), 4),
            "max": round(max(values), 4)}


def summarize(records):
    """Aggregate per-frame records into per-state detection and per-predicate scores."""
    by_state = defaultdict(list)
    for record in records:
        by_state[record["state"]].append(record)
    states = {}
    for state, rows in sorted(by_state.items()):
        grounded = [row for row in rows if row["mug_status"] == "grounded"]
        states[state] = {"frames": len(rows), "mug_grounded": len(grounded),
                         "mug_detection_rate": _rate(len(grounded), len(rows)),
                         "mug_camera": dict(Counter(row["mug_camera"] for row in grounded)),
                         "mug_miss_reasons": dict(Counter(row["mug_status"] for row in rows
                                                          if row["mug_status"] != "grounded")),
                         "mug_xy_error_m": _distribution([row["mug_xy_error_m"] for row in grounded]),
                         "mug_z_offset_m": _distribution([row["mug_z_offset_m"] for row in grounded])}

    predicates = {}
    for name in EVALUATED_PREDICATES:
        table = {"truth_true": Counter(), "truth_false": Counter()}
        for record in records:
            item = record["predicates"][name]
            answer = "unknown" if item["public"] is None else str(bool(item["public"])).lower()
            table["truth_true" if item["truth"] else "truth_false"][answer] += 1
        true_positive = table["truth_true"]["true"]
        false_true = table["truth_false"]["true"]
        answered_true = true_positive + false_true
        total = sum(table["truth_true"].values()) + sum(table["truth_false"].values())
        unknown = table["truth_true"]["unknown"] + table["truth_false"]["unknown"]
        predicates[name] = {"confusion": {key: dict(value) for key, value in table.items()},
                            "precision": _rate(true_positive, answered_true),
                            "recall": _rate(true_positive, sum(table["truth_true"].values())),
                            "unknown_rate": _rate(unknown, total),
                            "false_true": false_true}

    poses = {state: pose_summary(rows) for state, rows in sorted(by_state.items())}
    toppled = [record for record in records if record["state"] == "toppled"]
    toppled_supported = [record for record in toppled
                         if record["predicates"]["release_ready"]["public"] is True
                         or record["predicates"]["mug_staged_on_table"]["public"] is True]
    return {"frames": len(records), "captures": sorted({record["capture"] for record in records}),
            "by_state": states, "pose_by_state": poses, "pose_overall": pose_summary(records),
            "predicates": predicates,
            "toppled_reported_as_supported": {"toppled_frames": len(toppled),
                                              "reported_supported": len(toppled_supported)}}


def pose_summary(rows):
    """Coverage, accuracy and wrong confident answers of the fitted mug pose."""
    fitted = [row["pose"] for row in rows if row["pose"]["axis_error_deg"] is not None]
    based = [pose for pose in fitted if pose["base_error_m"] is not None]
    wrong = [pose for pose in fitted if pose["axis_error_deg"] > POSE_WRONG_AXIS_DEG
             or (pose["base_error_m"] is not None and pose["base_error_m"] > POSE_WRONG_BASE_M)]

    def toppled(tilt):
        return tilt > TOPPLED_TILT_DEG

    return {"frames": len(rows), "posed": len(fitted), "coverage": _rate(len(fitted), len(rows)),
            "with_base": len(based),
            "axis_error_deg": _distribution([pose["axis_error_deg"] for pose in fitted]),
            "base_error_m": _distribution([pose["base_error_m"] for pose in based]),
            "base_xy_error_m": _distribution([pose["base_xy_error_m"] for pose in based]),
            "base_z_error_m": _distribution([pose["base_z_error_m"] for pose in based]),
            "upright_called_toppled": sum(toppled(pose["pose_tilt_deg"]) and not toppled(pose["true_tilt_deg"])
                                          for pose in fitted),
            "toppled_called_upright": sum(not toppled(pose["pose_tilt_deg"]) and toppled(pose["true_tilt_deg"])
                                          for pose in fitted),
            "confident_wrong": len(wrong),
            "unknown_reasons": dict(Counter(row["pose"]["pose_reason"] for row in rows
                                            if row["pose"]["axis_error_deg"] is None))}


def markdown_report(summary):
    lines = [f"# Perception evaluation ({summary['frames']} frames, {len(summary['captures'])} captures)", "",
             f"Ignored cameras: {', '.join(summary.get('ignored_cameras') or []) or 'none'}", "",
             "## Mug detection by state", "",
             "| State | Frames | Detected | Rate | Cameras | Median xy error (m) | Misses |",
             "|---|---|---|---|---|---|---|"]
    for state, row in summary["by_state"].items():
        error = row["mug_xy_error_m"]["median"] if row["mug_xy_error_m"] else "-"
        lines.append(f"| {state} | {row['frames']} | {row['mug_grounded']} | {row['mug_detection_rate']} | "
                     f"{row['mug_camera']} | {error} | {row['mug_miss_reasons']} |")
    lines += ["", "## Mug pose by state", "",
              "| State | Frames | Posed | Median axis error (deg) | With base | Median base error (m) | "
              "Upright called toppled | Toppled called upright | Confident wrong | Unknown |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for state, row in [*summary["pose_by_state"].items(), ("all", summary["pose_overall"])]:
        axis = row["axis_error_deg"]["median"] if row["axis_error_deg"] else "-"
        base = row["base_error_m"]["median"] if row["base_error_m"] else "-"
        lines.append(f"| {state} | {row['frames']} | {row['posed']} | {axis} | {row['with_base']} | {base} | "
                     f"{row['upright_called_toppled']} | {row['toppled_called_upright']} | "
                     f"{row['confident_wrong']} | {row['unknown_reasons']} |")
    lines += ["", "## Predicates", "",
              "| Predicate | Precision | Recall | Unknown rate | False true | Truth true: t/f/? | Truth false: t/f/? |",
              "|---|---|---|---|---|---|---|"]
    for name, row in summary["predicates"].items():
        cells = []
        for key in ("truth_true", "truth_false"):
            counts = row["confusion"][key]
            cells.append(f"{counts.get('true', 0)}/{counts.get('false', 0)}/{counts.get('unknown', 0)}")
        lines.append(f"| {name} | {row['precision']} | {row['recall']} | {row['unknown_rate']} | "
                     f"{row['false_true']} | {cells[0]} | {cells[1]} |")
    toppled = summary["toppled_reported_as_supported"]
    lines += ["", f"Toppled frames reported as table-supported: {toppled['reported_supported']} "
                  f"of {toppled['toppled_frames']}.", ""]
    return "\n".join(lines)


def runtime_provider(cache_path):
    """Build the provider an episode uses, with SAM 3 answered from the cache."""
    from harness.perception.anchors import AnchorRegistry
    from harness.perception.modal_sam3 import ModalSam3Detector
    from harness.runtime.assembly import perception_provider_for

    worker = CachedSam3Worker(cache_path)
    profile = json.loads(Path(cache_path).read_text())["rack_prompt_profile"]
    return perception_provider_for(ModalSam3Detector(worker=worker), AnchorRegistry(),
                                   rack_prompt_profile=profile, rack_grounding_camera=RACK_GROUNDING_CAMERA)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, action="append", required=True,
                        help="labeled_frames capture folder with a sam3_cache.json; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    parser.add_argument("--ignore-camera", action="append", default=[], choices=PERCEPTION_CAMERAS,
                        help="score as if this perception-only camera had not been captured; repeatable")
    args = parser.parse_args(argv)
    missing = [str(folder / CACHE_FILE) for folder in args.frames if not (folder / CACHE_FILE).is_file()]
    if missing:
        parser.error("run development.replays.sam3_cache first; missing " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for folder in args.frames:
        # A fresh provider per capture: lift latches and initial heights are per episode.
        records += evaluate_capture(folder, runtime_provider(folder / CACHE_FILE),
                                    ignore_cameras=tuple(args.ignore_camera))
    summary = {**summarize(records), "ignored_cameras": sorted(args.ignore_camera)}
    (args.output / "frames.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown_report(summary))
    print(markdown_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
