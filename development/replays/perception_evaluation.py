"""Score public RGB-D perception against offline simulator labels; no simulator, no Modal.

Replays captured frames (``development.capture.labeled_frames``) through the
runtime perception provider, with SAM 3 answers served from each capture's
``sam3_cache.json`` (``development.replays.sam3_cache``), then compares the
public evidence with the privileged labels:

    python -m development.replays.perception_evaluation \\
      --frames data/perception/frames/seed_0 --frames data/perception/frames/seed_1 \\
      --output data/perception/reports/baseline

Reports detection rate and 3D error per mug state, and for each predicate a
confusion table with precision, recall, unknown rate and false-true count.
Labels never reach the provider; they are only read for scoring and to tell
the detector which phase the runtime planner would be in.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

from harness.perception.replay import load_public_observation
from development.capture.labeled_frames import LIFTED_HEIGHT_M, PRIVILEGED_FILE, TOPPLED_TILT_DEG
from development.replays.sam3_cache import CACHE_FILE, CachedSam3Worker

EVALUATED_PREDICATES = ("mug_lifted", "release_ready", "mug_staged_on_table", "right_mug_lifted")
# The full-task configs ground the rack from the right wrist camera.
RACK_GROUNDING_CAMERA = "right_camera"


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


def truth_predicates(label):
    """What each public predicate should answer, by its documented meaning.

    - mug_lifted: the mug has been lifted at some point (the public signal latches).
    - release_ready: the mug is upright and supported by the table, so opening a
      gripper would not drop it.
    - mug_staged_on_table: release_ready after an earlier lift (the handoff point).
    - right_mug_lifted: the right gripper alone holds the mug off the table.
    """
    contacts = label["contacts"]
    upright = label["mug"]["tilt_deg"] <= TOPPLED_TILT_DEG
    supported = contacts["table"] and upright
    return {"mug_lifted": lifted_so_far(label),
            "release_ready": supported,
            "mug_staged_on_table": supported and label["lifted_before"],
            "right_mug_lifted": (contacts["right_gripper"] and not contacts["left_gripper"]
                                 and not contacts["table"]
                                 and label["mug"]["height_change_m"] >= LIFTED_HEIGHT_M)}


def mug_error(grounded, label):
    """Horizontal and vertical offset of the grounded mug point from the true mug origin.

    The grounded point is the median of visible surface points, so a horizontal
    offset of about one mug radius is expected even for perfect perception.
    """
    estimate = np.asarray(grounded.position, dtype=float)
    truth = np.asarray(label["mug"]["pose"][:3], dtype=float)
    return float(np.linalg.norm(estimate[:2] - truth[:2])), float(estimate[2] - truth[2])


def evaluate_capture(frames_dir, provider):
    """Replay one capture in sequence order; return one record per labelled frame."""
    frames_dir = Path(frames_dir)
    labels = sorted((json.loads(line) for line in (frames_dir / PRIVILEGED_FILE).read_text().splitlines()),
                    key=lambda item: item["sequence"])
    records = []
    for label in labels:
        observation = load_public_observation(frames_dir / label["observation"])
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
        truth = truth_predicates(label)
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

    toppled = [record for record in records if record["state"] == "toppled"]
    toppled_supported = [record for record in toppled
                         if record["predicates"]["release_ready"]["public"] is True
                         or record["predicates"]["mug_staged_on_table"]["public"] is True]
    return {"frames": len(records), "captures": sorted({record["capture"] for record in records}),
            "by_state": states, "predicates": predicates,
            "toppled_reported_as_supported": {"toppled_frames": len(toppled),
                                              "reported_supported": len(toppled_supported)}}


def markdown_report(summary):
    lines = [f"# Perception evaluation ({summary['frames']} frames, {len(summary['captures'])} captures)", "",
             "## Mug detection by state", "",
             "| State | Frames | Detected | Rate | Cameras | Median xy error (m) | Misses |",
             "|---|---|---|---|---|---|---|"]
    for state, row in summary["by_state"].items():
        error = row["mug_xy_error_m"]["median"] if row["mug_xy_error_m"] else "-"
        lines.append(f"| {state} | {row['frames']} | {row['mug_grounded']} | {row['mug_detection_rate']} | "
                     f"{row['mug_camera']} | {error} | {row['mug_miss_reasons']} |")
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
    args = parser.parse_args(argv)
    missing = [str(folder / CACHE_FILE) for folder in args.frames if not (folder / CACHE_FILE).is_file()]
    if missing:
        parser.error("run development.replays.sam3_cache first; missing " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for folder in args.frames:
        # A fresh provider per capture: lift latches and initial heights are per episode.
        records += evaluate_capture(folder, runtime_provider(folder / CACHE_FILE))
    summary = summarize(records)
    (args.output / "frames.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown_report(summary))
    print(markdown_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
