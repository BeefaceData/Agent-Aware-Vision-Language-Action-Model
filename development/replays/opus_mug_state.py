"""Run bounded Opus mug-state assessment on saved RGB-D observations only.

Example:
  python -m development.replays.opus_mug_state \
    --model claude-opus-5 --output data/validation/opus_mug_state_replay_RUN

No simulator, robot motion, local vision model, or privileged simulator data is used.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from harness.perception.opus_mug_state import OpusMugStateAssessor
from harness.perception.replay import load_public_observation


ROOT = Path(__file__).resolve().parents[2]
SAM3_RESULTS = ROOT / "data/validation/multicamera_mug_replay_20261022_expanded/results.json"
SAMPLES = (
    ("grasp", "data/validation/uncertainty_ab_20261022T1040Z/abort_limit_seed3/observations", 48),
    ("grasp", "data/validation/uncertainty_ab_20261022T1040Z/abort_limit_seed3/observations", 52),
    ("grasp", "data/validation/uncertainty_ab_20261022T1040Z/abort_limit_seed3/observations", 68),
    ("grasp", "data/validation/uncertainty_ab_20261022T1040Z/abort_limit_seed3/observations", 84),
    ("grasp", "data/validation/uncertainty_ab_20261022T1040Z/abort_limit_seed3/observations", 108),
    ("handoff", "data/validation/fulltask_vla_followthrough_memory_20261021T1213Z/observations", 425),
    ("hang", "data/validation/fulltask_vla_followthrough_memory_20261021T1213Z/observations", 1596),
    ("hang", "data/validation/fulltask_vla_followthrough_memory_20261021T1213Z/observations", 1612),
)


def replay(model, output, *, client=None):
    output = Path(output)
    if output.exists():
        raise ValueError("output_already_exists")
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY_not_set")
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                           base_url="https://api.anthropic.com")
    assessor = OpusMugStateAssessor(client, model)
    sam3 = json.loads(SAM3_RESULTS.read_text(encoding="utf-8"))
    sam3_by_key = {(row["phase"], row["sequence"]): row for row in sam3["samples"]}
    rows = []
    for phase, relative_directory, sequence in SAMPLES:
        archive = ROOT / relative_directory / f"observation_{sequence:08d}.npz"
        observation = load_public_observation(archive)
        started = time.perf_counter()
        assessment = assessor.assess(observation, phase=phase)
        row = {"phase": phase, "sequence": sequence,
               "source_archive": str(archive.relative_to(ROOT)),
               "source_sha256": sha256(archive.read_bytes()).hexdigest(),
               "opus": assessment, "latency_seconds": time.perf_counter()-started}
        prior = sam3_by_key.get((phase, sequence))
        if prior is not None:
            row["sam3"] = {"head_only": prior["head_only"],
                           "phase_aware": prior["phase_aware"]}
        rows.append(row)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "mode": "saved_rgb_opus_mug_state_replay",
        "model": model, "robot_motion": False, "simulator_started": False,
        "local_gpu_used": False, "api_key_logged": False,
        "camera_policy": "head plus phase-relevant wrist camera(s)",
        "single_frame_grasp_claims_are_not_confirmation": True,
        "sample_count": len(rows), "samples": rows}
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Offline Opus mug-state comparison", "",
        "Saved RGB only; no simulator or local GPU was used. Opus outputs are visual assessments, not confirmed task predicates.", "",
        "| Phase | Observation | Opus visual state | Confidence | SAM 3 head | SAM 3 multi-camera | Overlap max |",
        "|---|---:|---|---:|---|---|---:|"]
    for row in rows:
        opus = row["opus"]
        sam = row.get("sam3", {})
        head = sam.get("head_only", {}).get("status", "n/a")
        multi = sam.get("phase_aware", {}).get("status", "n/a")
        overlaps = [view["gripper_overlap_fraction_of_mug"] for view in opus["views"]
                    if view["gripper_overlap_fraction_of_mug"] is not None]
        max_overlap = max(overlaps) if overlaps else None
        overlap_text = "n/a" if max_overlap is None else f"{max_overlap:.2f}"
        lines.append(f"| {row['phase']} | {row['sequence']} | {opus['visual_state']} | "
                     f"{opus['confidence']:.2f} | {head} | {multi} | {overlap_text} |")
    lines += ["", "A 2D overlap is an image-plane cue only. It cannot establish contact or grasp without temporal corroboration.",
              "See `results.json` for per-camera labels, boxes, overlap fractions, evidence, and latency."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Explicit Anthropic model ID")
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    args = parser.parse_args(argv)
    report = replay(args.model, args.output)
    print(json.dumps({"sample_count": report["sample_count"],
                      "output": str(args.output / "REPORT.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
