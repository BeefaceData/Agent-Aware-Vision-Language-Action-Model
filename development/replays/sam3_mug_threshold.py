"""Replay selected public head-camera RGB frames through Modal SAM 3.

This bounded diagnostic isolates the mug score cutoff on the remote worker.
It does not start RoboTwin, use local vision models, or read privileged state.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from harness.perception.modal_sam3 import ModalSam3Detector, MUG_SCORE_THRESHOLD
from harness.perception.replay import load_public_observation


ROOT = Path(__file__).resolve().parents[2]
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


def replay(output, *, detector=None):
    output = Path(output)
    if output.exists():
        raise ValueError("output_already_exists")
    detector = detector or ModalSam3Detector()
    detector.set_static_rack_reference(True)
    rows = []
    output.mkdir(parents=True)
    overlays = output / "overlays"
    overlays.mkdir()
    for phase, directory, sequence in SAMPLES:
        archive = ROOT / directory / f"observation_{sequence:08d}.npz"
        observation = load_public_observation(archive)
        image = observation.images["head_camera"]
        batch = detector.detect({"head_camera": image},
                                observation_sequence=observation.sequence)
        mugs = [item for item in batch.detections if item.label == "mug"]
        best = max(mugs, key=lambda item: item.confidence, default=None)
        row = {"phase": phase, "sequence": sequence,
               "source_archive": str(archive.relative_to(ROOT)),
               "camera": "head_camera", "cutoff": MUG_SCORE_THRESHOLD,
               "status": "detected" if best is not None else "not_detected",
               "candidate_count": len(mugs),
               "confidence": None if best is None else best.confidence,
               "mask_area_fraction": None if best is None else float(best.mask.mean())}
        if best is not None:
            ys, xs = np.nonzero(best.mask)
            row["mask_bbox_xyxy"] = [int(xs.min()), int(ys.min()),
                                     int(xs.max() + 1), int(ys.max() + 1)]
        else:
            row["mask_bbox_xyxy"] = None
        canvas = Image.fromarray(image).convert("RGBA")
        if best is not None:
            rgba = np.zeros((*best.mask.shape, 4), dtype=np.uint8)
            rgba[best.mask] = (40, 220, 70, 110)
            canvas = Image.alpha_composite(canvas, Image.fromarray(rgba))
            draw = ImageDraw.Draw(canvas)
            x1, y1, x2, y2 = row["mask_bbox_xyxy"]
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 80, 255), width=3)
            draw.text((x1 + 3, max(0, y1 - 14)),
                      f"mug {best.confidence:.2f}", fill=(0, 0, 0, 255),
                      stroke_width=2, stroke_fill=(0, 255, 80, 255))
        ImageDraw.Draw(canvas).text((8, 8),
            f"{phase} seq={sequence} cutoff={MUG_SCORE_THRESHOLD:.2f} {row['status']}",
            fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        filename = f"seq_{sequence:08d}_head_camera.png"
        canvas.convert("RGB").save(overlays / filename)
        row["overlay"] = f"overlays/{filename}"
        rows.append(row)

    report = {"schema_version": 1, "mode": "saved_public_rgb_sam3_mug_cutoff_replay",
              "robot_motion": False, "simulator_started": False,
              "local_gpu_used": False, "worker_gpu": "T4",
              "threshold": MUG_SCORE_THRESHOLD, "camera": "head_camera",
              "rack_inference": False, "sample_count": len(rows), "samples": rows}
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# SAM 3 mug threshold replay", "",
        "Saved public head-camera RGB only; sequential Modal T4 calls, no simulator or local GPU.",
        "Mug score cutoff: **0.40**. Green is the returned mask; outlines show its image-space bounds.", "",
        "| Phase | Observation | Status | Score | Mask area | Overlay |",
        "|---|---:|---|---:|---:|---|"]
    for row in rows:
        score = "n/a" if row["confidence"] is None else f"{row['confidence']:.3f}"
        area = "n/a" if row["mask_area_fraction"] is None else f"{row['mask_area_fraction']:.3f}"
        lines.append(f"| {row['phase']} | {row['sequence']} | {row['status']} | {score} | {area} | [{row['overlay']}]({row['overlay']}) |")
    lines += ["", "A mask is a visual segmentation, not proof of a grasp or task milestone."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = replay(args.output)
    print(json.dumps({"sample_count": report["sample_count"],
                      "output": str(args.output / "REPORT.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
