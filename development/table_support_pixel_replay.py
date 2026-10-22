"""Replay Opus mug-base pixels against a reset-cached public RGB-D table plane.

No simulator is started. The reset table mask uses the Modal SAM 3 service;
Opus receives only saved head-camera RGB frames and returns a pixel.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from harness.perception.modal_sam3 import ModalSam3Detector
from harness.perception.replay import load_public_observation
from harness.perception.table_plane import OpusMugSupportGrounder, StaticTablePlaneCache


ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ROOT / "data/validation/sam3_rack_fallback_live_20261020T1935Z/observations"
SAMPLES = (
    ("table_at_reset", 0, "mug rests on the table before arm contact"),
    ("clearly_raised_transfer", 112, "mug is visibly held above the table during transfer"),
    ("table_after_transfer", 624, "mug is visibly resting on the table after left transfer"),
)


def replay(output, *, detector=None, opus_client=None, model="claude-opus-5",
           repeats=3):
    output = Path(output)
    if output.exists():
        raise ValueError("output_already_exists")
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("repeats_must_be_between_1_and_5")
    output.mkdir(parents=True)
    if opus_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY_not_set")
        from anthropic import Anthropic
        opus_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                base_url="https://api.anthropic.com")
    detector = detector or ModalSam3Detector()
    detector.set_static_rack_reference(True)
    reset = load_public_observation(OBSERVATIONS / "observation_00000000.npz")
    table_batch = detector.detect({"head_camera": reset.images["head_camera"]},
                                  observation_sequence=reset.sequence)
    table = max((item for item in table_batch.detections if item.label == "table"),
                key=lambda item: item.confidence, default=None)
    if table is None:
        raise ValueError("reset_table_not_detected")
    plane = StaticTablePlaneCache()
    plane_metadata = plane.cache(reset, table.mask)
    grounder = OpusMugSupportGrounder(opus_client, model)
    rows = []
    for label, sequence, visual_basis in SAMPLES:
        observation = load_public_observation(
            OBSERVATIONS / f"observation_{sequence:08d}.npz")
        frame_rows = []
        for repeat in range(1, repeats + 1):
            started = time.perf_counter()
            report = grounder.locate(observation)
            latency = time.perf_counter() - started
            distance = None
            if report["status"] == "grounded":
                distance = plane.signed_distance(observation, report["support_pixel"])
            row = {"sample": label, "visual_basis": visual_basis,
                "observation_sequence": sequence, "repeat": repeat,
                "report": report, "support_height_above_table_m": distance,
                "latency_seconds": latency}
            rows.append(row)
            frame_rows.append(row)
        image = Image.fromarray(observation.images["head_camera"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        colors = ((255, 40, 40), (30, 110, 255), (20, 190, 70),
                  (255, 140, 20), (180, 50, 200))
        for row in frame_rows:
            if row["report"]["status"] == "grounded":
                x, y = map(float, row["report"]["support_pixel"])
                color = colors[(row["repeat"] - 1) % len(colors)]
                draw.ellipse((x-5, y-5, x+5, y+5), fill=color,
                             outline="white", width=2)
                draw.text((x+5, y-8), str(row["repeat"]), fill=color,
                          stroke_width=1, stroke_fill="black")
        heights = [row["support_height_above_table_m"] for row in frame_rows
                   if row["support_height_above_table_m"] is not None]
        median = float(np.median(heights)) if heights else None
        draw.text((8, 8), f"{label}; median height={median if median is not None else 'unknown'}",
                  fill="white", stroke_width=2, stroke_fill="black")
        image.save(output / f"{label}_{sequence:08d}.png")

    result = {"schema_version": 1, "mode": "saved_rgbd_table_support_pixel_pilot",
        "simulator_started": False, "robot_motion": False, "local_gpu_used": False,
        "sam3_table_mask": "Modal SAM 3 on reset head frame only",
        "opus_input": "saved head-camera RGB only",
        "repeat_count_per_frame": repeats,
        "table_reference": plane_metadata, "samples": rows}
    (output / "results.json").write_text(json.dumps(result, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Opus support-pixel table-plane pilot", "",
        "The reset head-camera SAM 3 mask and public RGB-D were used to cache one table plane. Opus received only the saved head RGB for each sample and returned a 2D mug support pixel.",
        f"Each of {len(SAMPLES)} visually reviewed states was queried {repeats} times using the exact same saved RGB frame. This measures support-pixel repeatability; it does not validate grasp or generalization.", "",
        "| Visual state | Observation | Pixel reports | Heights above table (m) | Median | Spread |", "|---|---:|---:|---|---:|---:|"]
    for label, sequence, visual_basis in SAMPLES:
        frame_rows = [row for row in rows if row["sample"] == label]
        heights = [row["support_height_above_table_m"] for row in frame_rows
                   if row["support_height_above_table_m"] is not None]
        median = float(np.median(heights)) if heights else None
        spread = max(heights) - min(heights) if heights else None
        report_statuses = ", ".join(row["report"]["status"] for row in frame_rows)
        height_text = ", ".join(f"{height:.3f}" for height in heights) or "unknown"
        lines.append(f"| {label}: {visual_basis} | {sequence} | {report_statuses} | {height_text} | {median if median is not None else 'unknown'} | {spread if spread is not None else 'unknown'} |")
    lines += ["", "Red dots mark Opus's reported support pixels. See the PNG overlays for visual review.",
        "An unknown/missing support point remains unknown; this pilot does not enable the runtime predicate."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    result = replay(args.output, model=args.model, repeats=args.repeats)
    print(json.dumps({"sample_count": len(result["samples"]),
                      "output": str(args.output / "REPORT.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
