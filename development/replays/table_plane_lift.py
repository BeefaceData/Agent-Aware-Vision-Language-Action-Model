"""Test public table-plane lift evidence on saved RGB-D observations.

The reset table mask is requested once per saved episode from Modal SAM 3;
Opus mug boxes are reused from the saved assessment. No simulator is started.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from harness.perception.modal_sam3 import ModalSam3Detector
from harness.perception.opus_mug_state import OpusMugStateAssessor
from harness.perception.replay import load_public_observation


ROOT = Path(__file__).resolve().parents[2]
OPUS_RESULTS = ROOT / "data/validation/opus_mug_state_replay_generic_20261022/results.json"
LIFTED_SAMPLES = (
    ("grasp", "data/validation/fulltask_vla_followthrough_memory_20261021T1213Z/observations", 284),
    ("handoff", "data/validation/fulltask_vla_followthrough_memory_20261021T1213Z/observations", 300),
)


def _world_points(camera, mask, *, limit=12000):
    ys, xs = np.nonzero(mask & np.isfinite(camera.depth_m) & (camera.depth_m > 0))
    if len(xs) > limit:
        indices = np.linspace(0, len(xs) - 1, limit, dtype=int)
        xs, ys = xs[indices], ys[indices]
    if not len(xs):
        return np.empty((0, 3), dtype=float)
    pixels = np.stack((xs, ys, np.ones_like(xs)), axis=0).astype(float)
    rays = np.linalg.solve(camera.intrinsic, pixels)
    camera_xyz = rays * camera.depth_m[ys, xs][None, :]
    camera_h = np.vstack((camera_xyz, np.ones((1, len(xs)))))
    world_h = np.linalg.solve(camera.world_to_camera, camera_h)
    return (world_h[:3] / world_h[3:4]).T


def fit_table_plane(camera, mask):
    """Fit a robust world-coordinate plane from reset table-mask depth."""
    points = _world_points(camera, np.asarray(mask, dtype=bool))
    if len(points) < 100:
        raise ValueError("insufficient_table_depth_points")
    active = points
    for _ in range(4):
        center = np.median(active, axis=0)
        _, _, vh = np.linalg.svd(active - center, full_matrices=False)
        normal = vh[-1]
        residual = np.abs((points - center) @ normal)
        cutoff = max(0.008, float(np.quantile(residual, 0.70)))
        selected = points[residual <= cutoff]
        if len(selected) < 100:
            raise ValueError("table_plane_fit_degenerate")
        active = selected
    center = np.median(active, axis=0)
    _, _, vh = np.linalg.svd(active - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    residual = (active - center) @ normal
    return {"normal_world": normal.tolist(),
            "offset_world": float(-normal @ center),
            "inlier_count": int(len(active)),
            "residual_median_abs_m": float(np.median(np.abs(residual))),
            "residual_p95_abs_m": float(np.quantile(np.abs(residual), .95))}


def measure_box_lower_edge(observation, camera_name, bbox, plane):
    """Measure depth-backed plane distance near the bottom of an Opus box."""
    camera = observation.camera_frames[camera_name]
    height, width = observation.images[camera_name].shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    points = []
    for fx in (.25, .50, .75):
        for fy in (.90, .97):
            u = min(width - 1, max(0, int(round(x1 + fx * (x2 - x1)))))
            v = min(height - 1, max(0, int(round(y1 + fy * (y2 - y1)))))
            try:
                xyz = np.asarray(camera.unproject(u, v, radius=1))
            except ValueError:
                continue
            distance = float(np.dot(plane["normal_world"], xyz) + plane["offset_world"])
            points.append({"pixel": [u, v], "signed_distance_m": distance})
    return points


def replay(output, *, detector=None, opus_client=None, model=None):
    output = Path(output)
    if output.exists():
        raise ValueError("output_already_exists")
    output.mkdir(parents=True)
    detector = detector or ModalSam3Detector()
    detector.set_static_rack_reference(True)
    opus = json.loads(OPUS_RESULTS.read_text(encoding="utf-8"))
    opus_rows = list(opus["samples"])
    if opus_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY_not_set")
        from anthropic import Anthropic
        opus_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                base_url="https://api.anthropic.com")
    model = model or opus["model"]
    assessor = OpusMugStateAssessor(opus_client, model)
    for phase, directory, sequence in LIFTED_SAMPLES:
        archive = ROOT / directory / f"observation_{sequence:08d}.npz"
        observation = load_public_observation(archive)
        started = time.perf_counter()
        assessment = assessor.assess(observation, phase=phase,
                                     cameras=("head_camera",))
        opus_rows.append({"phase": phase, "sequence": sequence,
            "source_archive": str(archive.relative_to(ROOT)), "opus": assessment,
            "opus_latency_seconds": time.perf_counter() - started})
    episode_cache, rows = {}, []
    overlays = output / "overlays"
    overlays.mkdir()
    for row in opus_rows:
        archive = ROOT / row["source_archive"]
        observation = load_public_observation(archive)
        episode_key = str(archive.parent)
        if episode_key not in episode_cache:
            reset_path = archive.parent / "observation_00000000.npz"
            reset = load_public_observation(reset_path)
            image = reset.images["head_camera"]
            batch = detector.detect({"head_camera": image},
                                    observation_sequence=reset.sequence)
            tables = [item for item in batch.detections if item.label == "table"]
            table = max(tables, key=lambda item: item.confidence, default=None)
            if table is None:
                raise ValueError(f"reset_table_not_detected:{episode_key}")
            plane = fit_table_plane(reset.camera_frames["head_camera"], table.mask)
            episode_cache[episode_key] = {"plane": plane, "table_confidence": table.confidence,
                "table_mask_area_fraction": float(table.mask.mean()),
                "reset_sequence": reset.sequence}
            rgba = np.zeros((*table.mask.shape, 4), dtype=np.uint8)
            rgba[table.mask] = (0, 160, 255, 95)
            base = Image.fromarray(image).convert("RGBA")
            Image.alpha_composite(base, Image.fromarray(rgba)).convert("RGB").save(
                overlays / f"reset_table_{len(episode_cache)}.png")

        assessment = row.get("opus", row)
        view = next(view for view in assessment["views"]
                    if view["camera"] == "head_camera")
        bbox = view["mug_bbox_px"]
        lower = (None if bbox is None else measure_box_lower_edge(
            observation, "head_camera", bbox, episode_cache[episode_key]["plane"]))
        valid = [] if lower is None else lower
        distances = [point["signed_distance_m"] for point in valid]
        # This is an exploratory separation only, not a runtime predicate.
        # A future predicate needs several fresh, stable observations and a
        # calibrated margin around the table plane.
        classification = "unknown"
        if len(distances) >= 3:
            near = sum(abs(value) <= .025 for value in distances)
            above = sum(value >= .04 for value in distances)
            if above >= 3:
                classification = "lower_box_points_above_table"
            elif near >= 3:
                classification = "lower_box_points_at_table"
        row_out = {"phase": row["phase"], "sequence": row["sequence"],
            "source_archive": row["source_archive"], "opus_state": assessment["visual_state"],
            "opus_confidence": assessment["confidence"], "mug_bbox_px": bbox,
            "opus_latency_seconds": row.get("opus_latency_seconds"),
            "lower_edge_depth_points": lower, "table_plane_relation": classification,
            "table_reference": episode_cache[episode_key]}
        rows.append(row_out)

        image = Image.fromarray(observation.images["head_camera"]).convert("RGBA")
        draw = ImageDraw.Draw(image)
        if bbox:
            draw.rectangle(tuple(bbox), outline=(60, 235, 90, 255), width=3)
            for point in valid:
                x, y = point["pixel"]
                color = (255, 80, 50, 255) if point["signed_distance_m"] >= .04 else (0, 220, 255, 255)
                draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
            draw.text((6, 6), f"seq={row['sequence']} table_relation={classification}",
                      fill=(255, 255, 255, 255), stroke_width=2,
                      stroke_fill=(0, 0, 0, 255))
        image.convert("RGB").save(overlays / f"seq_{row['sequence']:08d}.png")

    report = {"schema_version": 1, "mode": "saved_rgbd_table_plane_lift_pilot",
              "robot_motion": False, "simulator_started": False,
              "local_gpu_used": False, "table_masks_from": "Modal SAM 3 at reset only",
              "mug_boxes_from": "saved Opus assessment plus two head-only Opus calls for lifted/staged samples",
              "sample_count": len(rows), "samples": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Public table-plane mug-lift pilot", "",
        "Table masks were generated once per saved episode from its reset head RGB; depth/calibration and saved Opus mug boxes are public observation data.",
        "This is an offline feasibility check, not a runtime predicate or task-success confirmation.", "",
        "| Phase | Obs | Table relation | Valid depth points | Distances (m) |", "|---|---:|---|---:|---|"]
    for row in rows:
        values = [p["signed_distance_m"] for p in (row["lower_edge_depth_points"] or [])]
        lines.append(f"| {row['phase']} | {row['sequence']} | {row['table_plane_relation']} | {len(values)} | {', '.join(f'{v:.3f}' for v in values) or 'n/a'} |")
    lines += ["", "The green rectangle is the Opus mug box. Cyan points lie within 2.5 cm of the fitted table plane; red points are at least 4 cm above it.",
        "Overlay examples are in `overlays/`; reset table-mask overlays are named `reset_table_*.png`.",
        "Do not use one frame or a gripper-overlap label as a grasp predicate. Runtime use requires temporal stability, missing-depth handling, and held-out validation."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Optional explicit Opus model ID")
    args = parser.parse_args(argv)
    report = replay(args.output, model=args.model)
    print(json.dumps({"sample_count": report["sample_count"],
                      "output": str(args.output / "REPORT.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
