"""Regenerate every figure in pose_estimation_guide.pdf from captures, SAM 3 caches, meshes and reports.

Run from the repository root:

    conda run -n robotwin env PYTHONPATH=$PWD python docs/Collaboration/Olusola/guides/pose_estimation/make_figures.py

Then rebuild the PDF:

    cd docs/Collaboration/Olusola/guides/pose_estimation && conda run -n texbuild tectonic -X compile pose_estimation_guide.tex

What must exist first
---------------------
Nothing under data/ is committed. A figure whose inputs are missing is skipped with a message.

1. The `robotwin` conda env (numpy, matplotlib, Pillow, trimesh) and the RoboTwin checkout with
   its assets (RoboTwin/assets/objects/039_mug), for the mesh figure.
2. Captures with SAM 3 caches, made as described in ../perception/make_figures.py:
   data/perception/frames_cam/seed_0..4 and data/perception/frames_holdout/seed_5..9.
3. Evaluation reports (CPU, under a minute each), written by the pose-aware evaluator:
       python -m development.replays.perception_evaluation \\
         --frames data/perception/frames_cam/seed_0 ... --frames data/perception/frames_cam/seed_4 \\
         --output data/perception/reports/pose_dev_with_camera
   and likewise pose_holdout_with_camera (frames_holdout/seed_5..9). The before-numbers come
   from the earlier with_camera and holdout_with_camera reports.
"""

import base64
from io import BytesIO
import json
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from harness.perception.object_pose import MIN_INLIER_FRACTION, MUG, fit_cylinder, masked_surface, object_pose
from harness.perception.replay import load_public_observation
from harness.runtime.assembly import MUG_MASK_MIN_CONFIDENCE

ROOT = Path(__file__).resolve().parents[5]  # the repository root
DATA = ROOT / "data" / "perception"
REPORTS = DATA / "reports"
MUG_MESHES = ROOT / "RoboTwin" / "assets" / "objects" / "039_mug"
OUT = Path(__file__).resolve().parent / "figures"

INK, MUTED, BLUE, ORANGE, GREEN, RED, GRID = "#1f2937", "#6b7280", "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
# The episode's provider keeps SAM 3 masks at or above these scores (harness/runtime/assembly.py).
MIN_MASK_CONFIDENCE = {"mug": MUG_MASK_MIN_CONFIDENCE, "table": 0.60}
ON_TABLE_STATES = ("resting", "gripped_on_table", "staged")


# ---------------------------------------------------------------- inputs
def capture_dir(seed):
    return DATA / ("frames_holdout" if seed >= 5 else "frames_cam") / f"seed_{seed}"


def observation(seed, sequence):
    return load_public_observation(capture_dir(seed) / f"observations/observation_{sequence:08d}.npz")


def best_mask(seed, sequence, camera, label):
    """The highest-scoring cached SAM 3 mask of ``label``, or None below the provider's floor."""
    cache = json.loads((capture_dir(seed) / "sam3_cache.json").read_text())
    best = None
    for detection in cache["frames"][str(sequence)].get(camera, {}).get("detections", []):
        if detection["label"] == label and detection["confidence"] >= MIN_MASK_CONFIDENCE[label]:
            if best is None or detection["confidence"] > best[0]:
                best = (detection["confidence"], detection["mask_png"])
    if best is None:
        return None
    with Image.open(BytesIO(base64.b64decode(best[1]))) as image:
        return np.asarray(image) > 0


def table_height(seed, sequence, frame_obs):
    """Median height of the head camera's table mask, as the provider uses; None without one."""
    mask = best_mask(seed, sequence, "head_camera", "table")
    if mask is None:
        return None
    frame = frame_obs.camera_frames["head_camera"]
    points, _normals = frame.surface_points_and_normals(mask, erode_px=0, step_px=1)
    return float(np.median(points[:, 2]))


def labels(seed):
    return [json.loads(line) for line in (capture_dir(seed) / "labels.jsonl").read_text().splitlines()]


def report_frames(report):
    return [json.loads(line) for line in (REPORTS / report / "frames.jsonl").read_text().splitlines()]


def summary(report):
    return json.loads((REPORTS / report / "summary.json").read_text())


# ---------------------------------------------------------------- drawing helpers
def overlay(rgb, mask, color, alpha=0.5):
    image = rgb.astype(float) / 255
    image[mask] = image[mask] * (1 - alpha) + np.array(matplotlib.colors.to_rgb(color)) * alpha
    return image


def project(frame, points):
    camera = (np.c_[points, np.ones(len(points))] @ frame.world_to_camera.T)[:, :3]
    pixels = camera @ frame.intrinsic.T
    return pixels[:, :2] / pixels[:, 2:3]


def draw_cylinder(ax, frame, pose, low, high, color):
    """Outline of a fitted cylinder: its two end circles and the axis, projected into the image."""
    axis = np.asarray(pose.axis)
    side = np.cross(axis, (0, 0, 1) if abs(axis[2]) < 0.9 else (1, 0, 0))
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    angles = np.linspace(0, 2 * np.pi, 60)
    ring = pose.radius_m * (np.outer(np.cos(angles), side) + np.outer(np.sin(angles), other))
    for along in (low, high):
        centre = np.asarray(pose.centre) + along * axis
        uv = project(frame, centre + ring)
        ax.plot(uv[:, 0], uv[:, 1], color=color, lw=1.6)
    ends = project(frame, np.asarray(pose.centre) + np.outer((low - 0.02, high + 0.02), axis))
    ax.plot(ends[:, 0], ends[:, 1], color=color, lw=2.2, ls="--")


def crop_to(ax, mask, pad_px=34):
    ys, xs = np.nonzero(mask)
    ax.set_xlim(max(xs.min() - pad_px, 0), xs.max() + pad_px)
    ax.set_ylim(ys.max() + pad_px, max(ys.min() - pad_px, 0))
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def normal_z_map(frame, mask, step_px):
    """|n_z| per usable mask pixel (NaN elsewhere): the image form of surface_points_and_normals."""
    depth = frame.depth_m.astype(float)
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    xyz = (np.stack((u, v, np.ones_like(u)), -1).astype(float) @ np.linalg.inv(frame.intrinsic).T) * depth[..., None]
    s = step_px
    normals = np.cross(xyz[s:-s, 2 * s:] - xyz[s:-s, :-2 * s], xyz[2 * s:, s:-s] - xyz[:-2 * s, s:-s])
    normals = normals @ frame.world_to_camera[:3, :3]
    length = np.linalg.norm(normals, axis=-1)
    usable = mask[s:-s, s:-s] & (length > 1e-12)
    result = np.full((height, width), np.nan)
    result[s:-s, s:-s][usable] = np.abs(normals[usable, 2] / length[usable])
    return result


# ---------------------------------------------------------------- figures
def mesh_offset():
    """Seed 0's mug (model 1) seen from above: the label origin is not on the body axis."""
    import trimesh

    data = json.loads((MUG_MESHES / "model_data1.json").read_text())
    mesh = trimesh.load(MUG_MESHES / "visual" / "base1.glb", force="mesh")
    vertices = mesh.vertices * data["scale"][0]
    radius = vertices[:, 0].max()
    body_z = float(np.median(vertices[np.abs(vertices[:, 0]) > 0.95 * radius, 2]))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(vertices[:, 0] * 100, vertices[:, 2] * 100, s=0.4, color=MUTED, alpha=0.35)
    circle = plt.Circle((0, body_z * 100), radius * 100, fill=False, color=GREEN, lw=2)
    ax.add_patch(circle)
    ax.plot(0, 0, "x", color=RED, ms=12, mew=3, label="label origin (box centre)")
    ax.plot(0, body_z * 100, "o", color=GREEN, ms=9, label="body axis (true base centre)")
    ax.annotate("", xy=(0.15, body_z * 100), xytext=(0.15, 0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(0.6, body_z * 50, f"{body_z * 100:.1f} cm", fontsize=10, color=INK, va="center")
    ax.text(0, vertices[:, 2].min() * 100 - 0.6, "handle side", ha="center", va="top", color=MUTED, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("model x (cm)")
    ax.set_ylabel("model z (cm)")
    ax.set_title("Mug model 1 from above (mesh vertices)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, "mesh_offset.png")


def depth_steps():
    """Depth is stored in whole millimetres; a 1-pixel normal step turns that into noise."""
    seed, sequence, camera = 2, 0, "head_camera"
    frame_obs = observation(seed, sequence)
    frame = frame_obs.camera_frames[camera]
    mask = best_mask(seed, sequence, camera, "mug")
    ys, xs = np.nonzero(mask)
    row = int(np.median(ys))
    columns = np.arange(xs.min() - 6, xs.max() + 7)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    axes[0].step(columns, frame.depth_m[row, columns] * 1000, where="mid", color=BLUE)
    axes[0].set_title("One image row across the mug", loc="left", fontsize=10.5)
    axes[0].set_xlabel("pixel column")
    axes[0].set_ylabel("stored depth (mm)")
    axes[0].yaxis.grid(True, color=GRID)
    for ax, step in zip(axes[1:], (1, 2)):
        values = normal_z_map(frame, mask, step)
        ax.imshow(frame_obs.images[camera] / 255 * 0.5)
        ax.imshow(values, cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"|n_z| with a {step}-pixel step", fontsize=9.5)
        crop_to(ax, mask, 8)
    save(fig, "depth_steps.png")


def fit_on_frame(ax, seed, sequence, camera, title):
    frame_obs = observation(seed, sequence)
    frame = frame_obs.camera_frames[camera]
    mask = best_mask(seed, sequence, camera, "mug")
    table_z = table_height(seed, sequence, frame_obs)
    points, normals = masked_surface(frame, mask, table_height_m=table_z)
    fit, _reason = fit_cylinder(points, normals, MUG)
    pose, reason = object_pose({camera: (points, normals)}, MUG, table_height_m=table_z,
                               observation_sequence=sequence)
    ax.imshow(overlay(frame_obs.images[camera], mask, ORANGE, 0.35))
    color = GREEN if pose.trusted else ORANGE
    draw_cylinder(ax, frame, fit, *fit.extent, color)
    trust = "trusted" if pose.trusted else "untrusted"
    ax.set_title(f"{title}\ntilt {pose.tilt_deg():.0f} deg, explains {pose.quality:.2f} ({trust})",
                 fontsize=9, color=color)
    crop_to(ax, mask)


def fits_on_frames():
    cases = ((2, 0, "head_camera", "resting, head camera"),
             (0, 208, "perception_camera", "staged, perception camera"),
             (1, 720, "perception_camera", "toppled, perception camera"),
             (6, 640, "head_camera", "toppled end-on (held-out seed 6)"))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, case in zip(axes, cases):
        fit_on_frame(ax, *case)
    save(fig, "fits_on_frames.png")


def shadow():
    seed, sequence, camera = 0, 384, "perception_camera"
    frame_obs = observation(seed, sequence)
    mask = best_mask(seed, sequence, camera, "mug")
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.imshow(overlay(frame_obs.images[camera], mask, ORANGE, 0.6))
    ys, xs = np.nonzero(mask)
    ax.annotate("SAM 3 'mug' mask:\nthe mug's shadow", xy=(xs.mean(), ys.mean()), xytext=(xs.mean() - 120, ys.mean() - 10),
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="->", color=INK))
    ax.annotate("the real mug,\nin the right gripper", xy=(150, 75), xytext=(20, 30), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "shadow.png")


def quality_gap():
    """Share of points each per-camera fit explains on seeds 0-4, split by whether its axis was right."""
    from development.replays.perception_evaluation import true_mug_axis

    good, bad = [], []
    for seed in range(5):
        for label in labels(seed):
            frame_obs = observation(seed, label["sequence"])
            table_z = table_height(seed, label["sequence"], frame_obs)
            truth = true_mug_axis(label)
            for camera, frame in frame_obs.camera_frames.items():
                mask = best_mask(seed, label["sequence"], camera, "mug")
                if mask is None:
                    continue
                fit, _reason = fit_cylinder(*masked_surface(frame, mask, table_height_m=table_z), MUG)
                if fit is None:
                    continue
                error = np.degrees(np.arccos(min(1.0, abs(float(fit.axis @ truth)))))
                (bad if error > 20 else good).append(fit.inlier_fraction)
    frame_obs = observation(6, 640)
    seed6, _reason = fit_cylinder(*masked_surface(frame_obs.camera_frames["head_camera"],
                                                  best_mask(6, 640, "head_camera", "mug"),
                                                  table_height_m=table_height(6, 640, frame_obs)), MUG)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = np.linspace(0.3, 1.0, 36)
    ax.hist(good, bins=bins, color=GREEN, alpha=0.85, label=f"axis within 20 deg ({len(good)} fits)")
    ax.hist(bad, bins=bins, color=RED, alpha=0.9, label=f"axis more than 20 deg off ({len(bad)} fits)")
    ax.axvline(MIN_INLIER_FRACTION, color=INK, lw=2, ls="--")
    ax.text(MIN_INLIER_FRACTION + 0.005, ax.get_ylim()[1] * 0.9, f"trusted from {MIN_INLIER_FRACTION:.2f}",
            fontsize=9, color=INK)
    ax.axvline(seed6.inlier_fraction, color=ORANGE, lw=2)
    ax.text(seed6.inlier_fraction - 0.005, ax.get_ylim()[1] * 0.55, "held-out seed 6,\nend-on toppled",
            fontsize=8.5, color=ORANGE, ha="right")
    ax.set_xlabel("share of the camera's mug points that the fitted cylinder explains")
    ax.set_ylabel("fits (seeds 0-4)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "quality_gap.png")


def position_results():
    """Median distance to the true base centre, before (median of points) and after (cylinder fit)."""
    rows = report_frames("pose_dev_with_camera")
    old, new = [], []
    for state in ON_TABLE_STATES:
        here = [row for row in rows if row["state"] == state]
        old.append(100 * statistics.median(np.hypot(row["mug_xy_error_m"], row["mug_z_offset_m"])
                                           for row in here if row["mug_xy_error_m"] is not None))
        new.append(100 * statistics.median(row["pose"]["base_error_m"] for row in here
                                           if row["pose"]["base_error_m"] is not None))
    fig, ax = plt.subplots(figsize=(7, 3.3))
    x = np.arange(len(ON_TABLE_STATES))
    for offset, values, color, name in ((-0.2, old, MUTED, "before: median of mask points"),
                                        (0.2, new, GREEN, "now: cylinder fit, base on the table")):
        drawn = ax.bar(x + offset, values, 0.38, color=color, label=name)
        for bar, value in zip(drawn, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1, f"{value:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, [state.replace("_", " ") for state in ON_TABLE_STATES])
    ax.set_ylabel("median error (cm)")
    ax.set_title("Mug position error on the table, seeds 0-4", loc="left", fontsize=11)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "position_results.png")


def axis_results():
    states = ("resting", "staged", "gripped_on_table", "held_left", "held_right", "toppled", "touching_rack")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    for offset, report, color, name in ((-0.2, "pose_dev_with_camera", BLUE, "seeds 0-4"),
                                        (0.2, "pose_holdout_with_camera", ORANGE, "seeds 5-9 (held out)")):
        by_state = summary(report)["pose_by_state"]
        values = [(by_state.get(state, {}).get("axis_error_deg") or {}).get("median", 0) for state in states]
        coverage = [100 * (by_state.get(state, {}).get("coverage") or 0) for state in states]
        x = np.arange(len(states))
        axes[0].bar(x + offset, values, 0.38, color=color, label=name)
        axes[1].bar(x + offset, coverage, 0.38, color=color, label=name)
    for ax, title, ylabel in ((axes[0], "Median axis error", "degrees"),
                              (axes[1], "Frames with a trusted pose", "%")):
        ax.set_xticks(np.arange(len(states)), [state.replace("_", "\n") for state in states], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "axis_results.png")


def release_results():
    pairs = (("seeds 0-4", "with_camera", "pose_dev_with_camera"),
             ("seeds 5-9\n(held out)", "holdout_with_camera", "pose_holdout_with_camera"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for ax, predicate in zip(axes, ("release_ready", "mug_staged_on_table")):
        x = np.arange(len(pairs))
        for offset, index, color, name in ((-0.2, 1, MUTED, "surface direction only"),
                                           (0.2, 2, GREEN, "plus the fitted axis")):
            scores = [summary(pair[index])["predicates"][predicate] for pair in pairs]
            drawn = ax.bar(x + offset, [score["precision"] for score in scores], 0.38, color=color, label=name)
            for bar, score in zip(drawn, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, score["precision"] + 0.01,
                        f"{score['precision']:.2f}\n{score['false_true']} wrong", ha="center", fontsize=8.5)
        ax.set_xticks(x, [pair[0] for pair in pairs])
        ax.set_ylim(0.5, 1.12)
        ax.set_title(f"{predicate}: precision", loc="left", fontsize=11)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=2)
    save(fig, "release_results.png")


FIGURES = (mesh_offset, depth_steps, fits_on_frames, shadow, quality_gap, position_results, axis_results,
           release_results)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for figure in FIGURES:
        try:
            figure()
        except (FileNotFoundError, ImportError) as missing:
            skipped += 1
            print(f"skipped {figure.__name__}: {missing}")
            continue
        print(f"wrote {figure.__name__}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
