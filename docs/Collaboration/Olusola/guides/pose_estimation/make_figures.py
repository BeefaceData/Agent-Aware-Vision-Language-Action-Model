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
4. Steps 6-9: review_dev / review_holdout (frames_cam, frames_holdout scored at commit 4257459),
   step9_dev / step9_holdout (the same, at the step 9 code), and for the fresh seeds
   data/perception/frames_fresh/seed_10..14: fresh_with_camera and fresh_without_camera
   (commit 4257459), fresh_before_pose (commit aa6a0b1, before any pose code) and fresh_step9
   (step 9 code).
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
from harness.runtime.assembly import EXPLORE_RACK_MASK_MIN_CONFIDENCE, MUG_MASK_MIN_CONFIDENCE

ROOT = Path(__file__).resolve().parents[5]  # the repository root
DATA = ROOT / "data" / "perception"
REPORTS = DATA / "reports"
MUG_MESHES = ROOT / "RoboTwin" / "assets" / "objects" / "039_mug"
OUT = Path(__file__).resolve().parent / "figures"

INK, MUTED, BLUE, ORANGE, GREEN, RED, GRID = "#1f2937", "#6b7280", "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
# The episode's provider keeps SAM 3 masks at or above these scores (harness/runtime/assembly.py);
# the caches use the explore_mug_rack prompt profile, with its own rack floor.
MIN_MASK_CONFIDENCE = {"mug": MUG_MASK_MIN_CONFIDENCE, "table": 0.60, "rack": EXPLORE_RACK_MASK_MIN_CONFIDENCE}
ON_TABLE_STATES = ("resting", "gripped_on_table", "staged")


# ---------------------------------------------------------------- inputs
def capture_dir(seed):
    folder = "frames_fresh" if seed >= 10 else "frames_holdout" if seed >= 5 else "frames_cam"
    return DATA / folder / f"seed_{seed}"


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
    surface = masked_surface(frame, mask, table_height_m=table_z)
    fit, _reason = fit_cylinder(surface.points, surface.normals, MUG)
    pose, reason = object_pose({camera: surface}, MUG, table_height_m=table_z,
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
                surface = masked_surface(frame, mask, table_height_m=table_z)
                fit, _reason = fit_cylinder(surface.points, surface.normals, MUG)
                if fit is None:
                    continue
                error = np.degrees(np.arccos(min(1.0, abs(float(fit.axis @ truth)))))
                (bad if error > 20 else good).append(fit.inlier_fraction)
    frame_obs = observation(6, 640)
    surface = masked_surface(frame_obs.camera_frames["head_camera"], best_mask(6, 640, "head_camera", "mug"),
                             table_height_m=table_height(6, 640, frame_obs))
    seed6, _reason = fit_cylinder(surface.points, surface.normals, MUG)
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


def mug_surface(seed, sequence, camera):
    frame_obs = observation(seed, sequence)
    table_z = table_height(seed, sequence, frame_obs)
    surface = masked_surface(frame_obs.camera_frames[camera], best_mask(seed, sequence, camera, "mug"),
                             table_height_m=table_z)
    pose, _reason = object_pose({camera: surface}, MUG, table_height_m=table_z, observation_sequence=sequence)
    fit, _reason = fit_cylinder(surface.points, surface.normals, MUG)
    return frame_obs, surface, fit, pose


def handle_band(surface, fit):
    """Points 0.8-3.5 cm outside the fitted radius: what the pose reads as the handle."""
    offsets = surface.points - fit.centre
    radial = offsets - np.outer(offsets @ fit.axis, fit.axis)
    distance = np.linalg.norm(radial, axis=1)
    return (distance >= fit.radius_m + 0.008) & (distance <= fit.radius_m + 0.035)


def handle_and_base():
    """Step 6 on real frames: the handle beside an upright mug, and the inside wall of a lying one."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    # Upright: handle points sit just outside the fitted radius, on one side.
    seed, sequence, camera = 0, 0, "head_camera"
    frame_obs, surface, fit, pose = mug_surface(seed, sequence, camera)
    frame = frame_obs.camera_frames[camera]
    mask = best_mask(seed, sequence, camera, "mug")
    axes[0].imshow(overlay(frame_obs.images[camera], mask, ORANGE, 0.25))
    draw_cylinder(axes[0], frame, fit, *fit.extent, GREEN)
    uv = project(frame, surface.points[handle_band(surface, fit)])
    axes[0].scatter(uv[:, 0], uv[:, 1], s=5, color=BLUE, label="points just outside the radius: the handle")
    middle = np.asarray(pose.key_points["handle"])
    on_axis = np.asarray(fit.centre) + float((middle - fit.centre) @ fit.axis) * fit.axis
    start, end = project(frame, np.array([on_axis, on_axis + 0.07 * np.asarray(pose.heading)]))
    axes[0].annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
    axes[0].set_title("Upright (seed 0, head camera)\nred arrow = heading, from the axis to the handle",
                      fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    crop_to(axes[0], mask)
    # Lying down: wall normals that face the axis are the inside, seen through the opening.
    seed, sequence, camera = 1, 720, "perception_camera"
    frame_obs, surface, fit, pose = mug_surface(seed, sequence, camera)
    frame = frame_obs.camera_frames[camera]
    mask = best_mask(seed, sequence, camera, "mug")
    axes[1].imshow(np.asarray(frame_obs.images[camera]) / 255 * 0.55 + 0.45)
    offsets = surface.points - fit.centre
    radial = offsets - np.outer(offsets @ fit.axis, fit.axis)
    distance = np.linalg.norm(radial, axis=1)
    wall = (np.abs(surface.normals @ fit.axis) < np.sin(np.radians(12))) & (distance > 0.25 * fit.radius_m)
    inward = wall & (np.sum(surface.normals * radial, axis=1) < -0.5 * distance)
    for chosen, color, name in ((wall & ~inward, MUTED, "outside wall"), (inward, BLUE, "inside wall (faces the axis)")):
        uv = project(frame, surface.points[chosen])
        axes[1].scatter(uv[:, 0], uv[:, 1], s=4, color=color, label=name)
    low, high = fit.extent
    base = np.asarray(pose.position)
    opening = base + (high - low) * np.asarray(pose.axis)
    for point, color, name in ((base, GREEN, "base (away from the opening)"), (opening, ORANGE, "opening")):
        u, v = project(frame, point[None])[0]
        axes[1].plot(u, v, "o", ms=11, color=color, mec="white", mew=1.5, label=name)
    axes[1].set_title(f"Lying down (seed 1, perception camera), tilt {pose.tilt_deg():.0f} deg\n"
                      "we see inside, so the opening faces us", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    crop_to(axes[1], mask)
    save(fig, "handle_and_base.png")


def rack_fit():
    """Step 7 and 9 on a real frame: pole line, peg line, peg tip, seat, and RoboTwin's success target."""
    from development.replays.perception_evaluation import hang_success_target
    from harness.perception.object_pose import RACK, pole_with_peg_pose

    seed, sequence = 0, 0
    frame_obs = observation(seed, sequence)
    table_z = table_height(seed, sequence, frame_obs)
    surfaces = {camera: masked_surface(frame, best_mask(seed, sequence, camera, "rack"))
                for camera, frame in frame_obs.camera_frames.items()
                if best_mask(seed, sequence, camera, "rack") is not None}
    pose, reason = pole_with_peg_pose(surfaces, RACK, table_height_m=table_z, observation_sequence=sequence)
    label = labels(seed)[0]
    # RoboTwin checks only the target's xy (and a height floor), so draw it at the seat's height.
    target = np.r_[hang_success_target(label)[:2], pose.key_points["peg_seat"][2]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, camera in zip(axes, ("right_camera", "head_camera")):
        frame = frame_obs.camera_frames[camera]
        mask = best_mask(seed, sequence, camera, "rack")
        ax.imshow(overlay(frame_obs.images[camera], mask, ORANGE, 0.3))
        foot, tip, seat = (np.asarray(pose.position), np.asarray(pose.key_points["peg_tip"]),
                           np.asarray(pose.key_points["peg_seat"]))
        top = foot + 0.22 * np.asarray(pose.axis)
        for start, end, color, name in ((foot, top, BLUE, "pole line"), (seat, tip, GREEN, "peg line")):
            uv = project(frame, np.array([start, end]))
            ax.plot(uv[:, 0], uv[:, 1], color=color, lw=2.6, label=name)
        for point, marker, color, name in ((tip, "o", RED, "peg tip"), (seat, "o", GREEN, "peg seat (step 9)"),
                                           (target, "x", INK, "success target xy (label, offline)")):
            u, v = project(frame, point[None])[0]
            ax.plot(u, v, marker, ms=10, mew=2.5, color=color, mec="white" if marker == "o" else color,
                    label=name)
        ax.set_title(camera.replace("_", " "), fontsize=10)
        crop_to(ax, mask, 20)
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"Rack from depth, seed 0 frame 0 ({reason.replace('_', ' ')})", x=0.02, ha="left", fontsize=11)
    save(fig, "rack_fit.png")


# (name, report before the pose work, report at the frozen commit 4257459) per seed set.
SEED_SETS = (("seeds 0-4\n(tuning)", "with_camera", "review_dev"),
             ("seeds 5-9\n(seen once)", "holdout_with_camera", "review_holdout"),
             ("seeds 10-14\n(fresh)", "fresh_before_pose", "fresh_with_camera"))


def fresh_results():
    """Step 8: the frozen code on fresh seeds 10-14, beside the seeds it was built on."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), gridspec_kw={"width_ratios": [1.9, 1, 1, 1]})
    x = np.arange(len(SEED_SETS))
    for offset, index, color, name in ((-0.2, 1, MUTED, "before the pose work"), (0.2, 2, GREEN, "with the fitted axis")):
        scores = [summary(seed_set[index])["predicates"]["release_ready"] for seed_set in SEED_SETS]
        drawn = axes[0].bar(x + offset, [score["precision"] for score in scores], 0.36, color=color, label=name)
        for bar, score in zip(drawn, scores):
            axes[0].text(bar.get_x() + bar.get_width() / 2, score["precision"] + 0.01,
                         f"{score['precision']:.2f}\n{score['false_true']} wrong", ha="center", fontsize=7.5)
    axes[0].set_ylim(0.5, 1.18)
    axes[0].set_title("release_ready precision", loc="left", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    overall = [summary(seed_set[2])["pose_overall"] for seed_set in SEED_SETS]
    # Medians share an axis; the worst frame goes in the label, because one 173-degree frame would flatten the bars.
    for ax, key, scale, unit, title in ((axes[1], "axis_error_deg", 1, "deg", "Mug axis error, median"),
                                        (axes[2], "base_error_m", 1000, "mm", "Mug base error, median")):
        medians = [scale * pose[key]["median"] for pose in overall]
        ax.bar(x, medians, 0.5, color=BLUE)
        for position, value, pose in zip(x, medians, overall):
            ax.text(position, value, f"{value:.1f}\nworst {scale * pose[key]['max']:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(medians) * 1.5)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontsize=10.5)
    wrong = [pose["confident_wrong"] for pose in overall]
    drawn = axes[3].bar(x, wrong, 0.5, color=RED)
    axes[3].bar_label(drawn, fontsize=9)
    axes[3].axhline(0, color=INK, lw=1)
    axes[3].set_ylim(0, 5.5)
    axes[3].set_ylabel("frames")
    axes[3].set_title("Confident wrong poses (target: 0)", loc="left", fontsize=10.5)
    for ax in axes:
        ax.set_xticks(x, [seed_set[0] for seed_set in SEED_SETS], fontsize=8)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    save(fig, "fresh_results.png")


def fresh_failures():
    """Step 8's confident wrong poses: a rim read as a handle, and an upside-down mug read as upright."""
    from development.replays.perception_evaluation import true_handle_heading, true_mug_base

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    seed, sequence, camera = 11, 208, "head_camera"
    frame_obs, surface, fit, pose = mug_surface(seed, sequence, camera)
    frame = frame_obs.camera_frames[camera]
    mask = best_mask(seed, sequence, camera, "mug")
    label = next(item for item in labels(seed) if item["sequence"] == sequence)
    axes[0].imshow(overlay(frame_obs.images[camera], mask, ORANGE, 0.2))
    band = surface.points[handle_band(surface, fit)]
    uv = project(frame, band)
    axes[0].scatter(uv[:, 0], uv[:, 1], s=5, color=BLUE, label="band read as the handle")
    on_axis = np.asarray(fit.centre) + float((np.median(band, axis=0) - fit.centre) @ fit.axis) * fit.axis
    for direction, color, name in ((pose.heading, RED, "fitted heading"), (true_handle_heading(label), GREEN, "true heading (label, offline)")):
        start, end = project(frame, np.array([on_axis, on_axis + 0.08 * np.asarray(direction)]))
        axes[0].annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4))
        axes[0].plot([], [], color=color, lw=2.4, label=name)
    axes[0].set_title("Rim read as a handle (seed 11, frame 208)\nthe real handle is out of view, behind the mug", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    crop_to(axes[0], mask, 42)
    seed, sequence, camera = 13, 464, "head_camera"
    frame_obs, surface, fit, pose = mug_surface(seed, sequence, camera)
    frame = frame_obs.camera_frames[camera]
    mask = best_mask(seed, sequence, camera, "mug")
    label = next(item for item in labels(seed) if item["sequence"] == sequence)
    axes[1].imshow(overlay(frame_obs.images[camera], mask, ORANGE, 0.3))
    for point, color, name in ((pose.position, RED, "base we reported (the rim end)"), (true_mug_base(label), GREEN, "true base (label, offline)")):
        u, v = project(frame, np.asarray(point, dtype=float)[None])[0]
        axes[1].plot(u, v, "o", ms=11, color=color, mec="white", mew=1.5, label=name)
    axes[1].set_title(f"Upside down while tipping (seed 13, frame 464)\ntrue tilt {label['mug']['tilt_deg']:.0f} deg; the axis is within 45 deg of vertical, so it was read as 'up'", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    crop_to(axes[1], mask, 42)
    save(fig, "fresh_failures.png")


def camera_coverage():
    """Live runs have no perception camera: frames with a trusted pose with and without it."""
    sets = (("seeds 0-4", "pose_dev_with_camera", "pose_dev_without_camera"),
            ("seeds 5-9", "pose_holdout_with_camera", "pose_holdout_without_camera"),
            ("seeds 10-14", "fresh_with_camera", "fresh_without_camera"))
    fig, ax = plt.subplots(figsize=(7, 3.1))
    x = np.arange(len(sets))
    for offset, index, color, name in ((-0.2, 2, BLUE, "head and wrists (what live runs have today)"),
                                       (0.2, 1, ORANGE, "plus the perception camera")):
        values = [100 * summary(item[index])["pose_overall"]["coverage"] for item in sets]
        drawn = ax.bar(x + offset, values, 0.36, color=color, label=name)
        ax.bar_label(drawn, fmt="%.0f", fontsize=9)
    ax.set_xticks(x, [item[0] for item in sets])
    ax.set_ylim(0, 100)
    ax.set_ylabel("frames with a trusted pose (%)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "camera_coverage.png")


def seat_results():
    """Step 9: the depth peg seat against RoboTwin's hang target, every episode, beside the peg tip."""
    episodes = {}
    for report in ("step9_dev", "step9_holdout", "fresh_step9"):
        episodes.update(summary(report)["rack"]["first_pose"])
    names = sorted(episodes, key=lambda name: int(name.split("_")[1]))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(names))
    for offset, key, color, name in ((-0.2, "peg_tip_error_m", MUTED, "peg tip vs the label's peg end (3D)"),
                                     (0.2, "seat_xy_error_m", GREEN, "peg seat vs the hang target (xy)")):
        values = [100 * episodes[name][key] if episodes[name] else np.nan for name in names]
        ax.bar(x + offset, values, 0.38, color=color, label=name)
    ax.axhline(2.0, color=RED, lw=1.6, ls="--")
    ax.text(len(names) - 0.5, 2.05, "RoboTwin's hang tolerance: 2 cm", color=INK, fontsize=8.5, ha="right", va="bottom")
    ax.axvspan(9.5, len(names) - 0.5, color=GREEN, alpha=0.06)
    ax.text(12, 2.6, "fresh seeds", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [name.replace("seed_", "") for name in names])
    ax.set_xlabel("seed (first confirmed rack pose)")
    ax.set_ylabel("error (cm)")
    ax.set_ylim(0, 3.0)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "seat_results.png")


FIGURES = (mesh_offset, depth_steps, fits_on_frames, shadow, quality_gap, position_results, axis_results,
           release_results, handle_and_base, rack_fit, fresh_results, fresh_failures, camera_coverage, seat_results)


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
