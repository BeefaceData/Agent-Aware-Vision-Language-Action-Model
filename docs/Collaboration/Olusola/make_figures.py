"""Regenerate every figure in perception_guide.pdf from captures, SAM 3 caches and reports.

Run from the repository root:

    conda run -n robotwin env PYTHONPATH=$PWD python docs/Collaboration/Olusola/make_figures.py

Then rebuild the PDF:

    cd docs/Collaboration/Olusola && conda run -n texbuild tectonic -X compile perception_guide.tex

What must exist first
---------------------
Nothing under data/ is committed, so a fresh clone has none of the inputs below. A figure
whose inputs are missing is skipped with a message; the others are still written.

1. Environments
   - `robotwin` conda env (numpy, matplotlib, Pillow and this repository on PYTHONPATH).
   - `texbuild` conda env with tectonic, only to rebuild the PDF:
         conda create -n texbuild -c conda-forge tectonic

2. Labelled captures (GPU: Modal or the local simulator; one GPU job at a time)
   - data/perception/frames/seed_0..4         first capture, head + wrist cameras
   - data/perception/frames_cam/seed_0..4     second capture, with the perception camera
   - data/perception/frames_holdout/seed_5..9 held-out capture, with the perception camera
   For example:
         SIM_GPU=L4,A10G,T4 modal run modal_apps/simulator.py::frames --seeds 0,1,2,3,4 \
           --perception-camera default --output-root data/perception/frames_cam
         conda run -n robotwin python -m development.capture.labeled_frames --seed 5 \
           --output data/perception/frames_holdout/seed_5 --perception-camera default

3. SAM 3 answers for every capture folder (Modal GPU, after `modal deploy modal_apps/sam3.py`)
         python -m development.replays.sam3_cache --frames data/perception/frames_cam/seed_0
   Each folder then has sam3_cache.json, which the frame figures read for the mug masks.

4. Evaluation reports (CPU, seconds each). Each figure reads <report>/summary.json:
   - baseline             --frames data/perception/frames/seed_0..4, with the rules before the toppled fix
   - fix_toppled          the same frames, with the toppled fix
   - without_camera       --frames data/perception/frames_cam/seed_0..4 --ignore-camera perception_camera
   - with_camera          --frames data/perception/frames_cam/seed_0..4
   - holdout_with_camera  --frames data/perception/frames_holdout/seed_5..9
         python -m development.replays.perception_evaluation \
           --frames data/perception/frames_cam/seed_0 ... --frames data/perception/frames_cam/seed_4 \
           --output data/perception/reports/with_camera
   `baseline` and `fix_toppled` describe code that no longer exists on this branch; their
   summaries are kept, not regenerated. The evaluator refuses an existing --output folder,
   so move an old report aside before re-scoring it.

The probe and render-count figures use fixed numbers (see their functions); update those
by hand from sam3_mug_probe.py output or a new timing measurement.
"""

import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from harness.perception.replay import load_public_observation

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "perception"
REPORTS = DATA / "reports"
OUT = Path(__file__).resolve().parent / "figures"

INK, MUTED, BLUE, ORANGE, GREEN, RED, GRID = "#1f2937", "#6b7280", "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
CAMERA_TITLES = {"head_camera": "head camera", "left_camera": "left wrist", "right_camera": "right wrist",
                 "perception_camera": "perception camera (new)"}
PREDICATES = ("mug_lifted", "release_ready", "mug_staged_on_table", "right_mug_lifted")
PREDICATE_LABELS = ["mug_lifted", "release_ready", "mug_staged\n_on_table", "right_mug\n_lifted"]
# The provider keeps SAM 3 mug masks at or above this score (RGBDPerceptionProvider min_confidence).
MUG_MIN_CONFIDENCE = 0.60


# ---------------------------------------------------------------- inputs
def summary(report):
    return json.loads((REPORTS / report / "summary.json").read_text())


def observation(capture, sequence):
    return load_public_observation(DATA / capture / f"observations/observation_{sequence:08d}.npz")


def mug_masks(capture, sequence, camera):
    """(confidence, mask) for every cached SAM 3 mug detection in one frame and camera."""
    cache = json.loads((DATA / capture / "sam3_cache.json").read_text())
    masks = []
    for detection in cache["frames"][str(sequence)][camera]["detections"]:
        if detection["label"] == "mug":
            with Image.open(BytesIO(base64.b64decode(detection["mask_png"]))) as image:
                masks.append((detection["confidence"], np.asarray(image) > 0))
    return masks


def vertical_normal_map(frame, mask, erode_px=2, max_depth_step_m=0.01):
    """|n_z| per mask pixel, NaN elsewhere: the image form of RGBDCamera.vertical_normal_components."""
    inner = mask.copy()
    for _ in range(erode_px):
        eroded = inner.copy()
        eroded[1:] &= inner[:-1]
        eroded[:-1] &= inner[1:]
        eroded[:, 1:] &= inner[:, :-1]
        eroded[:, :-1] &= inner[:, 1:]
        inner = eroded
    depth = frame.depth_m.astype(float)
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    camera_xyz = (np.stack((u, v, np.ones_like(u)), -1).astype(float) @ np.linalg.inv(frame.intrinsic).T) * depth[..., None]
    normals = np.cross(camera_xyz[1:-1, 2:] - camera_xyz[1:-1, :-2],
                       camera_xyz[2:, 1:-1] - camera_xyz[:-2, 1:-1]) @ frame.world_to_camera[:3, :3]
    length = np.linalg.norm(normals, axis=-1)
    neighbours = (depth[1:-1, 2:], depth[1:-1, :-2], depth[2:, 1:-1], depth[:-2, 1:-1])
    usable = (inner[1:-1, 1:-1] & (depth[1:-1, 1:-1] > 0) & (length > 1e-12)
              & np.all([value > 0 for value in neighbours], axis=0)
              & (np.abs(neighbours[0] - neighbours[1]) <= max_depth_step_m)
              & (np.abs(neighbours[2] - neighbours[3]) <= max_depth_step_m))
    result = np.full((height, width), np.nan)
    result[1:-1, 1:-1][usable] = np.abs(normals[usable, 2] / length[usable])
    return result


# ---------------------------------------------------------------- drawing helpers
def overlay(rgb, mask, color, alpha=0.55):
    image = rgb.astype(float) / 255
    image[mask] = image[mask] * (1 - alpha) + np.array(matplotlib.colors.to_rgb(color)) * alpha
    return image


def draw_normals(ax, capture, sequence, camera, title, color, pad_px=18):
    """Crop around the best mug mask and colour it by |n_z|; return the image artist."""
    frame = observation(capture, sequence)
    _confidence, mask = max(mug_masks(capture, sequence, camera), key=lambda item: item[0])
    normal_z = vertical_normal_map(frame.camera_frames[camera], mask)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad_px, 0), ys.max() + pad_px
    x0, x1 = max(xs.min() - pad_px, 0), xs.max() + pad_px
    ax.imshow(frame.images[camera][y0:y1, x0:x1] / 255 * 0.55)
    artist = ax.imshow(normal_z[y0:y1, x0:x1], cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{title}\nmedian |n_z| = {np.nanmedian(normal_z):.2f}", fontsize=9.5, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def bars(ax, labels, series, colors, names, title, ylabel, fmt="{:.2f}", ylim=None):
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for index, (values, color, name) in enumerate(zip(series, colors, names)):
        drawn = ax.bar(x + (index - (len(series) - 1) / 2) * width, values, width * 0.92, color=color, label=name)
        for bar, value in zip(drawn, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=len(series))


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figures
def four_cameras():
    """Section 7.1: one moment from every camera; only the new camera finds the right-held mug."""
    capture, sequence = "frames_cam/seed_0", 416
    frame = observation(capture, sequence)
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.7))
    for ax, camera in zip(axes, ("head_camera", "left_camera", "right_camera", "perception_camera")):
        found = [item for item in mug_masks(capture, sequence, camera) if item[0] >= MUG_MIN_CONFIDENCE]
        image = frame.images[camera] / 255
        for _confidence, mask in found:
            image = overlay(frame.images[camera], mask, ORANGE)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GREEN if found else GRID)
            spine.set_linewidth(3)
        ax.set_title(CAMERA_TITLES[camera] + (f"\nmug found (score {found[0][0]:.2f})" if found else "\nno mug found"),
                     fontsize=9.5, color=GREEN if found else MUTED)
    save(fig, "four_cameras.png")


def normals():
    """Section 7.2: upright vs toppled from the head camera, and the end-on wrist view that fooled it."""
    cases = [("frames/seed_0", 32, "head_camera", "Upright, head camera", GREEN, "supported: OK"),
             ("frames/seed_3", 560, "head_camera", "Toppled, head camera", ORANGE, "rejected: faces up"),
             ("frames_cam/seed_1", 560, "right_camera", "Toppled, right wrist (end-on)", RED,
              "looks upright: wrist not trusted")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (capture, sequence, camera, title, color, verdict) in zip(axes, cases):
        artist = draw_normals(ax, capture, sequence, camera, title, color)
        ax.set_title(ax.get_title() + f"\n{verdict}", fontsize=9.5, color=color)
    fig.colorbar(artist, ax=axes, fraction=0.02, pad=0.01).set_label("|n_z|: 0 = side wall, 1 = faces up")
    save(fig, "normals.png")


def holdout_end_on():
    """Section 7.2 caveat: the same head-camera rule on two held-out toppled mugs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    artist = draw_normals(axes[0], "frames_holdout/seed_7", 640, "head_camera",
                          "Seed 7: toppled, side-on to head camera\n-> rejected (faces up)", GREEN, pad_px=20)
    draw_normals(axes[1], "frames_holdout/seed_6", 640, "head_camera",
                 "Seed 6: toppled, base facing head camera\n-> looks upright: wrong 'release_ready'", RED, pad_px=20)
    fig.colorbar(artist, ax=axes, fraction=0.03, pad=0.02).set_label("|n_z|: 0 sideways, 1 up")
    save(fig, "holdout_end_on.png")


def toppled_results():
    """Section 7.2: wrong trues and precision before and after the toppled fix (seeds 0-4)."""
    before, after = summary("baseline")["predicates"], summary("fix_toppled")["predicates"]
    names = ("release_ready", "mug_staged_on_table")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    bars(axes[0], list(names), [[before[n]["false_true"] for n in names], [after[n]["false_true"] for n in names]],
         [RED, GREEN], ["baseline", "after toppled fix"], "Wrong 'true' answers (seeds 0-4)", "frames", "{:.0f}",
         (0, 1.2 * max(before[n]["false_true"] for n in names) + 1))
    bars(axes[1], list(names), [[before[n]["precision"] for n in names], [after[n]["precision"] for n in names]],
         [RED, GREEN], ["baseline", "after toppled fix"], "Precision: when it says true, is it true?", "precision",
         ylim=(0, 1.15))
    save(fig, "toppled_results.png")


def camera_results():
    """Section 7.1: detection and recall on the same frames, without and with the perception camera."""
    without, with_camera = summary("without_camera"), summary("with_camera")
    states = ["gripped_on_table", "held_left", "held_right", "toppled", "touching_rack"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7))
    bars(axes[0], [state.replace("_", "\n") for state in states],
         [[without["by_state"][s]["mug_detection_rate"] for s in states],
          [with_camera["by_state"][s]["mug_detection_rate"] for s in states]],
         [MUTED, BLUE], ["head + wrists", "+ perception camera"], "Mug found, by situation", "detection rate",
         ylim=(0, 1.18))
    bars(axes[1], PREDICATE_LABELS,
         [[without["predicates"][p]["recall"] for p in PREDICATES],
          [with_camera["predicates"][p]["recall"] for p in PREDICATES]],
         [MUTED, BLUE], ["head + wrists", "+ perception camera"], "Recall by predicate", "recall", ylim=(0, 1.05))
    save(fig, "camera_results.png")


def holdout_results():
    """Section 10: development seeds 0-4 against held-out seeds 5-9, both with the camera."""
    development, held_out = summary("with_camera")["predicates"], summary("holdout_with_camera")["predicates"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7))
    for ax, (metric, title) in zip(axes, (("precision", "Precision: when it says true, is it true?"),
                                          ("recall", "Recall: how many true frames it catches"))):
        bars(ax, PREDICATE_LABELS,
             [[development[p][metric] for p in PREDICATES], [held_out[p][metric] for p in PREDICATES]],
             [BLUE, ORANGE], ["seeds 0-4 (development)", "seeds 5-9 (held out)"], title, metric, ylim=(0, 1.15))
    save(fig, "holdout_results.png")


def probe():
    """Section 7.6: SAM 3 prompt options on seeds 0-4.

    Fixed numbers: the first three rows come from `python -m development.replays.sam3_mug_probe`
    (single-camera options and the fallback used only where "mug" finds nothing), the last from
    the two-camera agreement analysis (agreement within 0.08 m, side prompt at >= 0.20).
    """
    options = [("'mug' at 0.30", 6, 8), ("'mug lying on its side' at 0.40", 13, 24),
               ("side prompt fallback at 0.60", 8, 4), ("side fallback + 2 cameras agree", 6, 0)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(options))[::-1]
    ax.barh(y + 0.18, [found for _name, found, _wrong in options], 0.34, color=GREEN, label="missed mugs found (of 16)")
    ax.barh(y - 0.18, [wrong for _name, _found, wrong in options], 0.34, color=RED, label="wrong groundings added")
    for row, (_name, found, wrong) in zip(y, options):
        ax.text(found + 0.3, row + 0.18, str(found), va="center", fontsize=9)
        ax.text(wrong + 0.3, row - 0.18, str(wrong), va="center", fontsize=9)
    ax.set_yticks(y, [name for name, _found, _wrong in options])
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("SAM 3 prompt probe (seeds 0-4)", loc="left", fontsize=11)
    save(fig, "probe.png")


def renders():
    """Section 7.7: RGB-D renders in one 900-action capture.

    Fixed numbers: before, two renders per action (1800); after, one per saved frame, which
    also serves the policy (58). Minutes assume ~0.8 s per render on the laptop RTX 2050.
    """
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.barh([1, 0], [1800, 58], color=[RED, GREEN])
    ax.set_yticks([1, 0], ["before: 2 renders per action", "after: only frames that are used"])
    for row, value, text in ((1, 1800, "1800 renders, ~24 min"), (0, 58, "58 renders, ~1 min")):
        ax.text(value + 20, row, text, va="center", fontsize=9)
    ax.set_xlim(0, 3300)
    ax.set_xlabel("RGB-D renders per 900-action capture (~0.8 s each on a laptop GPU)")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "renders.png")


FIGURES = (four_cameras, normals, holdout_end_on, toppled_results, camera_results, holdout_results, probe, renders)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for figure in FIGURES:
        try:
            figure()
        except FileNotFoundError as missing:
            skipped += 1
            print(f"skipped {figure.__name__}: missing {missing.filename}")
            continue
        print(f"wrote {figure.__name__}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
