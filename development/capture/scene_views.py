"""Save what every RoboTwin camera sees after one reset, for visual inspection.

Starts only the local RoboTwin simulator through the same ``LocalSimulator``
reset that ``run_skills.py`` uses. It does not start the policy, planner or
Modal clients, and it never moves the robot. Unlike ``rgbd_snapshot``, it keeps
all cameras RoboTwin renders (including ``front_camera`` and the third-person
observer, which the harness ignores) and writes viewable PNGs:

    python -m development.capture.scene_views --seed 3 --output data/scenes/seed_3
    python -m development.capture.scene_views --seed 3 --output data/scenes/seed_3_viewer --viewer

Each output folder gets ``<camera>_rgb.png``, ``<camera>_depth.png``,
``observer_camera_rgb.png``, an ``overview.png`` contact sheet and ``cameras.json``.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from harness.environment.layout import CAMERAS

# Depth colour scale; the table and robot sit roughly 0.2-2 m from the cameras.
DEPTH_DISPLAY_RANGE_MM = (200.0, 2000.0)
CONTACT_SHEET_COLUMNS = 3


def depth_to_rgb(depth_mm):
    """Colour depth (millimetres) near-blue to far-red; pixels without depth stay black."""
    from matplotlib import colormaps

    depth_mm = np.asarray(depth_mm, dtype=float)
    low, high = DEPTH_DISPLAY_RANGE_MM
    scaled = np.clip((depth_mm - low) / (high - low), 0.0, 1.0)
    rgb = (colormaps["turbo"](scaled)[..., :3] * 255).astype(np.uint8)
    rgb[depth_mm <= 0] = 0
    return rgb


def camera_summary(name, data):
    """Describe one camera from RoboTwin's ``get_obs()['observation'][name]``."""
    rgb = np.asarray(data["rgb"])
    summary = {"image_hw": list(rgb.shape[:2]), "used_by_harness": name in CAMERAS,
               "position_m": [round(float(v), 3) for v in np.asarray(data["cam2world_gl"])[:3, 3]],
               "focal_px": [round(float(np.asarray(data["intrinsic_cv"])[i, i]), 1) for i in (0, 1)]}
    if "depth" in data:
        valid = np.asarray(data["depth"])
        valid = valid[valid > 0]
        summary["depth_range_m"] = ([round(float(valid.min()) / 1000, 3), round(float(valid.max()) / 1000, 3)]
                                    if valid.size else None)
    return summary


def save_contact_sheet(tiles, path, seed):
    os.environ.setdefault("MPLBACKEND", "Agg")  # files only; no display needed
    import matplotlib.pyplot as plt

    rows = (len(tiles) + CONTACT_SHEET_COLUMNS - 1) // CONTACT_SHEET_COLUMNS
    figure, axes = plt.subplots(rows, CONTACT_SHEET_COLUMNS, figsize=(4.2 * CONTACT_SHEET_COLUMNS, 3.4 * rows),
                                squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (title, image) in zip(axes.ravel(), tiles):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
    figure.suptitle(f"hanging_mug, seed {seed}  (* = used by the harness)")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def capture(robotwin_root, output, seed, *, viewer=False):
    """Reset once, save every camera to ``output`` and return the report.

    With ``viewer``, also open SAPIEN's interactive window and block until it is closed.
    """
    from PIL import Image
    from development.simulator.local_simulator import LocalSimulator

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    simulator = LocalSimulator(robotwin_root, rgbd_enabled=True)
    if viewer:
        simulator.setup["render_freq"] = 1  # build_task sets 0 (headless)
    try:
        payload, _timings = simulator("127.0.0.1", 0, "/reset", {"seed": seed})
        task = simulator.task
        # observation_of() keeps only the harness cameras; ask RoboTwin for all of them.
        rendered = task.get_obs()["observation"]

        cameras, tiles = {}, []
        for name, data in sorted(rendered.items()):
            cameras[name] = camera_summary(name, data)
            marker = " *" if name in CAMERAS else ""
            rgb = np.ascontiguousarray(np.asarray(data["rgb"], dtype=np.uint8)[..., :3])
            Image.fromarray(rgb).save(output / f"{name}_rgb.png")
            tiles.append((f"{name}{marker}  rgb", rgb))
            if "depth" in data:
                depth = depth_to_rgb(data["depth"])
                Image.fromarray(depth).save(output / f"{name}_depth.png")
                tiles.append((f"{name}{marker}  depth", depth))

        observer = np.asarray(task.cameras.get_observer_rgb(), dtype=np.uint8)
        Image.fromarray(observer).save(output / "observer_camera_rgb.png")
        cameras["observer_camera"] = {"image_hw": list(observer.shape[:2]), "used_by_harness": False}
        tiles.insert(0, ("observer_camera (third person)", observer))
        save_contact_sheet(tiles, output / "overview.png", seed)

        report = {"schema_version": 1, "mode": "scene_views", "task": "hanging_mug", "seed": seed,
                  "robot_actions_after_reset": 0, "policy_started": False, "planner_started": False,
                  "robot_state_14": [round(float(v), 4) for v in payload["state"]],
                  "cameras": cameras}
        (output / "cameras.json").write_text(json.dumps(report, indent=2) + "\n")

        if viewer:
            print("Viewer open: right-drag to orbit, middle-drag to pan, scroll to zoom. "
                  "Close the window to exit.", flush=True)
            while not task.viewer.closed:
                task.viewer.render()
        return report
    finally:
        simulator.close()


def print_camera_table(report, output):
    print(f"seed {report['seed']}: saved to {output}")
    print(f"{'camera':16s} {'size':8s} {'harness':8s} {'depth range (m)':17s} position (m)")
    for name, entry in report["cameras"].items():
        size = "x".join(map(str, entry["image_hw"][::-1]))
        print(f"{name:16s} {size:8s} {'yes' if entry['used_by_harness'] else 'no':8s} "
              f"{str(entry.get('depth_range_m', '-')):17s} {entry.get('position_m', '-')}")


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing directories are never overwritten")
    parser.add_argument("--robotwin-root", type=Path, default=root / "RoboTwin")
    parser.add_argument("--viewer", action="store_true",
                        help="also open SAPIEN's interactive 3D window (needs a display)")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    # RoboTwin changes the working directory during start-up, so resolve paths first.
    output = args.output.resolve()
    report = capture(args.robotwin_root.resolve(), output, args.seed, viewer=args.viewer)
    print_camera_table(report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
