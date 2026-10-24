"""Capture one public RGB-D observation after a RoboTwin reset.

Starts only the local RoboTwin simulator. It does not start the policy, planner,
Modal client, or advance the robot after reset.
"""

import argparse
import json
from pathlib import Path

from harness.environment.base import Observation
from harness.perception.observations import ObservationStore
from harness.perception.rgbd import camera_frames_from_payload
from harness.environment.base import CAMERAS


def capture(robotwin_root, output, seed):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from development.simulator.local_simulator import LocalSimulator

    simulator = LocalSimulator(robotwin_root, rgbd_enabled=True)
    try:
        payload, _timings = simulator("127.0.0.1", 0, "/reset", {"seed": seed})
        images = {camera: payload[camera] for camera in CAMERAS}
        frames = camera_frames_from_payload(payload, images)
        if not frames:
            raise ValueError("simulator_returned_no_rgbd_frames")
        observation = Observation(0, images, tuple(payload["state"]), frames)
        ObservationStore(output / "observations").save(observation)
        manifest = {"schema_version": 1, "mode": "rgbd_snapshot", "seed": seed,
                    "simulator_started": True, "robot_actions_after_reset": 0,
                    "policy_started": False, "planner_started": False,
                    "rgbd_cameras": sorted(frames)}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        simulator.close()


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing directories are never overwritten")
    parser.add_argument("--robotwin-root", type=Path, default=root / "RoboTwin")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    manifest = capture(args.robotwin_root.resolve(), args.output, args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
