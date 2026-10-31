"""Single-use simulator owner for the unverified entry-660 comparison."""

import json
from pathlib import Path

from development.hanging.validate_entry660 import load_candidate, validate_entry
from harness.environment.perception_cameras import DEFAULT_PERCEPTION_CAMERA, describe


def validate_package(path, *, allow_unverified=False):
    checkpoint, _, _ = load_candidate(path, allow_unverified=allow_unverified)
    provenance = json.loads((Path(path) / checkpoint.provenance_path).read_text())
    if provenance["manifest"].get("perception_camera") != describe(DEFAULT_PERCEPTION_CAMERA):
        raise ValueError("original_perception_camera_required")
    return checkpoint


def run(path, output, robotwin_root, *, allow_unverified=False):
    validate_package(path, allow_unverified=allow_unverified)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Native imports and GPU initialization follow package validation.
    from development.simulator.local_simulator import LocalSimulator
    from harness.environment.local import LocalEnvironmentClient
    from harness.perception.observations import ObservationStore
    from development.demos.episode_video import EpisodeVideo
    from harness.environment.layout import CAMERAS

    simulator = None
    last = None
    video = None
    video_error = None
    report = dict(status="initialization_error", full_state_restored=False,
                  capture_provenance="unknown", executed_steps=0)
    try:
        simulator = LocalSimulator(robotwin_root, rgbd_enabled=True,
                                  perception_cameras=(DEFAULT_PERCEPTION_CAMERA,))
        environment = LocalEnvironmentClient(simulator, render_once_per_step=True)
        try:
            video = EpisodeVideo(output, cameras=(*CAMERAS, DEFAULT_PERCEPTION_CAMERA.name))
        except Exception as exc:
            video_error = type(exc).__name__

        def record(kind):
            nonlocal video_error
            if video is not None and video_error is None:
                try:
                    video.append(last, kind, last.sequence, 0)
                except Exception as exc:
                    # Recording failure must not change the recorded action prefix.
                    video_error = type(exc).__name__

        class RecordingBoundary:
            def reset(self, seed):
                nonlocal last
                last = environment.reset(seed)
                record("unverified_replay_reset")
                return last

            def step(self, action):
                nonlocal last
                last = environment.step(action)
                record("unverified_recorded_action_replay")
                return last

        report = validate_entry(path, RecordingBoundary(), allow_unverified=allow_unverified,
                                time_limit_seconds=1800)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        if video is not None:
            try:
                video.close()
                report["video"] = video.metadata()
            except Exception as exc:
                video_error = type(exc).__name__
        if video_error is not None:
            report["video_error_type"] = video_error
        if last is not None:
            try:
                report["last_confirmed_observation"] = ObservationStore(output / "observations").save(last)
                report["last_confirmed_sequence"] = last.sequence
            except Exception as exc:
                report["observation_save_error"] = type(exc).__name__
        if simulator is not None:
            try:
                simulator.close()
            except Exception as exc:
                report["cleanup_error"] = type(exc).__name__
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
