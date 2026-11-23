"""Explicitly unverified, development-only comparison of the recorded entry 660.

No policy, recovery or simulator initialization. The caller owns one reserved
environment and must not treat image agreement as proof of grasp/support.
"""

import json
import time

import numpy as np

from development.checkpoints.bank import CheckpointBank, load_prefix
from development.checkpoints.schemas import ReplayOptions
from harness.perception.replay import load_public_observation


def load_candidate(path, *, allow_unverified=False):
    if allow_unverified is not True:
        raise ValueError("explicit_unverified_replay_opt_in_required")
    bank = CheckpointBank(path)
    checkpoint = bank.get("unverified_hang_660")
    if checkpoint.step != 660 or checkpoint.seed != 3:
        raise ValueError("only_seed3_entry660_supported")
    bank.verify_artifacts(checkpoint)
    frames = load_prefix(bank, checkpoint)
    provenance = json.loads(bank.resolve_artifact(checkpoint.provenance_path).read_text())
    if (provenance.get("local") is not None
            or provenance.get("capture_provenance") != "unknown"
            or provenance.get("verified_replay_eligible") is not False):
        raise ValueError("expected_explicit_unknown_capture_provenance")
    boundary = load_public_observation(bank.resolve_artifact(checkpoint.images["head_camera"]))
    if boundary.sequence != 660 or boundary.command_state != checkpoint.state:
        raise ValueError("entry_boundary_mismatch")
    return checkpoint, frames, boundary


def validate_entry(path, environment, *, allow_unverified=False,
                   time_limit_seconds=1800, clock=time.monotonic):
    checkpoint, frames, expected = load_candidate(path, allow_unverified=allow_unverified)
    if type(time_limit_seconds) is not int or not 1 <= time_limit_seconds <= 1800:
        raise ValueError("invalid_validation_time_limit")
    limits = ReplayOptions()
    deadline = clock() + time_limit_seconds
    result = dict(status="unverified", capture_provenance="unknown",
                  full_state_restored=False, grasp_validity="unknown",
                  support_validity="unknown", executed_steps=0,
                  execution_uncertain=False, command_state_max_abs_error=0.,
                  image_mean_abs_errors={}, tcp_position_errors_m={})

    def check_state(observation, frame):
        error = float(np.max(np.abs(np.asarray(observation.command_state) - frame.command_state)))
        result["command_state_max_abs_error"] = max(error, result["command_state_max_abs_error"])
        return observation.sequence == frame.step and np.isfinite(error) and error <= limits.state_atol

    try:
        if clock() >= deadline:
            result["status"] = "time_limit"
            return result
        observation = environment.reset(checkpoint.seed)
        if not check_state(observation, frames[0]):
            result["status"] = "reset_drift"
            return result
        for frame in frames[1:]:
            if clock() >= deadline:
                result["status"] = "time_limit"
                return result
            # Never retry: an exception may follow an executed action.
            result["execution_uncertain"] = True
            observation = environment.step(frame.action)
            result["execution_uncertain"] = False
            result["executed_steps"] += 1
            if not check_state(observation, frame):
                result["status"] = "command_state_drift"
                return result
        for camera, image in expected.images.items():
            actual = observation.images.get(camera)
            if actual is None or actual.shape != image.shape:
                result["status"] = "camera_mismatch"
                return result
            result["image_mean_abs_errors"][camera] = float(
                np.mean(np.abs(actual.astype(float) - image.astype(float))))
        if expected.measured_tcp_poses and observation.measured_tcp_poses:
            for arm in ("left", "right"):
                result["tcp_position_errors_m"][arm] = float(np.linalg.norm(
                    np.asarray(expected.measured_tcp_poses[arm][:3]) -
                    observation.measured_tcp_poses[arm][:3]))
        result["status"] = ("image_drift" if any(
            error > limits.image_mae_tolerance for error in result["image_mean_abs_errors"].values())
            else "observations_agree_unverified")
        return result
    except Exception as exc:
        result.update(status="execution_error", error_type=type(exc).__name__)
        return result
