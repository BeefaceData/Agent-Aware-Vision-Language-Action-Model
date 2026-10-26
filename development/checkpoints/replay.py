"""Restore development checkpoints via seeded reset and recorded actions, never policy inference."""

import argparse
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from harness.environment.base import copy_images, vector14
from development.checkpoints.bank import CheckpointBank, load_prefix, write_json_atomic
from development.checkpoints.provenance import local_provenance
from development.checkpoints.schemas import CAMERAS, FailureCheckpoint, ReplayOptions, ReplayReport


class ReplayEnvironment:
    """Development-only adapter. Reports observed agreement, not full physics restoration.

    The caller must reserve exclusive use of the matching simulator. Artifact,
    prefix and local provenance checks run before any simulator mutation.
    """

    def __init__(self, bank, sim_client=None, *, host="127.0.0.1", port=8110, options=None):
        self.bank, self.host, self.port = bank, host, port
        if sim_client is None:
            from harness.environment.robotwin import call_sim
            sim_client = call_sim
        self.sim_client = sim_client
        self.options = ReplayOptions.model_validate((options or ReplayOptions()).model_dump())

    def restore_checkpoint(self, checkpoint):
        checkpoint = FailureCheckpoint.model_validate(checkpoint.model_dump())
        executed = 0
        provenance_status = "unchecked"
        maximum = None
        image_errors = {}

        def report(status, reason, error_type=None):
            return ReplayReport(
                checkpoint_id=checkpoint.checkpoint_id, status=status,
                expected_steps=checkpoint.step, executed_steps=executed,
                state_max_abs_error=maximum, image_mean_abs_errors=image_errors,
                local_provenance=provenance_status, reason=reason, error_type=error_type,
            )

        try:
            self.bank.verify_artifacts(checkpoint)
            frames = load_prefix(self.bank, checkpoint)
            expected_images = self.bank.load_images(checkpoint)
            provenance = json.loads(self.bank.resolve_artifact(checkpoint.provenance_path).read_text())
            if not isinstance(provenance, dict) or provenance.get("schema_version") != 1 or provenance.get("local") != local_provenance():
                provenance_status = "mismatch"
                return report("unavailable", "Local source/config/package provenance differs; no simulator request sent.")
            provenance_status = "matched"
        except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
            return report("unavailable", "Checkpoint assets or complete action prefix failed validation; no simulator request sent.", type(exc).__name__)

        try:
            observation, _ = self.sim_client(self.host, self.port, "/reset", {"seed": checkpoint.seed})
            maximum = float(np.max(np.abs(np.array(vector14(observation["state"])) - frames[0].command_state)))
            if maximum > self.options.state_atol:
                return report("drifted", "Reset command state differs; replay stopped before executing actions.")
            for frame in frames[1:]:
                observation, _ = self.sim_client(self.host, self.port, "/step", {"action": list(frame.action)})
                executed += 1
                maximum = max(maximum, float(np.max(np.abs(np.array(vector14(observation["state"])) - frame.command_state))))
                if maximum > self.options.state_atol:
                    return report("drifted", "Command-state drift detected in action prefix; replay stopped.")
            actual_images = copy_images({name: observation[name] for name in CAMERAS})
            for name in CAMERAS:
                if actual_images[name].shape != expected_images[name].shape:
                    return report("drifted", f"Camera shape differs: {name}.")
                image_errors[name] = float(np.mean(np.abs(actual_images[name].astype(float) - expected_images[name].astype(float))))
            if any(error > self.options.image_mae_tolerance for error in image_errors.values()):
                return report("drifted", "Final camera images exceed configured mean absolute error tolerance.")
            return report("matched", "Recorded command-state prefix and final RGB images agree within tolerances; full physics/task state is not verified.")
        except Exception as exc:
            return report("error", "Simulator replay failed; live state may include an unconfirmed action.", type(exc).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_id")
    parser.add_argument("--path", default="data/checkpoints")
    parser.add_argument("--sim-host", default="127.0.0.1")
    parser.add_argument("--sim-port", type=int, default=8110)
    parser.add_argument("--state-atol", type=float, default=1e-5)
    parser.add_argument("--image-mae-tolerance", type=float, default=5.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        options = ReplayOptions(state_atol=args.state_atol, image_mae_tolerance=args.image_mae_tolerance)
        bank = CheckpointBank(args.path)
        report = bank.get(args.checkpoint_id).restore(ReplayEnvironment(
            bank, host=args.sim_host, port=args.sim_port, options=options,
        ))
        result = report.model_dump(mode="json")
        if args.report:
            write_json_atomic(args.report, result)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"replay operation failed: {type(exc).__name__}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if report.status != "matched":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
