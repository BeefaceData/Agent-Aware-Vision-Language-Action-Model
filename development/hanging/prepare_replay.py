"""Convert a baseline prefix offline; capture provenance remains explicitly unknown."""

import argparse
import json
from pathlib import Path
import shutil

from development.checkpoints.bank import CheckpointBank, sha256, write_json_atomic
from development.checkpoints.schemas import FailureCheckpoint, RecordedFrame
from harness.environment.layout import CAMERAS
from harness.perception.replay import load_public_observation


def prepare_replay(original, recovered, output, *, step=660):
    original, recovered, output = map(Path, (original, recovered, output))
    if type(step) is not int or step < 1:
        raise ValueError("positive_boundary_required")
    manifest_path = recovered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary_path = original / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (manifest.get("mode") != "frozen_policy_baseline"
            or summary.get("mode") != "frozen_policy_baseline"
            or type(manifest.get("seed")) is not int or manifest["seed"] < 0
            or summary.get("seed") != manifest["seed"]
            or summary.get("execution_uncertain") is not False
            or summary.get("error_type") is not None
            or type(summary.get("policy_actions")) is not int
            or step > summary["policy_actions"]):
        raise ValueError("inconsistent_baseline_metadata")

    def observation_path(sequence):
        name = Path("observations") / f"observation_{sequence:08d}.npz"
        return recovered / name if (recovered / name).is_file() else original / name

    reset_path, boundary_path = observation_path(0), observation_path(step)
    reset = load_public_observation(reset_path)
    boundary = load_public_observation(boundary_path)
    if reset.sequence != 0 or boundary.sequence != step:
        raise ValueError("wrong_observation_boundary")
    frames = [RecordedFrame(step=0, action=None, command_state=reset.command_state)]
    journal_path = original / "actions.jsonl"
    with journal_path.open() as stream:
        for index in range(1, step + 1):
            lines = (stream.readline(), stream.readline())
            if not all(line.strip() for line in lines):
                raise ValueError("incomplete_or_reordered_prefix")
            request, confirmed = map(json.loads, lines)
            if (request.get("event") != "action_requested"
                    or confirmed.get("event") != "action_confirmed"
                    or request.get("index") != index or confirmed.get("index") != index
                    or request.get("sequence") != index - 1
                    or confirmed.get("sequence") != index):
                raise ValueError("incomplete_or_reordered_prefix")
            frames.append(RecordedFrame(step=index, action=request["action"],
                                        command_state=confirmed["command_state"]))
    if frames[-1].command_state != tuple(boundary.command_state):
        raise ValueError("boundary_state_mismatch")

    # Validate everything above before publishing a new, never-overwritten bank.
    output.mkdir(parents=True, exist_ok=False)
    image_path = output / "boundary.npz"
    shutil.copyfile(boundary_path, image_path)
    trace_path = output / "prefix.jsonl"
    trace_path.write_text("".join(frame.model_dump_json() + "\n" for frame in frames))
    provenance_path = output / "provenance.json"
    write_json_atomic(provenance_path, dict(schema_version=1, local=None,
        capture_provenance="unknown", verified_replay_eligible=False,
        manifest=manifest,
        source_sha256={"manifest": sha256(manifest_path), "summary": sha256(summary_path),
                       "journal": sha256(journal_path), "reset": sha256(reset_path),
                       "boundary": sha256(boundary_path)},
        hash_scope="computed during offline conversion; not capture-time attestation",
        full_state_restored=False, remote_policy_state_restored=False))
    evaluation_path = output / "evaluation.json"
    write_json_atomic(evaluation_path, dict(schema_version=1,
        reason="candidate_hanging_entry_requires_simulator_validation",
        entry_verified=False, replay_executed=False, full_state_restored=False))
    assets = [image_path, trace_path, provenance_path, evaluation_path]
    checkpoint = FailureCheckpoint(checkpoint_id=f"unverified_hang_{step}",
        run_id=original.name, episode_id=original.name, seed=manifest["seed"], step=step,
        state=frames[-1].command_state,
        images={camera: image_path.name for camera in CAMERAS},
        recent_states=tuple(frame.command_state for frame in frames[-40:]),
        recent_actions=tuple(frame.action for frame in frames[1:][-40:]),
        trace_path=trace_path.name, evaluation_path=evaluation_path.name,
        provenance_path=provenance_path.name,
        artifact_hashes={path.name: sha256(path) for path in assets},
        notes="UNVERIFIED: capture provenance unknown; existing verified replay must reject.")
    return CheckpointBank(output).add(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--recovered", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", type=int, default=660)
    args = parser.parse_args()
    checkpoint = prepare_replay(args.original, args.recovered, args.output, step=args.step)
    print(f"Prepared {checkpoint.checkpoint_id}; provenance unknown; no replay executed.")


if __name__ == "__main__":
    main()
