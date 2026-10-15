"""Extract an earlier recorded camera boundary without simulator or policy calls."""

import argparse
import io
import json
from pathlib import PurePosixPath
from zipfile import BadZipFile

import numpy as np

from harness.trace import copy_images
from .checkpoint_bank import CheckpointBank, load_prefix, sha256, write_json_atomic
from .schemas import CAMERAS, FailureCheckpoint


def extract_checkpoint(bank, source, step, *, history=40):
    if type(step) is not int or not 0 <= step < source.step:
        raise ValueError("boundary must precede the source checkpoint")
    if type(history) is not int or history < 1:
        raise ValueError("history must be a positive integer")
    bank.verify_artifacts(source)
    frames = load_prefix(bank, source)[:step + 1]
    # CaptureRecorder's canonical layout; missing keyframes cannot be interpolated.
    image_relative = str(PurePosixPath(source.trace_path).parent / f"images_{step:06d}.npz")
    image_source = bank.resolve_artifact(image_relative)
    image_bytes = image_source.read_bytes()
    with np.load(io.BytesIO(image_bytes), allow_pickle=False) as archive:
        copy_images({name: archive[name] for name in CAMERAS})
    provenance = json.loads(bank.resolve_artifact(source.provenance_path).read_text())
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        raise ValueError("unsupported source provenance")
    identity = f"{source.episode_id}_step_{step:06d}"
    destination = bank.resolve_artifact(f"extracted/{identity}")
    destination.mkdir(parents=True, exist_ok=False)
    image_path = destination / f"images_{step:06d}.npz"
    image_path.write_bytes(image_bytes)
    trace_path = destination / "actions.jsonl"
    trace_path.write_text("".join(frame.model_dump_json() + "\n" for frame in frames))
    relative = lambda path: path.relative_to(bank.path).as_posix()
    provenance_path = destination / "provenance.json"
    write_json_atomic(provenance_path, {**provenance, "extraction": {
        "source_checkpoint_id": source.checkpoint_id,
        "source_label_revision": source.label_revision,
        "source_artifact_hashes": source.artifact_hashes,
        "selected_step": step, "source_image_path": image_relative,
        "source_image_sha256": sha256(image_path),
        "image_integrity_scope": "periodic image first hashed at extraction; requires replay validation",
        "simulator_called": False, "policy_called": False,
    }})
    evaluation_path = destination / "evaluation.json"
    write_json_atomic(evaluation_path, {
        "schema_version": 1, "reason": "retrospective_recorded_boundary",
        "source_checkpoint_id": source.checkpoint_id, "confirmed_steps": step,
        "success_at_last_confirmed_frame": None,
        "note": "Source terminal failure and labels are not inherited at this earlier boundary.",
    })
    assets = [image_path, trace_path, provenance_path, evaluation_path]
    return bank.add(FailureCheckpoint(
        checkpoint_id=identity, run_id=source.run_id, episode_id=source.episode_id,
        seed=source.seed, step=step, state=frames[-1].command_state,
        images={name: relative(image_path) for name in CAMERAS},
        recent_states=tuple(f.command_state for f in frames[-history:]),
        recent_actions=tuple(f.action for f in frames[1:][-history:]),
        trace_path=relative(trace_path), provenance_path=relative(provenance_path),
        evaluation_path=relative(evaluation_path),
        artifact_hashes={relative(path): sha256(path) for path in assets},
        notes="Extracted recorded boundary; review labels and verify replay before recovery.",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_id")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--path", default="data/checkpoints")
    args = parser.parse_args()
    bank = CheckpointBank(args.path)
    try:
        checkpoint = extract_checkpoint(bank, bank.get(args.checkpoint_id), args.step)
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        parser.exit(1, f"extraction rejected: {type(exc).__name__}\n")
    print(json.dumps({"checkpoint_id": checkpoint.checkpoint_id, "step": checkpoint.step}))


if __name__ == "__main__":
    main()
