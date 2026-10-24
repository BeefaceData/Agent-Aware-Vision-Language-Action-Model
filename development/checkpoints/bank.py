"""Filesystem bank with immutable assets and atomic checkpoint metadata writes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from harness.legacy.schemas import FailureType, TaskPhase
from harness.environment.base import copy_images
from development.checkpoints.schemas import FailureCheckpoint, RecordedFrame, SafeID, relative_path
from pydantic import TypeAdapter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prefix(bank, checkpoint):
    frames = [RecordedFrame.model_validate_json(line) for line in
              bank.resolve_artifact(checkpoint.trace_path).read_text().splitlines()]
    if len(frames) != checkpoint.step + 1 or [frame.step for frame in frames] != list(range(checkpoint.step + 1)):
        raise ValueError("checkpoint requires a complete contiguous reset/action prefix")
    if frames[-1].command_state != checkpoint.state:
        raise ValueError("checkpoint state does not match recorded prefix")
    if tuple(frame.command_state for frame in frames[-len(checkpoint.recent_states):]) != checkpoint.recent_states:
        raise ValueError("checkpoint state history does not match recorded prefix")
    if checkpoint.recent_actions and tuple(frame.action for frame in frames[-len(checkpoint.recent_actions):]) != checkpoint.recent_actions:
        raise ValueError("checkpoint action history does not match recorded prefix")
    return frames


def write_json_atomic(path: Path, data: dict, *, replace=False):
    encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic no-overwrite publication.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CheckpointBank:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def resolve_artifact(self, relative: str) -> Path:
        relative_path(relative)
        path = (self.path / relative).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError("artifact escapes the checkpoint bank")
        return path

    def _metadata_path(self, checkpoint_id):
        TypeAdapter(SafeID).validate_python(checkpoint_id)
        return self.resolve_artifact(f"records/{checkpoint_id}.json")

    def verify_artifacts(self, checkpoint):
        for relative, expected in checkpoint.artifact_hashes.items():
            if sha256(self.resolve_artifact(relative)) != expected:
                raise ValueError(f"checkpoint artifact checksum mismatch: {relative}")

    def add(self, checkpoint: FailureCheckpoint):
        checkpoint = FailureCheckpoint.model_validate(checkpoint.model_dump(warnings=False))
        self.verify_artifacts(checkpoint)
        self.load_images(checkpoint)
        load_prefix(self, checkpoint)
        write_json_atomic(self._metadata_path(checkpoint.checkpoint_id), checkpoint.model_dump(mode="json"))
        return checkpoint

    def get(self, checkpoint_id: str) -> FailureCheckpoint:
        checkpoint = FailureCheckpoint.model_validate_json(self._metadata_path(checkpoint_id).read_text())
        if checkpoint.checkpoint_id != checkpoint_id:
            raise ValueError("checkpoint filename/ID mismatch")
        return checkpoint

    def query(self, *, phase=None, failure_type=None):
        phase = None if phase is None else TaskPhase(phase)
        failure_type = None if failure_type is None else FailureType(failure_type)
        results = []
        for path in sorted((self.path / "records").glob("*.json")):
            record = self.get(path.stem)
            if (phase is None or record.phase == phase) and (failure_type is None or record.failure_type == failure_type):
                results.append(record)
        return results

    def load_images(self, checkpoint):
        images = {}
        for camera, relative in checkpoint.images.items():
            # Own the file handle even if NumPy fails while opening a damaged ZIP.
            with self.resolve_artifact(relative).open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
                images[camera] = archive[camera]
        return copy_images(images)

    def relabel(self, checkpoint_id, *, phase, failure_type, notes=None):
        current = self.get(checkpoint_id)
        updated = FailureCheckpoint.model_validate({
            **current.model_dump(), "phase": phase, "failure_type": failure_type,
            "notes": current.notes if notes is None else notes,
            "label_revision": current.label_revision + 1,
        })
        self.verify_artifacts(updated)
        # Preserve prior labels before replacing only the metadata, never assets.
        archive = self.resolve_artifact(f"label_history/{checkpoint_id}/{current.label_revision}.json")
        write_json_atomic(archive, current.model_dump(mode="json"))
        write_json_atomic(self._metadata_path(checkpoint_id), updated.model_dump(mode="json"), replace=True)
        return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "show", "verify", "label"])
    parser.add_argument("checkpoint_id", nargs="?")
    parser.add_argument("--path", default="data/checkpoints")
    parser.add_argument("--phase", choices=[x.value for x in TaskPhase])
    parser.add_argument("--failure-type", choices=[x.value for x in FailureType])
    parser.add_argument("--notes")
    args = parser.parse_args()
    bank = CheckpointBank(args.path)
    if args.command != "list" and args.checkpoint_id is None:
        parser.error("checkpoint_id is required")
    try:
        if args.command == "list":
            result = [x.model_dump(mode="json") for x in bank.query(phase=args.phase, failure_type=args.failure_type)]
        elif args.command == "label":
            if args.phase is None or args.failure_type is None:
                parser.error("label requires --phase and --failure-type")
            result = bank.relabel(args.checkpoint_id, phase=args.phase, failure_type=args.failure_type,
                                  notes=args.notes).model_dump(mode="json")
        else:
            checkpoint = bank.get(args.checkpoint_id)
            result = checkpoint.model_dump(mode="json")
            if args.command == "verify":
                bank.verify_artifacts(checkpoint)
                load_prefix(bank, checkpoint)
                result = {"checkpoint_id": checkpoint.checkpoint_id, "artifacts_verified": True,
                          "camera_shapes": {k: list(v.shape) for k, v in bank.load_images(checkpoint).items()}}
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        parser.exit(1, f"checkpoint operation failed: {type(exc).__name__}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
