"""Synchronous append-only JSONL; logging failures are propagated to the caller."""

import json
from pathlib import Path


def append_jsonl(path: Path, record: dict) -> None:
    # Encode before opening: invalid data must not leave a partial JSON record.
    line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
