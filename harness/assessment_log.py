"""Synchronous append-only JSONL; logging failures are propagated to the runner."""

import json
from pathlib import Path

from .schemas import AssessmentRecord


def append_jsonl(path: Path, record: dict) -> None:
    # Encode before opening: invalid data must not leave a partial JSON record.
    line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()


class AssessmentLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: AssessmentRecord) -> None:
        # Revalidate nested containers even if the caller mutated a model's dict.
        validated = AssessmentRecord.model_validate(record.model_dump(warnings=False))
        append_jsonl(self.path, validated.model_dump(mode="json"))
