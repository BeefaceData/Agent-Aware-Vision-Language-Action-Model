"""Validated assessment records for the legacy pass-through supervisor."""

from pathlib import Path

from harness.jsonl import append_jsonl
from harness.legacy.schemas import AssessmentRecord


class AssessmentLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: AssessmentRecord) -> None:
        # Revalidate nested containers even if the caller mutated a model's dict.
        validated = AssessmentRecord.model_validate(record.model_dump(warnings=False))
        append_jsonl(self.path, validated.model_dump(mode="json"))
