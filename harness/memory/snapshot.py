"""Canonical snapshots and deterministic, public-only, bounded retrieval.

Snapshots are copied into immutable canonical bytes on load. Changing a source
file or a returned model cannot change memory during an episode. Hashes detect
changes; they are not signatures or proof that an evidence claim is truthful.
"""

from hashlib import sha256
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictStr, model_validator
from harness.schemas import Model
from .task_memory import TaskMemory
from .global_memory import GlobalMemory

MAX_BYTES = 2_000_000
MAX_RETRIEVAL_BYTES = 12000
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class MemoryCatalog(Model):
    schema_version: Literal[1] = 1
    task_memory: Annotated[tuple[TaskMemory, ...], Field(max_length=500)] = ()
    global_memory: Annotated[tuple[GlobalMemory, ...], Field(max_length=500)] = ()

    @model_validator(mode="after")
    def unique_ids(self):
        for records in (self.task_memory, self.global_memory):
            if len({r.id for r in records}) != len(records):
                raise ValueError("duplicate_memory_id")
        return self


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class SnapshotEnvelope(Model):
    sha256: Digest
    catalog: MemoryCatalog

    @model_validator(mode="after")
    def integrity(self):
        if sha256(canonical(self.catalog.model_dump(mode="json"))).hexdigest() != self.sha256:
            raise ValueError("memory_snapshot_hash_mismatch")
        return self


class MemorySelection(Model):
    snapshot_sha256: Digest
    task_memory: Annotated[tuple[TaskMemory, ...], Field(max_length=2)] = ()
    global_memory: Annotated[tuple[GlobalMemory, ...], Field(max_length=4)] = ()

    def planner_payload(self):
        # Revalidate/projection at the provider boundary too; don't trust a
        # model_copy or manually injected planner context to bypass provenance.
        checked = MemorySelection.model_validate(self.model_dump())
        records = [r.planner_payload() for r in (*checked.task_memory, *checked.global_memory) if r.eligible]
        if len(canonical(records)) > MAX_RETRIEVAL_BYTES:
            raise ValueError("memory_context_too_large")
        return records

    def references(self):
        return [{"kind": r["kind"], "id": r["id"], "revision": r["revision"]} for r in self.planner_payload()]


class MemorySnapshot:
    def __init__(self, catalog):
        catalog = MemoryCatalog.model_validate(catalog.model_dump() if isinstance(catalog, MemoryCatalog) else catalog)
        self._data = canonical(catalog.model_dump(mode="json"))
        if len(self._data) > MAX_BYTES:
            raise ValueError("memory_catalog_too_large")
        self._sha256 = sha256(self._data).hexdigest()

    @property
    def sha256(self):
        return self._sha256

    def catalog(self):
        """Return an independent validated copy for offline freezing/export."""
        return MemoryCatalog.model_validate_json(self._data)

    @classmethod
    def load(cls, path, *, expected_sha256=None):
        path = Path(path)
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("memory_snapshot_too_large")
        envelope = SnapshotEnvelope.model_validate_json(raw)
        if expected_sha256 is not None and envelope.sha256 != expected_sha256:
            raise ValueError("unexpected_memory_snapshot")
        return cls(envelope.catalog)

    @classmethod
    def migrate_legacy(cls, source, destination):
        """Verify old bytes before normalizing field aliases into a NEW snapshot.

        No in-place rewrite, provenance change, eligibility upgrade or evaluation
        manifest re-signing. Historical frozen evaluations must be frozen anew.
        """
        with Path(source).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("memory_snapshot_too_large")
        envelope = json.loads(raw)
        if (set(envelope) != {"sha256", "catalog"}
                or sha256(canonical(envelope["catalog"])).hexdigest() != envelope["sha256"]):
            raise ValueError("memory_snapshot_hash_mismatch")
        snapshot = cls(MemoryCatalog.model_validate(envelope["catalog"]))
        snapshot.export(destination)
        return snapshot

    def export(self, path):
        envelope = canonical({"sha256": self.sha256, "catalog": json.loads(self._data)})
        if len(envelope) + 1 > MAX_BYTES:
            raise ValueError("memory_snapshot_too_large")
        # Never overwrite an existing snapshot. Offline tooling owns this write;
        # the runtime only loads or copies its already selected snapshot.
        with Path(path).open("xb") as stream:
            stream.write(envelope + b"\n")

    def retrieve(self, *, task, phase, available_targets=(
            "current_ee", "mug_pregrasp", "rack_approach", "table_approach")):
        catalog = MemoryCatalog.model_validate_json(self._data)
        selected = {"task_memory": [], "global_memory": []}
        size = 2
        for field, limit in (("task_memory", 2), ("global_memory", 4)):
            records = [r for r in getattr(catalog, field) if r.eligible
                       and r.task in (task, "all") and (phase in r.phases or "all" in r.phases)]
            records.sort(key=lambda r: ({"repeated": 0, "single_case": 1, "hypothesis": 2}[r.evidence_level],
                                        r.task != task, phase not in r.phases, r.id))
            for record in records:
                # Stored recipes must still ground into the currently supported
                # anchor set. Memory never creates new runtime capabilities.
                targets = ([s.command.parameters.target for s in record.trace if s.command.action == "move_to"]
                           if field == "task_memory" else
                           [record.recommended_staging.target] if record.recommended_staging else [])
                if any(t not in available_targets for t in targets):
                    continue
                length = len(canonical(record.planner_payload())) + 1
                if size + length > MAX_RETRIEVAL_BYTES:
                    continue
                selected[field].append(record)
                size += length
                if len(selected[field]) == limit:
                    break
        return MemorySelection(snapshot_sha256=self.sha256, **selected)
