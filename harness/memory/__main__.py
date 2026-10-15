"""Offline snapshot creation/inspection. No simulator, provider or key access."""

import argparse
import json
from pathlib import Path
from .snapshot import MemoryCatalog, MemorySnapshot, MAX_BYTES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    build = sub.add_parser("build", help="Validate a catalog and create a new canonical snapshot")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("inspect", help="Show public retrieval without raw evidence references")
    inspect.add_argument("snapshot", type=Path)
    inspect.add_argument("--phase", default="APPROACH_MUG")
    migrate = sub.add_parser("migrate", help="Verify and copy an older snapshot using current skill field names")
    migrate.add_argument("--snapshot", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "build":
        with args.catalog.open("rb") as stream:
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            parser.error("catalog too large")
        snapshot = MemorySnapshot(MemoryCatalog.model_validate_json(data))
        snapshot.export(args.output)
        print(json.dumps({"sha256": snapshot.sha256, "output": str(args.output)}))
    elif args.operation == "migrate":
        snapshot = MemorySnapshot.migrate_legacy(args.snapshot, args.output)
        print(json.dumps({"sha256": snapshot.sha256, "output": str(args.output)}))
    else:
        snapshot = MemorySnapshot.load(args.snapshot)
        selected = snapshot.retrieve(task="hanging_mug", phase=args.phase)
        print(json.dumps({"sha256": snapshot.sha256, "memory": selected.planner_payload()}, indent=2))


if __name__ == "__main__":
    main()
