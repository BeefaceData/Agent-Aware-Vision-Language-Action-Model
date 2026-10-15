"""Freeze evaluation inputs offline, then run that exact oracle-assisted condition."""

import argparse
import json
from pathlib import Path
import yaml

from harness.evaluation.manifest import EvaluationSettings, FrozenEvaluation, freeze
from harness.evaluation.controller import EvaluationController
from harness.exploration.process import ProcessAttemptRunner
from harness.memory import MemorySnapshot


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("freeze", help="Create a new immutable evaluation contract; no API/GPU use")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--memory-snapshot", type=Path)
    prepare.add_argument("--exploration-summary", type=Path, action="append", default=[])
    prepare.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run", help="Run the frozen contract without configuration overrides")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--sha256", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--development-oracle", action="store_true")
    args = parser.parse_args(argv)
    if args.operation == "freeze":
        with args.config.open() as stream:
            settings = EvaluationSettings.model_validate(yaml.safe_load(stream))
        memory = MemorySnapshot.load(args.memory_snapshot) if args.memory_snapshot else None
        frozen = freeze(settings, root=root, memory=memory, exploration_summaries=args.exploration_summary)
        frozen.save(args.output)
        print(json.dumps({"sha256": frozen.digest, "condition": settings.condition,
                          "excluded_seeds": frozen.excluded_seeds, "seeds": settings.seeds}))
        return 0
    if not args.development_oracle:
        parser.error("This evaluation condition requires explicit --development-oracle")
    frozen = FrozenEvaluation.load(args.manifest, expected_sha256=args.sha256)
    runner = ProcessAttemptRunner(robotwin_root=root/"RoboTwin", planner=frozen.settings.planner)
    summary = EvaluationController(frozen, root=root, output=args.output, runner=runner).run()
    print(json.dumps(summary, indent=2))
    return 0 if summary["protocol_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
