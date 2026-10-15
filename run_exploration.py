"""R8 bounded exploration campaigns using the existing single-episode runner."""

import argparse
import json
from pathlib import Path
import yaml

from harness.exploration.config import ExploreSettings
from harness.exploration.controller import ExplorationController
from harness.exploration.process import ProcessAttemptRunner
from harness.memory import MemorySnapshot
from harness.planner.anthropic import AnthropicSettings


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/explore.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robotwin-root", type=Path, default=root / "RoboTwin")
    parser.add_argument("--planner", choices=("scripted", "anthropic"), default="scripted")
    parser.add_argument("--anthropic-model")
    parser.add_argument("--development-oracle", action="store_true")
    parser.add_argument("--memory-snapshot", type=Path)
    parser.add_argument("--memory-sha256")
    args = parser.parse_args(argv)
    if not args.development_oracle:
        parser.error("Exploration currently requires --development-oracle; visual grounding is not implemented")
    with args.config.open() as stream:
        settings = ExploreSettings.model_validate(yaml.safe_load(stream))
    if args.planner == "anthropic":
        if args.anthropic_model:
            AnthropicSettings(model=args.anthropic_model)  # Validate before any simulator startup.
        elif any(v.settings.anthropic is None for v in settings.variants):
            parser.error("Every Anthropic variant needs a model, or pass --anthropic-model")
    if args.memory_sha256 and not args.memory_snapshot:
        parser.error("--memory-sha256 requires --memory-snapshot")
    memory = None
    snapshot = None
    if args.memory_snapshot:
        path = args.memory_snapshot.resolve()
        snapshot = MemorySnapshot.load(path, expected_sha256=args.memory_sha256)
        memory = (path, snapshot.sha256)
    runner = ProcessAttemptRunner(robotwin_root=args.robotwin_root, planner=args.planner,
                                  model=args.anthropic_model, memory=memory)
    summary = ExplorationController(settings, runner=runner, output=args.output,
        memory=snapshot,
        metadata=dict(planner=args.planner, model_override=args.anthropic_model,
                      memory_snapshot_sha256=memory[1] if memory else None,
                      development_oracle=True)).run()
    print(json.dumps(summary, indent=2))
    return 0 if summary["stop_reason"] in ("schedule_complete", "attempt_budget_exhausted",
                                           "acquisition_stop_requested") else 1


if __name__ == "__main__":
    raise SystemExit(main())
