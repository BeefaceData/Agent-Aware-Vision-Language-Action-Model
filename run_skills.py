"""Scripted or Anthropic acquisition development runner; baselines stay independent."""

import argparse
from functools import partial
import json
from pathlib import Path
import yaml

from harness.assessment_log import append_jsonl
from harness.planner.scripted import GraspRetryPlanner
from harness.planner.anthropic import AnthropicPlanner, AnthropicSettings
from harness.planner.prompt_builder import prompt_bundle
from harness.skill_config import RunSettings
from harness.memory import MemorySnapshot


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this episode")
    parser.add_argument("--config", type=Path, default=root / "configs/skill_runtime.yaml")
    parser.add_argument("--robotwin-root", type=Path, default=root / "RoboTwin")
    parser.add_argument("--planner", choices=("scripted", "anthropic"), default="scripted")
    parser.add_argument("--anthropic-model", help="Explicit API model ID; may also be set in config.anthropic.model")
    parser.add_argument("--memory-snapshot", type=Path, help="Optional fixed task/global memory snapshot")
    parser.add_argument("--memory-sha256", help="Optional expected snapshot digest; reject mismatches")
    parser.add_argument("--evaluation-manifest", type=Path, help="Internal frozen evaluation episode contract")
    parser.add_argument("--evaluation-sha256", help="Required expected evaluation manifest hash")
    parser.add_argument("--development-oracle", action="store_true",
                        help="Explicitly allow simulator geometry/contact evidence; not evaluation")
    args = parser.parse_args(argv)
    if not args.development_oracle:
        parser.error("This development recipe requires --development-oracle; visual grounding is not implemented")
    with args.config.open() as stream:
        settings = RunSettings.model_validate(yaml.safe_load(stream))
    if args.memory_sha256 and not args.memory_snapshot:
        parser.error("--memory-sha256 requires --memory-snapshot")
    memory = (MemorySnapshot.load(args.memory_snapshot.resolve(), expected_sha256=args.memory_sha256)
              if args.memory_snapshot else None)
    planner_settings = settings.anthropic
    if args.planner == "anthropic":
        if planner_settings is None and not args.anthropic_model:
            parser.error("Anthropic requires --anthropic-model or config.anthropic.model; there is no fallback model")
        options = (planner_settings.model_dump() if planner_settings else
                   dict(max_calls=min(100, settings.max_turns),
                        max_vla_attempts=settings.recipe.max_retries+1, vla=settings.recipe.vla.model_dump()))
        if args.anthropic_model:
            options["model"] = args.anthropic_model
        planner_settings = AnthropicSettings.model_validate(options)
    check = lambda: None
    frozen = None
    if args.evaluation_manifest:
        if not args.evaluation_sha256 or not args.memory_snapshot:
            parser.error("Frozen evaluation requires a manifest digest and memory snapshot")
        from harness.evaluation.manifest import FrozenEvaluation
        from harness.evaluation.guard import episode_guard
        frozen = FrozenEvaluation.load(args.evaluation_manifest.resolve(), expected_sha256=args.evaluation_sha256)
        check = episode_guard(frozen, root=root, seed=args.seed, settings=settings,
            planner=args.planner, model_override=args.anthropic_model, memory_path=args.memory_snapshot.resolve(),
            manifest_path=args.evaluation_manifest.resolve(), robotwin_root=args.robotwin_root)
    elif args.evaluation_sha256:
        parser.error("--evaluation-sha256 requires --evaluation-manifest")
    output, robotwin_root = args.output.resolve(), args.robotwin_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=2, seed=args.seed, mode=f"{args.planner}_development",
                    planner=args.planner, development_oracle=True, completion_predicate="mug_lifted",
                    memory_snapshot_sha256=memory.sha256 if memory else None,
                    settings=settings.model_dump(mode="json"))
    if frozen:
        manifest.update(mode="eval", evaluation_sha256=frozen.digest,
                        condition=frozen.settings.condition, remote_policy_weights_verified=False)
    if memory:
        memory.export(output / "memory_snapshot.json")
    if args.planner == "anthropic":
        manifest.update(anthropic=planner_settings.model_dump(mode="json"),
                        prompt_sha256=prompt_bundle()[1], planner_oracle_inputs=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Simulator imports/startup occur only after CLI/config validation. Resolve
    # paths first: existing RoboTwin startup changes the process working directory.
    from development.local_simulator import LocalSimulator
    from development.skill_oracle import oracle_predicates, oracle_anchor_preparer
    from harness.environment.local import LocalEnvironmentClient
    from harness.perception.observations import ObservationStore
    from harness.skill_orchestrator import SkillOrchestrator
    from harness.skills import SkillExecutor
    from harness.vla.modal_client import ModalVLAClient

    simulator = None
    planner = None
    try:
        # Missing credentials/SDK fail before allocating the simulator GPU.
        planner = (AnthropicPlanner(planner_settings) if args.planner == "anthropic"
                   else GraspRetryPlanner(settings.recipe))
        simulator = LocalSimulator(robotwin_root)
        environment = LocalEnvironmentClient(simulator)
        if frozen:
            from harness.evaluation.guard import SingleResetEnvironment
            environment = SingleResetEnvironment(environment)
        environment.reset(args.seed)  # Exactly one reset, owned by this runner.
        oracle_log = partial(append_jsonl, output / "oracle.jsonl")
        executor = SkillExecutor(environment, ModalVLAClient(),
            log=partial(append_jsonl, output / "skill_events.jsonl"), budget=settings.budget,
            predicates=oracle_predicates(simulator.task, log=oracle_log),
            allow_development_oracle=True)
        controller = SkillOrchestrator(executor, planner,
            log=partial(append_jsonl, output / "assessments.jsonl"),
            save_observation=ObservationStore(output / "observations").save,
            episode_id=output.name, seed=args.seed, max_turns=settings.max_turns,
            memory=memory,
            invariant_check=check,
            prepare=oracle_anchor_preparer(simulator.task, executor.anchors, log=oracle_log))
        summary = controller.run()
        check()
    except Exception as exc:
        summary = dict(outcome="runner_error", task_success=None,
                       milestone_reached=False, error_type=type(exc).__name__)
    finally:
        if simulator is not None:
            try:
                simulator.close()
            except Exception as exc:
                # Preserve the episode result even if native cleanup fails.
                summary["cleanup_error_type"] = type(exc).__name__
        if isinstance(planner, AnthropicPlanner):
            try:
                planner.close()
            except Exception as exc:
                summary["planner_cleanup_error_type"] = type(exc).__name__
    if frozen:
        summary.update(mode="eval", evaluation_sha256=frozen.digest, condition=frozen.settings.condition)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("outcome") in ("acquisition_confirmed", "task_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
