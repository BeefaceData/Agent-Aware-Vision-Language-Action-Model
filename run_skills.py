"""Scripted or Anthropic skill runner; baseline inference stays independent."""

import argparse
from datetime import datetime
from functools import partial
import json
from pathlib import Path
import yaml

from harness.jsonl import append_jsonl
from harness.memory import MemorySnapshot
from harness.planner.anthropic import AnthropicPlanner
from harness.planner.prompt_builder import prompt_bundle
from harness.runtime import assembly
from harness.runtime.settings import RunSettings
from harness.exploration.candidates import candidates_from_attempt

RUNS_DIRECTORY = Path("data/runs")
# Memory candidate ids are "<run name>.observation" and must fit the 80-character
# memory Name limit (harness/memory/base.py); 3 more characters stay free for a
# same-minute "-NN" suffix.
MAX_RUN_NAME_LENGTH = 80 - len(".observation") - 3
# Dropped first, in order, when a config name would make the run name too long.
CONFIG_FILLER_WORDS = ("task", "table")

SUCCESSFUL_OUTCOMES = frozenset({
    "acquisition_confirmed", "left_transfer_complete", "handoff_confirmed", "task_success"})


def successful_outcome(summary):
    """Recognize an explicitly configured development milestone as success."""
    return summary.get("outcome") in SUCCESSFUL_OUTCOMES


def validate_evidence_mode(*, development_oracle, perception_factory, planner_pixel_grounding):
    """Validate evidence sources before simulator initialization."""
    if not (development_oracle or perception_factory or planner_pixel_grounding):
        raise ValueError("evidence_provider_required")
    if development_oracle and perception_factory:
        raise ValueError("oracle_and_public_perception_are_mutually_exclusive")


def default_output(root, config, seed, *, where, now=None):
    """Return a new, readable run folder under ``data/runs``.

    The name leads with a sortable date, then weekday and local time, then what
    ran: ``2026-10-28_Wed_07-55am_opus_full_task_rgbd_table_lift_seed3_local``.
    A ``-2``, ``-3``... suffix keeps runs started in the same minute apart. Long
    config names are shortened so the name fits ``MAX_RUN_NAME_LENGTH``.
    """
    now = now or datetime.now()
    when = now.strftime("%Y-%m-%d_%a_%I-%M") + now.strftime("%p").lower()
    config_name = Path(config).stem
    if config_name.startswith("skill_runtime_"):
        config_name = config_name.removeprefix("skill_runtime_")
    tail = f"_seed{seed}_{where}"
    config_name = _shorten_config_name(config_name, MAX_RUN_NAME_LENGTH - len(when) - 1 - len(tail))
    folder = Path(root) / RUNS_DIRECTORY / f"{when}_{config_name}{tail}"
    candidate, count = folder, 1
    while candidate.exists():
        count += 1
        candidate = folder.with_name(f"{folder.name}-{count}")
    return candidate


def _shorten_config_name(name, room):
    """Fit ``name`` into ``room`` characters, dropping filler words before truncating."""
    words = name.split("_")
    for filler in CONFIG_FILLER_WORDS:
        if len("_".join(words)) <= room:
            break
        words = [word for word in words if word != filler] or words
    return "_".join(words)[:max(room, 1)].rstrip("_")


def option_error(args, settings):
    """Return the first incompatible CLI/config combination, or None.

    Checked before any output directory, simulator or provider is created.
    """
    full_task = settings.anthropic is not None and settings.anthropic.task_scope == "full_task"
    for enabled, name in ((settings.perception.opus_rack_grounding, "opus_rack_grounding"),
                          (settings.perception.opus_mug_lift_grounding, "opus_mug_lift_grounding")):
        if enabled and args.planner != "anthropic":
            return f"perception.{name} requires --planner anthropic"
        if enabled and not args.perception_factory:
            return f"perception.{name} requires --perception-factory"
    if args.perception_factory and not settings.perception.rgbd_enabled:
        return "--perception-factory requires perception.rgbd_enabled: true"
    if args.perception_factory and args.evaluation_manifest:
        return "perception-factory evaluation requires a frozen detector snapshot and is not enabled yet"
    if args.planner_pixel_grounding and not settings.perception.rgbd_enabled:
        return "--planner-pixel-grounding requires perception.rgbd_enabled: true"
    if args.planner_pixel_grounding and args.planner != "anthropic":
        return "--planner-pixel-grounding requires --planner anthropic"
    if args.planner_pixel_grounding and args.evaluation_manifest:
        return "pixel-grounding evaluation requires a frozen prompt/schema snapshot and is not enabled yet"
    if settings.handoff is not None and args.evaluation_manifest:
        return "Handoff evaluation requires a frozen full-task contract and is not enabled yet"
    if args.evaluation_manifest and full_task:
        return "Full-task Anthropic evaluation requires a frozen full-task contract and is not enabled yet"
    if full_task and args.planner != "anthropic":
        return "Full-task scope requires --planner anthropic"
    if args.memory_sha256 and not args.memory_snapshot:
        return "--memory-sha256 requires --memory-snapshot"
    if args.planner == "anthropic" and settings.anthropic is None and not args.anthropic_model:
        return "Anthropic requires --anthropic-model or config.anthropic.model; there is no fallback model"
    return None


def build_parser(root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        help="New directory for this episode (default: data/runs/<date>_<weekday>_<time>"
                             "_<config>_seed<N>_local)")
    parser.add_argument("--config", type=Path, default=root / "configs/skill_runtime.yaml")
    parser.add_argument("--robotwin-root", type=Path, default=root / "RoboTwin")
    parser.add_argument("--planner", choices=("scripted", "anthropic"), default="scripted")
    parser.add_argument("--anthropic-model", help="Explicit API model ID; may also be set in config.anthropic.model")
    parser.add_argument("--memory-snapshot", type=Path, help="Optional fixed task/global memory snapshot")
    parser.add_argument("--memory-sha256", help="Optional expected snapshot digest; reject mismatches")
    parser.add_argument("--evaluation-manifest", type=Path, help="Internal frozen evaluation episode contract")
    parser.add_argument("--evaluation-sha256", help="Required expected evaluation manifest hash")
    parser.add_argument("--development-oracle", action="store_true",
                        help="Use simulator evidence only for internal skill gates; not evaluation")
    parser.add_argument("--perception-factory",
                        help="RGB-only detector factory as package.module:callable")
    parser.add_argument("--planner-pixel-grounding", action="store_true",
                        help="Allow Opus-selected RGB pixels plus same-frame depth for mug/rack anchors")
    return parser


def parse_arguments(parser, argv=None):
    """Parse the CLI and load its config; exit through ``parser.error`` on any conflict.

    Has no side effects, so remote launchers can validate before allocating a GPU.
    """
    args = parser.parse_args(argv)
    try:
        validate_evidence_mode(development_oracle=args.development_oracle,
            perception_factory=args.perception_factory,
            planner_pixel_grounding=args.planner_pixel_grounding)
    except ValueError as exc:
        parser.error(str(exc))
    with args.config.open() as stream:
        settings = RunSettings.model_validate(yaml.safe_load(stream))
    error = option_error(args, settings)
    if error:
        parser.error(error)
    return args, settings


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = build_parser(root)
    args, settings = parse_arguments(parser, argv)
    memory =(MemorySnapshot.load(args.memory_snapshot.resolve(), expected_sha256=args.memory_sha256)
              if args.memory_snapshot else None)
    planner_settings = (assembly.anthropic_settings(settings, args.anthropic_model)
                        if args.planner == "anthropic" else None)
    from harness.perception.anchors import AnchorRegistry
    anchors = AnchorRegistry()
    perception = None
    if args.perception_factory:
        # Validate/import the detector before allocating simulator resources.
        try:
            perception = assembly.load_perception_provider(args.perception_factory, anchors,
                rack_prompt_profile=settings.perception.rack_prompt_profile,
                rack_grounding_camera=settings.perception.rack_grounding_camera)
        except Exception as exc:
            parser.error(f"cannot initialize perception detector: {type(exc).__name__}: {exc}")
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

    output = (args.output or default_output(root, args.config, args.seed, where="local")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Run folder: {output}", flush=True)
    write_manifest(output, args, settings, planner_settings=planner_settings,
                   memory=memory, frozen=frozen)
    if memory:
        memory.export(output / "memory_snapshot.json")

    summary = run_episode(args, settings, output=output, planner_settings=planner_settings,
                          anchors=anchors, perception=perception, memory=memory,
                          frozen=frozen, check=check)
    # Live development runs retain reviewable task/global success and failure
    # candidates beside the trace. Frozen evaluation stays write-restricted.
    if not frozen:
        record_memory_candidates(output, summary, seed=args.seed)
    write_summary(output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if successful_outcome(summary) else 1


def write_manifest(output, args, settings, *, planner_settings, memory, frozen):
    manifest = dict(schema_version=2, seed=args.seed, mode=f"{args.planner}_development",
                    planner=args.planner, development_oracle=args.development_oracle,
                    perception_factory=args.perception_factory,
                    planner_pixel_grounding=args.planner_pixel_grounding,
                    opus_rack_grounding=settings.perception.opus_rack_grounding,
                    completion_predicate=settings.completion_predicate,
                    memory_snapshot_sha256=memory.sha256 if memory else None,
                    settings=settings.model_dump(mode="json"))
    if frozen:
        manifest.update(mode="eval", evaluation_sha256=frozen.digest,
                        condition=frozen.settings.condition, remote_policy_weights_verified=False)
    if planner_settings is not None:
        manifest.update(anthropic=planner_settings.model_dump(mode="json"),
                        prompt_sha256=prompt_bundle(planner_settings.task_scope)[1],
                        planner_oracle_inputs=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def record_memory_candidates(output, summary, *, seed):
    """Write unreviewed memory candidates and note them (or the failure) in ``summary``."""
    try:
        candidates = candidates_from_attempt(output, attempt_id=output.name,
            seed=seed, execution_kind="live")
        candidate_path = output / "candidate_catalog.json"
        candidate_path.write_text(json.dumps(candidates.model_dump(mode="json"),
            indent=2, allow_nan=False) + "\n")
        summary.update(memory_candidates={"path": candidate_path.name,
            "task": len(candidates.task_memory), "global": len(candidates.global_memory),
            "auto_promoted": False})
    except Exception as exc:
        summary["memory_candidate_error_type"] = type(exc).__name__


def run_episode(args, settings, *, output, planner_settings, anchors, perception, memory,
                frozen, check):
    """Own one simulator for exactly one reset and episode; always clean up."""
    # Simulator imports/startup occur only after CLI/config validation. Paths are
    # already resolved: RoboTwin startup changes the process working directory.
    from development.simulator.local_simulator import LocalSimulator
    from harness.perception.observations import ObservationStore
    from harness.runtime.orchestrator import SkillOrchestrator
    from harness.skills import SkillExecutor
    from harness.vla.modal_client import ModalVLAClient

    simulator = planner = video = None
    summary = dict(outcome="runner_interrupted", task_success=None, milestone_reached=False)
    rack_grounding = {"status": "disabled"}
    log_assessment = partial(append_jsonl, output / "assessments.jsonl")

    def log_initialization(event):
        log_assessment({"schema_version": 2, "episode": output.name, "seed": args.seed, **event})

    try:
        # Missing credentials/SDK fail before allocating the simulator GPU.
        planner = assembly.build_planner(args.planner, settings, planner_settings=planner_settings,
                                         pixel_grounding=args.planner_pixel_grounding)
        if settings.record_video:
            from development.demos.episode_video import EpisodeVideo
            video = EpisodeVideo(output)
        simulator = LocalSimulator(args.robotwin_root.resolve(),
            rgbd_enabled=settings.perception.rgbd_enabled,
            diagnostics=partial(append_jsonl, output / "simulator_timing.jsonl"))
        environment = open_environment(simulator, video=video, frozen=frozen)
        initial_observation = environment.reset(args.seed)  # Exactly one reset, owned by this runner.
        if settings.perception.rgbd_enabled:
            from harness.perception.rgbd import require_camera_frames
            required_cameras = {"head_camera"}
            if settings.perception.opus_rack_grounding:
                required_cameras.add(settings.perception.rack_grounding_camera)
            if args.planner_pixel_grounding:
                required_cameras.update(("head_camera", "left_camera", "right_camera"))
            require_camera_frames(initial_observation, required_cameras)

        rack_frame_cache = None
        if settings.perception.opus_rack_grounding:
            rack_frame_cache, rack_grounding = assembly.initialize_rack_frame(
                environment, perception, planner,
                camera=settings.perception.rack_grounding_camera, log=log_initialization)
        if settings.perception.opus_mug_lift_grounding:
            assembly.initialize_mug_lift_evidence(environment, perception, planner,
                log=log_initialization,
                minimum_policy_actions=settings.perception.mug_lift_first_check_after_actions)
        predicates, prepare = build_evidence(
            args, settings, environment=environment, simulator=simulator, anchors=anchors,
            perception=perception, planner=planner, rack_frame_cache=rack_frame_cache,
            oracle_log=partial(append_jsonl, output / "oracle.jsonl"))
        executor = SkillExecutor(environment, ModalVLAClient(),
            log=partial(append_jsonl, output / "skill_events.jsonl"), budget=settings.budget,
            predicates=predicates, anchors=anchors,
            allow_development_oracle=args.development_oracle)
        controller = SkillOrchestrator(executor, planner, log=log_assessment,
            save_observation=ObservationStore(output / "observations").save,
            episode_id=output.name, seed=args.seed, max_turns=settings.max_turns,
            completion_predicate=settings.completion_predicate,
            memory=memory, invariant_check=check, prepare=prepare)
        summary = controller.run()
        check()
    except KeyboardInterrupt:
        # The live-validation watchdog uses SIGINT to stop cleanly. Preserve
        # partial episode artifacts and still finalize the video/summary.
        summary = dict(outcome="interrupted", task_success=None,
                       milestone_reached=False, error_type="KeyboardInterrupt")
    except Exception as exc:
        summary = dict(outcome="runner_error", task_success=None,
                       milestone_reached=False, error_type=type(exc).__name__)
    finally:
        close_resources(summary, video=video, simulator=simulator, planner=planner)
    if frozen:
        summary.update(mode="eval", evaluation_sha256=frozen.digest, condition=frozen.settings.condition)
    if settings.perception.opus_rack_grounding:
        summary["rack_grounding_initialization"] = rack_grounding
    return summary


def open_environment(simulator, *, video, frozen):
    from harness.environment.local import LocalEnvironmentClient

    if video is not None:
        from development.demos.episode_video import RecordingEnvironment
        environment = RecordingEnvironment(simulator, video)
    else:
        environment = LocalEnvironmentClient(simulator)
    if frozen:
        from harness.evaluation.guard import SingleResetEnvironment
        environment = SingleResetEnvironment(environment)
    return environment


def build_evidence(args, settings, *, environment, simulator, anchors, perception, planner,
                   rack_frame_cache, oracle_log):
    """Choose predicates and anchor preparation for the selected evidence mode.

    Development-oracle evidence may gate execution locally, but the planner
    prompt/result projection excludes that information source.
    """
    from development.simulator.skill_oracle import oracle_anchor_preparer, oracle_predicates
    from harness.perception.preparation import PlannerPixelAnchorPreparer
    from harness.predicates.rack_seat_gate import RackSeatReleaseGate
    from harness.predicates.registry import PredicateRegistry

    # Release at a grounded rack seat is only possible with a cached rack frame.
    release_gate = RackSeatReleaseGate(environment) if rack_frame_cache is not None else None
    if perception is not None:
        return assembly.public_perception_evidence(
            perception, environment, anchors, planner=planner,
            rack_frame_cache=rack_frame_cache, release_gate=release_gate,
            hang_camera=settings.perception.rack_grounding_camera,
            pixel_grounding=args.planner_pixel_grounding)
    predicates = (oracle_predicates(simulator.task, log=oracle_log) if args.development_oracle
                  else PredicateRegistry())
    predicates.rack_seat_gate = release_gate
    if args.planner_pixel_grounding:
        return predicates, PlannerPixelAnchorPreparer(environment, anchors)
    return predicates, oracle_anchor_preparer(simulator.task, anchors, log=oracle_log)


def close_resources(summary, *, video, simulator, planner):
    """Close video, simulator and planner; record cleanup errors without raising."""
    if video is not None:
        try:
            video.close()
        except Exception as exc:
            summary["video_cleanup_error_type"] = type(exc).__name__
        summary["video"] = video.metadata()
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


if __name__ == "__main__":
    raise SystemExit(main())
