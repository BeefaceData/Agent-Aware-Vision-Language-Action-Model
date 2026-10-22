"""Scripted or Anthropic skill runner; baseline inference stays independent."""

import argparse
import importlib
from functools import partial
import json
from pathlib import Path
import time
import yaml

from harness.assessment_log import append_jsonl
from harness.planner.scripted import GraspRetryPlanner
from harness.planner.anthropic import AnthropicPlanner, AnthropicSettings
from harness.planner.prompt_builder import prompt_bundle
from harness.skill_config import RunSettings
from harness.memory import MemorySnapshot
from harness.exploration.candidates import candidates_from_attempt


def successful_outcome(summary):
    """Recognize an explicitly configured development milestone as success."""
    return summary.get("outcome") in (
        "acquisition_confirmed", "left_transfer_complete", "handoff_confirmed", "task_success")


def validate_evidence_mode(*, development_oracle, perception_factory, planner_pixel_grounding):
    """Validate evidence sources before simulator initialization."""
    if not (development_oracle or perception_factory or planner_pixel_grounding):
        raise ValueError("evidence_provider_required")
    if development_oracle and perception_factory:
        raise ValueError("oracle_and_public_perception_are_mutually_exclusive")


def load_perception_provider(factory_spec, anchors, *, rack_prompt_profile="default",
                             rack_grounding_camera="head_camera"):
    """Load an explicit RGB-only detector factory for public RGB-D grounding.

    ``factory_spec`` is ``package.module:callable``. The callable returns a
    detector implementing ``detect(images, observation_sequence=...)``; camera
    depth and calibration remain inside the harness perception provider.
    """
    from harness.perception.grounding import RGBDPerceptionProvider

    if not isinstance(factory_spec, str) or factory_spec.count(":") != 1:
        raise ValueError("perception_factory_must_be_module_callable")
    module_name, callable_name = factory_spec.split(":")
    if not module_name or not callable_name or not all(
            part.isidentifier() for part in module_name.split(".")) or not callable_name.isidentifier():
        raise ValueError("perception_factory_must_be_module_callable")
    factory = getattr(importlib.import_module(module_name), callable_name, None)
    if not callable(factory):
        raise ValueError("perception_factory_not_found")
    if rack_prompt_profile not in ("default", "explore_mug_rack"):
        raise ValueError("unsupported_rack_prompt_profile")
    detector = factory()
    if not callable(getattr(detector, "detect", None)):
        raise ValueError("perception_detector_missing_detect")
    configure_profile = getattr(detector, "configure_rack_prompt_profile", None)
    if callable(configure_profile):
        configure_profile(rack_prompt_profile)
    elif rack_prompt_profile != "default":
        raise ValueError("detector_does_not_support_rack_prompt_profile")
    configure_camera = getattr(detector, "configure_rack_grounding_camera", None)
    if callable(configure_camera):
        configure_camera(rack_grounding_camera)
    elif rack_grounding_camera != "head_camera":
        raise ValueError("detector_does_not_support_rack_grounding_camera")
    thresholds = {"mug": 0.40}
    if rack_prompt_profile == "explore_mug_rack":
        thresholds["rack"] = 0.18
    return RGBDPerceptionProvider(detector, anchors,
                                  min_confidence_by_label=thresholds)


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
                        help="Use simulator evidence only for internal skill gates; not evaluation")
    parser.add_argument("--perception-factory",
                          help="RGB-only detector factory as package.module:callable")
    parser.add_argument("--planner-pixel-grounding", action="store_true",
                        help="Allow Opus-selected RGB pixels plus same-frame depth for mug/rack anchors")
    args = parser.parse_args(argv)
    try:
        validate_evidence_mode(development_oracle=args.development_oracle,
            perception_factory=args.perception_factory,
            planner_pixel_grounding=args.planner_pixel_grounding)
    except ValueError as exc:
        parser.error(str(exc))
    with args.config.open() as stream:
        settings = RunSettings.model_validate(yaml.safe_load(stream))
    if settings.perception.opus_rack_grounding:
        if args.planner != "anthropic":
            parser.error("perception.opus_rack_grounding requires --planner anthropic")
        if not args.perception_factory:
            parser.error("perception.opus_rack_grounding requires --perception-factory")
    if settings.perception.opus_mug_lift_grounding:
        if args.planner != "anthropic":
            parser.error("perception.opus_mug_lift_grounding requires --planner anthropic")
        if not args.perception_factory:
            parser.error("perception.opus_mug_lift_grounding requires --perception-factory")
    if args.perception_factory and not settings.perception.rgbd_enabled:
        parser.error("--perception-factory requires perception.rgbd_enabled: true")
    if args.perception_factory and args.evaluation_manifest:
        parser.error("perception-factory evaluation requires a frozen detector snapshot and is not enabled yet")
    if args.planner_pixel_grounding and not settings.perception.rgbd_enabled:
        parser.error("--planner-pixel-grounding requires perception.rgbd_enabled: true")
    if args.planner_pixel_grounding and args.planner != "anthropic":
        parser.error("--planner-pixel-grounding requires --planner anthropic")
    if args.planner_pixel_grounding and args.evaluation_manifest:
        parser.error("pixel-grounding evaluation requires a frozen prompt/schema snapshot and is not enabled yet")
    if settings.handoff is not None and args.evaluation_manifest:
        parser.error("Handoff evaluation requires a frozen full-task contract and is not enabled yet")
    if (args.evaluation_manifest and settings.anthropic is not None
            and settings.anthropic.task_scope == "full_task"):
        parser.error("Full-task Anthropic evaluation requires a frozen full-task contract and is not enabled yet")
    if (settings.anthropic is not None and settings.anthropic.task_scope == "full_task"
            and args.planner != "anthropic"):
        parser.error("Full-task scope requires --planner anthropic")
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
                        task_scope="full_task" if settings.handoff is not None else "acquisition",
                        vla=(settings.handoff or settings.recipe.vla).model_dump()))
        if args.anthropic_model:
            options["model"] = args.anthropic_model
        planner_settings = AnthropicSettings.model_validate(options)
    from harness.perception.anchors import AnchorRegistry
    anchors = AnchorRegistry()
    perception = None
    if args.perception_factory:
        # Validate/import the detector before allocating simulator resources.
        try:
            perception = load_perception_provider(args.perception_factory, anchors,
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
    output, robotwin_root = args.output.resolve(), args.robotwin_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
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
    if memory:
        memory.export(output / "memory_snapshot.json")
    if args.planner == "anthropic":
        manifest.update(anthropic=planner_settings.model_dump(mode="json"),
                        prompt_sha256=prompt_bundle(planner_settings.task_scope)[1],
                        planner_oracle_inputs=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Simulator imports/startup occur only after CLI/config validation. Resolve
    # paths first: existing RoboTwin startup changes the process working directory.
    from development.local_simulator import LocalSimulator
    from development.skill_oracle import oracle_predicates, oracle_anchor_preparer
    from harness.environment.local import LocalEnvironmentClient
    from harness.perception.preparation import (PerceptionAnchorPreparer,
        PlannerPixelAnchorPreparer, HybridAnchorPreparer)
    from harness.perception.observations import ObservationStore
    from harness.predicates.registry import PredicateRegistry
    from harness.predicates.rack_seat_gate import RackSeatReleaseGate
    from harness.skill_orchestrator import SkillOrchestrator
    from harness.skills import SkillExecutor
    from harness.vla.modal_client import ModalVLAClient

    simulator = None
    planner = None
    video = None
    summary = dict(outcome="runner_interrupted", task_success=None,
                   milestone_reached=False)
    rack_grounding_initialization = {"status": "disabled"}
    rack_frame_cache = None
    try:
        # Missing credentials/SDK fail before allocating the simulator GPU.
        planner = (AnthropicPlanner(planner_settings,
                                    require_pixel_grounding=args.planner_pixel_grounding)
                   if args.planner == "anthropic"
                   else GraspRetryPlanner(settings.recipe))
        if settings.handoff is not None and args.planner == "scripted":
            from harness.planner.handoff import HandoffPlanner
            planner = HandoffPlanner(settings.recipe, settings.handoff)
        if settings.record_video:
            from development.episode_video import EpisodeVideo, RecordingEnvironment
            video = EpisodeVideo(output)
        simulator = LocalSimulator(robotwin_root, rgbd_enabled=settings.perception.rgbd_enabled,
            diagnostics=partial(append_jsonl, output / "simulator_timing.jsonl"))
        environment = (RecordingEnvironment(simulator, video) if video is not None
                       else LocalEnvironmentClient(simulator))
        if frozen:
            from harness.evaluation.guard import SingleResetEnvironment
            environment = SingleResetEnvironment(environment)
        environment.reset(args.seed)  # Exactly one reset, owned by this runner.
        if settings.perception.opus_rack_grounding:
            from harness.perception.hang_targets import StaticRackFrameCache
            from harness.perception.opus_frame import OpusObjectFrameGrounder
            started = time.perf_counter()
            try:
                initial_observation = environment.observe()
                # Capture the public SAM mask and Opus keypoints against exactly
                # the reset observation before any policy or arm motion.
                perception.observe(initial_observation)
                rack_report = OpusObjectFrameGrounder(
                    planner.client, planner.settings.model, target="rack_peg",
                    camera=settings.perception.rack_grounding_camera).locate(initial_observation)
                rack_frame_cache = StaticRackFrameCache()
                rack_mask = perception.detection_mask("rack", initial_observation,
                    camera=settings.perception.rack_grounding_camera)
                rack_grounding_initialization = dict(rack_report)
                rack_grounding_initialization["frame_cache"] = rack_frame_cache.cache(
                    initial_observation, rack_report, rack_mask)
                rack_grounding_initialization["world_anchor_available"] = True
                rack_grounding_initialization["world_anchor_reason"] = "mask_supported_rack_frame"
                perception.set_static_rack_frame_cache(rack_frame_cache)
                rack_grounding_initialization["latency_seconds"] = (
                    time.perf_counter() - started)
                append_jsonl(output / "assessments.jsonl", {
                    "schema_version": 2, "episode": output.name, "seed": args.seed,
                    "event_type": "rack_grounding_initialized",
                    "status": rack_grounding_initialization["status"],
                    "camera": rack_grounding_initialization.get("camera"),
                    "observation_sequence": initial_observation.sequence,
                    "frame_keypoints_px": rack_grounding_initialization.get("keypoints_px"),
                    "confidence": rack_grounding_initialization.get("confidence"),
                    "world_anchor_available": rack_grounding_initialization.get(
                        "world_anchor_available", False),
                    "world_anchor_reason": rack_grounding_initialization.get(
                        "world_anchor_reason"),
                    "world_anchor_depth_mad_m": rack_grounding_initialization.get(
                        "world_anchor_depth_mad_m"),
                    "world_anchor_valid_depth_samples": rack_grounding_initialization.get(
                        "world_anchor_valid_depth_samples"),
                    "world_anchor_peg_to_mask_px": rack_grounding_initialization.get(
                        "world_anchor_peg_to_mask_px"),
                    "camera_calibration_sha256": rack_grounding_initialization.get(
                        "camera_calibration_sha256"),
                    "latency_seconds": rack_grounding_initialization["latency_seconds"],
                    "model": planner.settings.model,
                })
            except Exception as exc:
                rack_frame_cache = None
                rack_grounding_initialization = {
                    "status": "error", "error_type": type(exc).__name__,
                    "latency_seconds": time.perf_counter() - started}
                append_jsonl(output / "assessments.jsonl", {
                    "schema_version": 2, "episode": output.name, "seed": args.seed,
                    "event_type": "rack_grounding_initialized", **rack_grounding_initialization,
                    "model": planner.settings.model,
                })
        if settings.perception.opus_mug_lift_grounding:
            from harness.perception.table_plane import (
                MugLiftEvidenceTracker, OpusMugSupportGrounder,
                StaticTablePlaneCache,
            )
            started = time.perf_counter()
            initialization = {"status": "error"}
            try:
                initial_observation = environment.observe()
                # Reuses the reset-frame SAM masks if rack grounding already ran.
                perception.observe(initial_observation)
                table_mask = perception.detection_mask("table", initial_observation,
                    camera="head_camera")
                table_plane = StaticTablePlaneCache()
                initialization = {"status": "grounded",
                    **table_plane.cache(initial_observation, table_mask,
                                        camera="head_camera")}
                grounder = OpusMugSupportGrounder(planner.client,
                    planner.settings.model, camera="head_camera")
                perception.set_mug_lift_evidence_tracker(
                    MugLiftEvidenceTracker(table_plane, grounder))
            except Exception as exc:
                # Do not fall back to SAM mug-centroid lift evidence when this
                # explicitly selected public table-plane path cannot initialize.
                perception.set_mug_lift_evidence_tracker(None)
                initialization = {"status": "error",
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:300]}
            initialization["latency_seconds"] = time.perf_counter() - started
            append_jsonl(output / "assessments.jsonl", {
                "schema_version": 2, "episode": output.name, "seed": args.seed,
                "event_type": "table_plane_initialized", **initialization,
                "model": planner.settings.model,
            })
        rack_seat_gate = (RackSeatReleaseGate(environment)
                          if rack_frame_cache is not None else None)
        oracle_log = partial(append_jsonl, output / "oracle.jsonl")
        if perception is not None:
            predicates = PredicateRegistry(perception=perception,
                rack_seat_gate=rack_seat_gate)
            mug_frame_grounder = None
            if rack_frame_cache is not None:
                from harness.perception.opus_frame import OpusObjectFrameGrounder
                mug_frame_grounder = OpusObjectFrameGrounder(
                    planner.client, planner.settings.model, target="mug_handle",
                    camera=settings.perception.rack_grounding_camera)
            semantic_prepare = PerceptionAnchorPreparer(perception, environment, anchors,
                hang_target_cache=rack_frame_cache,
                mug_frame_grounder=mug_frame_grounder,
                mug_frame_camera=settings.perception.rack_grounding_camera)
            if args.planner_pixel_grounding:
                prepare = HybridAnchorPreparer(semantic_prepare,
                    PlannerPixelAnchorPreparer(environment, anchors))
            else:
                prepare = semantic_prepare
        elif args.planner_pixel_grounding:
            # Development oracle evidence may gate execution locally, but the
            # planner prompt/result projection excludes this information source.
            predicates = (oracle_predicates(simulator.task, log=oracle_log)
                          if args.development_oracle else PredicateRegistry())
            predicates.rack_seat_gate = rack_seat_gate
            prepare = PlannerPixelAnchorPreparer(environment, anchors)
        else:
            predicates = oracle_predicates(simulator.task, log=oracle_log)
            predicates.rack_seat_gate = rack_seat_gate
            prepare = oracle_anchor_preparer(simulator.task, anchors, log=oracle_log)
        executor = SkillExecutor(environment, ModalVLAClient(),
            log=partial(append_jsonl, output / "skill_events.jsonl"), budget=settings.budget,
            predicates=predicates, anchors=anchors,
            allow_development_oracle=args.development_oracle)
        controller = SkillOrchestrator(executor, planner,
            log=partial(append_jsonl, output / "assessments.jsonl"),
            save_observation=ObservationStore(output / "observations").save,
            episode_id=output.name, seed=args.seed, max_turns=settings.max_turns,
            completion_predicate=settings.completion_predicate,
            memory=memory,
            invariant_check=check,
            prepare=prepare)
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
    if frozen:
        summary.update(mode="eval", evaluation_sha256=frozen.digest, condition=frozen.settings.condition)
    if settings.perception.opus_rack_grounding:
        summary["rack_grounding_initialization"] = rack_grounding_initialization
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Live development runs retain reviewable task/global success and failure
    # candidates beside the trace. Frozen evaluation stays write-restricted.
    if not frozen:
        try:
            candidates = candidates_from_attempt(output, attempt_id=output.name,
                seed=args.seed, execution_kind="live")
            candidate_path = output / "candidate_catalog.json"
            candidate_path.write_text(json.dumps(candidates.model_dump(mode="json"),
                indent=2, allow_nan=False) + "\n")
            summary.update(memory_candidates={"path": candidate_path.name,
                "task": len(candidates.task_memory), "global": len(candidates.global_memory),
                "auto_promoted": False})
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        except Exception as exc:
            summary["memory_candidate_error_type"] = type(exc).__name__
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if successful_outcome(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
