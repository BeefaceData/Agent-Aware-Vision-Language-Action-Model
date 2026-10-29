"""Build the swappable parts of one supervised episode.

Each builder takes its dependencies as arguments and returns one component, so
a planner, detector or evidence source can be replaced without touching the
turn loop (``harness.runtime.orchestrator``) or the CLI (``run_skills.py``).
Development-oracle wiring deliberately lives outside this package.
"""

import importlib
import time

from harness.perception.grounding import RGBDPerceptionProvider
from harness.perception.hang_targets import StaticRackFrameCache
from harness.perception.opus_frame import OpusObjectFrameGrounder
from harness.perception.preparation import (HybridAnchorPreparer, PerceptionAnchorPreparer,
                                            PlannerPixelAnchorPreparer)
from harness.perception.table_plane import (MugLiftEvidenceTracker, OpusMugSupportGrounder,
                                            StaticTablePlaneCache)
from harness.planner.anthropic import AnthropicPlanner, AnthropicSettings
from harness.planner.handoff import HandoffPlanner
from harness.planner.scripted import GraspRetryPlanner
from harness.predicates.registry import PredicateRegistry

# SAM 3 mask confidence floors. The exploration rack prompt ("mug rack") scores
# lower than the default prompt, so it is paired with its own calibrated floor.
MUG_MASK_MIN_CONFIDENCE = 0.40
EXPLORE_RACK_MASK_MIN_CONFIDENCE = 0.18
RACK_PROMPT_PROFILES = ("default", "explore_mug_rack")
TABLE_PLANE_CAMERA = "head_camera"


def load_perception_provider(factory_spec, anchors, *, rack_prompt_profile="default",
                             rack_grounding_camera="head_camera"):
    """Load an explicit RGB-only detector factory for public RGB-D grounding.

    ``factory_spec`` is ``package.module:callable``. The callable returns a
    detector implementing ``detect(images, observation_sequence=...)``; camera
    depth and calibration remain inside the harness perception provider.
    """
    if not isinstance(factory_spec, str) or factory_spec.count(":") != 1:
        raise ValueError("perception_factory_must_be_module_callable")
    module_name, callable_name = factory_spec.split(":")
    if not module_name or not callable_name or not all(
            part.isidentifier() for part in module_name.split(".")) or not callable_name.isidentifier():
        raise ValueError("perception_factory_must_be_module_callable")
    factory = getattr(importlib.import_module(module_name), callable_name, None)
    if not callable(factory):
        raise ValueError("perception_factory_not_found")
    if rack_prompt_profile not in RACK_PROMPT_PROFILES:
        raise ValueError("unsupported_rack_prompt_profile")
    return perception_provider_for(factory(), anchors, rack_prompt_profile=rack_prompt_profile,
                                   rack_grounding_camera=rack_grounding_camera)


def perception_provider_for(detector, anchors, *, rack_prompt_profile="default",
                            rack_grounding_camera="head_camera"):
    """Wrap a detector exactly as an episode does: prompt profile, camera and thresholds.

    Shared by ``load_perception_provider`` and offline evaluation, so both see
    the same confidence floors and detector configuration.
    """
    if rack_prompt_profile not in RACK_PROMPT_PROFILES:
        raise ValueError("unsupported_rack_prompt_profile")
    if not callable(getattr(detector, "detect", None)):
        raise ValueError("perception_detector_missing_detect")
    _configure_optional(detector, "configure_rack_prompt_profile", rack_prompt_profile,
                        default="default", error="detector_does_not_support_rack_prompt_profile")
    _configure_optional(detector, "configure_rack_grounding_camera", rack_grounding_camera,
                        default="head_camera", error="detector_does_not_support_rack_grounding_camera")
    thresholds = {"mug": MUG_MASK_MIN_CONFIDENCE}
    if rack_prompt_profile == "explore_mug_rack":
        thresholds["rack"] = EXPLORE_RACK_MASK_MIN_CONFIDENCE
    return RGBDPerceptionProvider(detector, anchors, min_confidence_by_label=thresholds)


def _configure_optional(detector, method_name, value, *, default, error):
    """Apply an optional detector setting; a non-default value requires support."""
    configure = getattr(detector, method_name, None)
    if callable(configure):
        configure(value)
    elif value != default:
        raise ValueError(error)


def anthropic_settings(run_settings, model_override=None):
    """Resolve planner settings from config plus an optional CLI model override.

    There is no fallback model: a missing model is a configuration error.
    """
    configured = run_settings.anthropic
    if configured is None and not model_override:
        raise ValueError("anthropic_model_required")
    if configured is not None:
        options = configured.model_dump()
    else:
        options = dict(max_calls=min(100, run_settings.max_turns),
                       task_scope="full_task" if run_settings.handoff is not None else "acquisition",
                       vla=(run_settings.handoff or run_settings.recipe.vla).model_dump())
    if model_override:
        options["model"] = model_override
    return AnthropicSettings.model_validate(options)


def build_planner(kind, run_settings, *, planner_settings=None, pixel_grounding=False):
    """Return the planner for ``kind`` ("anthropic" or "scripted")."""
    if kind == "anthropic":
        return AnthropicPlanner(planner_settings, require_pixel_grounding=pixel_grounding)
    if kind != "scripted":
        raise ValueError("unsupported_planner")
    if run_settings.handoff is not None:
        return HandoffPlanner(run_settings.recipe, run_settings.handoff)
    return GraspRetryPlanner(run_settings.recipe)


def initialize_rack_frame(environment, perception, planner, *, camera, log):
    """Ground the rack peg frame once on the reset observation, before any motion.

    Returns ``(cache, report)``; ``cache`` is None when grounding failed, in
    which case rack-seat targets stay unavailable for the episode.
    """
    started = time.perf_counter()
    try:
        observation = environment.observe()
        report = dict(OpusObjectFrameGrounder(
            planner.client, planner.settings.model, target="rack_peg", camera=camera,
        ).locate(observation))
        cache = StaticRackFrameCache()
        report["frame_cache"] = cache.cache(observation, report)
        report["world_anchor_available"] = True
        report["world_anchor_reason"] = "opus_keypoints_rgbd_frame"
        perception.set_static_rack_frame_cache(cache)
        report["latency_seconds"] = time.perf_counter() - started
        log({"event_type": "rack_grounding_initialized",
             "status": report["status"],
             "camera": report.get("camera"),
             "observation_sequence": observation.sequence,
             "frame_keypoints_px": report.get("keypoints_px"),
             "confidence": report.get("confidence"),
             "world_anchor_available": report.get("world_anchor_available", False),
             "world_anchor_reason": report.get("world_anchor_reason"),
             "world_anchor_depth_mad_m": report.get("world_anchor_depth_mad_m"),
             "world_anchor_valid_depth_samples": report.get("world_anchor_valid_depth_samples"),
             "world_anchor_depth_gap_m": report.get("world_anchor_depth_gap_m"),
             "camera_calibration_sha256": report.get("camera_calibration_sha256"),
             "latency_seconds": report["latency_seconds"],
             "model": planner.settings.model})
        return cache, report
    except Exception as exc:
        report = {"status": "error", "error_type": type(exc).__name__,
                  "latency_seconds": time.perf_counter() - started}
        log({"event_type": "rack_grounding_initialized", **report,
             "model": planner.settings.model})
        return None, report


def initialize_mug_lift_evidence(environment, perception, planner, *, log,
                                 minimum_policy_actions=0):
    """Cache the reset table plane and enable table-relative mug-lift evidence."""
    started = time.perf_counter()
    try:
        observation = environment.observe()
        # Reuses the reset-frame SAM masks if rack grounding already ran.
        perception.observe(observation)
        table_mask = perception.detection_mask("table", observation, camera=TABLE_PLANE_CAMERA)
        table_plane = StaticTablePlaneCache()
        initialization = {"status": "grounded",
                          **table_plane.cache(observation, table_mask, camera=TABLE_PLANE_CAMERA)}
        grounder = OpusMugSupportGrounder(planner.client, planner.settings.model,
                                          camera=TABLE_PLANE_CAMERA)
        perception.set_mug_lift_evidence_tracker(MugLiftEvidenceTracker(
            table_plane, grounder, minimum_policy_actions=minimum_policy_actions))
    except Exception as exc:
        # Do not fall back to SAM mug-centroid lift evidence when this
        # explicitly selected public table-plane path cannot initialize.
        perception.set_mug_lift_evidence_tracker(None)
        initialization = {"status": "error", "error_type": type(exc).__name__,
                          "reason": str(exc)[:300]}
    initialization["latency_seconds"] = time.perf_counter() - started
    log({"event_type": "table_plane_initialized", **initialization,
         "model": planner.settings.model})
    return initialization


def public_perception_evidence(perception, environment, anchors, *, planner,
                               rack_frame_cache=None, release_gate=None,
                               hang_camera="head_camera", pixel_grounding=False):
    """Predicates and anchor preparation from public RGB-D perception.

    Returns ``(predicates, prepare)``. With a cached rack frame, hang targets
    are composed from a freshly grounded mug-handle frame on each request.
    """
    predicates = PredicateRegistry(perception=perception, rack_seat_gate=release_gate)
    mug_frame_grounder = None
    if rack_frame_cache is not None:
        mug_frame_grounder = OpusObjectFrameGrounder(
            planner.client, planner.settings.model, target="mug_handle", camera=hang_camera)
    prepare = PerceptionAnchorPreparer(perception, environment, anchors,
                                       hang_target_cache=rack_frame_cache,
                                       mug_frame_grounder=mug_frame_grounder,
                                       mug_frame_camera=hang_camera)
    if pixel_grounding:
        prepare = HybridAnchorPreparer(prepare, PlannerPixelAnchorPreparer(environment, anchors))
    return predicates, prepare
