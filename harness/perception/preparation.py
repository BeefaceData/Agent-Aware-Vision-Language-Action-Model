"""Perception-backed symbolic anchor preparation for validated MOVE_TO skills."""

import numpy as np

from .anchors import Anchor
from .geometry import pose7
from .grounding import RGBDPerceptionProvider, SYMBOLIC_ANCHORS


class PerceptionAnchorPreparer:
    """Refresh one fixed symbolic target from the current public observation.

    The environment is used only for its measured arm TCP pose. Object geometry
    comes from the injected public RGB-D perception provider. The provider and
    executor must share one ``AnchorRegistry`` so the fresh result is exactly
    the anchor the skill later resolves.
    """

    def __init__(self, provider, environment, anchors, *, approach_offset_m=0.12,
                 hang_target_cache=None, mug_frame_grounder=None,
                 mug_frame_camera="head_camera"):
        if not isinstance(provider, RGBDPerceptionProvider):
            raise ValueError("invalid_perception_provider")
        if provider.anchors is not anchors:
            raise ValueError("perception_anchor_registry_mismatch")
        if (isinstance(approach_offset_m, bool)
                or not 0 <= approach_offset_m <= provider.max_approach_offset_m):
            raise ValueError("approach_offset_out_of_bounds")
        self.provider, self.environment, self.anchors = provider, environment, anchors
        self.approach_offset_m = float(approach_offset_m)
        if (hang_target_cache is None) != (mug_frame_grounder is None):
            raise ValueError("hang_cache_and_mug_grounder_must_be_paired")
        if mug_frame_camera not in ("head_camera", "left_camera", "right_camera"):
            raise ValueError("invalid_hang_frame_camera")
        self.hang_target_cache = hang_target_cache
        self.mug_frame_grounder = mug_frame_grounder
        self.mug_frame_camera = mug_frame_camera

    def __call__(self, command, observation):
        action = command.action.value if hasattr(command.action, "value") else command.action
        if action != "move_to":
            return None
        target = command.parameters.target
        if target == "current_ee":
            return None
        if target not in SYMBOLIC_ANCHORS:
            raise ValueError("unsupported_perception_anchor")

        # Hang targets are composed once from the episode rack frame, the
        # current mug handle frame, and measured TCP pose. Preserve that
        # same-observation result instead of replacing it with a generic
        # point approach derived from the semantic detector.
        if target in ("rack_approach", "rack_seat"):
            try:
                return self.anchors.resolve(target, command.arm, observation)
            except ValueError as exc:
                if str(exc) != "anchor_missing_or_stale":
                    raise
        if target == "rack_seat" and self.hang_target_cache is None:
            raise ValueError("hang_target_not_prepared")

        if target in ("rack_approach", "rack_seat") and self.hang_target_cache is not None:
            return self._prepare_hang_targets(command.arm, observation, target)

        # Repeated observation of the same sequence reuses cached detections.
        result = self.provider.observe(observation)
        if result.observation_sequence != observation.sequence:
            raise ValueError("grounding_missing_or_stale")
        measured = self.environment.arm_state(command.arm)["pose"]
        if (target == "rack_approach"
                and self.provider.static_rack_world_anchor_available(observation)):
            return self.provider.bind_static_rack_approach(
                target, command.arm, observation, measured_tcp_pose=measured,
                offset_m=self.approach_offset_m)
        return self.provider.bind_approach_anchor(target, command.arm, observation,
            measured_tcp_pose=measured, offset_m=self.approach_offset_m)

    def _prepare_hang_targets(self, arm, observation, requested_target):
        from .hang_targets import bind_hang_targets

        self.hang_target_cache.resolve(observation)
        result = self.provider.observe(observation)
        if result.observation_sequence != observation.sequence:
            raise ValueError("grounding_missing_or_stale")
        report = self.mug_frame_grounder.locate(observation)
        mask = self.provider.detection_mask("mug", observation,
                                            camera=self.mug_frame_camera)
        measured = self.environment.arm_state(arm)["pose"]
        targets = bind_hang_targets(self.hang_target_cache, observation, report, mask,
            tcp_pose=measured, anchors=self.anchors, arm=arm)
        return targets[requested_target]


def perception_anchor_preparer(provider, environment, anchors, *, approach_offset_m=0.12):
    """Build the runner ``prepare(command, observation)`` hook for perception."""
    return PerceptionAnchorPreparer(provider, environment, anchors,
                                    approach_offset_m=approach_offset_m)


class PlannerPixelAnchorPreparer:
    """Resolve one planner-selected RGB pixel through same-frame metric depth.

    The planner supplies only a view and normalized image coordinate. World
    geometry is computed locally from that exact observation; distance and
    approach offsets are bounded before binding the symbolic anchor.
    """

    MAX_LOCAL_DEPTH_GAP_M = 0.06
    DEPTH_SEARCH_RADIUS_PX = 16

    def __init__(self, environment, anchors, *, approach_offset_m=0.12,
                 max_anchor_distance_m=0.65):
        if (not np.isfinite((approach_offset_m, max_anchor_distance_m)).all()
                or not 0 <= approach_offset_m <= 0.20 or max_anchor_distance_m <= 0):
            raise ValueError("invalid_pixel_grounding_limits")
        self.environment, self.anchors = environment, anchors
        self.approach_offset_m = float(approach_offset_m)
        self.max_anchor_distance_m = float(max_anchor_distance_m)

    def __call__(self, command, observation):
        action = command.action.value if hasattr(command.action, "value") else command.action
        if action != "move_to" or command.parameters.target == "current_ee":
            return None
        target_name = command.parameters.target
        if target_name not in ("mug_pregrasp", "rack_approach"):
            raise ValueError("unsupported_pixel_grounding_target")
        grounding = command.parameters.grounding
        if grounding is None:
            raise ValueError("grounding_pixel_required")
        frames = observation.camera_frames or {}
        frame = frames.get(grounding.camera)
        image = observation.images.get(grounding.camera)
        if frame is None or image is None or frame.depth_m.shape != np.asarray(image).shape[:2]:
            raise ValueError("grounding_camera_frame_unavailable")

        height, width = frame.depth_m.shape
        u = float(grounding.uv[0]) * (width - 1)
        v = float(grounding.uv[1]) * (height - 1)
        if frame.local_depth_gap(u, v, search_radius=self.DEPTH_SEARCH_RADIUS_PX) > self.MAX_LOCAL_DEPTH_GAP_M:
            raise ValueError("grounding_pixel_behind_local_surface")
        target = np.asarray(frame.unproject(u, v, radius=1), dtype=float)
        measured_pose = pose7(self.environment.arm_state(command.arm)["pose"])
        delta = target - measured_pose[:3]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6 or distance > self.max_anchor_distance_m:
            raise ValueError("grounding_target_out_of_reach")
        waypoint = target - delta / distance * self.approach_offset_m
        anchor_pose = np.concatenate((waypoint, measured_pose[3:]))
        self.anchors.bind(target_name, command.arm,
                          Anchor(tuple(map(float, anchor_pose)), observation.sequence,
                                 source="perception"))
        return self.anchors.resolve(target_name, command.arm, observation)


class HybridAnchorPreparer:
    """Use planner pixels for explicit targets and semantic masks otherwise."""

    def __init__(self, semantic_preparer, pixel_preparer):
        if not callable(semantic_preparer) or not callable(pixel_preparer):
            raise ValueError("invalid_hybrid_anchor_preparers")
        self.semantic_preparer = semantic_preparer
        self.pixel_preparer = pixel_preparer

    def __call__(self, command, observation):
        action = command.action.value if hasattr(command.action, "value") else command.action
        if (action == "move_to" and command.parameters.grounding is not None
                and command.parameters.target in ("mug_pregrasp", "rack_approach")):
            return self.pixel_preparer(command, observation)
        return self.semantic_preparer(command, observation)
