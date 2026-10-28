"""Public RGB-D semantic grounding with bounded, sequence-fresh outputs.

The detector is injected and sees only RGB camera images. This module combines
its masks with the matching public depth/calibration frame; it never imports or
queries RoboTwin object state. Optional measured robot TCP poses are used only
for arm attribution; they are not planner-facing object or contact oracles.
Missing or ambiguous evidence stays unknown and cannot create an anchor.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .anchors import Anchor, AnchorRegistry
from .geometry import pose7
from .rgbd import RGBDCamera, calibration_sha256
from harness.environment.layout import CAMERAS, PERCEPTION_CAMERAS
from harness.predicates.base import PredicateResult


SEMANTIC_LABELS = frozenset({"mug", "rack", "table"})
SYMBOLIC_ANCHORS = frozenset({
    "mug_pregrasp", "rack_approach", "rack_seat", "table_approach"})


@dataclass(frozen=True)
class Detection:
    """One semantic instance mask in the pixel coordinates of an RGB image."""

    label: str
    confidence: float
    mask: np.ndarray

    def __post_init__(self):
        if self.label not in SEMANTIC_LABELS:
            raise ValueError("unsupported_detection_label")
        if (isinstance(self.confidence, (bool, np.bool_))
                or not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("invalid_detection_confidence")
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or not mask.size:
            raise ValueError("invalid_detection_mask")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class DetectionBatch:
    """Detector output stamped with the source observation sequence."""

    observation_sequence: int
    camera: str
    detections: tuple[Detection, ...]

    def __post_init__(self):
        if (type(self.observation_sequence) is not int or self.observation_sequence < 0
                or not isinstance(self.camera, str) or not self.camera
                or not isinstance(self.detections, (tuple, list))
                or any(not isinstance(item, Detection) for item in self.detections)):
            raise ValueError("invalid_detection_batch")
        object.__setattr__(self, "detections", tuple(self.detections))


class SemanticDetector(Protocol):
    def detect(self, images: dict[str, np.ndarray], *, observation_sequence: int):
        """Return one or more camera-stamped batches from public RGB only."""


@dataclass(frozen=True)
class GroundedObject:
    label: str
    position: tuple[float, float, float]
    confidence: float
    observation_sequence: int
    camera: str
    valid_depth_fraction: float
    # Internal public RGB-D samples retained for geometric support checks.
    # These are world points derived only from this object's detector mask.
    surface_points: tuple[tuple[float, float, float], ...] = ()
    boundary_points: tuple[tuple[float, float, float], ...] = ()
    # Descriptive, view-dependent features for planner comparison. These are
    # deliberately not converted into a canonical mug identity or class.
    normalized_bbox: tuple[float, float, float, float] | None = None
    median_rgb_0_255: tuple[float, float, float] | None = None
    visible_surface_extent_m: tuple[float, float, float] | None = None
    # Median |z| of the mask's world surface normals (mug only): near 0 when the
    # visible surface faces sideways like an upright mug's wall; None when too
    # few pixels allow a normal.
    median_normal_z: float | None = None


@dataclass(frozen=True)
class GroundingResult:
    observation_sequence: int
    objects: dict[str, GroundedObject]
    status: dict[str, str]
    release_ready: bool | None = None
    release_ready_reason: str = "not_provable_from_semantic_rgbd_alone"
    mug_lift_seen: bool = False
    mug_staged_on_table: bool | None = None
    mug_staged_reason: str = "not_provable_from_semantic_rgbd_alone"


class RGBDPerceptionProvider:
    """Convert detections to current-observation world points and bounded anchors.

    The provider is deliberately model-agnostic: a segmentation-capable detector
    can be supplied later without changing controller code. A mask is required
    so depth at a mug's hollow center cannot silently ground the mug to the
    table behind it.
    """

    source = "perception"
    # The measured pose is the arm's end-effector frame behind the fingers, and
    # the policy grips the mug's rim or handle: a right-held mug's true centre
    # sat 0.20-0.22 m from it (median) in the seed 0-4 captures, so the earlier
    # 0.16 m almost never associated. Table-height mugs near the gripper stay
    # excluded by the lift and table-support checks before association.
    RIGHT_ARM_ASSOCIATION_MAX_M = 0.24
    RIGHT_ARM_ASSOCIATION_MARGIN_M = 0.06
    RIGHT_LIFT_STABLE_OBSERVATIONS = 2
    # Measured on the 236 seed 0-4 frames at 320x240: every upright mug that met
    # the table had >= 775 normal pixels; the two toppled frames below 0.25 were
    # 45- and 80-pixel slivers.
    MIN_ORIENTATION_PIXELS = 300
    # A mug being lifted or set down moves between policy chunks (16 actions):
    # 2.6-4.0 cm in the seed 0-4 captures, while 95% of mugs resting on the
    # table moved less than 2 cm between observations.
    MUG_SETTLED_MAX_STEP_M = 0.02

    def __init__(self, detector: SemanticDetector, anchors: AnchorRegistry | None = None,
                 *, camera_order=(*CAMERAS, *PERCEPTION_CAMERAS),
                 min_confidence=0.60, ambiguity_margin=0.10,
                 min_valid_depth_fraction=0.35, max_anchor_distance_m=0.65,
                 max_approach_offset_m=0.20, sample_limit=256,
                 min_confidence_by_label=None):
        numeric = (min_confidence, ambiguity_margin, min_valid_depth_fraction,
                   max_anchor_distance_m, max_approach_offset_m)
        if not np.isfinite(numeric).all():
            raise ValueError("invalid_grounding_limits")
        if (not 0 <= min_confidence <= 1 or not 0 <= ambiguity_margin <= 1
                or not 0 < min_valid_depth_fraction <= 1 or max_anchor_distance_m <= 0
                or not 0 <= max_approach_offset_m <= 0.20
                or type(sample_limit) is not int or not 1 <= sample_limit <= 2048):
            raise ValueError("invalid_grounding_limits")
        if not camera_order or any(not isinstance(name, str) or not name for name in camera_order):
            raise ValueError("invalid_camera_order")
        self.detector = detector
        self.anchors = anchors or AnchorRegistry()
        self.camera_order = tuple(camera_order)
        self.min_confidence = float(min_confidence)
        if min_confidence_by_label is None:
            min_confidence_by_label = {}
        if (not isinstance(min_confidence_by_label, dict)
                or any(label not in SEMANTIC_LABELS for label in min_confidence_by_label)
                or any(isinstance(value, bool) or not np.isfinite(value)
                       or not 0 <= value <= 1 for value in min_confidence_by_label.values())):
            raise ValueError("invalid_label_confidence_thresholds")
        self.min_confidence_by_label = {
            label: float(value) for label, value in min_confidence_by_label.items()}
        self.ambiguity_margin = float(ambiguity_margin)
        self.min_valid_depth_fraction = float(min_valid_depth_fraction)
        self.max_anchor_distance_m = float(max_anchor_distance_m)
        self.max_approach_offset_m = float(max_approach_offset_m)
        self.sample_limit = sample_limit
        self._last_sequence = -1
        self._last_result = None
        self._initial_mug_height = None
        self._previous_mug_height = None
        self._mug_lift_seen = False
        self._right_lift_confirmed = False
        self._right_lift_candidate_count = 0
        self._right_lift_last_sequence = None
        self._static_rack_reference = None
        self._static_rack_frame_cache = None
        self._mug_lift_tracker = None
        self._mug_lift_tracker_enabled = False
        self._last_batches = ()

    def set_mug_lift_evidence_tracker(self, tracker):
        """Use an optional table-relative public lift source for mug_lifted."""
        if tracker is not None and not callable(getattr(tracker, "evaluate", None)):
            raise ValueError("invalid_mug_lift_evidence_tracker")
        self._mug_lift_tracker = tracker
        self._mug_lift_tracker_enabled = True

    def set_policy_action_count(self, count):
        """Share confirmed VLA progress with action-scheduled public checks."""
        if self._mug_lift_tracker is not None:
            self._mug_lift_tracker.set_policy_action_count(count)

    def set_static_rack_reference(self, reference, *, observation=None):
        """Cache a one-shot RGB rack reference and, when possible, a 3D anchor.

        The 3D anchor comes from the Opus-selected rack peg pixel and matched
        public depth/calibration. Coordinates remain inside this provider;
        planner summaries expose availability and quality metadata, not world
        coordinates.
        """
        if reference is None:
            self._static_rack_reference = None
            self._set_detector_rack_reference(False)
            return {"world_anchor_available": False,
                    "world_anchor_reason": "reference_cleared"}
        if (not isinstance(reference, dict)
                or reference.get("status") not in ("grounded", "not_detected")
                or reference.get("source") != "opus_rgb"
                or reference.get("camera") != "head_camera"):
            raise ValueError("invalid_static_rack_reference")
        if reference["status"] == "not_detected":
            self._static_rack_reference = {
                "status": "not_detected", "source": "opus_rgb", "camera": "head_camera",
                "observation_sequence": reference.get("observation_sequence"),
                "image_shape_px": reference.get("image_shape_px"),
                "confidence": reference.get("confidence"),
                "world_anchor_available": False,
                "world_anchor_reason": "initial_rack_detection_failed",
            }
            self._set_detector_rack_reference(False)
            return {"world_anchor_available": False,
                    "world_anchor_reason": "initial_rack_detection_failed"}
        shape = reference.get("image_shape_px")
        bbox = reference.get("bbox_normalized")
        peg = reference.get("peg_uv_normalized")
        sequence = reference.get("observation_sequence")
        confidence = reference.get("confidence")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                or any(type(x) is not int or x <= 0 for x in shape)
                or not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or not isinstance(peg, (list, tuple)) or len(peg) != 2
                or type(sequence) is not int or sequence < 0
                or isinstance(confidence, bool)):
            raise ValueError("invalid_static_rack_reference")
        try:
            bbox, peg, confidence = (np.asarray(bbox, dtype=float),
                                     np.asarray(peg, dtype=float), float(confidence))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("invalid_static_rack_reference") from None
        if (not np.isfinite(bbox).all() or not np.isfinite(peg).all()
                or not np.isfinite(confidence) or not 0 <= confidence <= 1
                or not 0 <= bbox[0] < bbox[2] <= 1
                or not 0 <= bbox[1] < bbox[3] <= 1
                or not bbox[0] <= peg[0] <= bbox[2]
                or not bbox[1] <= peg[1] <= bbox[3]):
            raise ValueError("invalid_static_rack_reference")
        self._static_rack_reference = {
            "status": "grounded", "source": "opus_rgb", "camera": "head_camera",
            "observation_sequence": sequence,
            "image_shape_px": list(shape),
            "bbox_normalized": [float(x) for x in bbox],
            "peg_uv_normalized": [float(x) for x in peg],
            "confidence": float(confidence),
            "camera_calibration_sha256": reference.get("camera_calibration_sha256"),
            "world_anchor_available": False,
        }
        quality = {"world_anchor_available": False,
                   "world_anchor_reason": "initial_rgbd_not_available"}
        if observation is not None:
            if observation.sequence != sequence:
                raise ValueError("static_rack_observation_sequence_mismatch")
            minimum = self.min_confidence_by_label.get("rack", self.min_confidence)
            if confidence < minimum:
                quality = {"world_anchor_available": False,
                           "world_anchor_reason": "rack_keypoint_confidence_too_low"}
            else:
                quality = self._estimate_static_rack_point(self._static_rack_reference,
                                                           observation)
            self._static_rack_reference.update(quality)
        # Continue live rack inference only when the one-shot RGB-D reference
        # is unusable; otherwise the fixed world anchor survives occlusion.
        self._set_detector_rack_reference(bool(quality.get("world_anchor_available")))
        return {key: value for key, value in quality.items() if key != "world_position_xyz"}

    def _estimate_static_rack_point(self, reference, observation):
        """Estimate one Opus-selected rack peg pixel from local RGB-D only."""
        sequence = reference["observation_sequence"]
        frame = (observation.camera_frames or {}).get("head_camera")
        image = observation.images.get("head_camera")
        if (frame is None or image is None or not isinstance(frame, RGBDCamera)
                or frame.depth_m.shape != np.asarray(image).shape[:2]):
            return {"world_anchor_available": False,
                    "world_anchor_reason": "head_rgbd_unavailable"}
        height, width = frame.depth_m.shape
        peg_norm = np.asarray(reference["peg_uv_normalized"], dtype=float)
        peg_u, peg_v = peg_norm * np.asarray((width - 1, height - 1), dtype=float)
        if not (0 <= peg_u < width and 0 <= peg_v < height):
            return {"world_anchor_available": False,
                    "world_anchor_reason": "rack_keypoint_out_of_bounds"}
        try:
            depth_gap = frame.local_depth_gap(peg_u, peg_v, search_radius=8,
                                              point_radius=2)
            if depth_gap > 0.06:
                return {"world_anchor_available": False,
                        "world_anchor_reason": "rack_keypoint_depth_discontinuity"}
            point = np.asarray(frame.unproject(peg_u, peg_v, radius=2), dtype=float)
            x, y = int(round(peg_u)), int(round(peg_v))
            local = frame.depth_m[max(0, y-2):min(height, y+3),
                                  max(0, x-2):min(width, x+3)]
            depths = local[np.isfinite(local) & (local > 0)]
        except ValueError as exc:
            return {"world_anchor_available": False,
                    "world_anchor_reason": str(exc)}
        if len(depths) < 8 or not np.isfinite(point).all():
            return {"world_anchor_available": False,
                    "world_anchor_reason": "rack_keypoint_depth_support_not_found"}
        median_depth = float(np.median(depths))
        mad_depth = float(np.median(np.abs(depths - median_depth)))
        return {
            "world_anchor_available": True,
            "world_anchor_reason": "opus_keypoint_local_depth",
            "world_position_xyz": [float(value) for value in point],
            "world_anchor_depth_m": median_depth,
            "world_anchor_depth_mad_m": float(mad_depth),
            "world_anchor_valid_depth_samples": int(len(depths)),
            "world_anchor_keypoint_confidence": float(reference["confidence"]),
            "world_anchor_depth_gap_m": float(depth_gap),
        }

    def bind_static_rack_approach(self, name, arm, observation, *, measured_tcp_pose,
                                 offset_m=0.12):
        """Bind a fresh measured-start staging pose from the episode's cached rack point."""
        if name != "rack_approach" or arm not in ("left", "right"):
            raise ValueError("unsupported_static_rack_anchor")
        reference = self._static_rack_reference
        if reference is None or not reference.get("world_anchor_available"):
            raise ValueError("static_rack_world_anchor_unavailable")
        if (isinstance(offset_m, bool) or not np.isfinite(offset_m)
                or not 0 <= offset_m <= self.max_approach_offset_m):
            raise ValueError("approach_offset_out_of_bounds")
        if not self._static_rack_reference_matches(reference, observation):
            self._set_detector_rack_reference(False)
            raise ValueError("static_rack_camera_changed")
        start = pose7(measured_tcp_pose)
        point = np.asarray(reference["world_position_xyz"], dtype=float)
        delta = point - start[:3]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6 or distance > self.max_anchor_distance_m:
            raise ValueError("grounded_target_out_of_reach")
        waypoint = point - delta / distance * float(offset_m)
        anchor_pose = np.concatenate((waypoint, start[3:]))
        self.anchors.bind(name, arm, Anchor(tuple(anchor_pose), observation.sequence,
                                            "perception"))
        return self.anchors.resolve(name, arm, observation)

    def static_rack_world_anchor_available(self, observation):
        reference = self._static_rack_reference
        available = bool(reference is not None
                         and reference.get("world_anchor_available")
                         and self._static_rack_reference_matches(reference, observation))
        if reference is not None and reference.get("world_anchor_available") and not available:
            self._set_detector_rack_reference(False)
        return available

    def set_static_rack_frame_cache(self, cache):
        """Attach an episode-local metric rack frame and stop redundant rack inference."""
        if cache is None:
            self._static_rack_frame_cache = None
            if self._static_rack_reference is None:
                self._set_detector_rack_reference(False)
            return
        if not callable(getattr(cache, "resolve", None)) or not callable(
                getattr(cache, "metadata", None)):
            raise ValueError("invalid_static_rack_frame_cache")
        self._static_rack_frame_cache = cache
        self._set_detector_rack_reference(True)

    def set_task_phase(self, phase):
        """Tell a phase-aware detector which wrist view should recover a lost mug."""
        configure = getattr(self.detector, "set_task_phase", None)
        if callable(configure):
            configure(phase)

    def detection_mask(self, label, observation, *, camera):
        """Return one unambiguous, confidence-qualified mask from a fresh batch."""
        if label not in SEMANTIC_LABELS or camera not in self.camera_order:
            raise ValueError("unsupported_detection_mask_request")
        result = self.observe(observation)
        if result.observation_sequence != observation.sequence:
            raise ValueError("detection_mask_stale_observation")
        batches = [batch for batch in self._last_batches
                   if batch.camera == camera
                   and batch.observation_sequence == observation.sequence]
        if not batches:
            raise ValueError("detection_mask_camera_unavailable")
        candidates, reason = self._select(
            [detection for batch in batches for detection in batch.detections], label)
        if candidates is None:
            raise ValueError("detection_mask_" + reason)
        if candidates.mask.shape != np.asarray(observation.images[camera]).shape[:2]:
            raise ValueError("detection_mask_shape_mismatch")
        return candidates.mask.copy()

    @staticmethod
    def _static_rack_reference_matches(reference, observation):
        image = observation.images.get("head_camera")
        if (image is None or list(np.asarray(image).shape[:2]) != reference["image_shape_px"]):
            return False
        expected = reference.get("camera_calibration_sha256")
        frame = (observation.camera_frames or {}).get("head_camera")
        if expected is None or frame is None:
            return False
        return calibration_sha256(frame) == expected

    def _set_detector_rack_reference(self, initialized):
        configure = getattr(self.detector, "set_static_rack_reference", None)
        if callable(configure):
            configure(initialized)

    def observe(self, observation):
        """Ground one public observation; duplicate or old sequences fail closed."""
        sequence = observation.sequence
        if type(sequence) is not int:
            raise ValueError("invalid_observation_sequence")
        if sequence == self._last_sequence and self._last_result is not None:
            return self._last_result
        if sequence < self._last_sequence:
            return GroundingResult(sequence, {}, {label: "stale_observation" for label in SEMANTIC_LABELS})
        self._last_sequence = sequence
        self._last_batches = ()
        frames = observation.camera_frames or {}
        if not frames:
            result = GroundingResult(sequence, {}, {label: "rgbd_unavailable" for label in SEMANTIC_LABELS})
            self._last_result = result
            return result

        public_images = {}
        for name, image in observation.images.items():
            snapshot = np.asarray(image).copy()
            snapshot.setflags(write=False)
            public_images[name] = snapshot
        try:
            batch = self.detector.detect(public_images, observation_sequence=sequence)
        except Exception:
            result = GroundingResult(sequence, {}, {label: "detector_unavailable" for label in SEMANTIC_LABELS})
            self._last_result = result
            return result
        batches = ((batch,) if isinstance(batch, DetectionBatch)
                   else tuple(batch) if isinstance(batch, (tuple, list)) else ())
        if not batches or any(not isinstance(item, DetectionBatch) for item in batches):
            result = GroundingResult(sequence, {}, {label: "stale_detection" for label in SEMANTIC_LABELS})
            self._last_result = result
            return result
        if any(item.observation_sequence != sequence for item in batches):
            result = GroundingResult(sequence, {}, {label: "stale_detection" for label in SEMANTIC_LABELS})
            self._last_result = result
            return result
        if any(item.camera not in self.camera_order or item.camera not in frames
               or item.camera not in observation.images for item in batches):
            result = GroundingResult(sequence, {}, {label: "camera_frame_unavailable" for label in SEMANTIC_LABELS})
            self._last_result = result
            return result
        self._last_batches = batches

        objects, status = {}, {label: "not_detected" for label in SEMANTIC_LABELS}
        for label in SEMANTIC_LABELS:
            estimates, reasons = [], []
            for item in batches:
                frame = frames[item.camera]
                image = observation.images[item.camera]
                image_shape = np.asarray(image).shape[:2]
                if not isinstance(frame, RGBDCamera) or frame.depth_m.shape != image_shape:
                    reasons.append("rgbd_shape_mismatch")
                    continue
                selected, reason = self._select(item.detections, label)
                if selected is None:
                    reasons.append(reason)
                    continue
                if selected.mask.shape != image_shape:
                    reasons.append("mask_shape_mismatch")
                    continue
                estimate, reason = self._estimate(selected, frame, sequence, item.camera, image)
                if estimate is None:
                    reasons.append(reason)
                else:
                    estimates.append(estimate)
            if estimates:
                objects[label] = max(estimates, key=lambda item: item.confidence)
                status[label] = "grounded"
            elif reasons:
                # Prefer specific failures over the expected no-detection
                # result from a secondary rack-only view.
                priority = ("ambiguous_instances", "mask_shape_mismatch", "rgbd_shape_mismatch",
                            "occluded_or_invalid_depth", "occluded_or_empty_mask",
                            "low_confidence", "stale_observation", "not_detected")
                status[label] = next((reason for reason in priority if reason in reasons), reasons[0])

        mug = objects.get("mug")
        previous_mug_height = self._previous_mug_height
        if mug is not None:
            self._previous_mug_height = mug.position[2]
            if self._initial_mug_height is None:
                self._initial_mug_height = mug.position[2]
            elif mug.position[2] >= self._initial_mug_height + 0.03:
                # This is a public temporal event latch. It lets later table
                # support distinguish a completed transfer from the mug's
                # initial resting pose without consulting simulator state.
                self._mug_lift_seen = True
        release_ready, release_reason = self._table_support_evidence(objects)
        # A few mug silhouette points can overlap the table mask while the
        # mug is still being carried. A returned mug must also be at roughly
        # its episode-start table height before support can authorize release.
        if (release_ready is True and mug is not None
                and self._initial_mug_height is not None
                and abs(mug.position[2] - self._initial_mug_height) > 0.03):
            release_ready, release_reason = None, "mug_above_initial_table_height"
        # A mug the gripper is still lifting or lowering can sit within those
        # tolerances; support also needs it to have stopped moving since the
        # last observation that grounded it.
        elif release_ready is True and (
                previous_mug_height is None
                or abs(mug.position[2] - previous_mug_height) > self.MUG_SETTLED_MAX_STEP_M):
            release_ready, release_reason = None, "mug_not_yet_settled"
        if release_ready is True and self._mug_lift_seen:
            staged, staged_reason = True, "public_lift_then_table_support"
        elif release_ready is True:
            staged, staged_reason = False, "mug_lift_not_previously_observed"
        else:
            staged, staged_reason = release_ready, release_reason
        result = GroundingResult(sequence, objects, status, release_ready, release_reason,
                                 self._mug_lift_seen, staged, staged_reason)
        self._last_result = result
        return result

    def planner_summary(self, observation):
        """Return bounded public visual descriptors without world coordinates.

        Color and visible extent depend on lighting, view, occlusion, and mask
        quality. The payload is descriptive evidence, not a variant label.
        """
        result = self.observe(observation)
        objects = {}
        for label, item in result.objects.items():
            objects[label] = {
                "camera": item.camera,
                "confidence": float(item.confidence),
                "valid_depth_fraction": float(item.valid_depth_fraction),
                "normalized_bbox": item.normalized_bbox,
                "median_rgb_0_255": item.median_rgb_0_255,
                "visible_surface_extent_m": item.visible_surface_extent_m,
            }
        summary = {"source": "public_rgbd", "observation_sequence": result.observation_sequence,
                   "status": dict(result.status), "objects": objects,
                   "interpretation": "view_dependent_descriptors_not_variant_labels"}
        frame_cache = self._static_rack_frame_cache
        if frame_cache is not None:
            metadata = frame_cache.metadata()
            try:
                frame_cache.resolve(observation)
                valid = True
                validity_reason = "same_static_camera_frame"
            except ValueError:
                valid = False
                validity_reason = "camera_shape_or_calibration_changed"
                self._set_detector_rack_reference(False)
            summary["rack_static_reference"] = {
                "initialized": valid,
                "world_anchor_available": bool(valid and metadata.get("available")),
                "source": "opus_keypoints_rgbd",
                "camera": metadata.get("camera"),
                "source_observation_sequence": metadata.get("observation_sequence"),
                "confidence": metadata.get("confidence"),
                "current_visibility": result.status.get("rack", "unknown"),
                "world_anchor_reason": ("opus_keypoints_rgbd_frame" if valid else validity_reason),
                "validity_reason": validity_reason,
            }
            return summary
        reference = self._static_rack_reference
        if reference is not None:
            if reference["status"] == "not_detected":
                summary["rack_static_reference"] = {
                    "initialized": False, "source": reference["source"],
                    "world_anchor_available": False,
                    "camera": reference["camera"],
                    "source_observation_sequence": reference["observation_sequence"],
                    "current_visibility": result.status.get("rack", "unknown"),
                    "validity_reason": "initial_rack_detection_failed",
                }
                return summary
            image = observation.images.get("head_camera")
            shape_matches = (image is not None
                             and list(np.asarray(image).shape[:2]) == reference["image_shape_px"])
            frame = (observation.camera_frames or {}).get("head_camera")
            expected_calibration = reference.get("camera_calibration_sha256")
            calibration_matches = (expected_calibration is None
                                   or (frame is not None
                                       and calibration_sha256(frame) == expected_calibration))
            valid = bool(shape_matches and calibration_matches)
            if not valid:
                self._set_detector_rack_reference(False)
            summary["rack_static_reference"] = {
                "initialized": valid,
                "world_anchor_available": bool(
                    valid and reference.get("world_anchor_available", False)),
                "source": reference["source"],
                "camera": reference["camera"],
                "source_observation_sequence": reference["observation_sequence"],
                "bbox_normalized": reference["bbox_normalized"],
                "peg_uv_normalized": reference["peg_uv_normalized"],
                "confidence": reference["confidence"],
                "world_anchor_reason": (reference.get("world_anchor_reason") if valid
                                         else "camera_shape_or_calibration_changed"),
                "world_anchor_depth_mad_m": reference.get("world_anchor_depth_mad_m"),
                "world_anchor_valid_depth_samples": reference.get(
                    "world_anchor_valid_depth_samples"),
                "world_anchor_depth_gap_m": reference.get("world_anchor_depth_gap_m"),
                "current_visibility": result.status.get("rack", "unknown"),
                "validity_reason": ("same_static_camera_frame" if valid else
                                    "camera_shape_or_calibration_changed"),
            }
        return summary

    def bind_approach_anchor(self, name, arm, observation, *, measured_tcp_pose,
                             offset_m=0.12):
        """Bind a fresh semantic approach point relative to the measured TCP.

        The detector supplies the world target. This method only moves the
        waypoint back along the measured TCP-to-object direction by a bounded
        standoff; it never accepts planner-generated world coordinates.
        """
        labels = {"mug_pregrasp": "mug", "rack_approach": "rack",
                  "table_approach": "table"}
        if name not in SYMBOLIC_ANCHORS or arm not in ("left", "right"):
            raise ValueError("unsupported_symbolic_anchor")
        if (isinstance(offset_m, bool) or not np.isfinite(offset_m)
                or not 0 <= offset_m <= self.max_approach_offset_m):
            raise ValueError("approach_offset_out_of_bounds")
        start = pose7(measured_tcp_pose)
        result = self._last_result
        if result is None or result.observation_sequence != observation.sequence:
            raise ValueError("grounding_missing_or_stale")
        target = result.objects.get(labels[name])
        if target is None or result.status.get(labels[name]) != "grounded":
            raise ValueError("grounded_target_unavailable")
        point = np.asarray(target.position, dtype=float)
        delta = point - start[:3]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6 or distance > self.max_anchor_distance_m:
            raise ValueError("grounded_target_out_of_reach")
        waypoint = point - delta / distance * float(offset_m)
        anchor_pose = np.concatenate((waypoint, start[3:]))
        self.anchors.bind(name, arm, Anchor(tuple(anchor_pose), observation.sequence, "perception"))
        return self.anchors.resolve(name, arm, observation)

    def predicate_evidence(self, name, observation, *, near_rack_m=0.10,
                           lift_delta_m=0.03):
        """Return tri-state visual evidence; unknown where RGB-D cannot prove it."""
        if (not np.isfinite(near_rack_m) or not 0 < near_rack_m <= 0.30
                or not np.isfinite(lift_delta_m) or not 0 < lift_delta_m <= 0.15):
            raise ValueError("visual_predicate_threshold_out_of_bounds")
        result = self._last_result
        if result is None or result.observation_sequence != observation.sequence:
            return {"name": name, "value": None, "source": "perception",
                    "observation_sequence": observation.sequence, "reason": "stale_or_missing_grounding"}
        if name == "mug_visible":
            value = "mug" in result.objects if result.status.get("mug") in ("grounded", "not_detected") else None
            reason = result.status.get("mug", "unknown")
        elif name == "rack_visible":
            value = "rack" in result.objects if result.status.get("rack") in ("grounded", "not_detected") else None
            reason = result.status.get("rack", "unknown")
        elif name == "mug_near_rack":
            mug, rack = result.objects.get("mug"), result.objects.get("rack")
            value = (None if mug is None or rack is None else
                     float(np.linalg.norm(np.asarray(mug.position)-np.asarray(rack.position))) <= near_rack_m)
            reason = "joint_rgbd_geometry" if value is not None else "mug_or_rack_geometry_unavailable"
        elif name == "mug_lifted":
            if self._mug_lift_tracker_enabled:
                evidence = (self._mug_lift_tracker.evaluate(observation)
                            if self._mug_lift_tracker is not None else
                            {"value": None, "reason": "table_plane_initialization_unavailable"})
                value, reason = evidence.get("value"), evidence.get("reason", "unknown")
            else:
                mug = result.objects.get("mug")
                value = None if mug is None else self._mug_lift_seen
                reason = ("public_rgbd_lift_event_latched" if mug is not None
                          else "mug_geometry_unavailable")
        elif name == "mug_staged_on_table":
            value, reason = result.mug_staged_on_table, result.mug_staged_reason
        elif name == "right_mug_lifted":
            mug = result.objects.get("mug")
            if self._right_lift_confirmed:
                value, reason = True, "right_gripper_lift_association_latched"
            elif mug is None:
                # Not seeing the mug (often the gripper hides it) does not
                # contradict an earlier right-arm association, so it does not
                # reset the count; contradicting observations below do.
                value, reason = None, "mug_geometry_unavailable"
            elif self._initial_mug_height is None:
                value, reason = None, "initial_mug_height_unavailable"
            elif mug.position[2] < self._initial_mug_height + lift_delta_m:
                self._right_lift_candidate_count = 0
                value, reason = False, "mug_not_lifted"
            elif result.release_ready is True:
                self._right_lift_candidate_count = 0
                value, reason = False, "mug_still_supported_by_table"
            else:
                poses = observation.measured_tcp_poses
                if poses is None:
                    self._right_lift_candidate_count = 0
                    value, reason = None, "measured_arm_tcp_poses_unavailable"
                else:
                    mug_xyz = np.asarray(mug.position, dtype=float)
                    distances = {arm: float(np.linalg.norm(
                        mug_xyz - np.asarray(poses[arm][:3], dtype=float)))
                        for arm in ("left", "right")}
                    right_is_nearest = (
                        distances["right"] <= self.RIGHT_ARM_ASSOCIATION_MAX_M
                        and distances["left"] - distances["right"]
                        >= self.RIGHT_ARM_ASSOCIATION_MARGIN_M)
                    left_is_nearest = (
                        distances["left"] <= self.RIGHT_ARM_ASSOCIATION_MAX_M
                        and distances["right"] - distances["left"]
                        >= self.RIGHT_ARM_ASSOCIATION_MARGIN_M)
                    if self._right_lift_last_sequence != observation.sequence:
                        self._right_lift_last_sequence = observation.sequence
                        self._right_lift_candidate_count = (
                            self._right_lift_candidate_count + 1
                            if right_is_nearest else 0)
                    if left_is_nearest:
                        value, reason = False, "mug_associated_with_left_gripper"
                    elif not right_is_nearest:
                        value, reason = None, "mug_gripper_association_ambiguous"
                    elif self._right_lift_candidate_count < self.RIGHT_LIFT_STABLE_OBSERVATIONS:
                        value, reason = None, "right_gripper_lift_candidate_not_stable"
                    else:
                        self._right_lift_confirmed = True
                        value, reason = True, "stable_right_gripper_lift_association"
        elif name in {"mug_grasp_confirmed", "handoff_confirmed", "rack_contact_detected",
                      "rack_seat_aligned", "mug_released", "mug_placed_on_table",
                      "left_transfer_complete",
                      "grasp_attempt_failed", "task_success", "chunk_budget_exhausted"}:
            value, reason = None, "not_provable_from_semantic_rgbd_alone"
        elif name == "release_ready":
            value, reason = result.release_ready, result.release_ready_reason
        else:
            raise ValueError("unknown_visual_predicate")
        return {"name": name, "value": value, "source": "perception",
                "observation_sequence": observation.sequence, "reason": reason}

    def evaluate(self, name, observation, *, arm="auto"):
        """Predicate-provider adapter used by ``PredicateRegistry``.

        Repeated checks for one sensor sequence use the cached detections, so a
        planner turn and the execution stop gate cannot see different detector
        results for the same observation.
        """
        # Mug-lift evidence uses an Opus-selected public support pixel and its
        # same-frame depth against the reset-cached table plane. It does not
        # need SAM masks on every observation; retain SAM for semantic masks
        # used by other predicates and metric anchor preparation.
        if name == "mug_lifted" and self._mug_lift_tracker_enabled:
            evidence = (self._mug_lift_tracker.evaluate(observation)
                        if self._mug_lift_tracker is not None else
                        {"value": None, "reason": "table_plane_initialization_unavailable"})
            return PredicateResult(name=name, value=evidence.get("value"),
                source="perception", observation_sequence=observation.sequence,
                reason=evidence.get("reason", "unknown"))
        if self._last_result is None or observation.sequence > self._last_sequence:
            self.observe(observation)
        evidence = self.predicate_evidence(name, observation)
        return PredicateResult.model_validate(evidence)

    def _select(self, detections, label):
        minimum = self.min_confidence_by_label.get(label, self.min_confidence)
        candidates = sorted((item for item in detections if item.label == label
                             and item.confidence >= minimum),
                            key=lambda item: item.confidence, reverse=True)
        if not candidates:
            below = any(item.label == label for item in detections)
            return None, "low_confidence" if below else "not_detected"
        if len(candidates) > 1 and candidates[0].confidence-candidates[1].confidence < self.ambiguity_margin:
            return None, "ambiguous_instances"
        return candidates[0], "grounded"

    def _estimate(self, detection, frame, sequence, camera, image):
        ys, xs = np.nonzero(detection.mask)
        if not len(xs):
            return None, "occluded_or_empty_mask"
        image = np.asarray(image)
        image_height, image_width = detection.mask.shape
        normalized_bbox = (float(xs.min()/image_width), float(ys.min()/image_height),
                           float((xs.max()+1)/image_width), float((ys.max()+1)/image_height))
        median_rgb = None
        if image.shape == (image_height, image_width, 3) and image.dtype.kind in "uif":
            pixels = image[ys, xs].astype(float)
            if np.isfinite(pixels).all() and np.all((pixels >= 0) & (pixels <= 255)):
                median_rgb = tuple(float(value) for value in np.median(pixels, axis=0))
        depths = frame.depth_m[ys, xs]
        valid = np.isfinite(depths) & (depths > 0)
        fraction = float(valid.mean())
        if fraction < self.min_valid_depth_fraction:
            return None, "occluded_or_invalid_depth"
        xs, ys = xs[valid], ys[valid]
        depths = depths[valid]
        if len(xs) > self.sample_limit:
            indices = np.linspace(0, len(xs)-1, self.sample_limit, dtype=int)
            xs, ys, depths = xs[indices], ys[indices], depths[indices]
        points = np.asarray([frame.unproject(int(x), int(y)) for x, y in zip(xs, ys)])
        point = np.median(points, axis=0)
        if not np.isfinite(point).all():
            return None, "invalid_world_geometry"
        low, high = np.percentile(points, (10, 90), axis=0)
        visible_extent = tuple(float(value) for value in high-low)
        boundary_points = ()
        median_normal_z = None
        if detection.label == "mug":
            normal_z = frame.vertical_normal_components(detection.mask)
            if len(normal_z) >= self.MIN_ORIENTATION_PIXELS:
                median_normal_z = float(np.median(normal_z))
            # The outer silhouette avoids treating depth seen through the mug's
            # hollow interior as evidence that the mug itself touches the table.
            mask = detection.mask
            interior = mask.copy()
            interior[1:, :] &= mask[:-1, :]
            interior[:-1, :] &= mask[1:, :]
            interior[:, 1:] &= mask[:, :-1]
            interior[:, :-1] &= mask[:, 1:]
            boundary = mask[ys, xs] & ~interior[ys, xs]
            boundary_points = tuple(
                tuple(float(value) for value in sample) for sample in points[boundary])
        return GroundedObject(detection.label, tuple(float(x) for x in point),
            float(detection.confidence), sequence, camera, fraction,
            tuple(tuple(float(value) for value in sample) for sample in points),
            boundary_points, normalized_bbox, median_rgb, visible_extent, median_normal_z), "grounded"

    @staticmethod
    def _table_support_evidence(objects, *, neighborhood_m=0.14, tolerance_m=0.025,
                                min_local_points=3, min_mug_points=3, upright_max_normal_z=0.25,
                                support_view_cameras=("head_camera", *PERCEPTION_CAMERAS)):
        """Conservatively detect an upright mug resting on the locally visible tabletop.

        A semantic overlap or object-center distance is not contact evidence.
        This compares outer-silhouette mug points with the local table-mask
        surface. Incomplete geometry remains unknown rather than being
        interpreted as safe to open the gripper.

        A mug on its side also meets the table, so support further needs the
        visible surface to face sideways like an upright mug's wall. From the
        head camera the median normal |z| of upright mugs that met the table was
        at most 0.18, of toppled mugs at least 0.29. A wrist camera close to a
        toppled mug can see only its flat base, which faces sideways too, so
        support is judged only from the elevated fixed views: across two
        five-seed captures every wrist-view release_ready=true (22) was wrong.
        """
        mug, table = objects.get("mug"), objects.get("table")
        if mug is None or table is None:
            return None, "mug_or_table_geometry_unavailable"
        if mug.camera not in support_view_cameras:
            return None, "table_support_not_judged_from_wrist_view"
        mug_points = np.asarray(mug.boundary_points, dtype=float)
        table_points = np.asarray(table.surface_points, dtype=float)
        if (mug_points.ndim != 2 or mug_points.shape[1:] != (3,)
                or table_points.ndim != 2 or table_points.shape[1:] != (3,)
                or len(mug_points) < min_mug_points):
            return None, "insufficient_support_surface_points"
        center = np.asarray(mug.position[:2], dtype=float)
        local = table_points[np.linalg.norm(table_points[:, :2] - center, axis=1)
                             <= neighborhood_m]
        if len(local) < min_local_points:
            return None, "local_table_surface_unavailable"
        table_z = float(np.median(local[:, 2]))
        # Require multiple outer-silhouette points to lie on the local table
        # plane. Interior pixels through a hollow mug are excluded above.
        supported = np.abs(mug_points[:, 2] - table_z) <= tolerance_m
        support_fraction = float(supported.mean())
        if int(supported.sum()) < min_mug_points or support_fraction < 0.05:
            return None, "mug_table_support_not_established"
        if mug.median_normal_z is None:
            return None, "mug_orientation_unavailable"
        if mug.median_normal_z > upright_max_normal_z:
            return False, "mug_surface_faces_up_not_upright"
        return True, "mug_silhouette_meets_local_table"
