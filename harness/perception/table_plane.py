"""Episode-local table plane and table-relative mug lift evidence.

Only public RGB-D pixels are used. The table plane is fitted once from a reset
table mask; later Opus support pixels are unprojected with their same-frame
depth and camera calibration.
"""

import numpy as np

from harness.prompts import read_prompt
from .rgbd import RGBDCamera, calibration_sha256
from .vision_request import (bounded_text, image_block, png_bytes, request_tool_input,
                             require_messages_client, rgb_view, unit_interval)


class StaticTablePlaneCache:
    """Fit and cache a horizontal table plane from one reset RGB-D mask."""

    def __init__(self, *, max_points=2048, inlier_tolerance_m=0.012,
                 max_p95_residual_m=0.012, min_points=50):
        if (type(max_points) is not int or not 50 <= max_points <= 10000
                or type(min_points) is not int or not 20 <= min_points <= max_points
                or not np.isfinite((inlier_tolerance_m, max_p95_residual_m)).all()
                or not 0 < inlier_tolerance_m <= 0.05
                or not 0 < max_p95_residual_m <= 0.05):
            raise ValueError("invalid_table_plane_limits")
        self.max_points = max_points
        self.inlier_tolerance_m = float(inlier_tolerance_m)
        self.max_p95_residual_m = float(max_p95_residual_m)
        self.min_points = min_points
        self._reference = None

    def cache(self, observation, mask, *, camera="head_camera"):
        image = observation.images.get(camera)
        frame = (observation.camera_frames or {}).get(camera)
        binary = np.asarray(mask, dtype=bool)
        if image is None or not isinstance(frame, RGBDCamera):
            raise ValueError("table_plane_rgbd_unavailable")
        shape = tuple(np.asarray(image).shape[:2])
        if binary.shape != shape or not binary.any() or frame.depth_m.shape != shape:
            raise ValueError("table_plane_mask_misaligned")
        ys, xs = np.nonzero(binary & (frame.depth_m > 0))
        if len(xs) < self.min_points:
            raise ValueError("table_plane_too_few_depth_points")
        if len(xs) > self.max_points:
            indexes = np.linspace(0, len(xs) - 1, self.max_points, dtype=int)
            xs, ys = xs[indexes], ys[indexes]
        points = []
        for u, v in zip(xs, ys):
            try:
                points.append(frame.unproject(int(u), int(v)))
            except ValueError:
                continue
        points = np.asarray(points, dtype=np.float64)
        if len(points) < self.min_points:
            raise ValueError("table_plane_too_few_grounded_points")
        normal, offset, residuals, inliers = _robust_plane(
            points, self.inlier_tolerance_m)
        if int(inliers.sum()) < self.min_points:
            raise ValueError("table_plane_too_few_inliers")
        p95 = float(np.percentile(residuals[inliers], 95))
        if p95 > self.max_p95_residual_m:
            raise ValueError("table_plane_residual_too_large")
        # The manipulation workspace uses +Z as up. Reject a non-table surface.
        if normal[2] < 0.90:
            raise ValueError("table_plane_not_horizontal")
        self._reference = {
            "normal": tuple(float(x) for x in normal),
            "offset": float(offset),
            "camera": camera,
            "sequence": observation.sequence,
            "image_shape": shape,
            "calibration_sha256": calibration_sha256(frame),
            "valid_points": int(len(points)),
            "inliers": int(inliers.sum()),
            "p95_residual_m": p95,
        }
        return self.metadata()

    def signed_distance(self, observation, pixel, *, camera=None, depth_radius=1):
        reference = self._require_reference()
        camera = camera or reference["camera"]
        image = observation.images.get(camera)
        frame = (observation.camera_frames or {}).get(camera)
        if camera != reference["camera"] or image is None or not isinstance(frame, RGBDCamera):
            raise ValueError("table_plane_camera_unavailable")
        if tuple(np.asarray(image).shape[:2]) != reference["image_shape"]:
            raise ValueError("table_plane_image_shape_changed")
        if calibration_sha256(frame) != reference["calibration_sha256"]:
            raise ValueError("table_plane_camera_calibration_changed")
        if (not isinstance(pixel, (tuple, list, np.ndarray)) or len(pixel) != 2
                or not np.isfinite(np.asarray(pixel, dtype=float)).all()):
            raise ValueError("invalid_support_pixel")
        point = np.asarray(frame.unproject(*pixel, radius=depth_radius), dtype=float)
        return float(np.dot(np.asarray(reference["normal"]), point) + reference["offset"])

    def clear(self):
        self._reference = None

    def metadata(self):
        if self._reference is None:
            return {"available": False}
        return {"available": True, "camera": self._reference["camera"],
                "observation_sequence": self._reference["sequence"],
                "image_shape_px": list(self._reference["image_shape"]),
                "valid_depth_points": self._reference["valid_points"],
                "plane_inliers": self._reference["inliers"],
                "p95_residual_m": self._reference["p95_residual_m"],
                "camera_calibration_sha256": self._reference["calibration_sha256"]}

    def _require_reference(self):
        if self._reference is None:
            raise ValueError("table_plane_not_cached")
        return self._reference


def _robust_plane(points, tolerance):
    """SVD fit with deterministic median-residual trimming."""
    active = np.ones(len(points), dtype=bool)
    normal = offset = None
    for _ in range(4):
        cloud = points[active]
        center = np.median(cloud, axis=0)
        _, _, vectors = np.linalg.svd(cloud - center, full_matrices=False)
        normal = vectors[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -float(np.dot(normal, center))
        residuals = np.abs(points @ normal + offset)
        updated = residuals <= tolerance
        if int(updated.sum()) < 20 or np.array_equal(updated, active):
            break
        active = updated
    residuals = np.abs(points @ normal + offset)
    return normal, offset, residuals, active


class OpusMugSupportGrounder:
    """Request one visible mug-base/support pixel from a single RGB frame."""

    def __init__(self, client, model, *, camera="head_camera"):
        require_messages_client(client, model)
        if camera != "head_camera":
            raise ValueError("mug_support_requires_head_camera")
        self.client, self.model, self.camera = client, model, camera
        self.system_prompt = read_prompt("mug_support_grounding.md")

    @staticmethod
    def _tool_schema():
        return {"name": "report_mug_support_pixel",
            "description": "Report the visible mug base pixel that meets the support surface.",
            "input_schema": {"type": "object", "properties": {
                "visible": {"type": "boolean"},
                "support_pixel": {"anyOf": [
                    {"type": "array", "items": {"type": "number"},
                     "minItems": 2, "maxItems": 2}, {"type": "null"}]},
                "description": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["visible", "support_pixel", "description", "confidence"],
                "additionalProperties": False}}

    def locate(self, observation):
        image = rgb_view(observation, self.camera,
                         missing_error="mug_support_camera_unavailable",
                         format_error="mug_support_requires_uint8_rgb")
        height, width = image.shape[:2]
        result = request_tool_input(
            self.client, model=self.model, max_tokens=250, system=self.system_prompt,
            tool=self._tool_schema(), content=[
                {"type": "text", "text": (
                    f"Camera={self.camera}; observation={observation.sequence}; "
                    f"image={width}x{height} pixels.")},
                image_block(png_bytes(image))])
        if not isinstance(result, dict) or type(result.get("visible")) is not bool:
            raise ValueError("mug_support_structured_result_invalid")
        confidence = unit_interval(result.get("confidence"), "mug_support_confidence_invalid")
        report = {"status": "grounded" if result["visible"] else "not_detected",
            "camera": self.camera, "observation_sequence": observation.sequence,
            "image_shape_px": [height, width], "confidence": confidence,
            "description": bounded_text(result.get("description"), 400,
                                        "mug_support_description_invalid")}
        pixel = result.get("support_pixel")
        if not result["visible"]:
            if pixel is not None:
                raise ValueError("invisible_mug_support_has_pixel")
            return report
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            raise ValueError("mug_support_pixel_invalid")
        uv = np.asarray(pixel, dtype=float)
        if not np.isfinite(uv).all() or not (0 <= uv[0] < width and 0 <= uv[1] < height):
            raise ValueError("mug_support_pixel_out_of_bounds")
        report["support_pixel"] = [float(uv[0]), float(uv[1])]
        return report


class MugLiftEvidenceTracker:
    """Turn repeated support-pixel/table distances into tri-state lift evidence."""

    def __init__(self, table_plane, grounder, *, minimum_confidence=0.55,
                 lifted_distance_m=0.04, table_distance_m=0.025,
                 consecutive_observations=2,
                 minimum_policy_actions=0):
        if (not isinstance(table_plane, StaticTablePlaneCache)
                or not callable(getattr(grounder, "locate", None))
                or not np.isfinite((minimum_confidence, lifted_distance_m,
                                    table_distance_m)).all()
                or not 0 <= minimum_confidence <= 1
                or not 0 < table_distance_m < lifted_distance_m <= 0.20
                or type(consecutive_observations) is not int
                or not 2 <= consecutive_observations <= 5
                or type(minimum_policy_actions) is not int
                or not 0 <= minimum_policy_actions <= 10000):
            raise ValueError("invalid_mug_lift_tracker_settings")
        self.table_plane, self.grounder = table_plane, grounder
        self.minimum_confidence = float(minimum_confidence)
        self.lifted_distance_m = float(lifted_distance_m)
        self.table_distance_m = float(table_distance_m)
        self.consecutive_observations = consecutive_observations
        self.minimum_policy_actions = minimum_policy_actions
        self.policy_actions_executed = 0
        self._last_sequence = -1
        self._last_result = None
        self._lift_streak = 0
        self._lift_latched = False

    def set_policy_action_count(self, count):
        if type(count) is not int or count < self.policy_actions_executed:
            raise ValueError("policy_action_count_must_be_monotonic")
        self.policy_actions_executed = count

    def evaluate(self, observation):
        if observation.sequence == self._last_sequence:
            return dict(self._last_result)
        if observation.sequence < self._last_sequence:
            return self._unknown(observation.sequence, "stale_observation")
        self._last_sequence = observation.sequence
        if self.policy_actions_executed < self.minimum_policy_actions:
            return self._store({"value": None,
                "reason": "mug_lift_check_deferred_until_feedback_window",
                "observation_sequence": observation.sequence,
                "policy_actions_executed": self.policy_actions_executed,
                "first_check_policy_actions": self.minimum_policy_actions})
        try:
            report = self.grounder.locate(observation)
            if (report.get("status") != "grounded"
                    or report.get("observation_sequence") != observation.sequence
                    or report.get("camera") != "head_camera"):
                raise ValueError("mug_support_not_visible")
            if report.get("confidence", 0.0) < self.minimum_confidence:
                raise ValueError("mug_support_confidence_too_low")
            distance = self.table_plane.signed_distance(
                observation, report.get("support_pixel"), camera="head_camera")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self._lift_streak = 0
            return self._store({"value": True if self._lift_latched else None,
                "reason": ("lift_event_previously_confirmed" if self._lift_latched
                           else str(exc)), "observation_sequence": observation.sequence})
        if distance >= self.lifted_distance_m:
            self._lift_streak += 1
            if self._lift_streak >= self.consecutive_observations:
                self._lift_latched = True
        elif distance <= self.table_distance_m:
            self._lift_streak = 0
        else:
            self._lift_streak = 0
        if self._lift_latched:
            value, reason = True, "stable_public_table_relative_lift"
        elif distance <= self.table_distance_m:
            value, reason = False, "mug_support_point_at_table"
        else:
            value, reason = None, "lift_evidence_not_yet_stable"
        return self._store({"value": value, "reason": reason,
            "support_height_above_table_m": float(distance),
            "lift_observations": self._lift_streak,
            "observation_sequence": observation.sequence})

    def _unknown(self, sequence, reason):
        return {"value": None, "reason": reason, "observation_sequence": sequence}

    def _store(self, result):
        self._last_result = result
        return dict(result)
