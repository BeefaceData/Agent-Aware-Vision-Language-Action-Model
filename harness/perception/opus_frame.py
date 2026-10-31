"""One-shot Opus image keypoints for public RGB-D object-frame grounding."""

from hashlib import sha256

import numpy as np

from harness.environment.layout import CAMERAS, PERCEPTION_CAMERAS
from harness.prompts import read_prompt
from .pose_grounding import grounded_frame_from_pixels
from .vision_request import (bounded_text, image_block, png_bytes, request_tool_input,
                             require_messages_client, rgb_view, unit_interval)


class OpusObjectFrameGrounder:
    """Ask Opus for visible semantic keypoints, never robot actions or poses."""

    PROMPTS = {
        "rack_peg": "rack_frame_grounding.md",
        "mug_handle": "mug_handle_grounding.md",
    }

    def __init__(self, client, model, *, target, camera):
        require_messages_client(client, model)
        if target not in self.PROMPTS or camera not in (*CAMERAS, *PERCEPTION_CAMERAS):
            raise ValueError("unsupported_frame_grounding_target")
        self.client, self.model = client, model
        self.target, self.camera = target, camera
        self.system_prompt = read_prompt(self.PROMPTS[target])

    @staticmethod
    def _tool_schema():
        return {
            "name": "report_object_frame_keypoints",
            "description": "Report three ordered visible pixels defining an object frame.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "visible": {"type": "boolean"},
                    "keypoints_px": {"anyOf": [
                        {"type": "array", "items": {"type": "array",
                            "items": {"type": "number"}, "minItems": 2,
                            "maxItems": 2}, "minItems": 3, "maxItems": 3},
                        {"type": "null"}]},
                    "description": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["visible", "keypoints_px", "description", "confidence"],
                "additionalProperties": False,
            },
        }

    def locate(self, observation):
        image = rgb_view(observation, self.camera,
                         missing_error="frame_grounding_camera_unavailable",
                         format_error="frame_grounding_requires_uint8_rgb")
        height, width = image.shape[:2]
        png = png_bytes(image)
        result = request_tool_input(
            self.client, model=self.model, max_tokens=350, system=self.system_prompt,
            tool=self._tool_schema(), content=[
                {"type": "text", "text": (
                    f"Target={self.target}; camera={self.camera}; "
                    f"observation={observation.sequence}; image={width}x{height} pixels.")},
                image_block(png)])
        if not isinstance(result, dict):
            raise ValueError("frame_grounding_structured_result_missing")
        visible = result.get("visible")
        if type(visible) is not bool:
            raise ValueError("frame_grounding_visibility_invalid")
        confidence = _confidence(result.get("confidence"))
        common = {"status": "grounded" if visible else "not_detected",
                  "target": self.target, "camera": self.camera,
                  "observation_sequence": observation.sequence,
                  "image_shape_px": [height, width], "confidence": confidence,
                  "description": _description(result.get("description")),
                  "image_sha256": sha256(png).hexdigest()}
        if not visible:
            if result.get("keypoints_px") is not None:
                raise ValueError("invisible_frame_has_keypoints")
            return common
        points = _keypoints(result.get("keypoints_px"))
        if any(not (0 <= u < width and 0 <= v < height) for u, v in points):
            raise ValueError("frame_grounding_keypoint_out_of_bounds")
        return {**common, "keypoints_px": [list(point) for point in points]}


def ground_public_object_frame(observation, report, mask, *, minimum_confidence=0.55,
                               max_mask_gap_px=8.0, max_depth_gap_m=0.06):
    """Validate same-frame Opus pixels against RGB-D and a matching SAM mask.

    This keeps language-model localization as a 2D proposal. Metric coordinates
    come only from the named observation's camera calibration and depth raster.
    """
    if not isinstance(report, dict) or report.get("status") != "grounded":
        raise ValueError("grounded_frame_report_required")
    camera_name = report.get("camera")
    if (camera_name not in observation.images
            or report.get("observation_sequence") != observation.sequence):
        raise ValueError("grounding_report_stale_or_wrong_camera")
    image = np.asarray(observation.images[camera_name])
    if list(image.shape[:2]) != report.get("image_shape_px"):
        raise ValueError("grounding_image_shape_changed")
    confidence = _confidence(report.get("confidence"))
    if (not np.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1
            or confidence < minimum_confidence):
        raise ValueError("grounding_confidence_too_low")
    if (not np.isfinite(max_mask_gap_px) or max_mask_gap_px < 0
            or type(max_mask_gap_px) is bool):
        raise ValueError("invalid_mask_keypoint_tolerance")
    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.shape != image.shape[:2] or not binary_mask.any():
        raise ValueError("grounding_mask_missing_or_misaligned")
    frames = observation.camera_frames or {}
    camera = frames.get(camera_name)
    if camera is None:
        raise ValueError("grounding_rgbd_camera_unavailable")
    points = np.asarray(_keypoints(report.get("keypoints_px")), dtype=float)
    ys, xs = np.nonzero(binary_mask)
    snapped = []
    for u, v in points:
        distance = np.hypot(xs - u, ys - v)
        if not len(distance) or float(distance.min()) > max_mask_gap_px:
            raise ValueError("grounding_keypoint_outside_object_mask")
        # Thin rack wires often put the model's semantic pixel just beside the
        # segmented material, where depth belongs to the background. Snap only
        # to a nearby mask pixel whose own local depth is surface-consistent.
        candidates = np.argsort(distance)
        selected = None
        for index in candidates:
            if float(distance[index]) > max_mask_gap_px:
                break
            pixel = (int(xs[index]), int(ys[index]))
            try:
                depth_gap = camera.local_depth_gap(
                    pixel[0], pixel[1], search_radius=8, point_radius=1)
            except ValueError:
                continue
            if depth_gap <= max_depth_gap_m:
                selected = pixel
                break
        if selected is None:
            raise ValueError("keypoint_depth_discontinuity")
        snapped.append(selected)
    return grounded_frame_from_pixels(camera, snapped,
                                      max_depth_gap_m=max_depth_gap_m)


def ground_public_object_frame_rgbd(observation, report, *, minimum_confidence=0.55,
                                    max_depth_gap_m=0.06):
    """Ground Opus keypoints directly with RGB-D, without a segmentation mask.

    The report remains a 2D semantic proposal. Each selected pixel must have a
    locally continuous depth surface, and the resulting 3D frame must pass the
    same baseline and handedness checks as mask-supported grounding.
    """
    if not isinstance(report, dict) or report.get("status") != "grounded":
        raise ValueError("grounded_frame_report_required")
    camera_name = report.get("camera")
    if (camera_name not in observation.images
            or report.get("observation_sequence") != observation.sequence):
        raise ValueError("grounding_report_stale_or_wrong_camera")
    image = np.asarray(observation.images[camera_name])
    if list(image.shape[:2]) != report.get("image_shape_px"):
        raise ValueError("grounding_image_shape_changed")
    confidence = _confidence(report.get("confidence"))
    if (not np.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1
            or confidence < minimum_confidence):
        raise ValueError("grounding_confidence_too_low")
    if not np.isfinite(max_depth_gap_m) or max_depth_gap_m < 0:
        raise ValueError("invalid_frame_limits")
    camera = (observation.camera_frames or {}).get(camera_name)
    if camera is None:
        raise ValueError("grounding_rgbd_camera_unavailable")
    points = _keypoints(report.get("keypoints_px"))
    return grounded_frame_from_pixels(camera, points,
                                      max_depth_gap_m=max_depth_gap_m)


def _keypoints(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("frame_grounding_requires_three_keypoints")
    points = []
    try:
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError
            uv = tuple(float(item) for item in point)
            if not np.isfinite(uv).all():
                raise ValueError
            points.append(uv)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("frame_grounding_keypoints_invalid") from None
    return tuple(points)


def _confidence(value):
    return unit_interval(value, "frame_grounding_confidence_invalid")


def _description(value):
    return bounded_text(value, 500, "frame_grounding_description_invalid")
