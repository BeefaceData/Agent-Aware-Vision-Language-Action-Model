"""One-shot Opus image keypoints for public RGB-D object-frame grounding."""

import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

from .pose_grounding import grounded_frame_from_pixels


class OpusObjectFrameGrounder:
    """Ask Opus for visible semantic keypoints, never robot actions or poses."""

    PROMPTS = {
        "rack_peg": "rack_frame_grounding.md",
        "mug_handle": "mug_handle_grounding.md",
    }

    def __init__(self, client, model, *, target, camera):
        if not callable(getattr(getattr(client, "messages", None), "create", None)):
            raise ValueError("anthropic_messages_client_required")
        if not isinstance(model, str) or not model:
            raise ValueError("anthropic_model_required")
        if target not in self.PROMPTS or camera not in (
                "head_camera", "left_camera", "right_camera"):
            raise ValueError("unsupported_frame_grounding_target")
        self.client, self.model = client, model
        self.target, self.camera = target, camera
        path = Path(__file__).resolve().parents[1] / "prompts" / self.PROMPTS[target]
        self.system_prompt = path.read_text(encoding="utf-8")

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
        from PIL import Image

        image = observation.images.get(self.camera)
        if image is None:
            raise ValueError("frame_grounding_camera_unavailable")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("frame_grounding_requires_uint8_rgb")
        height, width = image.shape[:2]
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG", optimize=True)
        png = stream.getvalue()
        tool = self._tool_schema()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=350,
            system=self.system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    f"Target={self.target}; camera={self.camera}; "
                    f"observation={observation.sequence}; image={width}x{height} pixels.")},
                {"type": "image", "source": {"type": "base64",
                    "media_type": "image/png", "data": base64.b64encode(png).decode("ascii")}},
            ]}],
        )
        result = next((item.input for item in response.content
                       if getattr(item, "type", None) == "tool_use"
                       and getattr(item, "name", None) == tool["name"]), None)
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
    if isinstance(value, bool):
        raise ValueError("frame_grounding_confidence_invalid")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("frame_grounding_confidence_invalid") from None
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("frame_grounding_confidence_invalid")
    return value


def _description(value):
    if not isinstance(value, str):
        raise ValueError("frame_grounding_description_invalid")
    return value[:500]
