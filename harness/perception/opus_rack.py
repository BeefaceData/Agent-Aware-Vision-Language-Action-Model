"""One-shot public RGB rack localization by the configured Anthropic planner."""

from hashlib import sha256

import numpy as np

from harness.prompts import read_prompt
from .rgbd import calibration_sha256
from .vision_request import (bounded_text, image_block, png_bytes, request_tool_input,
                             require_messages_client, rgb_view, unit_interval)


def _tool_schema():
    return {
        "name": "report_rack_localization",
        "description": "Report the visible rack box and one visible angled hanger peg.",
        "input_schema": {
            "type": "object",
            "properties": {
                "visible": {"type": "boolean"},
                "bbox_px": {"anyOf": [
                    {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    {"type": "null"}]},
                "peg_px": {"anyOf": [
                    {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    {"type": "null"}]},
                "description": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["visible", "bbox_px", "peg_px", "description", "confidence"],
            "additionalProperties": False,
        },
    }


class OpusRackGrounder:
    """Ask Opus once for a head-camera rack reference, never for robot actions."""

    CAMERA = "head_camera"

    def __init__(self, client, model):
        require_messages_client(client, model)
        self.client, self.model = client, model
        self.system_prompt = read_prompt("rack_grounding.md")

    def locate(self, observation):
        """Return a validated, normalized public image reference or a miss."""
        image = rgb_view(observation, self.CAMERA,
                         missing_error="rack_grounding_head_camera_unavailable",
                         format_error="rack_grounding_requires_uint8_rgb")
        height, width = image.shape[:2]
        png = png_bytes(image)
        result = request_tool_input(
            self.client, model=self.model, max_tokens=300, system=self.system_prompt,
            tool=_tool_schema(), content=[
                {"type": "text", "text": (
                    f"Initial episode frame, observation {observation.sequence}; "
                    f"image size is {width}x{height} pixels.")},
                image_block(png)])
        if not isinstance(result, dict):
            raise ValueError("rack_grounding_structured_result_missing")
        common = {"source": "opus_rgb", "camera": self.CAMERA,
                  "observation_sequence": observation.sequence,
                  "image_shape_px": [height, width],
                  "image_sha256": sha256(png).hexdigest(),
                  "camera_calibration_sha256": _camera_calibration_sha256(observation)}
        visible = result.get("visible")
        if type(visible) is not bool:
            raise ValueError("rack_grounding_visibility_invalid")
        if not visible:
            return {**common, "status": "not_detected",
                    "confidence": _confidence(result.get("confidence")),
                    "description": _description(result.get("description"))}

        bbox = _coordinates(result.get("bbox_px"), 4, "rack_grounding_bbox_invalid")
        peg = _coordinates(result.get("peg_px"), 2, "rack_grounding_peg_invalid")
        x1, y1, x2, y2 = bbox
        u, v = peg
        if (not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height
                or not x1 <= u <= x2 or not y1 <= v <= y2):
            raise ValueError("rack_grounding_coordinates_out_of_bounds")
        confidence = _confidence(result.get("confidence"))
        return {**common, "status": "grounded", "confidence": confidence,
                "bbox_normalized": [x1 / width, y1 / height, x2 / width, y2 / height],
                "peg_uv_normalized": [u / max(1, width - 1), v / max(1, height - 1)],
                "description": _description(result.get("description"))}


def _coordinates(value, size, error):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(error)
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(error) from None
    if not np.isfinite(values).all():
        raise ValueError(error)
    return values


def _confidence(value):
    return unit_interval(value, "rack_grounding_confidence_invalid")


def _description(value):
    return bounded_text(value, 500, "rack_grounding_description_invalid")


def _camera_calibration_sha256(observation):
    frame = (observation.camera_frames or {}).get("head_camera")
    return None if frame is None else calibration_sha256(frame)
