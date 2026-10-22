"""One-shot public RGB rack localization by the configured Anthropic planner."""

import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np


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
    PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "rack_grounding.md"

    def __init__(self, client, model):
        if not callable(getattr(getattr(client, "messages", None), "create", None)):
            raise ValueError("anthropic_messages_client_required")
        if not isinstance(model, str) or not model:
            raise ValueError("anthropic_model_required")
        self.client, self.model = client, model
        self.system_prompt = self.PROMPT_PATH.read_text(encoding="utf-8")

    def locate(self, observation):
        """Return a validated, normalized public image reference or a miss."""
        from PIL import Image

        image = observation.images.get(self.CAMERA)
        if image is None:
            raise ValueError("rack_grounding_head_camera_unavailable")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rack_grounding_requires_uint8_rgb")
        height, width = image.shape[:2]
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG", optimize=True)
        png = stream.getvalue()
        tool = _tool_schema()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=self.system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    f"Initial episode frame, observation {observation.sequence}; "
                    f"image size is {width}x{height} pixels.")},
                {"type": "image", "source": {"type": "base64",
                    "media_type": "image/png", "data": base64.b64encode(png).decode("ascii")}},
            ]}],
        )
        result = next((item.input for item in response.content
                       if getattr(item, "type", None) == "tool_use"
                       and getattr(item, "name", None) == tool["name"]), None)
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
    if isinstance(value, bool):
        raise ValueError("rack_grounding_confidence_invalid")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("rack_grounding_confidence_invalid") from None
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("rack_grounding_confidence_invalid")
    return value


def _description(value):
    if not isinstance(value, str):
        raise ValueError("rack_grounding_description_invalid")
    return value[:500]


def _camera_calibration_sha256(observation):
    frame = (observation.camera_frames or {}).get("head_camera")
    if frame is None:
        return None
    digest = sha256()
    digest.update(np.asarray(frame.intrinsic, dtype=np.float64).tobytes())
    digest.update(np.asarray(frame.world_to_camera, dtype=np.float64).tobytes())
    return digest.hexdigest()
