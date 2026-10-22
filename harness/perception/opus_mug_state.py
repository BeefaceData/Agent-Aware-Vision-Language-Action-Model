"""Structured visual mug-state assessment from phase-relevant public RGB views."""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np


PHASE_CAMERAS = {
    "grasp": ("head_camera", "left_camera"),
    "handoff": ("head_camera", "left_camera", "right_camera"),
    "hang": ("head_camera", "right_camera"),
}
VISUAL_STATES = frozenset({
    "mug_on_table", "mug_overlapping_gripper", "mug_between_grippers",
    "mug_on_rack", "mug_visible_elsewhere",
    "mug_not_visible", "uncertain",
})
VISIBILITY = frozenset({"visible", "partially_occluded", "not_visible", "uncertain"})


class OpusMugStateAssessor:
    """Ask Opus for visual state and 2D boxes, never metric poses or actions.

    The returned gripper relation is a single-frame visual judgment. It does
    not establish contact, grasp, or lift; those require temporal corroboration.
    """

    PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "mug_state_assessment.md"

    def __init__(self, client, model):
        if not callable(getattr(getattr(client, "messages", None), "create", None)):
            raise ValueError("anthropic_messages_client_required")
        if not isinstance(model, str) or not model:
            raise ValueError("anthropic_model_required")
        self.client, self.model = client, model
        self.system_prompt = self.PROMPT_PATH.read_text(encoding="utf-8")

    @staticmethod
    def _tool_schema():
        bbox = {"anyOf": [
            {"type": "array", "items": {"type": "number"}, "minItems": 4,
             "maxItems": 4},
            {"type": "null"},
        ]}
        camera = {
            "type": "object",
            "properties": {
                "camera": {"type": "string", "enum": [
                    "head_camera", "left_camera", "right_camera"]},
                "mug_visibility": {"type": "string", "enum": sorted(VISIBILITY)},
                "mug_bbox_px": bbox,
                "gripper_visibility": {"type": "string", "enum": sorted(VISIBILITY)},
                "gripper_bbox_px": bbox,
                "evidence": {"type": "string"},
            },
            "required": ["camera", "mug_visibility", "mug_bbox_px",
                         "gripper_visibility", "gripper_bbox_px", "evidence"],
            "additionalProperties": False,
        }
        return {
            "name": "report_mug_visual_state",
            "description": "Classify visible mug state and report approximate per-camera boxes.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string", "enum": list(PHASE_CAMERAS)},
                    "visual_state": {"type": "string", "enum": sorted(VISUAL_STATES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                    "views": {"type": "array", "items": camera,
                              "minItems": 1, "maxItems": 3},
                },
                "required": ["phase", "visual_state", "confidence", "evidence", "views"],
                "additionalProperties": False,
            },
        }

    def assess(self, observation, *, phase, cameras=None):
        from PIL import Image

        if phase not in PHASE_CAMERAS:
            raise ValueError("unsupported_mug_assessment_phase")
        if cameras is None:
            cameras = PHASE_CAMERAS[phase]
        if (not isinstance(cameras, (tuple, list)) or not cameras
                or len(cameras) > 3 or len(set(cameras)) != len(cameras)
                or any(camera not in PHASE_CAMERAS[phase] for camera in cameras)):
            raise ValueError("unsupported_mug_assessment_cameras")
        images = {}
        for camera in cameras:
            image = observation.images.get(camera)
            if image is None:
                raise ValueError("mug_assessment_camera_unavailable")
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("mug_assessment_requires_uint8_rgb")
            images[camera] = image

        content = [{"type": "text", "text": (
            f"Observation={observation.sequence}; phase={phase}. "
            "The following synchronized views are labelled by camera:")}]
        for camera, image in images.items():
            stream = BytesIO()
            Image.fromarray(image).save(stream, format="PNG", optimize=True)
            content.append({"type": "text", "text": camera})
            content.append({"type": "image", "source": {"type": "base64",
                "media_type": "image/png", "data": base64.b64encode(
                    stream.getvalue()).decode("ascii")}})

        tool = self._tool_schema()
        response = self.client.messages.create(model=self.model, max_tokens=800,
            system=self.system_prompt, tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": content}])
        result = next((item.input for item in response.content
                       if getattr(item, "type", None) == "tool_use"
                       and getattr(item, "name", None) == tool["name"]), None)
        return self._validate(result, phase, images, observation.sequence)

    @classmethod
    def _validate(cls, result, phase, images, sequence):
        if not isinstance(result, dict):
            raise ValueError("mug_assessment_structured_result_missing")
        if (result.get("phase") != phase or result.get("visual_state") not in VISUAL_STATES
                or not isinstance(result.get("views"), list)
                or len(result["views"]) != len(images)):
            raise ValueError("invalid_mug_assessment_result")
        confidence = result.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
                or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid_mug_assessment_confidence")
        by_camera = {}
        for view in result["views"]:
            if (not isinstance(view, dict) or view.get("camera") not in images
                    or view["camera"] in by_camera
                    or view.get("mug_visibility") not in VISIBILITY
                    or view.get("gripper_visibility") not in VISIBILITY
                    or not isinstance(view.get("evidence"), str)):
                raise ValueError("invalid_mug_assessment_view")
            camera = view["camera"]
            height, width = images[camera].shape[:2]
            validated = dict(view)
            validated["evidence"] = view["evidence"][:250]
            for label in ("mug", "gripper"):
                box = view.get(f"{label}_bbox_px")
                if box is None:
                    validated[f"{label}_bbox_px"] = None
                    continue
                try:
                    coords = np.asarray(box, dtype=float)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError("invalid_mug_assessment_bbox") from None
                if (coords.shape != (4,) or not np.isfinite(coords).all()
                        or not 0 <= coords[0] < coords[2] <= width
                        or not 0 <= coords[1] < coords[3] <= height):
                    raise ValueError("invalid_mug_assessment_bbox")
                validated[f"{label}_bbox_px"] = [float(x) for x in coords]
            mug_box = validated["mug_bbox_px"]
            gripper_box = validated["gripper_bbox_px"]
            validated["gripper_overlap_fraction_of_mug"] = (
                _overlap_fraction(mug_box, gripper_box)
                if mug_box and gripper_box else None)
            by_camera[camera] = validated
        if set(by_camera) != set(images):
            raise ValueError("mug_assessment_camera_set_mismatch")
        return {"observation_sequence": sequence, "phase": phase,
                "visual_state": result["visual_state"],
                "confidence": float(confidence),
                "evidence": result.get("evidence", "")[:500],
                "views": [by_camera[camera] for camera in images]}


def _overlap_fraction(mug_box, gripper_box):
    """Return intersection area as a fraction of mug-box area."""
    mx1, my1, mx2, my2 = mug_box
    gx1, gy1, gx2, gy2 = gripper_box
    intersection = max(0.0, min(mx2, gx2)-max(mx1, gx1)) * max(
        0.0, min(my2, gy2)-max(my1, gy1))
    return float(intersection / ((mx2-mx1) * (my2-my1)))
