"""Bounded public grounding before spending a simulator replay on API setup."""

from hashlib import sha256
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from development.checkpoints.bank import CheckpointBank
from development.hanging.run_entry660 import validate_package
from harness.perception.opus_frame import OpusObjectFrameGrounder, ground_public_object_frame_rgbd
from harness.perception.replay import load_public_observation
from harness.perception.vision_request import png_bytes


CROP_PADDING_PX = 24
CROP_LONG_SIDE_PX = 768


def frame_crop_retry(grounder, observation, proposal, output, *, artifact_prefix=None):
    """One magnified RGB retry; metric depth always remains in the original view."""
    camera = "perception_camera"
    rgb = observation.images[camera]
    height, width = rgb.shape[:2]
    points = np.asarray(proposal.get("keypoints_px"), dtype=float)
    target = proposal.get("target")
    if (target not in ("rack_peg", "mug_handle") or proposal.get("camera") != camera
            or proposal.get("observation_sequence") != observation.sequence
            or points.shape != (3, 2) or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] >= height)):
        raise ValueError("crop_requires_current_in_bounds_frame_proposal")
    lower = np.floor(points.min(axis=0)).astype(int) - CROP_PADDING_PX
    upper = np.ceil(points.max(axis=0)).astype(int) + CROP_PADDING_PX + 1
    left, top = max(0, int(lower[0])), max(0, int(lower[1]))
    right, bottom = min(width, int(upper[0])), min(height, int(upper[1]))
    crop = Image.fromarray(rgb).crop((left, top, right, bottom))
    scale = CROP_LONG_SIDE_PX / max(crop.size)
    shown = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                        resample=Image.Resampling.BICUBIC)
    output = Path(output)
    prefix = artifact_prefix or ("rack" if target == "rack_peg" else "mug_handle")
    shown.save(output / f"{prefix}_crop_source.png")
    # This observation is only for the image proposal, never for depth lookup.
    crop_observation = replace(observation,
        images={**observation.images, camera: np.asarray(shown)}, camera_frames=None)
    cropped_report = grounder.locate(crop_observation)
    transform = dict(box_xyxy=[left, top, right, bottom],
                     shown_shape_px=[shown.height, shown.width],
                     original_shape_px=[height, width], mapping="pixel_center")
    (output / f"{prefix}_crop_report.json").write_text(json.dumps(
        dict(proposal=cropped_report, transform=transform), indent=2) + "\n")
    mapped = {**cropped_report, "image_shape_px": [height, width],
              "image_sha256": sha256(png_bytes(rgb)).hexdigest()}
    if "keypoints_px" in cropped_report:
        mapped_points = []
        for u, v in cropped_report["keypoints_px"]:
            original_u = left + (u + .5) * crop.width / shown.width - .5
            original_v = top + (v + .5) * crop.height / shown.height - .5
            if not (0 <= original_u < width and 0 <= original_v < height):
                raise ValueError("mapped_crop_keypoint_out_of_bounds")
            mapped_points.append([original_u, original_v])
        mapped["keypoints_px"] = mapped_points
    return mapped, transform


def validate_with_crop_retry(grounder, observation, proposal, output, *, row,
                             artifact_prefix=None, on_retry=lambda: None):
    """One low-confidence retry for either functional frame; never relax geometry."""
    row["proposal"] = proposal
    target = proposal.get("target")
    try:
        frame = ground_public_object_frame_rgbd(observation, proposal)
    except ValueError as exc:
        if target not in ("rack_peg", "mug_handle") or str(exc) != "grounding_confidence_too_low":
            raise
        row["initial_proposal"] = proposal
        prefix = artifact_prefix or ("rack" if target == "rack_peg" else "mug_handle")
        image = Image.fromarray(observation.images["perception_camera"])
        draw = ImageDraw.Draw(image)
        for index, (u, v) in enumerate(proposal.get("keypoints_px", ())):
            draw.ellipse((u-3, v-3, u+3, v+3), outline="red")
            draw.text((u+4, v), str(index+1), fill="red")
        image.save(Path(output) / f"{prefix}_initial.png")
        row["status"] = "crop_retry"
        on_retry()
        proposal, transform = frame_crop_retry(grounder, observation, proposal, output,
                                                artifact_prefix=artifact_prefix)
        row["proposal"], row["crop_transform"] = proposal, transform
        image = Image.fromarray(observation.images["perception_camera"])
        draw = ImageDraw.Draw(image)
        for index, (u, v) in enumerate(proposal.get("keypoints_px", ())):
            draw.ellipse((u-3, v-3, u+3, v+3), outline="red")
            draw.text((u+4, v), str(index+1), fill="red")
        image.save(Path(output) / f"{artifact_prefix or target}.png")
        row["status"] = "validating_geometry"
        frame = ground_public_object_frame_rgbd(observation, proposal)
    row["camera_derived_frame"] = list(frame)
    return proposal


def sanitized_failure(exc):
    """Keep provider error text/IDs, never request bodies, headers or images."""
    result = {"error_type": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if type(status) is int:
        result["http_status"] = status
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}

    def clean(value):
        text = str(value)
        text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
        text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
        text = re.sub(r"data:image/[^\s\"']+", "[IMAGE REDACTED]", text)
        text = re.sub(r"[A-Za-z0-9+/=_-]{100,}", "[LONG VALUE REDACTED]", text)
        return text[:700]

    if isinstance(error, dict):
        for source, destination in (("type", "provider_error_type"), ("message", "provider_message")):
            if isinstance(error.get(source), str):
                result[destination] = clean(error[source])
    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str):
        result["request_id"] = clean(request_id)
    if isinstance(exc, ValueError):
        result["validation_error"] = clean(str(exc))
    return result


def grounding_client(client):
    """Opus 5.5 uses adaptive thinking and cannot be forced to call a tool."""
    def create(**request):
        tool_name = request["tools"][0]["name"]
        request["system"] += (
            f"\nRespond by calling {tool_name} exactly once with your keypoint report. "
            "If landmarks are uncertain, use that tool to report visible=false; "
            "do not infer hidden landmarks. Do not respond with prose alone."
        )
        request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        request["max_tokens"] = 4096  # Includes adaptive thinking and the short report.
        # extra_body also works with the simulator's older Anthropic SDK.
        request["extra_body"] = {"thinking": {"type": "adaptive"},
                                 "output_config": {"effort": "low"}}
        response = client.messages.create(**request)
        calls = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
        if len(calls) != 1 or calls[0].name != tool_name:
            raise ValueError("grounding_requires_exactly_one_keypoint_report")
        return response
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def preflight(package, output, model, *, client=None):
    checkpoint = validate_package(package, allow_unverified=True)
    if not isinstance(model, str) or not model.startswith("claude-") or "MODEL" in model:
        raise ValueError("real_claude_model_id_required")
    bank = CheckpointBank(package)
    observation = load_public_observation(bank.resolve_artifact(checkpoint.images["head_camera"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="checking", model=model, camera="perception_camera", sequence=660,
        rgb_sha256=sha256(observation.images["perception_camera"].tobytes()).hexdigest(),
        targets={}, maximum_requests=4, requests_attempted=0,
        simulator_started=False, capture_provenance="unknown")
    try:
        if client is None:
            from anthropic import Anthropic
            client = Anthropic(max_retries=0, timeout=60.)
        client = grounding_client(client)
        for target in ("rack_peg", "mug_handle"):
            row = dict(status="requesting")
            report["targets"][target] = row
            report["requests_attempted"] += 1
            grounder = OpusObjectFrameGrounder(client, model, target=target,
                                              camera="perception_camera")
            proposal = grounder.locate(observation)
            row["proposal"] = proposal
            image = Image.fromarray(observation.images["perception_camera"])
            draw = ImageDraw.Draw(image)
            for index, (u, v) in enumerate(proposal.get("keypoints_px", ())):
                draw.ellipse((u-3, v-3, u+3, v+3), outline="red")
                draw.text((u+4, v), str(index+1), fill="red")
            image.save(output / f"{target}.png")
            row["status"] = "validating_geometry"
            def count_retry():
                report["requests_attempted"] += 1
            validate_with_crop_retry(grounder, observation, proposal, output,
                                     row=row, on_retry=count_retry)
            row["status"] = "passed"
        # These tool reports establish surface frames only. In the handle
        # prompt, X runs along the loop; the target composer treats X as the
        # peg insertion axis. Passing depth/baseline checks cannot establish
        # the missing aperture origin and surface-to-mating transform.
        # Do not spend a simulator replay on this known semantic mismatch.
        raise ValueError("handle_surface_frame_is_not_a_reviewed_mating_frame")
    except Exception as exc:
        report.update(status="preflight_failed", failure=sanitized_failure(exc))
        if report["targets"] and row["status"] != "passed":
            row["status"] = "failed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
