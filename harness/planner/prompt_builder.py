"""Explicit public projection; no raw provider transcripts or oracle inputs."""

import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from harness.trace import CAMERAS
from harness.predicates.registry import PREDICATE_NAMES


def prompt_bundle(task_scope="acquisition"):
    directory = Path(__file__).resolve().parent.parent / "prompts"
    if task_scope not in ("acquisition", "full_task"):
        raise ValueError("unsupported_task_scope")
    scope_prompt = "acquisition.md" if task_scope == "acquisition" else "full_task.md"
    prompt = "\n\n".join((directory / name).read_text() for name in ("common.md", scope_prompt))
    return prompt, sha256(prompt.encode()).hexdigest()


def public_result(result):
    if result is None:
        return None
    # Whitelist only execution metadata/proprioception. In particular, even the
    # stop reason can encode a private predicate value and must be generalized.
    data = {name: getattr(result, name, None) for name in (
        "call_id", "completed", "executed_policy_actions", "vla_chunks_requested",
        "executed_physics_ticks", "discarded_actions", "execution_uncertain",
        "controlled_arms", "policy_history_reset", "arm_overlap_detected")}
    data["skill"] = result.skill.value
    data["motion_accounting"] = {
        "policy_actions_confirmed": result.executed_policy_actions,
        "analytic_physics_ticks": result.executed_physics_ticks,
        "physics_ticks_scope": "analytic_skills_only",
        "zero_physics_ticks_means_no_vla_motion": False,
    }
    private = result.information_source == "development_oracle"
    data["stop_reason"] = "private_gate_handback" if private else result.stop_reason
    data["objective_satisfied"] = None if private else result.objective_satisfied
    data["position_error_m"] = None if private else result.position_error_m
    data["orientation_error_rad"] = None if private else result.orientation_error_rad
    data["tracking_error"] = (None if result.tracking_error is None
                              else result.tracking_error.model_dump(mode="json"))
    predicate = result.predicate
    data["predicate"] = (predicate.model_dump(mode="json") if predicate is not None
                         and predicate.source in ("perception", "runtime") else None)
    return data


def user_content(context, *, limits, history, pixel_grounding=False):
    from PIL import Image
    observation = context.observation.copy()
    available_targets = ["current_ee", "mug_pregrasp"]
    cached_rack_target = False
    if limits.get("task_scope") == "full_task":
        available_targets = ["current_ee"]
        visual = context.visual_context or {}
        status = visual.get("status", {})
        rack_reference = visual.get("rack_static_reference", {})
        cached_rack_target = (limits.get("vla_phase") == "hang"
                              and rack_reference.get("world_anchor_available") is True)
        public = {item.name: item.value for item in context.predicates
                  if item.source in ("perception", "runtime")}
        if pixel_grounding or status.get("mug") == "grounded":
            available_targets.append("mug_pregrasp")
        if pixel_grounding or status.get("rack") == "grounded" or cached_rack_target:
            available_targets.append("rack_approach")
        if cached_rack_target:
            available_targets.append("rack_seat")
        if status.get("table") == "grounded":
            available_targets.append("table_approach")
    if limits.get("analytic_skills_available") is False:
        available_targets = []
    data = dict(turn=context.turn, observation_sequence=observation.sequence,
        command_state=observation.command_state, remaining_budget=context.remaining_budget,
        runtime_limits=limits, recent_commands=history, previous_result=public_result(context.previous),
        predicates=[p.model_dump(mode="json") for p in context.predicates
                    if p.source in ("perception", "runtime")
                    and p.name in PREDICATE_NAMES],
        visual_context=context.visual_context,
        available_targets=available_targets,
        pixel_grounding_required=bool(pixel_grounding),
        rack_pixel_grounding_exempt=bool(cached_rack_target),
        pixel_coordinate_convention=("u,v normalized to [0,1] over the full named source RGB view; "
                                      "for a mug choose its solid outer wall or rim, never its hollow opening; "
                                      "for a rack choose a visible solid hanger or support bar, not empty space")
            if pixel_grounding else None,
        image_sizes_px={name: list(observation.images[name].shape[:2]) for name in CAMERAS},
        memory_retrieved=context.memory.planner_payload() if context.memory else [],
        memory_snapshot_sha256=context.memory.snapshot_sha256 if context.memory else None,
        grounding_feedback=context.grounding_feedback,
        grounding_camera_to_avoid=context.grounding_camera_to_avoid)
    content = []
    for name in CAMERAS:
        with BytesIO() as stream:
            picture = Image.fromarray(observation.images[name])
            picture.thumbnail((640, 480))
            picture.save(stream, format="PNG")
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        content.extend([{"type": "text", "text": name},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}])
    content.append({"type": "text", "text": json.dumps(data, allow_nan=False)})
    return content
