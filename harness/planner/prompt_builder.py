"""Explicit public projection; no raw provider transcripts or oracle inputs."""

import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from harness.trace import CAMERAS


def prompt_bundle():
    directory = Path(__file__).resolve().parent.parent / "prompts"
    prompt = "\n\n".join((directory / name).read_text() for name in ("common.md", "acquisition.md"))
    return prompt, sha256(prompt.encode()).hexdigest()


def public_result(result):
    if result is None:
        return None
    # Whitelist only execution metadata/proprioception. In particular, even the
    # stop reason can encode a private predicate value and must be generalized.
    data = {name: getattr(result, name) for name in (
        "call_id", "completed", "executed_policy_actions", "vla_chunks_requested",
        "executed_physics_ticks", "discarded_actions", "execution_uncertain",
        "controlled_arms", "policy_history_reset")}
    data["skill"] = result.skill.value
    private = result.information_source == "development_oracle"
    data["stop_reason"] = "private_gate_handback" if private else result.stop_reason
    data["objective_satisfied"] = None if private else result.objective_satisfied
    predicate = result.predicate
    data["predicate"] = (predicate.model_dump(mode="json") if predicate is not None
                         and predicate.source in ("perception", "runtime") else None)
    return data


def user_content(context, *, limits, history):
    from PIL import Image
    observation = context.observation.copy()
    data = dict(turn=context.turn, observation_sequence=observation.sequence,
        command_state=observation.command_state, remaining_budget=context.remaining_budget,
        runtime_limits=limits, recent_commands=history, previous_result=public_result(context.previous),
        predicates=[p.model_dump(mode="json") for p in context.predicates
                    if p.source in ("perception", "runtime")],
        available_targets=["current_ee", "mug_pregrasp"],
        memory_retrieved=context.memory.planner_payload() if context.memory else [],
        memory_snapshot_sha256=context.memory.snapshot_sha256 if context.memory else None)
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
