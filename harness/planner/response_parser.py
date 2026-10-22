"""Typed provider tools; reject malformed/multiple/truncated calls before actuation."""

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from harness.skills.schemas import PARAMETERS, SkillCommand, SkillType
from harness.predicates.registry import PREDICATE_NAMES
from .diagnostics import validation_issues


class InvalidPlannerResponse(ValueError):
    # Only local diagnostic codes may reach logs, never provider response text.
    CODES = frozenset({
        "expected_completed_tool_call", "invalid_content", "expected_one_tool_call",
        "invalid_tool_arguments", "invalid_skill_command", "unsupported_anchor",
        "missing_grounding_point",
        "vla_attempt_budget_exhausted", "vla_call_outside_acquisition_limits",
        "invalid_phase_assessment",
        "phase_outside_task_scope", "vla_call_outside_task_phase_limits",
        "inter_arm_recovery_requires_right_retreat", "inter_arm_retreat_outside_bounds",
        "release_not_publicly_authorized",
        "analytic_skill_not_recovery_eligible",
    })

    def __init__(self, code, *, validation_error=None, prefix=()):
        self.code = code if isinstance(code, str) and code in self.CODES else "invalid_planner_response"
        self.issues = (validation_issues(validation_error, prefix=prefix)
                       if isinstance(validation_error, ValidationError) else [])
        super().__init__(self.code)


DESCRIPTIONS = {
    "move_to": "Stage or retreat one arm from measured state to a fresh symbolic target plus bounded offset. When requested, provide one normalized pixel on the visible solid target surface; never provide world coordinates. Returns achieved pose error; completion is not grasp success.",
    "rotate_wrist": "Rotate one wrist by bounded world-Z yaw while preserving EE position. Requires clearance for the swept hand and any held object.",
    "set_gripper": "Ramp one gripper to value 0 (closed) through 1 (open), holding measured arm positions. Opening requires public release_ready=true; closing is not grasp verification.",
    "release": "Guarded full opening of one gripper. Use authorization=observed_support only with public release_ready=true. Use authorization=rack_seat_alignment only when the runtime reports rack_seat_aligned=true after a completed grounded rack-seat move. Alignment is not proof of contact or task success.",
    "vla_act": "Invoke frozen Modal Neotix for a bounded segment controlling both arms. The runtime returns after at most the configured feedback chunks with fresh images; inspect them and choose one bounded next skill. Stop on the named objective or budget and discard the unused chunk tail. No custom joint actions or language.",
    "abort": "End the episode without issuing robot motion. Use for uncertainty, unsupported operations, exhausted retries or unsafe next steps.",
}


@dataclass(frozen=True)
class ParsedPlannerResponse:
    command: SkillCommand
    task_phase: Literal["grasp", "handoff", "hang"]
    phase_evidence: str
    reason_truncated: bool = False


def tool_schemas(vla_limits=None, *, task_scope="acquisition", vla_stop_predicates=None,
                 release_authorized=True, rack_seat_authorized=False):
    if task_scope not in ("acquisition", "full_task"):
        raise ValueError("unsupported_task_scope")
    tools = []
    for action, parameters in PARAMETERS.items():
        if action == SkillType.RELEASE and not (release_authorized or rack_seat_authorized):
            continue
        parameter_schema = parameters.model_json_schema()
        if action == SkillType.SET_GRIPPER and not release_authorized:
            # Closing remains available for acquisition. Opening is reserved
            # for contexts with explicit public support/empty-hand evidence.
            parameter_schema["properties"]["value"]["maximum"] = 0.0
        if action == SkillType.VLA_ACT:
            # Runtime-internal fields must not be selectable by Opus. In
            # particular, it cannot request a simulator-only failure monitor
            # or alter per-arm execution masking.
            for name in ("lock_arm", "failure_stop", "failure_arm"):
                parameter_schema["properties"].pop(name, None)
        if action == SkillType.MOVE_TO:
            # Anthropic's tool schema needs a self-contained object shape here;
            # keep the optional nested Pydantic type inlined and omit the
            # nullable alternative because absent optional arguments are valid.
            pixel_schema = parameter_schema.pop("$defs")["GroundingPixel"]
            parameter_schema["properties"]["grounding"] = pixel_schema
        if action == SkillType.RELEASE:
            modes = []
            if release_authorized:
                modes.append("observed_support")
            if rack_seat_authorized:
                modes.append("rack_seat_alignment")
            parameter_schema["properties"]["authorization"]["enum"] = modes
        arms = (["auto", "both"] if action == SkillType.VLA_ACT else
                ["auto"] if action == SkillType.ABORT else ["left", "right"])
        properties = {"arm": {"type": "string", "enum": arms},
                      "parameters": parameter_schema,
                      "reason": {"type": "string", "maxLength": 1000},
                      "task_phase": {"type": "string", "enum":
                          ["grasp"] if task_scope == "acquisition" else ["grasp", "handoff", "hang"]},
                      "phase_evidence": {"type": "string", "minLength": 1, "maxLength": 300}}
        if action == SkillType.VLA_ACT:
            properties["stop"] = {"type": "string", "enum": sorted(PREDICATE_NAMES)}
            if vla_limits is not None:
                properties["stop"]["enum"] = list(vla_stop_predicates or (
                    ("mug_lifted",) if task_scope == "acquisition" else
                    ("mug_lifted", "chunk_budget_exhausted", "task_success")))
                properties["parameters"]["required"] = ["max_chunks", "chunk_size", "max_policy_actions"]
                for name in properties["parameters"]["required"]:
                    schema = properties["parameters"]["properties"][name]
                    schema["maximum"] = schema["default"] = getattr(vla_limits, name)
        required = ["arm", "parameters", "reason", "task_phase", "phase_evidence"]
        if action == SkillType.VLA_ACT:
            required.append("stop")
        tools.append({"name": action.value, "description": DESCRIPTIONS[action.value],
            "input_schema": {"type": "object", "properties": properties,
                             "required": required, "additionalProperties": False}})
    return tools


def parse_response(response):
    return parse_planner_response(response).command


def parse_planner_response(response):
    data = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(data, dict) or data.get("stop_reason") != "tool_use":
        raise InvalidPlannerResponse("expected_completed_tool_call")
    content = data.get("content")
    if not isinstance(content, list) or any(not isinstance(b, dict) for b in content):
        raise InvalidPlannerResponse("invalid_content")
    calls = [b for b in content if b.get("type") == "tool_use"]
    if len(calls) != 1 or any(b.get("type") not in ("text", "tool_use", "thinking", "redacted_thinking") for b in content):
        raise InvalidPlannerResponse("expected_one_tool_call")
    call = calls[0]
    arguments = call.get("input")
    if (not isinstance(call.get("name"), str) or call["name"] not in DESCRIPTIONS or not isinstance(arguments, dict)
            or not {"arm", "parameters", "reason", "task_phase", "phase_evidence"} <= arguments.keys()
            or not isinstance(arguments.get("parameters"), dict)
            or not isinstance(arguments.get("reason"), str)
            or arguments.get("task_phase") not in ("grasp", "handoff", "hang")
            or not isinstance(arguments.get("phase_evidence"), str)
            or not 1 <= len(arguments["phase_evidence"]) <= 300
            or "action" in arguments):
        raise InvalidPlannerResponse("invalid_phase_assessment" if isinstance(arguments, dict)
                                     and ("task_phase" in arguments or "phase_evidence" in arguments)
                                     else "invalid_tool_arguments")
    # The rationale is descriptive only; an overlong explanation must not
    # discard an otherwise valid, bounded skill request. Clip a local copy so
    # provider response objects remain untouched and logs stay schema-bounded.
    reason_truncated = len(arguments["reason"]) > 1000
    arguments = dict(arguments)
    if reason_truncated:
        arguments["reason"] = arguments["reason"][:1000]
    # Validate the selected parameter model separately so its field paths retain
    # the parameters prefix (the command's before-validator otherwise loses it).
    try:
        parameters = PARAMETERS[SkillType(call["name"])].model_validate(arguments["parameters"])
    except ValidationError as exc:
        raise InvalidPlannerResponse("invalid_skill_command", validation_error=exc,
                                     prefix=("parameters",)) from None
    try:
        command = SkillCommand.model_validate({"action": call["name"],
            **{key: value for key, value in arguments.items()
               if key not in ("task_phase", "phase_evidence")}, "parameters": parameters})
        return ParsedPlannerResponse(command=command, task_phase=arguments["task_phase"],
                                     phase_evidence=arguments["phase_evidence"],
                                     reason_truncated=reason_truncated)
    except ValidationError as exc:
        raise InvalidPlannerResponse("invalid_skill_command", validation_error=exc) from None
    except (ValueError, TypeError):
        raise InvalidPlannerResponse("invalid_skill_command") from None
