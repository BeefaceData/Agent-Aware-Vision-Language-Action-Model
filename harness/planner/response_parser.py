"""Six provider tools; reject malformed/multiple/truncated calls before actuation."""

from pydantic import ValidationError

from harness.skills.schemas import PARAMETERS, SkillCommand, SkillType
from harness.predicates.registry import PREDICATE_NAMES
from .diagnostics import validation_issues


class InvalidPlannerResponse(ValueError):
    # Only local diagnostic codes may reach logs, never provider response text.
    CODES = frozenset({
        "expected_completed_tool_call", "invalid_content", "expected_one_tool_call",
        "invalid_tool_arguments", "invalid_skill_command", "unsupported_anchor",
        "vla_attempt_budget_exhausted", "vla_call_outside_acquisition_limits",
    })

    def __init__(self, code, *, validation_error=None, prefix=()):
        self.code = code if isinstance(code, str) and code in self.CODES else "invalid_planner_response"
        self.issues = (validation_issues(validation_error, prefix=prefix)
                       if isinstance(validation_error, ValidationError) else [])
        super().__init__(self.code)


DESCRIPTIONS = {
    "move_to": "Stage or retreat one arm from measured state to a fresh symbolic target plus bounded offset. Returns achieved pose error; completion is not grasp success.",
    "rotate_wrist": "Rotate one wrist by bounded world-Z yaw while preserving EE position. Requires clearance for the swept hand and any held object.",
    "set_gripper": "Ramp one gripper to value 0 (closed) through 1 (open), holding measured arm positions. Opening requires runtime support authorization; closing is not grasp verification.",
    "release": "Guarded full opening of one gripper. Requires support or an empty hand; does not imply successful hanging.",
    "vla_act": "Invoke frozen Modal Neotix for bounded chunks controlling both arms. Stop on the named objective or budget and discard the unused chunk tail. No custom joint actions or language.",
    "abort": "End the episode without issuing robot motion. Use for uncertainty, unsupported operations, exhausted retries or unsafe next steps.",
}


def tool_schemas(vla_limits=None):
    tools = []
    for action, parameters in PARAMETERS.items():
        arms = (["auto", "both"] if action == SkillType.VLA_ACT else
                ["auto"] if action == SkillType.ABORT else ["left", "right"])
        properties = {"arm": {"type": "string", "enum": arms},
                      "parameters": parameters.model_json_schema(),
                      "reason": {"type": "string", "maxLength": 1000}}
        if action == SkillType.VLA_ACT:
            properties["stop"] = {"type": "string", "enum": sorted(PREDICATE_NAMES)}
            if vla_limits is not None:
                properties["stop"]["enum"] = ["mug_lifted"]
                properties["parameters"]["required"] = ["max_chunks", "chunk_size", "max_policy_actions"]
                for name in properties["parameters"]["required"]:
                    schema = properties["parameters"]["properties"][name]
                    schema["maximum"] = schema["default"] = getattr(vla_limits, name)
        required = ["arm", "parameters", "reason"]
        if action == SkillType.VLA_ACT:
            required.append("stop")
        tools.append({"name": action.value, "description": DESCRIPTIONS[action.value],
            "input_schema": {"type": "object", "properties": properties,
                             "required": required, "additionalProperties": False}})
    return tools


def parse_response(response):
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
            or not {"arm", "parameters", "reason"} <= arguments.keys()
            or not isinstance(arguments.get("parameters"), dict)
            or not isinstance(arguments.get("reason"), str)
            or "action" in arguments):
        raise InvalidPlannerResponse("invalid_tool_arguments")
    # Validate the selected parameter model separately so its field paths retain
    # the parameters prefix (the command's before-validator otherwise loses it).
    try:
        parameters = PARAMETERS[SkillType(call["name"])].model_validate(arguments["parameters"])
    except ValidationError as exc:
        raise InvalidPlannerResponse("invalid_skill_command", validation_error=exc,
                                     prefix=("parameters",)) from None
    try:
        return SkillCommand.model_validate({"action": call["name"], **arguments, "parameters": parameters})
    except ValidationError as exc:
        raise InvalidPlannerResponse("invalid_skill_command", validation_error=exc) from None
    except (ValueError, TypeError):
        raise InvalidPlannerResponse("invalid_skill_command") from None
