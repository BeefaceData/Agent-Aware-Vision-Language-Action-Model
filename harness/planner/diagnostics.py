"""Bounded validation metadata; never retain provider values or error messages."""

FIELDS = frozenset({
    "action", "arm", "parameters", "stop", "reason", "max_physics_ticks",
    "target", "offset_xyz", "delta_yaw_deg", "value", "ramp_ticks",
    "max_chunks", "chunk_size", "max_policy_actions",
})
CODES = frozenset({
    "missing", "extra_forbidden", "literal_error", "enum", "int_type",
    "float_type", "string_type", "string_too_long", "string_too_short",
    "greater_than_equal", "less_than_equal", "finite_number", "tuple_type",
    "too_long", "too_short", "model_type", "value_error",
})


def validation_issues(error, *, prefix=()):
    issues = []
    for item in error.errors(include_input=False, include_context=False, include_url=False)[:8]:
        path = []
        for part in (*prefix, *item["loc"])[:6]:
            if isinstance(part, str) and part in FIELDS:
                path.append(part)
            elif type(part) is int and 0 <= part <= 2:
                path.append(part)
            else:
                path.append("unknown_field")
        code = item["type"] if item["type"] in CODES else "validation_error"
        issues.append({"path": path, "code": code})
    return issues
