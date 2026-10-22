"""Load small public-contract replay cases used by CPU-only tests."""

import json
from pathlib import Path


_PATH = Path(__file__).parent / "fixtures" / "hanging_mug_scenarios.json"


def scenario(name):
    data = json.loads(_PATH.read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("scenarios"), dict):
        raise ValueError("unsupported_scenario_fixture")
    value = data["scenarios"].get(name)
    if not isinstance(value, dict):
        raise ValueError("unknown_scenario_fixture")
    return value
