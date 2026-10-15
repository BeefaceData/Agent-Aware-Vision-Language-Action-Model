from pathlib import Path

import numpy as np

from harness.config import HarnessConfig
from harness.schemas import AssessmentRecord, SupervisorDecision
from harness.trace import CAMERAS


def images(value=0):
    return {name: np.full((2, 3, 3), value, dtype=np.uint8) for name in CAMERAS}


def decision(**updates):
    return SupervisorDecision.model_validate({
        "decision": "pass", "phase": "unknown", "failure_type": "none",
        "confidence": 1.0, "reason": "PASS-only mode", **updates,
    })


def assessment(**updates):
    return AssessmentRecord.model_validate({
        "run_id": "run", "episode_id": "episode", "event_id": "event",
        "event_type": "decision", "step": 0, "chunk_id": 0, "chunk_pos": 0,
        "timestamp": 1.0, "phase": "unknown", "failure_type": "none",
        "supervisor_confidence": 1.0, "policy_action": [0.0] * 14,
        "decision": "pass", "pre_state_summary": {"command_state": [0.0] * 14},
        **updates,
    })


def config(directory, enabled=True):
    root = Path(directory)
    return HarnessConfig.model_validate({
        "harness": {"enabled": enabled},
        "trace": {"state_history": 3, "action_history": 2,
                  "keyframes": 2, "keyframe_interval": 2},
        "logging": {"trace_dir": root / "traces", "assessment_dir": root / "assessments",
                    "evaluation_dir": root / "evaluation"},
    })
