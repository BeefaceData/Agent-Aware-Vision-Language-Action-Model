"""Attach declared, deployable arm state to an observation packet.

Specifications are adapter owned. Values come only from named observation keys;
this module never derives a missing measurement from actions or simulator truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Mapping, Sequence

from episode_harness import StateField


@dataclass(frozen=True)
class StateSpec:
    arm: str
    name: str
    source_key: str
    frame: str
    units: str
    dimensions: int


# LeRobot 0.4.3 LiberoEnv formats robot0 measurements under robot_state:
# https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/envs/libero.py
# These are deployable observations, not evaluator object poses or success.
LIBERO_STATE_SPECS = (
    StateSpec('robot0', 'end_effector_position', 'robot_state/eef/pos',
              'world', 'm', 3),
    StateSpec('robot0', 'end_effector_orientation', 'robot_state/eef/quat',
              'world', 'unit_quaternion_xyzw', 4),
    StateSpec('robot0', 'gripper_joint_position', 'robot_state/gripper/qpos',
              'gripper_joint', 'm', 2),
)


def state_fields_for_observation(
    observation: Mapping[str, Any], specs: Sequence[StateSpec],
    captured_at: datetime, captured_monotonic: float,
) -> tuple[StateField, ...]:
    """Return declared fields in spec order, preserving missing measurements.

    Timestamp is the adapter's observation-return time. A present but malformed
    measurement raises ValueError; it must not be presented as an absent sensor.
    Vector environment's single batch axis is accepted without importing numpy.
    """
    if not isinstance(observation, Mapping):
        raise ValueError('state observation must be a mapping')
    if (not isinstance(captured_at, datetime) or captured_at.utcoffset() is None or
        isinstance(captured_monotonic, bool) or
        not isinstance(captured_monotonic, (int, float)) or
        not isfinite(captured_monotonic)):
        raise ValueError('state observation requires valid return timestamps')
    fields = []
    seen = set()
    for spec in specs:
        if (not isinstance(spec, StateSpec) or
            any(not isinstance(part, str) or not part for part in
                (spec.arm, spec.name, spec.source_key, spec.frame, spec.units)) or
            type(spec.dimensions) is not int or spec.dimensions < 1 or
            (spec.arm, spec.name) in seen):
            raise ValueError('invalid or duplicate state specification')
        seen.add((spec.arm, spec.name))
        raw = observation
        for key in spec.source_key.split('/'):
            raw = raw.get(key) if isinstance(raw, Mapping) else None
        if raw is None:
            fields.append(StateField(spec.arm, spec.name, spec.frame, spec.units,
                                     'missing', None, None, None, None))
            continue
        if hasattr(raw, 'tolist'):
            raw = raw.tolist()
        if (isinstance(raw, (list, tuple)) and len(raw) == 1 and
            isinstance(raw[0], (list, tuple))):
            raw = raw[0]
        if (not isinstance(raw, (list, tuple)) or len(raw) != spec.dimensions or
            any(isinstance(item, bool) or not isinstance(item, (int, float)) or
                not isfinite(item) for item in raw)):
            raise ValueError(f'invalid state measurement: {spec.arm}.{spec.name}')
        fields.append(StateField(spec.arm, spec.name, spec.frame, spec.units,
                                 'available', tuple(float(item) for item in raw),
                                 captured_at, float(captured_monotonic),
                                 'observation_return'))
    return tuple(fields)
