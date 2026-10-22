"""Fresh, arm-specific symbolic anchors supplied by trusted perception code."""

from dataclasses import dataclass

from .geometry import pose7


@dataclass(frozen=True)
class Anchor:
    pose: tuple[float, ...]
    observation_sequence: int
    source: str = "perception"


class AnchorRegistry:
    def __init__(self):
        self._anchors = {}

    def bind(self, name, arm, anchor):
        if arm not in ("left", "right") or anchor.source not in (
                "perception", "runtime", "development_oracle"):
            raise ValueError("invalid_anchor")
        self._anchors[name, arm] = Anchor(tuple(pose7(anchor.pose)), anchor.observation_sequence, anchor.source)

    def resolve(self, name, arm, observation, *, allow_oracle=False):
        anchor = self._anchors.get((name, arm))
        if anchor is None or anchor.observation_sequence != observation.sequence:
            raise ValueError("anchor_missing_or_stale")
        if anchor.source == "development_oracle" and not allow_oracle:
            raise ValueError("oracle_anchor_not_allowed")
        return anchor
