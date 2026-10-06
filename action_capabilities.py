"""Host-declared native command semantics, independent of robot dimensionality."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ActionComponent:
    """One ordered native coordinate, with its arm and coordination group."""

    name: str
    arm: str
    group: str
    representation: str
    unit: str
    frame: str
    minimum: float
    maximum: float
    scale: float = 1.0

    def __post_init__(self):
        for value in (self.name, self.arm, self.group, self.representation,
                      self.unit, self.frame):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('action semantics require nonempty identifiers')
        for value in (self.minimum, self.maximum, self.scale):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError('action bounds and scale must be finite numbers')
        if self.minimum >= self.maximum or self.scale <= 0:
            raise ValueError('action bounds must increase and scale must be positive')


@dataclass(frozen=True)
class ActionCapabilities:
    """Ordered coordinates; scale maps native values into the declared unit.

    Operations describe native adapter support, not permission for corrections.
    Layout identifies transport (for example flat versus one-vector-batch).
    """

    components: tuple[ActionComponent, ...]
    control_frequency_hz: float
    operations: tuple[str, ...]
    layout: str = 'flat'

    def __post_init__(self):
        if (type(self.components) is not tuple or not self.components
                or any(type(c) is not ActionComponent for c in self.components)):
            raise ValueError('nonempty immutable action components required')
        names = [(c.arm, c.name) for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError('duplicate arm/component identity')
        hz = self.control_frequency_hz
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not isfinite(hz) or hz <= 0:
            raise ValueError('control frequency must be finite and positive')
        if (type(self.operations) is not tuple or not self.operations
                or any(not isinstance(op, str) or not op.strip() for op in self.operations)
                or len(set(self.operations)) != len(self.operations)):
            raise ValueError('unique nonempty supported operations required')
        if not isinstance(self.layout, str) or not self.layout.strip():
            raise ValueError('action transport layout required')


def validate_action_pair(policy, environment):
    """Reject mismatched declarations before reset, inference or dispatch.

    Two undeclared legacy adapters retain historical opaque replay behavior.
    Once either adapter declares native semantics, both must declare them.
    """
    required = getattr(policy, 'action_capabilities', None)
    supported = getattr(environment, 'action_capabilities', None)
    if required is None and supported is None:
        return
    if type(required) is not ActionCapabilities or type(supported) is not ActionCapabilities:
        raise ValueError('both adapters must declare ActionCapabilities')
    if (required.layout != supported.layout
            or required.control_frequency_hz != supported.control_frequency_hz
            or len(required.components) != len(supported.components)):
        raise ValueError('incompatible action layout, frequency or dimension')
    if not set(required.operations) <= set(supported.operations):
        raise ValueError('environment lacks required action operations')
    for wanted, actual in zip(required.components, supported.components):
        fields = ('name', 'arm', 'group', 'representation', 'unit', 'frame', 'scale')
        if any(getattr(wanted, key) != getattr(actual, key) for key in fields):
            raise ValueError('incompatible action component semantics')
        if wanted.minimum < actual.minimum or wanted.maximum > actual.maximum:
            raise ValueError('policy action range exceeds environment capability')


def libero_native_capabilities():
    """Pinned SmolVLA/relative OSC_POSE postprocessed native interface.

    This declaration describes baseline commands; it enables no recovery,
    numeric adjustment, arbitrary frame conversion or physical robot operation.
    """
    components = tuple(ActionComponent(
        name, 'panda', 'panda', representation, unit, frame, -1.0, 1.0, scale)
        for names, representation, unit, frame, scale in (
            (('x', 'y', 'z'), 'translation_delta', 'm', 'world', 0.05),
            (('rx', 'ry', 'rz'), 'axis_angle_delta', 'rad', 'world', 0.5),
            (('gripper',), 'gripper_command', 'normalized', 'actuator', 1.0))
        for name in names)
    return ActionCapabilities(components, 20.0, ('policy_action',), 'single-vector-batch')
