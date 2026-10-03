"""Observed gripper transitions request assessment, never establish a grasp."""

from dataclasses import dataclass
from datetime import datetime

from episode_harness import RobotStateCapture, supervisor_observation


@dataclass(frozen=True)
class GripperSample:
    proposal_id: str
    observation_sequence: int
    captured_at: datetime
    captured_monotonic: float | None
    robot_state_capture: RobotStateCapture | None
    closed: bool | None


@dataclass(frozen=True)
class GripperEvidence:
    episode_id: str
    samples: tuple[GripperSample, ...]
    transition: str | None
    limitations: tuple[str, ...]


class GripperTransitionTrigger:
    """Inspect ``evidence`` after every call, including unavailable-state calls.

    Accept only adapter-mapped scalar closed booleans or numeric 0/1. Do not
    infer closure from native qpos, action commands or evaluator outcomes.
    Every contiguous observed change requests assessment, including normal
    grasps/releases: additional temporal/visual evidence must resolve suspicion.
    Missing state, sequence gaps and episode changes break the two-sample chain.
    The caller retains evidence in its sink; this object keeps at most two
    samples. Packet timestamps do not claim measured gripper sensor freshness.
    """

    def __init__(self):
        self.evidence = None
        self._previous = None
        self._episode = None

    def __call__(self, proposal):
        packet = supervisor_observation(proposal.observation)
        value = packet.observation.get('robot_state', {}).get('gripper', {}).get('closed')
        closed = (bool(value) if type(value) in (bool, int, float)
                  and value in (0, 1) else None)
        sample = GripperSample(proposal.proposal_id, packet.sequence,
            packet.captured_at, packet.captured_monotonic,
            packet.robot_state_capture, closed)
        previous = self._previous
        contiguous = (packet.episode_id == self._episode and previous is not None
                      and packet.sequence == previous.observation_sequence + 1)
        limitations = ['gripper_state_alone_does_not_establish_object_grasp_or_loss']
        if (packet.robot_state_capture is None or
                (contiguous and previous.robot_state_capture is None)):
            limitations.append('gripper_sensor_capture_time_unverified')
        if closed is None:
            limitations.append('gripper_closed_state_unavailable_or_invalid')
        if not contiguous:
            limitations.append('no_contiguous_previous_gripper_state')
        samples = (previous, sample) if contiguous else (sample,)
        transition = None
        if contiguous and closed is not None and closed != previous.closed:
            transition = 'open_to_closed' if closed else 'closed_to_open'
        self.evidence = GripperEvidence(packet.episode_id, samples, transition,
                                        tuple(limitations))
        self._previous = sample if closed is not None else None
        self._episode = packet.episode_id
        return transition is not None
