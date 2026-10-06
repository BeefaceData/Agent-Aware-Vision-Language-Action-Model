"""Explicit synthetic sensor and adapter attestations for fallback tests."""

from dataclasses import replace

from baseline_fallback import BaselineFallback
from episode_harness import RobotStateCapture
from replay_adapters import ReplayEnvironment


def healthy_fallback():
    # These scripted adapters accept the fixture's native actions. No live
    # controller or real sensor validity is established by this attestation.
    return BaselineFallback({'robot_state': 10}, lambda: True, lambda p: True)


class MeasuredReplayEnvironment(ReplayEnvironment):
    """Synthetic state is generated at the scripted packet capture time."""

    @staticmethod
    def measured(packet):
        return replace(packet, robot_state_capture=RobotStateCapture(
            packet.sequence, packet.captured_at, packet.captured_monotonic))

    def reset(self, seed, episode_id):
        return self.measured(super().reset(seed, episode_id))

    def step(self, action):
        result = super().step(action)
        return replace(result, observation=self.measured(result.observation))
