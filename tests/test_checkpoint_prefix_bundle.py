"""Strict public-state validation for compact simulator replay prefixes."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from development.checkpoints.prefix_bundle import (
    from_runtime_episode, load_bundle, replay_prefix, write_bundle,
)
from harness.environment.base import Observation
from harness.environment.layout import CAMERAS
from tests.helpers import images


class PrefixEnvironment:
    def __init__(self, states, frames, *, drift_at=None):
        self.states, self.frames = states, frames
        self.drift_at = drift_at
        self.sequence = 0

    def observe(self):
        return Observation(self.sequence, images(self.frames[self.sequence]),
                           tuple(self.states[self.sequence]))

    def step(self, action):
        self.sequence += 1
        return self.observe()


class ReplayPrefixTests(unittest.TestCase):
    def make_bundle(self, path, *, final_value=2):
        states = np.zeros((3, 14), dtype=float)
        states[1, 0] = 1
        states[2, 0] = final_value
        frames = {camera: np.full((2, 3, 3), 7, dtype=np.uint8) for camera in CAMERAS}
        return write_bundle(path, seed=3, source_id="fixture", boundary_sequence=2,
            actions=states[1:], states=states, final_images=frames, vla_chunks=1), states, frames

    def test_replays_only_after_all_command_states_and_public_images_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefix.npz"
            metadata, states, frames = self.make_bundle(path)
            prefix = load_bundle(path)
            env = PrefixEnvironment(states, [0, 0, 7])

            report = replay_prefix(env, prefix, seed=metadata["seed"])

        self.assertEqual(report["status"], "matched")
        self.assertEqual(report["replayed_steps"], 2)
        self.assertEqual(report["vla_chunks_in_prefix"], 1)
        self.assertFalse(report["full_state_restored"])

    def test_replay_streams_measured_tcp_history_at_each_confirmed_boundary(self):
        class MeasuredPrefixEnvironment(PrefixEnvironment):
            def observe(self):
                observation = super().observe()
                pose = (float(self.sequence), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                return Observation(observation.sequence, observation.images,
                    observation.command_state,
                    measured_tcp_poses={"left": pose, "right": pose})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefix.npz"
            metadata, states, frames = self.make_bundle(path)
            env = MeasuredPrefixEnvironment(states, [0, 0, 7])
            captured = []

            report = replay_prefix(env, load_bundle(path), seed=metadata["seed"],
                observation_callback=captured.append)

        self.assertEqual(report["status"], "matched")
        self.assertEqual([item.sequence for item in captured], [0, 1, 2])
        self.assertEqual(captured[-1].measured_tcp_poses["left"][0], 2.0)
        self.assertEqual(captured[-1].measured_tcp_poses["right"][0], 2.0)

    def test_fast_replay_skips_intermediate_renders_and_reports_each_action(self):
        from types import SimpleNamespace

        class FastPrefixEnvironment(PrefixEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fast_steps = 0
                self.render_count = 0

            def observe(self):
                self.render_count += 1
                return super().observe()

            def step(self, action):
                raise AssertionError("fast replay must not render each action")

            def advance_for_replay(self, action):
                self.fast_steps += 1
                self.sequence += 1
                return SimpleNamespace(sequence=self.sequence,
                    command_state=tuple(self.states[self.sequence]),
                    measured_tcp_poses=None)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefix.npz"
            metadata, states, frames = self.make_bundle(path)
            env = FastPrefixEnvironment(states, [0, 0, 7])
            progress = []

            report = replay_prefix(env, load_bundle(path), seed=metadata["seed"],
                                   progress_callback=progress.append)

        self.assertEqual(report["status"], "matched")
        self.assertTrue(report["camera_render_skipped_during_actions"])
        self.assertEqual(env.fast_steps, 2)
        self.assertEqual(env.render_count, 2)  # Reset image and final RGB verification only.
        self.assertEqual([item["event_type"] for item in progress], [
            "replay_action_requested", "replay_action_confirmed",
            "replay_action_requested", "replay_action_confirmed"])
        self.assertEqual(progress[-1]["step"], 2)

    def test_state_drift_stops_before_planner_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefix.npz"
            _, states, _ = self.make_bundle(path)
            drifted = states.copy()
            drifted[1, 1] = 0.1
            env = PrefixEnvironment(drifted, [0, 0, 7])

            report = replay_prefix(env, load_bundle(path), seed=3)

        self.assertEqual(report["status"], "drifted")
        self.assertEqual(report["replayed_steps"], 1)
        self.assertEqual(report["reason"], "action_prefix_command_state_mismatch")

    def test_final_rgb_drift_rejects_the_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefix.npz"
            _, states, _ = self.make_bundle(path)
            env = PrefixEnvironment(states, [0, 0, 8])

            report = replay_prefix(env, load_bundle(path), seed=3, image_mae_tolerance=0)

        self.assertEqual(report["status"], "drifted")
        self.assertEqual(report["reason"], "final_public_rgb_mismatch")

    def test_runtime_episode_extraction_ignores_post_boundary_and_oracle_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "observations").mkdir()
            (root / "summary.json").write_text(json.dumps({"seed": 3}))
            state0 = [0.0] * 14
            state1, state2 = state0.copy(), state0.copy()
            state1[0], state2[0] = 1.0, 2.0
            events = [
                {"call_id": 1, "event_type": "skill_requested", "observation": 0,
                 "command_state": state0,
                 "command": {"action": "vla_act"}},
                {"call_id": 1, "event_type": "chunk_requested", "observation": 0},
                {"call_id": 1, "event_type": "action_requested", "observation": 0},
                {"call_id": 1, "event_type": "action_confirmed", "observation": 1,
                 "action": state1, "command_state": state1},
                {"call_id": 1, "event_type": "action_requested", "observation": 1},
                {"call_id": 1, "event_type": "action_confirmed", "observation": 2,
                 "action": state2, "command_state": state2},
                {"call_id": 3, "event_type": "skill_requested", "observation": 2,
                 "command_state": state2, "command": {"action": "abort"}},
                {"call_id": 3, "event_type": "measured_tick", "physics_tick": 1},
                {"call_id": 3, "event_type": "oracle", "hidden_pose": [999]},
            ]
            (root / "skill_events.jsonl").write_text("\n".join(map(json.dumps, events)))
            image = np.full((2, 3, 3), 11, dtype=np.uint8)
            with (root / "observations/observation_00000002.npz").open("wb") as stream:
                np.savez_compressed(stream, command_state=np.asarray(state2),
                    **{camera: image for camera in CAMERAS})
            output = root / "prefix.npz"

            metadata = from_runtime_episode(root, 2, output)
            prefix = load_bundle(output)

        self.assertEqual(metadata["action_count"], 2)
        self.assertEqual(metadata["vla_chunks"], 1)
        self.assertNotIn("oracle", json.dumps(prefix.metadata).lower())

    def test_runtime_episode_extraction_rejects_analytic_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "observations").mkdir()
            (root / "summary.json").write_text(json.dumps({"seed": 3}))
            events = [
                {"call_id": 1, "event_type": "skill_requested", "observation": 0,
                 "command_state": [0.0] * 14, "command": {"action": "move_to"}},
                {"call_id": 1, "event_type": "measured_tick"},
            ]
            (root / "skill_events.jsonl").write_text("\n".join(map(json.dumps, events)))

            with self.assertRaisesRegex(ValueError, "non_vla_skill"):
                from_runtime_episode(root, 1, root / "prefix.npz")


if __name__ == "__main__":
    unittest.main()
