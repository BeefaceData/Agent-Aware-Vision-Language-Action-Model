"""Build and validate compact action-prefix bundles for development replay."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from development.checkpoints.bank import CheckpointBank, load_prefix
from harness.environment.layout import CAMERAS, ACTION_DIM


SCHEMA_VERSION = 1
MAX_REPLAY_ACTIONS = 20_000


@dataclass(frozen=True)
class ReplayPrefix:
    """Previously confirmed actions and expected state/image at the boundary."""

    metadata: dict
    actions: np.ndarray
    states: np.ndarray
    final_images: dict[str, np.ndarray]


def write_bundle(path, *, seed, source_id, boundary_sequence, actions, states,
                 final_images, vla_chunks=0):
    actions = np.asarray(actions, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    if (actions.ndim != 2 or actions.shape[1] != ACTION_DIM
            or not 1 <= len(actions) <= MAX_REPLAY_ACTIONS
            or states.shape != (len(actions) + 1, ACTION_DIM)
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        raise ValueError("invalid_replay_prefix_vectors")
    if type(seed) is not int or seed < 0 or type(vla_chunks) is not int or vla_chunks < 0:
        raise ValueError("invalid_replay_prefix_metadata")
    if boundary_sequence != len(actions):
        raise ValueError("replay_prefix_boundary_mismatch")
    if set(final_images) != set(CAMERAS):
        raise ValueError("replay_prefix_requires_all_public_cameras")
    images = {}
    for camera in CAMERAS:
        image = np.asarray(final_images[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("invalid_replay_prefix_image")
        images[camera] = np.ascontiguousarray(image)
    metadata = {"schema_version": SCHEMA_VERSION, "seed": seed,
                "source_id": str(source_id), "boundary_sequence": boundary_sequence,
                "action_count": len(actions), "vla_chunks": vla_chunks}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, metadata_json=np.asarray(json.dumps(metadata)),
                            actions=actions, states=states, **images)
    return metadata


def from_checkpoint(bank_path, checkpoint_id, output):
    """Package one checkpoint after checksum and complete-prefix validation."""
    bank = CheckpointBank(bank_path)
    checkpoint = bank.get(checkpoint_id)
    bank.verify_artifacts(checkpoint)
    frames = load_prefix(bank, checkpoint)
    return write_bundle(output, seed=checkpoint.seed, source_id=checkpoint_id,
        boundary_sequence=checkpoint.step,
        actions=[frame.action for frame in frames[1:]],
        states=[frame.command_state for frame in frames],
        final_images=bank.load_images(checkpoint), vla_chunks=0)


def from_runtime_episode(episode_dir, boundary_sequence, output):
    """Package an all-VLA prefix from a recorded episode boundary.

    Historical simulator/oracle fields are deliberately ignored. Any analytic
    physics ticks before the selected boundary make the prefix ineligible.
    """
    episode = Path(episode_dir)
    boundary_sequence = int(boundary_sequence)
    if not 1 <= boundary_sequence <= MAX_REPLAY_ACTIONS:
        raise ValueError("invalid_replay_prefix_boundary")
    summary = json.loads((episode / "summary.json").read_text())
    events = [json.loads(line) for line in
              (episode / "skill_events.jsonl").read_text().splitlines()]
    if any(event.get("event_type") == "skill_requested"
           and type(event.get("observation")) is int
           and event["observation"] < boundary_sequence
           and event.get("command", {}).get("action") != "vla_act"
           for event in events):
        raise ValueError("replay_prefix_contains_non_vla_skill")
    confirmed, requests, reset_state = {}, set(), None
    last_call_id = max((event.get("call_id", -1) for event in events
        if event.get("event_type") == "skill_requested"
        and type(event.get("observation")) is int
        and event["observation"] < boundary_sequence
        and type(event.get("call_id")) is int), default=-1)
    for event in events:
        call_id = event.get("call_id")
        if event.get("event_type") == "skill_requested" and event.get("observation") == 0:
            state = event.get("command_state")
            if reset_state is None:
                reset_state = state
            elif state != reset_state:
                raise ValueError("replay_prefix_reset_state_ambiguous")
        if event.get("event_type") == "action_requested":
            sequence = event.get("observation")
            if type(sequence) is int and sequence < boundary_sequence:
                requests.add(sequence + 1)
        if event.get("event_type") == "action_confirmed":
            sequence = event.get("observation")
            action, state = event.get("action"), event.get("command_state")
            if type(sequence) is int and sequence <= boundary_sequence:
                if sequence in confirmed:
                    raise ValueError("replay_prefix_duplicate_confirmed_action")
                confirmed[sequence] = (action, state)
        if (event.get("event_type") == "measured_tick" and type(call_id) is int
                and call_id <= last_call_id):
            raise ValueError("replay_prefix_contains_analytic_physics_ticks")
    expected_sequences = set(range(1, boundary_sequence + 1))
    if reset_state is None or set(confirmed) != expected_sequences:
        raise ValueError("replay_prefix_action_trace_incomplete")
    if requests != expected_sequences:
        raise ValueError("replay_prefix_requested_action_trace_incomplete")
    final_path = episode / "observations" / f"observation_{boundary_sequence:08d}.npz"
    with np.load(final_path, allow_pickle=False) as archive:
        final_state = np.asarray(archive["command_state"], dtype=np.float64)
        final_images = {camera: np.asarray(archive[camera]) for camera in CAMERAS}
    states = [reset_state]
    actions = []
    for sequence in range(1, boundary_sequence + 1):
        action, state = confirmed[sequence]
        actions.append(action)
        states.append(state)
    if not np.array_equal(np.asarray(states[-1], dtype=np.float64), final_state):
        raise ValueError("replay_prefix_observation_state_mismatch")
    vla_chunks = sum(1 for event in events
        if (event.get("event_type") == "chunk_requested"
            and type(event.get("observation")) is int
            and event["observation"] < boundary_sequence))
    return write_bundle(output, seed=int(summary["seed"]), source_id=episode.name,
        boundary_sequence=boundary_sequence, actions=actions, states=states,
        final_images=final_images, vla_chunks=vla_chunks)


def load_bundle(path):
    """Load a bundle as untrusted input and validate every array before replay."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"metadata_json", "actions", "states", *CAMERAS}
        if set(archive.files) != required:
            raise ValueError("invalid_replay_prefix_bundle_fields")
        metadata = json.loads(str(archive["metadata_json"].item()))
        actions = np.asarray(archive["actions"], dtype=np.float64)
        states = np.asarray(archive["states"], dtype=np.float64)
        images = {camera: np.asarray(archive[camera]) for camera in CAMERAS}
    if (metadata.get("schema_version") != SCHEMA_VERSION
            or type(metadata.get("seed")) is not int or metadata["seed"] < 0
            or metadata.get("action_count") != len(actions)
            or metadata.get("boundary_sequence") != len(actions)
            or type(metadata.get("vla_chunks")) is not int or metadata["vla_chunks"] < 0
            or len(actions) > MAX_REPLAY_ACTIONS):
        raise ValueError("invalid_replay_prefix_bundle_metadata")
    if (actions.ndim != 2 or actions.shape[1] != ACTION_DIM
            or states.shape != (len(actions) + 1, ACTION_DIM)
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        raise ValueError("invalid_replay_prefix_bundle_vectors")
    for camera, image in images.items():
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("invalid_replay_prefix_bundle_image")
    return ReplayPrefix(metadata, actions, states, images)


def replay_prefix(environment, prefix, *, seed, state_atol=1e-5, image_mae_tolerance=5.0,
                  observation_callback=None, progress_callback=None):
    """Replay a verified prefix; optionally stream confirmed public observations.

    The callback can collect measured robot state such as TCP poses for a
    development retry. It never receives simulator object state or contacts.
    """
    if isinstance(prefix, (str, Path)):
        prefix = load_bundle(prefix)
    if prefix.metadata["seed"] != seed:
        raise ValueError("replay_prefix_seed_mismatch")
    if (not np.isfinite(state_atol) or state_atol < 0
            or not np.isfinite(image_mae_tolerance) or not 0 <= image_mae_tolerance <= 255):
        raise ValueError("invalid_replay_tolerance")

    observation = environment.observe()
    if observation_callback is not None:
        observation_callback(observation)
    max_state_error = float(np.max(np.abs(
        np.asarray(observation.command_state) - prefix.states[0])))
    if max_state_error > state_atol:
        return {"status": "drifted", "source_id": prefix.metadata["source_id"],
                "expected_steps": len(prefix.actions), "replayed_steps": 0,
                "max_state_error": max_state_error, "image_mae": {},
                "reason": "reset_command_state_mismatch"}

    replayed = 0
    started = time.perf_counter()
    fast_advance = getattr(environment, "advance_for_replay", None)
    for action, expected_state in zip(prefix.actions, prefix.states[1:]):
        next_step = replayed + 1
        action_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback({"event_type": "replay_action_requested", "step": next_step,
                "expected_steps": len(prefix.actions),
                "elapsed_seconds": time.perf_counter() - started,
                "render_skipped": callable(fast_advance)})
        observation = (fast_advance(action) if callable(fast_advance)
                       else environment.step(action))
        replayed += 1
        if observation_callback is not None:
            observation_callback(observation)
        error = float(np.max(np.abs(
            np.asarray(observation.command_state) - expected_state)))
        max_state_error = max(max_state_error, error)
        if error > state_atol:
            return {"status": "drifted", "source_id": prefix.metadata["source_id"],
                    "expected_steps": len(prefix.actions), "replayed_steps": replayed,
                    "max_state_error": max_state_error, "image_mae": {},
                    "reason": "action_prefix_command_state_mismatch"}
        if progress_callback is not None:
            progress_callback({"event_type": "replay_action_confirmed", "step": replayed,
                "expected_steps": len(prefix.actions),
                "step_elapsed_seconds": time.perf_counter() - action_started,
                "elapsed_seconds": time.perf_counter() - started,
                "max_state_error": max_state_error})

    if callable(fast_advance):
        # RGB is needed only at the checkpoint boundary. Refresh it once after
        # the state-checked replay instead of rendering all cameras per action.
        observation = environment.observe()

    image_mae = {}
    for camera in CAMERAS:
        actual = np.asarray(observation.images[camera])
        expected = prefix.final_images[camera]
        if actual.shape != expected.shape:
            return {"status": "drifted", "source_id": prefix.metadata["source_id"],
                    "expected_steps": len(prefix.actions), "replayed_steps": replayed,
                    "max_state_error": max_state_error, "image_mae": {},
                    "reason": f"camera_shape_mismatch:{camera}"}
        image_mae[camera] = float(np.mean(np.abs(actual.astype(np.float32)
                                                 - expected.astype(np.float32))))
    if any(error > image_mae_tolerance for error in image_mae.values()):
        return {"status": "drifted", "source_id": prefix.metadata["source_id"],
                "expected_steps": len(prefix.actions), "replayed_steps": replayed,
                "max_state_error": max_state_error, "image_mae": image_mae,
                "reason": "final_public_rgb_mismatch"}
    return {"status": "matched", "source_id": prefix.metadata["source_id"],
            "expected_steps": len(prefix.actions), "replayed_steps": replayed,
            "max_state_error": max_state_error, "image_mae": image_mae,
            "vla_chunks_in_prefix": prefix.metadata["vla_chunks"],
            "full_state_restored": False,
            "camera_render_skipped_during_actions": callable(fast_advance),
            "reason": "confirmed_action_states_and_final_public_rgb_match"}
