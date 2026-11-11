"""Clients for the deployed Neotix policy on Modal; weights and serving code are unchanged.

``ModalPolicyClient`` is the raw transport shared with the baseline runner.
``ModalVLAClient`` adapts it to the runtime's ``VLAClient`` protocol.
"""

from __future__ import annotations

import io
import math
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from harness.environment.layout import ACTION_DIM, CAMERAS

POLICY_TIMING_FIELDS = (
    "pack_ms", "modal_rpc_ms", "modal_compute_ms", "modal_overhead_ms",
    "cloud_preprocess_ms", "cloud_inference_ms", "cloud_postprocess_ms",
)

if TYPE_CHECKING:  # The baseline runner shares this module without pydantic.
    from harness.environment.base import Observation

# Modal's ``.remote()`` has no deadline: when the policy container vanished
# mid-request, a capture waited for hours on a closed connection. A cold start
# takes about 30-60 s and a warm chunk under a second, so this only fires on a
# lost call.
POLICY_CALL_TIMEOUT_SECONDS = 300.0


def call_with_timeout(function, timeout_seconds, *args, **kwargs):
    """Return ``function(*args, **kwargs)``, or raise ``TimeoutError`` after ``timeout_seconds``.

    The call runs on a daemon thread so a dead connection cannot keep the
    process alive. A timed-out call is abandoned, not cancelled: the remote side
    may still have run it, so the caller must not retry it blindly.
    """
    outcome = {}

    def run():
        try:
            outcome["value"] = function(*args, **kwargs)
        except BaseException as exc:  # Re-raised in the caller's thread below.
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"modal_policy_call_timed_out after {timeout_seconds:.0f} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def pack_observation(images, state):
    """Compress the three RGB views and 14-DoF state; return (bytes, pack_ms, size_kb)."""
    started = time.perf_counter()
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **{camera: np.asarray(images[camera], dtype=np.uint8) for camera in CAMERAS},
        state=np.asarray(state, dtype=np.float32),
    )
    payload = buffer.getvalue()
    return payload, (time.perf_counter() - started) * 1000.0, len(payload) / 1024.0


class ModalPolicyClient:
    """Remote handle for ``modal_apps/policy.py``; returns actions plus latency timings."""

    def __init__(self, app_name: str, class_name: str, *, timeout_seconds=POLICY_CALL_TIMEOUT_SECONDS):
        import modal

        self.worker = modal.Cls.from_name(app_name, class_name)()
        self.timeout_seconds = timeout_seconds

    def reset(self):
        started = time.perf_counter()
        info = call_with_timeout(self.worker.reset_episode.remote, self.timeout_seconds)
        return info, (time.perf_counter() - started) * 1000.0

    def act_chunk(self, images, state, chunk_size):
        packed, pack_ms, payload_kb = pack_observation(images, state)
        started = time.perf_counter()
        result = call_with_timeout(self.worker.act_chunk.remote, self.timeout_seconds,
                                   packed, requested_actions=chunk_size)
        rpc_ms = (time.perf_counter() - started) * 1000.0

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise RuntimeError(
                f"Modal returned invalid chunk shape {actions.shape}; expected [T, {ACTION_DIM}]")

        def remote_ms(name):
            return float(result.get(name, float("nan")))

        compute_ms = remote_ms("modal_compute_ms")
        timing = {
            "pack_ms": pack_ms,
            "payload_kb": payload_kb,
            "modal_rpc_ms": rpc_ms,
            "modal_compute_ms": compute_ms,
            "modal_overhead_ms": rpc_ms - compute_ms if np.isfinite(compute_ms) else float("nan"),
            "cloud_preprocess_ms": remote_ms("preprocess_ms"),
            "cloud_inference_ms": remote_ms("inference_ms"),
            "cloud_postprocess_ms": remote_ms("postprocess_ms"),
            "returned_actions": int(result["num_actions"]),
            "chunk_limit": int(result["chunk_limit"]),
        }
        return actions, timing


class ModalVLAClient:
    def __init__(self, worker=None, *, app_name="robotwin-policy", class_name="PolicyWorker"):
        if worker is None:
            worker = ModalPolicyClient(app_name, class_name)
        self.worker = worker
        self.last_timing = {}
        self.last_reset_ms = None

    def reset_history(self):
        self.last_reset_ms = None
        info, reset_ms = self.worker.reset()
        if not isinstance(info, dict) or info.get("ok") is not True:
            raise RuntimeError("incompatible Modal reset response")
        if type(reset_ms) in (int, float) and math.isfinite(reset_ms) and reset_ms >= 0:
            self.last_reset_ms = float(reset_ms)

    def infer(self, observation: Observation, requested_actions: int):
        self.last_timing = {}
        observation = observation.copy()
        actions, timing = self.worker.act_chunk(observation.images, observation.command_state,
                                               requested_actions)
        # Retain only numeric measurements, never provider payloads or error text.
        self.last_timing = {key: float(timing[key]) for key in POLICY_TIMING_FIELDS
                            if type(timing.get(key)) in (int, float)
                            and math.isfinite(timing[key]) and timing[key] >= 0}
        return actions
