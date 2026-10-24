"""Clients for the deployed Neotix policy on Modal; weights and serving code are unchanged.

``ModalPolicyClient`` is the raw transport shared with the baseline runner.
``ModalVLAClient`` adapts it to the runtime's ``VLAClient`` protocol.
"""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING

import numpy as np

from harness.environment.layout import ACTION_DIM, CAMERAS

if TYPE_CHECKING:  # The baseline runner shares this module without pydantic.
    from harness.environment.base import Observation


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

    def __init__(self, app_name: str, class_name: str):
        import modal

        self.worker = modal.Cls.from_name(app_name, class_name)()

    def reset(self):
        started = time.perf_counter()
        info = self.worker.reset_episode.remote()
        return info, (time.perf_counter() - started) * 1000.0

    def act_chunk(self, images, state, chunk_size):
        packed, pack_ms, payload_kb = pack_observation(images, state)
        started = time.perf_counter()
        result = self.worker.act_chunk.remote(packed, requested_actions=chunk_size)
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

    def reset_history(self):
        info, _ = self.worker.reset()
        if not isinstance(info, dict) or info.get("ok") is not True:
            raise RuntimeError("incompatible Modal reset response")

    def infer(self, observation: Observation, requested_actions: int):
        observation = observation.copy()
        actions, _ = self.worker.act_chunk(observation.images, observation.command_state,
                                          requested_actions)
        return actions
