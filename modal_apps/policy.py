"""Modal-hosted RoboTwin MultiTaskDiT policy worker with robust action chunking.

Deploy:
    modal deploy modal_apps/policy.py

Key idea:
- Call `policy.select_action(...)` ONCE. For MultiTaskDiT, that call performs
  the expensive model inference when the action queue is empty and caches the
  rest of the predicted action chunk internally.
- Return the first action plus additional actions already cached in the policy's
  ACTION queue.
- Clear unused cached actions before the next cloud request so each request uses
  the newest RoboTwin observation.

This avoids calling `predict_action_chunk()` directly and therefore preserves
the exact queue/history initialization logic implemented by LeRobot.
"""

import io
import os
from pathlib import Path
import time

import modal

APP_NAME = "robotwin-policy"
MODEL_REPO = "Neotix-Robotics/robotwin-hanging-mug-dit"

MOUNT = "/model-data"
WEIGHTS = f"{MOUNT}/weights"
HF_HOME = f"{MOUNT}/hf-cache"
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements"

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(
    "robotwin-model-data",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(REQUIREMENTS / "policy.txt"))
    .pip_install("huggingface_hub")
    .env(
        {
            "HF_HOME": HF_HOME,
            "HF_HUB_CACHE": f"{HF_HOME}/hub",
        }
    )
)


@app.cls(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
    volumes={MOUNT: volume},
)
class PolicyWorker:
    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import (
            get_policy_class,
            make_pre_post_processors,
        )

        os.makedirs(WEIGHTS, exist_ok=True)

        if not os.path.exists(os.path.join(WEIGHTS, "config.json")):
            print(f"Downloading {MODEL_REPO} ...", flush=True)
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=WEIGHTS,
            )
            volume.commit()

        config = PreTrainedConfig.from_pretrained(WEIGHTS)
        config.pretrained_path = WEIGHTS
        config.device = "cuda"

        self.config = config

        self.policy = (
            get_policy_class(config.type)
            .from_pretrained(
                WEIGHTS,
                config=config,
            )
        )
        self.policy.to("cuda")
        self.policy.eval()

        override = {"device": "cuda"}

        self.preprocessor, self.postprocessor = (
            make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=WEIGHTS,
                preprocessor_overrides={
                    "device_processor": override
                },
                postprocessor_overrides={
                    "device_processor": override
                },
            )
        )

        self.policy.reset()

        self.model_chunk_limit = int(
            getattr(
                config,
                "n_action_steps",
                getattr(config, "chunk_size", 1),
            )
        )

        print(
            f"Policy ready on {torch.cuda.get_device_name(0)} | "
            f"type={config.type} | "
            f"n_obs_steps={getattr(config, 'n_obs_steps', 'unknown')} | "
            f"chunk_limit={self.model_chunk_limit}",
            flush=True,
        )

    @modal.method()
    def reset_episode(self):
        self.policy.reset()

        return {
            "ok": True,
            "chunk_limit": self.model_chunk_limit,
        }

    def _prepare_input_batch(self, packed_observation: bytes):
        import numpy as np
        import torch

        with np.load(io.BytesIO(packed_observation)) as data:
            batch = {
                "observation.images.cam_high": data["head_camera"],
                "observation.images.cam_left_wrist": data["left_camera"],
                "observation.images.cam_right_wrist": data["right_camera"],
                "observation.state": data["state"],
            }

        batch = {
            key: torch.as_tensor(value)
            .to("cuda")
            .unsqueeze(0)
            for key, value in batch.items()
        }

        for key in list(batch):
            if "images" in key:
                batch[key] = (
                    batch[key]
                    .permute(0, 3, 1, 2)
                    .float()
                    / 255.0
                )

        batch["task"] = ["hang the mug on the rack"]

        return self.preprocessor(batch)

    @modal.method()
    def act_chunk(
        self,
        packed_observation: bytes,
        requested_actions: int = 8,
    ):
        """Return multiple actions from one expensive policy inference.

        We intentionally use `select_action()` once rather than calling
        `predict_action_chunk()` directly. MultiTaskDiT's `select_action()`
        correctly:
          1. prepares the observation,
          2. updates observation-history queues,
          3. runs one chunk prediction if the action queue is empty,
          4. caches the remaining predicted actions.

        We then read the already-cached actions from the ACTION queue without
        running the model again.
        """
        import numpy as np
        import torch
        from lerobot.utils.constants import ACTION

        total_start = time.perf_counter()

        requested_actions = max(
            1,
            min(
                int(requested_actions),
                self.model_chunk_limit,
            ),
        )

        # Ensure a fresh prediction from the CURRENT observation.
        # Preserve observation-history queues, but discard any stale actions
        # left from a previous shorter execution chunk.
        if (
            hasattr(self.policy, "_queues")
            and ACTION in self.policy._queues
        ):
            self.policy._queues[ACTION].clear()

        # --------------------------------------------------------------
        # 1. Decode + normal LeRobot preprocessing
        # --------------------------------------------------------------
        preprocess_start = time.perf_counter()

        processed_batch = self._prepare_input_batch(
            packed_observation
        )

        preprocess_ms = (
            time.perf_counter() - preprocess_start
        ) * 1000.0

        # --------------------------------------------------------------
        # 2. ONE select_action call -> ONE expensive chunk inference
        # --------------------------------------------------------------
        torch.cuda.synchronize()
        inference_start = time.perf_counter()

        with torch.inference_mode():
            first_action = self.policy.select_action(
                processed_batch
            )

        torch.cuda.synchronize()

        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000.0

        # --------------------------------------------------------------
        # 3. Pull additional actions from the chunk already cached by
        #    MultiTaskDiT. This does NOT invoke the neural network again.
        # --------------------------------------------------------------
        normalized_actions = [first_action]

        action_queue = self.policy._queues.get(ACTION)

        while (
            action_queue is not None
            and len(normalized_actions) < requested_actions
            and len(action_queue) > 0
        ):
            normalized_actions.append(
                action_queue.popleft()
            )

        # Discard actions from this old prediction that we chose not to execute.
        # Observation queues remain intact so n_obs_steps history still works.
        if action_queue is not None:
            action_queue.clear()

        actions = torch.stack(
            normalized_actions,
            dim=1,
        )

        # Expected here: [B, T, A].
        if actions.ndim != 3:
            raise RuntimeError(
                "Unexpected stacked action shape "
                f"{tuple(actions.shape)}; expected [B, T, A]"
            )

        # --------------------------------------------------------------
        # 4. Apply the checkpoint's action postprocessor to the chunk.
        # --------------------------------------------------------------
        postprocess_start = time.perf_counter()

        actions = self.postprocessor(actions)

        postprocess_ms = (
            time.perf_counter() - postprocess_start
        ) * 1000.0

        actions = (
            actions.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        return {
            "actions": actions.tolist(),
            "num_actions": int(actions.shape[0]),
            "chunk_limit": self.model_chunk_limit,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "modal_compute_ms": total_ms,
        }


@app.local_entrypoint()
def smoke_test():
    worker = PolicyWorker()
    info = worker.reset_episode.remote()
    print(f"Modal policy worker ready: {info}")
