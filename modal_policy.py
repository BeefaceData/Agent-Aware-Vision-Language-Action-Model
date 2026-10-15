"""Modal-hosted Neotix RoboTwin policy.

Deploy once:
    modal deploy modal_policy.py
"""

import io
import os
import modal

APP_NAME = "robotwin-policy"
MODEL_REPO = "Neotix-Robotics/robotwin-hanging-mug-dit"
MOUNT = "/model-data"
WEIGHTS = f"{MOUNT}/weights"
HF_HOME = f"{MOUNT}/hf-cache"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("robotwin-model-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements("requirements-policy.txt")
    .pip_install("huggingface_hub")
    .env({"HF_HOME": HF_HOME, "HF_HUB_CACHE": f"{HF_HOME}/hub"})
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
        from lerobot.policies import get_policy_class, make_pre_post_processors

        os.makedirs(WEIGHTS, exist_ok=True)
        if not os.path.exists(os.path.join(WEIGHTS, "config.json")):
            print(f"Downloading {MODEL_REPO} ...", flush=True)
            snapshot_download(repo_id=MODEL_REPO, local_dir=WEIGHTS)
            volume.commit()

        config = PreTrainedConfig.from_pretrained(WEIGHTS)
        config.pretrained_path = WEIGHTS
        config.device = "cuda"

        self.policy = get_policy_class(config.type).from_pretrained(WEIGHTS, config=config)
        self.policy.to("cuda")
        self.policy.eval()

        override = {"device": "cuda"}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=WEIGHTS,
            preprocessor_overrides={"device_processor": override},
            postprocessor_overrides={"device_processor": override},
        )
        self.policy.reset()
        print(f"Policy ready on {torch.cuda.get_device_name(0)}", flush=True)

    @modal.method()
    def reset_episode(self):
        self.policy.reset()
        return True

    @modal.method()
    def act(self, packed_observation: bytes):
        import numpy as np
        import torch

        with np.load(io.BytesIO(packed_observation)) as data:
            batch = {
                "observation.images.cam_high": data["head_camera"],
                "observation.images.cam_left_wrist": data["left_camera"],
                "observation.images.cam_right_wrist": data["right_camera"],
                "observation.state": data["state"],
            }

        batch = {k: torch.as_tensor(v).to("cuda").unsqueeze(0) for k, v in batch.items()}
        for key in list(batch):
            if "images" in key:
                batch[key] = batch[key].permute(0, 3, 1, 2).float() / 255.0
        batch["task"] = ["hang the mug on the rack"]

        with torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(self.preprocessor(batch)))

        return action.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()


@app.local_entrypoint()
def smoke_test():
    worker = PolicyWorker()
    worker.reset_episode.remote()
    print("Modal policy worker loaded successfully")
