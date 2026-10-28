"""Modal-hosted SAM 3 RGB segmentation service, isolated from the Neotix worker.

Deploy after configuring the Modal ``huggingface`` secret with ``HF_TOKEN`` and
receiving gated model access:
    modal deploy modal_apps/sam3.py
"""

from io import BytesIO
import math
import os
from pathlib import Path

import modal


APP_NAME = "robotwin-perception-sam3"
MODEL_REPO = "facebook/sam3"
MOUNT = "/perception-data"
WEIGHTS = f"{MOUNT}/sam3"
HF_HOME = f"{MOUNT}/hf-cache"
MAX_IMAGE_BYTES = 4_000_000
MAX_IMAGE_SIDE = 1280
MAX_DETECTIONS_PER_LABEL = 3
MUG_SCORE_THRESHOLD = 0.40
PROMPTS = {"mug": "mug", "rack": "rack", "table": "table surface"}
RACK_PROMPT_VARIANTS = ("rack", "wire rack", "mug rack", "hanging rack")
RACK_PROFILES = {"default": ("rack", 0.60),
                 "explore_mug_rack": ("mug rack", 0.18)}
# The camera name only labels the request; the policy cameras plus
# harness.environment.layout.PERCEPTION_CAMERAS (this image does not ship the harness).
CAMERAS = frozenset({"head_camera", "left_camera", "right_camera", "perception_camera"})
REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements"


def confidence_meets_threshold(score, threshold):
    """Apply the same label/profile cutoff used by SAM post-processing."""
    try:
        score, threshold = float(score), float(threshold)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(score) and math.isfinite(threshold) and score >= threshold

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("robotwin-sam3-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "pillow")
    .pip_install_from_requirements(str(REQUIREMENTS / "perception.txt"))
    .env({"HF_HOME": HF_HOME, "HF_HUB_CACHE": f"{HF_HOME}/hub"})
)


@app.cls(
    image=image,
    gpu="T4",
    timeout=10 * 60,
    scaledown_window=2 * 60,
    max_containers=1,
    volumes={MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
class Sam3PerceptionWorker:
    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Sam3Model, Sam3Processor

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("modal_secret_missing_HF_TOKEN")
        os.makedirs(WEIGHTS, exist_ok=True)
        has_weights = (Path(WEIGHTS, "config.json").is_file()
                       and any(Path(WEIGHTS).glob("*.safetensors")))
        if not has_weights:
            snapshot_download(repo_id=MODEL_REPO, local_dir=WEIGHTS, token=token)
            volume.commit()

        self.processor = Sam3Processor.from_pretrained(WEIGHTS)
        self.model = Sam3Model.from_pretrained(WEIGHTS).to("cuda").eval()
        self.torch = torch
        print(f"SAM 3 perception ready on {torch.cuda.get_device_name(0)}", flush=True)

    @modal.method()
    def detect(self, rgb_png: bytes, *, observation_sequence: int, camera: str,
               rack_prompt_profile: str = "default", include_rack: bool = True):
        import numpy as np
        from PIL import Image

        if (type(observation_sequence) is not int or observation_sequence < 0
                or camera not in CAMERAS or rack_prompt_profile not in RACK_PROFILES
                or type(include_rack) is not bool
                or not isinstance(rgb_png, bytes)
                or not rgb_png or len(rgb_png) > MAX_IMAGE_BYTES):
            raise ValueError("invalid_sam3_request")
        with Image.open(BytesIO(rgb_png)) as opened:
            image = np.asarray(opened.convert("RGB"))
        if (image.ndim != 3 or image.shape[2] != 3
                or max(image.shape[:2]) > MAX_IMAGE_SIDE):
            raise ValueError("sam3_image_size_out_of_bounds")

        detections = []
        for label, default_prompt in PROMPTS.items():
            if label == "rack" and not include_rack:
                continue
            if label == "rack":
                prompt, threshold = RACK_PROFILES[rack_prompt_profile]
            elif label == "mug":
                prompt, threshold = default_prompt, MUG_SCORE_THRESHOLD
            else:
                prompt, threshold = default_prompt, 0.60
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = self._to_device(inputs, "cuda")
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            sizes = inputs.get("original_sizes")
            if sizes is None:
                raise RuntimeError("sam3_original_image_size_missing")
            if hasattr(sizes, "tolist"):
                sizes = sizes.tolist()
            result = self.processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=0.5,
                target_sizes=sizes)[0]
            masks, scores = result.get("masks"), result.get("scores")
            if masks is None or scores is None:
                continue
            masks = self._numpy(masks)
            scores = self._numpy(scores).reshape(-1)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim != 3 or len(masks) != len(scores):
                raise RuntimeError("sam3_output_shape_invalid")
            ranked = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
            for index in ranked[:MAX_DETECTIONS_PER_LABEL]:
                score = float(scores[index])
                if not confidence_meets_threshold(score, threshold):
                    continue
                binary = np.asarray(masks[index], dtype=np.uint8) * 255
                if binary.shape != image.shape[:2] or not binary.any():
                    continue
                stream = BytesIO()
                Image.fromarray(binary).save(stream, format="PNG", optimize=True)
                detections.append({"label": label, "confidence": score,
                                   "mask_png": stream.getvalue()})

        return {"observation_sequence": observation_sequence, "camera": camera,
                "detections": detections}

    @modal.method()
    def detect_rack_prompt_variants(self, rgb_png: bytes, *, observation_sequence: int,
                                    camera: str):
        """Development-only prompt comparison on one saved RGB view.

        This endpoint is not called by runtime perception. It keeps the prompt
        set fixed and returns only each prompt's highest-scoring mask so saved
        frames can be compared on the existing Modal GPU without loading SAM 3
        beside the local simulator.
        """
        import numpy as np
        from PIL import Image

        if (type(observation_sequence) is not int or observation_sequence < 0
                or camera not in CAMERAS or not isinstance(rgb_png, bytes)
                or not rgb_png or len(rgb_png) > MAX_IMAGE_BYTES):
            raise ValueError("invalid_sam3_request")
        with Image.open(BytesIO(rgb_png)) as opened:
            image = np.asarray(opened.convert("RGB"))
        if (image.ndim != 3 or image.shape[2] != 3
                or max(image.shape[:2]) > MAX_IMAGE_SIDE):
            raise ValueError("sam3_image_size_out_of_bounds")

        variants = []
        for prompt in RACK_PROMPT_VARIANTS:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = self._to_device(inputs, "cuda")
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            sizes = inputs.get("original_sizes")
            if sizes is None:
                raise RuntimeError("sam3_original_image_size_missing")
            if hasattr(sizes, "tolist"):
                sizes = sizes.tolist()
            result = self.processor.post_process_instance_segmentation(
                outputs, threshold=0.01, mask_threshold=0.5, target_sizes=sizes)[0]
            masks, scores = result.get("masks"), result.get("scores")
            best = {"prompt": prompt, "confidence": None, "mask_png": None,
                    "mask_area_fraction": 0.0}
            if masks is not None and scores is not None:
                masks = self._numpy(masks)
                scores = self._numpy(scores).reshape(-1)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                if masks.ndim != 3 or len(masks) != len(scores):
                    raise RuntimeError("sam3_output_shape_invalid")
                if len(scores):
                    index = max(range(len(scores)), key=lambda i: float(scores[i]))
                    score = float(scores[index])
                    mask = np.asarray(masks[index], dtype=np.uint8)
                    if (np.isfinite(score) and mask.shape == image.shape[:2]
                            and mask.any()):
                        binary = mask * 255
                        stream = BytesIO()
                        Image.fromarray(binary).save(stream, format="PNG", optimize=True)
                        best.update(confidence=score, mask_png=stream.getvalue(),
                                    mask_area_fraction=float(mask.mean()))
            variants.append(best)
        return {"observation_sequence": observation_sequence, "camera": camera,
                "variants": variants}

    @staticmethod
    def _numpy(value):
        import numpy as np
        if callable(getattr(value, "detach", None)):
            value = value.detach()
        if callable(getattr(value, "cpu", None)):
            value = value.cpu()
        if callable(getattr(value, "numpy", None)):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _to_device(inputs, device):
        if callable(getattr(inputs, "to", None)):
            return inputs.to(device)
        return {key: value.to(device) if callable(getattr(value, "to", None)) else value
                for key, value in inputs.items()}


@app.local_entrypoint()
def smoke_test():
    import numpy as np
    from PIL import Image
    worker = Sam3PerceptionWorker()
    stream = BytesIO()
    Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8)).save(stream, format="PNG")
    result = worker.detect.remote(stream.getvalue(), observation_sequence=0,
                                  camera="head_camera")
    if result.get("observation_sequence") != 0 or result.get("camera") != "head_camera":
        raise RuntimeError("sam3_modal_smoke_response_invalid")
    print("SAM 3 Modal worker ready; mask_count=", len(result.get("detections", [])))
