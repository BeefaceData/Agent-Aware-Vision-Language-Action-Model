"""Compare fixed rack prompts on saved public RGB views using Modal SAM 3.

No simulator is started and SAM 3 is never loaded on the local GPU. Each
sequence/camera pair makes one request to the existing T4 worker.
"""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from harness.environment.layout import CAMERAS

PROMPTS = ("rack", "wire rack", "mug rack", "hanging rack")
MAX_FRAME_REQUESTS = 12


def validate_selection(sequences, cameras):
    if (not sequences or len(sequences) > MAX_FRAME_REQUESTS
            or any(type(seq) is not int or seq < 0 for seq in sequences)
            or len(set(sequences)) != len(sequences)):
        raise ValueError("invalid_saved_sequences")
    if (not cameras or any(camera not in CAMERAS for camera in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("invalid_camera_selection")
    if len(sequences) * len(cameras) > MAX_FRAME_REQUESTS:
        raise ValueError("sam3_prompt_replay_request_limit_exceeded")
    return tuple(sequences), tuple(cameras)


def replay(observations_dir, sequences, cameras, output, *, app_name="robotwin-perception-sam3",
           class_name="Sam3PerceptionWorker", worker=None):
    observations_dir, output = Path(observations_dir), Path(output)
    sequences, cameras = validate_selection(sequences, cameras)
    if not observations_dir.is_dir() or output.exists():
        raise ValueError("replay_input_missing_or_output_exists")
    if worker is None:
        import modal
        worker = modal.Cls.from_name(app_name, class_name)()
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for sequence in sequences:
        path = observations_dir / f"observation_{sequence:08d}.npz"
        if not path.is_file():
            raise ValueError("saved_observation_missing")
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["sequence"]) != sequence:
                raise ValueError("saved_observation_sequence_mismatch")
            for camera in cameras:
                image = np.asarray(archive[camera])
                if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3):
                    raise ValueError("saved_rgb_invalid")
                stream = BytesIO()
                Image.fromarray(image).save(stream, format="PNG", optimize=True)
                response = worker.detect_rack_prompt_variants.remote(
                    stream.getvalue(), observation_sequence=sequence, camera=camera)
                if (not isinstance(response, dict)
                        or response.get("observation_sequence") != sequence
                        or response.get("camera") != camera
                        or not isinstance(response.get("variants"), list)
                        or [item.get("prompt") for item in response["variants"]]
                            != list(PROMPTS)):
                    raise ValueError("invalid_sam3_prompt_response")
                variants = []
                for index, item in enumerate(response["variants"]):
                    mask_png = item.get("mask_png")
                    mask_path = None
                    if mask_png is not None:
                        if not isinstance(mask_png, bytes):
                            raise ValueError("invalid_sam3_prompt_mask")
                        mask_path = f"seq_{sequence:08d}_{camera}_prompt_{index}.png"
                        with Image.open(BytesIO(mask_png)) as mask_image:
                            mask_image.convert("L").save(output / mask_path)
                    variants.append({"prompt": item["prompt"],
                        "confidence": item.get("confidence"),
                        "mask_area_fraction": item.get("mask_area_fraction"),
                        "mask_path": mask_path})
                rows.append({"sequence": sequence, "camera": camera,
                    "source_sha256": sha256(path.read_bytes()).hexdigest(),
                    "variants": variants})
    report = {"schema_version": 1, "mode": "saved_frame_sam3_rack_prompt_replay",
        "robot_motion": False, "simulator_started": False,
        "local_sam3_loaded": False, "worker_gpu": "Modal T4",
        "prompt_variants": list(PROMPTS), "requests": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", required=True)
    parser.add_argument("--cameras", choices=CAMERAS, nargs="+", default=["head_camera"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = replay(args.observations, args.sequences, args.cameras, args.output)
    print(json.dumps({"requests": len(report["requests"]),
                      "output": str(args.output / "results.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
