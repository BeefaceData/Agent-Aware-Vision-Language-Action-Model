"""Two image-keypoint requests on saved entry 660; no simulator or motion."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from harness.perception.opus_frame import OpusObjectFrameGrounder, ground_public_object_frame_rgbd
from harness.perception.replay import load_public_observation


def check_grounding(archive, output, *, model, client=None):
    archive, output = Path(archive), Path(output)
    observation = load_public_observation(archive)
    camera = "perception_camera"
    if observation.sequence != 660 or camera not in observation.images:
        raise ValueError("entry660_perception_camera_required")
    if camera not in (observation.camera_frames or {}):
        raise ValueError("grounding_rgbd_camera_unavailable")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("explicit_model_required")
    if output.exists():
        raise FileExistsError(output)
    if client is None:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                           base_url="https://api.anthropic.com", max_retries=0, timeout=60.)
    grounders = {target: OpusObjectFrameGrounder(client, model, target=target, camera=camera)
                 for target in ("rack_peg", "mug_handle")}
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(observation.images[camera]).save(output / "source.png")
    report = dict(mode="saved_entry660_grounding_check", model=model, sequence=660,
        camera=camera, archive_sha256=sha256(archive.read_bytes()).hexdigest(),
        maximum_requests=2, targets={}, motion=False, support_verified=False,
        human_review_required=True)
    for target, grounder in grounders.items():
        row = dict(status="request_failed")
        try:
            proposal = grounder.locate(observation)
            row["proposal"] = proposal
            # Save proposals even when metric geometry is subsequently rejected.
            image = Image.fromarray(observation.images[camera])
            draw = ImageDraw.Draw(image)
            for index, (u, v) in enumerate(proposal.get("keypoints_px", ())):
                draw.ellipse((u-3, v-3, u+3, v+3), outline="red", width=1)
                draw.text((u+4, v), str(index+1), fill="red")
            image.save(output / f"{target}_proposal.png")
            row["status"] = "geometry_rejected"
            row["frame_pose_camera_derived"] = list(ground_public_object_frame_rgbd(observation, proposal))
            row["status"] = "geometry_passed_needs_review"
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            # API exception messages can include request details; retain only
            # local validation codes from our pure geometry layer.
            if isinstance(exc, ValueError) and "proposal" in row:
                row["validation_error"] = str(exc)
        report["targets"][target] = row
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    report = check_grounding(args.archive, args.output, model=args.model)
    print(json.dumps({target: row["status"] for target, row in report["targets"].items()}, indent=2))
    print(f"Saved review artifacts: {args.output}")


if __name__ == "__main__":
    main()
