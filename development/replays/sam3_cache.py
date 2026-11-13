"""Cache Modal SAM 3 answers for captured frames, then serve them back offline.

Each frame's camera images (three, or four with a perception camera) are sent to the deployed SAM 3 worker once;
the raw responses are stored beside the frames. ``CachedSam3Worker`` replays
them through the real ``ModalSam3Detector`` client, so offline evaluation uses
the runtime's own head-first/wrist-fallback rules and thresholds without
repeating Modal calls:

    python -m development.replays.sam3_cache --frames data/perception/frames/seed_3

The worker drops detections below its own score floors (mug 0.40, table 0.60),
so the cache cannot study lower thresholds.
"""

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from harness.environment.layout import CAMERAS, PERCEPTION_CAMERAS
from harness.perception.replay import load_public_observation

CACHE_FILE = "sam3_cache.json"
PARTIAL_CACHE_FILE = "sam3_cache.partial.json"
# The full-task configs use the "mug rack" prompt (configs/*rgbd*.yaml).
DEFAULT_RACK_PROMPT_PROFILE = "explore_mug_rack"


def encode_png(image):
    """PNG-encode RGB exactly as ModalSam3Detector does before calling the worker."""
    from PIL import Image

    stream = BytesIO()
    Image.fromarray(np.asarray(image)).save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _to_json(response):
    return {**response, "detections": [
        {**item, "mask_png": base64.b64encode(item["mask_png"]).decode("ascii")}
        for item in response["detections"]]}


def _from_json(response):
    return {**response, "detections": [
        {**item, "mask_png": base64.b64decode(item["mask_png"])}
        for item in response["detections"]]}


def build_cache(frames_dir, worker, *, rack_prompt_profile=DEFAULT_RACK_PROMPT_PROFILE, log=print,
                done_frames=None, save_progress=None):
    """Query ``worker`` once per frame and camera; return the cache document.

    ``done_frames`` (sequence -> camera answers, from an interrupted run) are kept
    and not queried again; ``save_progress(frames)`` is called after each new frame.
    """
    archives = sorted(Path(frames_dir, "observations").glob("observation_*.npz"))
    if not archives:
        raise ValueError("no_captured_observations")
    frames = dict(done_frames or {})
    cameras = {camera for answers in frames.values() for camera in answers}
    for index, path in enumerate(archives, start=1):
        observation = load_public_observation(path)
        if str(observation.sequence) not in frames:
            # Every camera the frame has, including a perception-only camera when captured.
            frames[str(observation.sequence)] = {
                camera: _to_json(worker.detect.remote(
                    encode_png(image), observation_sequence=observation.sequence,
                    camera=camera, rack_prompt_profile=rack_prompt_profile, include_rack=True))
                for camera, image in observation.images.items()}
            if save_progress is not None:
                save_progress(frames)
        cameras.update(observation.images)
        log(f"  {index}/{len(archives)} frames")
    return {"schema_version": 1, "rack_prompt_profile": rack_prompt_profile,
            "cameras": [name for name in (*CAMERAS, *PERCEPTION_CAMERAS) if name in cameras], "frames": frames}


def _write_json(path, document):
    """Write via a temporary file, so an interrupted write never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document) + "\n")
    temporary.replace(path)


class _CachedDetect:
    def __init__(self, cache):
        self._cache = cache

    def remote(self, rgb_png, *, observation_sequence, camera, rack_prompt_profile="default",
               include_rack=True):
        if rack_prompt_profile != self._cache["rack_prompt_profile"]:
            raise ValueError("sam3_cache_rack_prompt_profile_mismatch")
        try:
            response = _from_json(self._cache["frames"][str(observation_sequence)][camera])
        except KeyError:
            raise ValueError("sam3_cache_missing_frame_or_camera") from None
        if not include_rack:
            # The worker prompts each label separately, so skipping the rack
            # prompt is the same as dropping its detections.
            response["detections"] = [item for item in response["detections"] if item["label"] != "rack"]
        return response


class CachedSam3Worker:
    """Stands in for the deployed SAM 3 worker, answering from a cache document or file."""

    def __init__(self, cache):
        if isinstance(cache, (str, Path)):
            cache = json.loads(Path(cache).read_text())
        if not isinstance(cache, dict) or cache.get("schema_version") != 1:
            raise ValueError("unsupported_sam3_cache")
        self.detect = _CachedDetect(cache)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, required=True, help="a labeled_frames capture folder")
    parser.add_argument("--rack-prompt-profile", default=DEFAULT_RACK_PROMPT_PROFILE,
                        choices=("default", "explore_mug_rack"))
    args = parser.parse_args(argv)
    target = args.frames / CACHE_FILE
    if target.exists():
        parser.error(f"{target} already exists; delete it to query SAM 3 again")
    # A run that stops part-way (a laptop going to sleep, a preempted worker)
    # leaves its answers here; the next run continues from them.
    partial = args.frames / PARTIAL_CACHE_FILE
    done_frames = {}
    if partial.exists():
        progress = json.loads(partial.read_text())
        if progress["rack_prompt_profile"] != args.rack_prompt_profile:
            parser.error(f"{partial} used another rack prompt profile; delete it to start again")
        done_frames = progress["frames"]
        print(f"Continuing from {len(done_frames)} frames in {partial}")
    import modal

    worker = modal.Cls.from_name("robotwin-perception-sam3", "Sam3PerceptionWorker")()
    print(f"Querying SAM 3 for {args.frames} ...")
    cache = build_cache(args.frames, worker, rack_prompt_profile=args.rack_prompt_profile,
                        done_frames=done_frames,
                        save_progress=lambda frames: _write_json(
                            partial, {"rack_prompt_profile": args.rack_prompt_profile, "frames": frames}))
    _write_json(target, cache)
    partial.unlink(missing_ok=True)
    print(f"Saved {len(cache['frames'])} frames x {len(cache['cameras'])} cameras to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
