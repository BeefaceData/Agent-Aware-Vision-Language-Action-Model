"""SAM 3 answer cache: built once per frame and camera, replayed through the real client."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

from development.replays.sam3_cache import CachedSam3Worker, build_cache
from harness.environment.base import Observation
from harness.environment.layout import CAMERAS
from harness.perception.modal_sam3 import ModalSam3Detector
from harness.perception.observations import ObservationStore
from tests.helpers import images


def mask_png():
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 1] = 255
    stream = BytesIO()
    Image.fromarray(mask).save(stream, format="PNG")
    return stream.getvalue()


class FakeWorker:
    def __init__(self):
        self.calls = []
        self.detect = SimpleNamespace(remote=self.remote)

    def remote(self, rgb_png, *, observation_sequence, camera, rack_prompt_profile, include_rack):
        self.calls.append((observation_sequence, camera, include_rack))
        return {"observation_sequence": observation_sequence, "camera": camera,
                "detections": [{"label": "mug", "confidence": 0.9, "mask_png": mask_png()},
                               {"label": "rack", "confidence": 0.5, "mask_png": mask_png()}]}


def capture_with(sequences):
    directory = tempfile.TemporaryDirectory()
    store = ObservationStore(Path(directory.name) / "observations")
    for sequence in sequences:
        store.save(Observation(sequence, images(), tuple([0.0] * 14)))
    return directory


class Sam3CacheTests(unittest.TestCase):
    def test_queries_every_camera_of_every_frame_once(self):
        directory = capture_with([0, 16])
        worker = FakeWorker()

        cache = build_cache(directory.name, worker, log=lambda message: None)
        directory.cleanup()

        self.assertEqual(sorted(cache["frames"]), ["0", "16"])
        self.assertEqual(len(worker.calls), 2 * len(CAMERAS))
        self.assertTrue(all(include_rack for _sequence, _camera, include_rack in worker.calls))

    def test_cached_answers_replay_through_the_real_detector(self):
        directory = capture_with([0])
        cache = build_cache(directory.name, FakeWorker(), log=lambda message: None)
        directory.cleanup()
        detector = ModalSam3Detector(worker=CachedSam3Worker(cache),
                                     rack_prompt_profile=cache["rack_prompt_profile"])

        batch = detector.detect(images(), observation_sequence=0)

        self.assertEqual(batch.camera, "head_camera")
        mug = next(item for item in batch.detections if item.label == "mug")
        self.assertEqual(mug.mask.tolist(), [[False, True, False], [False, False, False]])

    def test_skipping_the_rack_prompt_drops_rack_detections(self):
        directory = capture_with([0])
        cache = build_cache(directory.name, FakeWorker(), log=lambda message: None)
        directory.cleanup()

        response = CachedSam3Worker(cache).detect.remote(b"", observation_sequence=0, camera="head_camera",
                                                         rack_prompt_profile=cache["rack_prompt_profile"],
                                                         include_rack=False)

        self.assertEqual([item["label"] for item in response["detections"]], ["mug"])

    def test_missing_frames_and_other_prompt_profiles_are_rejected(self):
        directory = capture_with([0])
        cache = build_cache(directory.name, FakeWorker(), log=lambda message: None)
        directory.cleanup()
        worker = CachedSam3Worker(cache)

        with self.assertRaisesRegex(ValueError, "missing_frame"):
            worker.detect.remote(b"", observation_sequence=99, camera="head_camera",
                                 rack_prompt_profile=cache["rack_prompt_profile"])
        with self.assertRaisesRegex(ValueError, "profile_mismatch"):
            worker.detect.remote(b"", observation_sequence=0, camera="head_camera", rack_prompt_profile="default")


if __name__ == "__main__":
    unittest.main()
