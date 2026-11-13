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
    def test_selection_limits_remote_calls_without_modifying_source_frames(self):
        directory = capture_with([0, 8, 16, 24, 32])
        self.addCleanup(directory.cleanup)
        worker = FakeWorker()
        cache = build_cache(directory.name, worker, every=2, max_frames=2,
                            camera="head_camera", log=lambda _: None)
        self.assertEqual(worker.calls, [(0, "head_camera", True), (16, "head_camera", True)])
        self.assertEqual(cache["cameras"], ["head_camera"])
        self.assertEqual(len(list(Path(directory.name, "observations").glob("*.npz"))), 5)

    def test_missing_requested_camera_does_not_query_worker(self):
        directory = capture_with([0])
        self.addCleanup(directory.cleanup)
        worker = FakeWorker()
        with self.assertRaisesRegex(ValueError, "sam3_replay_camera_missing"):
            build_cache(directory.name, worker, camera="perception_camera")
        self.assertEqual(worker.calls, [])

    def test_resume_preserves_frame_selection_and_completed_camera_answers(self):
        directory = capture_with([0, 8, 16, 24, 32])
        self.addCleanup(directory.cleanup)
        first = build_cache(directory.name, FakeWorker(), every=2, max_frames=2,
                            camera="head_camera", log=lambda _: None)
        worker = FakeWorker()
        cache = build_cache(directory.name, worker, every=2, max_frames=2,
                            camera="head_camera", done_frames={"0": first["frames"]["0"]},
                            log=lambda _: None)
        self.assertEqual(worker.calls, [(16, "head_camera", True)])
        self.assertEqual(cache, first)

    def test_resume_queries_missing_cameras_without_repeating_completed_answers(self):
        directory = capture_with([0])
        self.addCleanup(directory.cleanup)
        first = build_cache(directory.name, FakeWorker(), camera="head_camera", log=lambda _: None)
        worker = FakeWorker()
        cache = build_cache(directory.name, worker, done_frames=first["frames"], log=lambda _: None)
        self.assertEqual({camera for _, camera, _ in worker.calls}, set(CAMERAS)-{"head_camera"})
        self.assertEqual(cache["frames"]["0"]["head_camera"], first["frames"]["0"]["head_camera"])

    def test_invalid_selection_does_not_query_worker(self):
        worker = FakeWorker()
        for options in ({"every": 0}, {"max_frames": 0}, {"camera": "unknown"}):
            with self.assertRaisesRegex(ValueError, "invalid_sam3_replay_selection"):
                build_cache("not_read", worker, **options)
        self.assertEqual(worker.calls, [])

    def test_queries_every_camera_of_every_frame_once(self):
        directory = capture_with([0, 16])
        worker = FakeWorker()

        cache = build_cache(directory.name, worker, log=lambda message: None)
        directory.cleanup()

        self.assertEqual(sorted(cache["frames"]), ["0", "16"])
        self.assertEqual(len(worker.calls), 2 * len(CAMERAS))
        self.assertTrue(all(include_rack for _sequence, _camera, include_rack in worker.calls))

    def test_an_interrupted_run_continues_without_querying_done_frames_again(self):
        directory = capture_with([0, 16])
        first = build_cache(directory.name, FakeWorker(), log=lambda message: None)
        worker, saved = FakeWorker(), []

        cache = build_cache(directory.name, worker, log=lambda message: None,
                            done_frames={"0": first["frames"]["0"]}, save_progress=saved.append)
        directory.cleanup()

        self.assertEqual({sequence for sequence, _camera, _rack in worker.calls}, {16})
        self.assertEqual(cache, first)
        self.assertEqual(len(saved), 1)

    def test_a_captured_perception_camera_is_cached_too(self):
        directory = tempfile.TemporaryDirectory()
        store = ObservationStore(Path(directory.name) / "observations")
        store.save(Observation(0, {**images(), "perception_camera": images()["head_camera"]}, tuple([0.0] * 14)))
        worker = FakeWorker()

        cache = build_cache(directory.name, worker, log=lambda message: None)
        directory.cleanup()

        self.assertEqual(cache["cameras"], [*CAMERAS, "perception_camera"])
        self.assertIn("perception_camera", cache["frames"]["0"])

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
