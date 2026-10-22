"""CPU contracts for the RGB-only Modal SAM 3 client boundary."""

from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from harness.perception.modal_sam3 import ModalSam3Detector, create_detector


class RemoteMethod:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def remote(self, rgb_png, *, observation_sequence, camera,
               rack_prompt_profile="default", include_rack=True):
        self.calls.append((rgb_png, observation_sequence, camera, rack_prompt_profile,
                           include_rack))
        if isinstance(self.result, list):
            return self.result.pop(0)
        return self.result


class FakeWorker:
    def __init__(self, result):
        self.detect = RemoteMethod(result)


def png_mask(mask):
    stream = BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(stream, format="PNG")
    return stream.getvalue()


class ModalSam3ClientTests(unittest.TestCase):
    @staticmethod
    def response(sequence, camera, detections=()):
        return {"observation_sequence": sequence, "camera": camera,
                "detections": list(detections)}

    @staticmethod
    def mug_detection(mask, confidence=.9):
        return {"label": "mug", "confidence": confidence, "mask_png": png_mask(mask)}

    def test_serializes_only_rgb_and_decodes_sequence_fresh_masks(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[1:4, 2:6] = True
        response = {"observation_sequence": 12, "camera": "head_camera", "detections": [
            {"label": "mug", "confidence": .95, "mask_png": png_mask(mask)}]}
        worker = FakeWorker(response)
        detector = ModalSam3Detector(worker)
        image = np.full((5, 7, 3), 70, dtype=np.uint8)

        batch = detector.detect({"head_camera": image, "right_camera": image},
                                observation_sequence=12)

        self.assertEqual(batch.observation_sequence, 12)
        self.assertEqual(batch.camera, "head_camera")
        np.testing.assert_array_equal(batch.detections[0].mask, mask)
        self.assertEqual(len(worker.detect.calls), 1)
        payload, sequence, camera, profile, include_rack = worker.detect.calls[0]
        self.assertEqual((sequence, camera), (12, "head_camera"))
        self.assertEqual(profile, "default")
        self.assertTrue(include_rack)
        self.assertIsInstance(payload, bytes)
        self.assertLess(len(payload), 10_000)

    def test_factory_returns_remote_client_without_loading_model_locally(self):
        worker = FakeWorker({"observation_sequence": 0, "camera": "head_camera",
                             "detections": []})
        modal_module = SimpleNamespace(Cls=SimpleNamespace(
            from_name=Mock(return_value=lambda: worker)))
        with patch.dict("sys.modules", {"modal": modal_module}):
            detector = create_detector()
        self.assertIsInstance(detector, ModalSam3Detector)
        modal_module.Cls.from_name.assert_called_once_with(
            "robotwin-perception-sam3", "Sam3PerceptionWorker")

    def test_client_can_select_a_non_head_rgb_camera(self):
        worker = FakeWorker({"observation_sequence": 3, "camera": "right_camera",
                             "detections": []})
        detector = ModalSam3Detector(worker, camera="right_camera")
        batch = detector.detect({"right_camera": np.zeros((4, 5, 3), dtype=np.uint8)},
                                observation_sequence=3)
        self.assertEqual(batch.camera, "right_camera")
        self.assertEqual(worker.detect.calls[0][2:4], ("right_camera", "default"))

    def test_exploration_rack_profile_is_explicit_and_configurable(self):
        worker = FakeWorker({"observation_sequence": 3, "camera": "head_camera",
                             "detections": []})
        detector = ModalSam3Detector(worker)
        detector.configure_rack_prompt_profile("explore_mug_rack")
        detector.detect({"head_camera": np.zeros((4, 5, 3), dtype=np.uint8)},
                        observation_sequence=3)
        self.assertEqual(worker.detect.calls[0][3], "explore_mug_rack")
        with self.assertRaisesRegex(ValueError, "unsupported_rack_prompt_profile"):
            detector.configure_rack_prompt_profile("arbitrary_prompt")

    def test_grounding_camera_keeps_mug_table_on_primary_view(self):
        detector = ModalSam3Detector(FakeWorker({"observation_sequence": 3,
            "camera": "right_camera", "detections": []}))
        detector.configure_rack_prompt_profile("explore_mug_rack")
        detector.configure_rack_grounding_camera("right_camera")
        self.assertEqual(detector.camera, "head_camera")
        self.assertEqual(detector.rack_grounding_camera, "right_camera")
        self.assertEqual(detector.rack_fallback_camera, "right_camera")

    def test_mug_score_at_new_cutoff_suppresses_unneeded_wrist_fallback(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            {"observation_sequence": 3, "camera": "head_camera", "detections": [
                {"label": "mug", "confidence": .40, "mask_png": png_mask(mask)}]},
        ])
        detector = ModalSam3Detector(worker)
        batches = detector.detect({"head_camera": np.zeros((4, 5, 3), dtype=np.uint8),
                                   "left_camera": np.zeros((4, 5, 3), dtype=np.uint8)},
                                  observation_sequence=3)
        self.assertEqual(len(worker.detect.calls), 1)
        self.assertEqual(batches.camera, "head_camera")
        with self.assertRaisesRegex(ValueError, "unsupported_sam3_grounding_camera"):
            detector.configure_rack_grounding_camera("wrist_camera")

    def test_selected_rack_view_is_queried_once_even_if_head_sees_rack(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            {"observation_sequence": 8, "camera": "head_camera", "detections": [
                {"label": "mug", "confidence": .9, "mask_png": png_mask(mask)},
                {"label": "rack", "confidence": .9, "mask_png": png_mask(mask)}]},
            {"observation_sequence": 8, "camera": "right_camera", "detections": [
                {"label": "rack", "confidence": .8, "mask_png": png_mask(mask)}]},
            {"observation_sequence": 9, "camera": "head_camera", "detections": [
                {"label": "mug", "confidence": .9, "mask_png": png_mask(mask)}]},
        ])
        detector = ModalSam3Detector(worker)
        detector.configure_rack_grounding_camera("right_camera")
        images = {name: np.zeros((4, 5, 3), dtype=np.uint8)
                  for name in ("head_camera", "right_camera")}
        first = detector.detect(images, observation_sequence=8)
        self.assertEqual([batch.camera for batch in first],
                         ["head_camera", "right_camera"])
        self.assertEqual([item.label for item in first[0].detections], ["mug", "rack"])
        self.assertEqual([item.label for item in first[1].detections], ["rack"])
        detector.set_static_rack_reference(True)
        second = detector.detect(images, observation_sequence=9)
        self.assertEqual(second.camera, "head_camera")
        self.assertEqual([call[2] for call in worker.detect.calls],
                         ["head_camera", "right_camera", "head_camera"])
        self.assertFalse(worker.detect.calls[-1][4])

    def test_exploration_profile_uses_right_view_only_when_head_has_no_rack(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            {"observation_sequence": 8, "camera": "head_camera", "detections": [
                {"label": "mug", "confidence": .9, "mask_png": png_mask(mask)}]},
            {"observation_sequence": 8, "camera": "right_camera", "detections": [
                {"label": "rack", "confidence": .21, "mask_png": png_mask(mask)},
                {"label": "mug", "confidence": .9, "mask_png": png_mask(mask)}]},
        ])
        detector = ModalSam3Detector(worker)
        detector.configure_rack_prompt_profile("explore_mug_rack")
        batches = detector.detect({"head_camera": np.zeros((4, 5, 3), dtype=np.uint8),
                                   "right_camera": np.ones((4, 5, 3), dtype=np.uint8)},
                                  observation_sequence=8)
        self.assertEqual([batch.camera for batch in batches], ["head_camera", "right_camera"])
        self.assertEqual([item.label for item in batches[0].detections], ["mug"])
        self.assertEqual([item.label for item in batches[1].detections], ["rack"])
        self.assertEqual([call[2] for call in worker.detect.calls], ["head_camera", "right_camera"])
        self.assertEqual([call[3] for call in worker.detect.calls],
                         ["explore_mug_rack", "explore_mug_rack"])

    def test_exploration_profile_rechecks_ambiguous_head_rack_from_right_view(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            {"observation_sequence": 9, "camera": "head_camera", "detections": [
                {"label": "rack", "confidence": .46, "mask_png": png_mask(mask)},
                {"label": "rack", "confidence": .40, "mask_png": png_mask(mask)},
                {"label": "mug", "confidence": .9, "mask_png": png_mask(mask)}]},
            {"observation_sequence": 9, "camera": "right_camera", "detections": [
                {"label": "rack", "confidence": .31, "mask_png": png_mask(mask)}]},
        ])
        detector = ModalSam3Detector(worker)
        detector.configure_rack_prompt_profile("explore_mug_rack")

        batches = detector.detect({"head_camera": np.zeros((4, 5, 3), dtype=np.uint8),
                                   "right_camera": np.ones((4, 5, 3), dtype=np.uint8)},
                                  observation_sequence=9)

        self.assertEqual([batch.camera for batch in batches], ["head_camera", "right_camera"])
        self.assertEqual([item.label for item in batches[1].detections], ["rack"])
        self.assertEqual([call[2] for call in worker.detect.calls], ["head_camera", "right_camera"])

    def test_exploration_profile_skips_right_view_for_confident_head_rack(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker({"observation_sequence": 10, "camera": "head_camera",
                             "detections": [
                                 {"label": "rack", "confidence": .8,
                                  "mask_png": png_mask(mask)},
                                 {"label": "rack", "confidence": .6,
                                  "mask_png": png_mask(mask)}]})
        detector = ModalSam3Detector(worker)
        detector.configure_rack_prompt_profile("explore_mug_rack")

        batch = detector.detect({"head_camera": np.zeros((4, 5, 3), dtype=np.uint8),
                                 "right_camera": np.ones((4, 5, 3), dtype=np.uint8)},
                                observation_sequence=10)

        self.assertEqual(batch.camera, "head_camera")
        self.assertEqual(len(worker.detect.calls), 1)

    def test_cached_rack_reference_disables_repeated_remote_rack_inference(self):
        worker = FakeWorker({"observation_sequence": 11, "camera": "head_camera",
                             "detections": []})
        detector = ModalSam3Detector(worker)
        detector.set_static_rack_reference(True)

        detector.detect({"head_camera": np.zeros((4, 5, 3), dtype=np.uint8)},
                        observation_sequence=11)

        self.assertFalse(worker.detect.calls[0][4])

    def test_rejects_missing_camera_invalid_rgb_and_stale_response(self):
        detector = ModalSam3Detector(FakeWorker({
            "observation_sequence": 2, "camera": "head_camera", "detections": []}))
        with self.assertRaisesRegex(ValueError, "camera_unavailable"):
            detector.detect({}, observation_sequence=1)
        with self.assertRaisesRegex(ValueError, "uint8_rgb"):
            detector.detect({"head_camera": np.zeros((4, 4), dtype=np.uint8)},
                            observation_sequence=1)
        with self.assertRaisesRegex(ValueError, "invalid_modal_sam3_response"):
            detector.detect({"head_camera": np.zeros((4, 4, 3), dtype=np.uint8)},
                            observation_sequence=1)

    def test_head_is_primary_and_grasp_phase_falls_back_to_left_wrist_camera(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            self.response(21, "head_camera"),
            self.response(21, "left_camera", [self.mug_detection(mask)]),
        ])
        detector = ModalSam3Detector(worker)
        images = {name: np.zeros((4, 5, 3), dtype=np.uint8)
                  for name in ("head_camera", "left_camera", "right_camera")}

        batches = detector.detect(images, observation_sequence=21)

        self.assertEqual([batch.camera for batch in batches], ["head_camera", "left_camera"])
        self.assertEqual([item.label for item in batches[1].detections], ["mug"])
        self.assertEqual([call[2] for call in worker.detect.calls],
                         ["head_camera", "left_camera"])

    def test_handoff_phase_tries_left_then_right_wrist_until_mug_is_found(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            self.response(22, "head_camera"),
            self.response(22, "left_camera"),
            self.response(22, "right_camera", [self.mug_detection(mask)]),
        ])
        detector = ModalSam3Detector(worker)
        detector.set_task_phase("HANDOFF")
        images = {name: np.zeros((4, 5, 3), dtype=np.uint8)
                  for name in ("head_camera", "left_camera", "right_camera")}

        batches = detector.detect(images, observation_sequence=22)

        self.assertEqual([batch.camera for batch in batches],
                         ["head_camera", "right_camera"])
        self.assertEqual([call[2] for call in worker.detect.calls],
                         ["head_camera", "left_camera", "right_camera"])

    def test_hang_phase_uses_right_wrist_and_skips_fallback_when_head_sees_mug(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            self.response(23, "head_camera", [self.mug_detection(mask)]),
        ])
        detector = ModalSam3Detector(worker)
        detector.set_task_phase("HANG")
        images = {name: np.zeros((4, 5, 3), dtype=np.uint8)
                  for name in ("head_camera", "left_camera", "right_camera")}

        batch = detector.detect(images, observation_sequence=23)

        self.assertEqual(batch.camera, "head_camera")
        self.assertEqual([call[2] for call in worker.detect.calls], ["head_camera"])

    def test_hang_phase_uses_right_wrist_when_head_loses_mug(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            self.response(24, "head_camera"),
            self.response(24, "right_camera", [self.mug_detection(mask)]),
        ])
        detector = ModalSam3Detector(worker)
        detector.set_task_phase("RIGHT_PICK_AND_HANG")
        images = {name: np.zeros((4, 5, 3), dtype=np.uint8)
                  for name in ("head_camera", "left_camera", "right_camera")}

        batches = detector.detect(images, observation_sequence=24)

        self.assertEqual([batch.camera for batch in batches], ["head_camera", "right_camera"])
        self.assertEqual([call[2] for call in worker.detect.calls],
                         ["head_camera", "right_camera"])

    def test_low_confidence_head_mug_does_not_suppress_phase_fallback(self):
        mask = np.ones((4, 5), dtype=bool)
        worker = FakeWorker([
            self.response(25, "head_camera", [self.mug_detection(mask, confidence=.39)]),
            self.response(25, "left_camera", [self.mug_detection(mask, confidence=.9)]),
        ])
        detector = ModalSam3Detector(worker)
        images = {name: np.zeros((4, 5, 3), dtype=np.uint8)
                  for name in ("head_camera", "left_camera")}

        batches = detector.detect(images, observation_sequence=25)

        self.assertEqual([batch.camera for batch in batches], ["head_camera", "left_camera"])


if __name__ == "__main__":
    unittest.main()
