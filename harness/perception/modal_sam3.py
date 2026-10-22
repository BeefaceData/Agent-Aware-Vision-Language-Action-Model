"""Modal client for SAM 3 masks; no model weights or GPU code run locally."""

from io import BytesIO

import numpy as np
from PIL import Image

from .grounding import Detection, DetectionBatch, SEMANTIC_LABELS

_RACK_AMBIGUITY_MARGIN = 0.10
MUG_SCORE_THRESHOLD = 0.40


class ModalSam3Detector:
    """RGB-only semantic detector backed by a separately deployed Modal app."""

    def __init__(self, worker=None, *, app_name="robotwin-perception-sam3",
                 class_name="Sam3PerceptionWorker", camera="head_camera",
                 max_image_bytes=4_000_000, rack_prompt_profile="default",
                 rack_fallback_camera=None):
        if (not isinstance(app_name, str) or not app_name
                or not isinstance(class_name, str) or not class_name
                or not isinstance(camera, str) or not camera
                or type(max_image_bytes) is not int or not 1024 <= max_image_bytes <= 16_000_000
                or rack_prompt_profile not in ("default", "explore_mug_rack")
                or rack_fallback_camera not in (None, "head_camera", "left_camera", "right_camera")
                or rack_fallback_camera == camera):
            raise ValueError("invalid_modal_sam3_settings")
        if worker is None:
            import modal
            worker = modal.Cls.from_name(app_name, class_name)()
        if not callable(getattr(getattr(worker, "detect", None), "remote", None)):
            raise ValueError("modal_sam3_remote_method_missing")
        self.worker = worker
        self.camera = camera
        self.rack_grounding_camera = camera
        self.max_image_bytes = max_image_bytes
        self.rack_prompt_profile = rack_prompt_profile
        self.rack_fallback_camera = rack_fallback_camera
        self.include_rack = True
        self.task_phase = "grasp"

    def set_task_phase(self, phase):
        """Select wrist-view fallback order without changing the head primary."""
        aliases = {"APPROACH_MUG": "grasp", "GRASP": "grasp",
                   "TRANSFER": "handoff", "HANDOFF": "handoff",
                   "APPROACH_RACK": "hang", "RIGHT_PICK_AND_HANG": "hang",
                   "HANG": "hang", "COMPLETE": "hang"}
        value = aliases.get(phase, phase)
        if value not in ("grasp", "handoff", "hang"):
            raise ValueError("unsupported_perception_task_phase")
        self.task_phase = value

    def configure_rack_prompt_profile(self, profile):
        if profile not in ("default", "explore_mug_rack"):
            raise ValueError("unsupported_rack_prompt_profile")
        self.rack_prompt_profile = profile
        # One conditional second view, only for the rack, keeps mug and table
        # detections on the head camera and avoids extra calls when the primary
        # view already grounds a rack candidate.
        fallback = {"head_camera": "right_camera", "right_camera": "head_camera",
                    "left_camera": "right_camera"}[self.camera]
        self.rack_fallback_camera = (self.rack_grounding_camera
            if self.rack_grounding_camera != self.camera else
            fallback if profile == "explore_mug_rack" else None)

    def configure_rack_grounding_camera(self, camera):
        if camera not in ("head_camera", "left_camera", "right_camera"):
            raise ValueError("unsupported_sam3_grounding_camera")
        # Mug/table evidence remains on the primary camera. The selected rack
        # view is queried separately while the one-shot rack cache is built.
        self.rack_grounding_camera = camera
        self.configure_rack_prompt_profile(self.rack_prompt_profile)

    def set_static_rack_reference(self, initialized):
        """Skip repeated SAM rack inference after one-shot episode grounding."""
        if type(initialized) is not bool:
            raise ValueError("static_rack_reference_flag_must_be_boolean")
        self.include_rack = not initialized

    def detect(self, images, *, observation_sequence):
        if type(observation_sequence) is not int or observation_sequence < 0:
            raise ValueError("invalid_observation_sequence")
        if self.camera not in images:
            raise ValueError("sam3_camera_unavailable")
        image = np.asarray(images[self.camera])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
            raise ValueError("sam3_requires_uint8_rgb")
        queried = {}

        def query(camera):
            if camera not in queried:
                if camera not in images:
                    return None
                view = np.asarray(images[camera])
                if (view.ndim != 3 or view.shape[2] != 3
                        or view.dtype != np.uint8 or not view.size):
                    return None
                queried[camera] = self._detect_camera(view, camera, observation_sequence)
            return queried[camera]

        primary = query(self.camera)
        if primary is None:
            raise ValueError("sam3_camera_unavailable")
        batches = [primary]
        rack_candidates_all = sorted(
            (item for item in primary.detections if item.label == "rack"),
            key=lambda item: item.confidence, reverse=True)
        rack_is_ambiguous = (
            len(rack_candidates_all) > 1
            and rack_candidates_all[0].confidence - rack_candidates_all[1].confidence
            < _RACK_AMBIGUITY_MARGIN)
        required_rack_view = (self.include_rack
                              and self.rack_grounding_camera != self.camera)
        if (self.include_rack and self.rack_fallback_camera is not None
                and self.rack_fallback_camera in images
                and (required_rack_view or rack_is_ambiguous
                     or not rack_candidates_all)):
            try:
                rack_view = query(self.rack_fallback_camera)
                rack_only = (() if rack_view is None else
                    tuple(item for item in rack_view.detections if item.label == "rack"))
                if rack_only:
                    batches.append(DetectionBatch(observation_sequence,
                        self.rack_fallback_camera, rack_only))
            except Exception:
                # Optional rack inference must not invalidate primary
                # mug/table evidence or turn a transient RPC error into dropout.
                pass

        # Head RGB remains the cheap primary view. Only query phase-relevant
        # eye-in-hand views when it has no confident mug candidate. Each view
        # uses its own same-observation RGB-D calibration downstream.
        if not self._has_confident_mug(primary):
            for camera in self._mug_fallback_order():
                if camera == self.camera or camera not in images:
                    continue
                try:
                    fallback = query(camera)
                except Exception:
                    continue
                if fallback is None:
                    continue
                mug_only = tuple(item for item in fallback.detections
                                 if item.label == "mug" and item.confidence >= MUG_SCORE_THRESHOLD)
                if mug_only:
                    batches.append(DetectionBatch(observation_sequence, camera, mug_only))
                    break
        return batches[0] if len(batches) == 1 else tuple(batches)

    def _mug_fallback_order(self):
        return {"grasp": ("left_camera",),
                "handoff": ("left_camera", "right_camera"),
                "hang": ("right_camera",)}[self.task_phase]

    @staticmethod
    def _has_confident_mug(batch):
        return any(item.label == "mug" and item.confidence >= MUG_SCORE_THRESHOLD
                   for item in batch.detections)

    def _detect_camera(self, image, camera, observation_sequence):
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG", optimize=True)
        rgb_png = stream.getvalue()
        if len(rgb_png) > self.max_image_bytes:
            raise ValueError("sam3_rgb_payload_too_large")
        response = self.worker.detect.remote(rgb_png,
            observation_sequence=observation_sequence, camera=camera,
            rack_prompt_profile=self.rack_prompt_profile,
            include_rack=self.include_rack)
        if (not isinstance(response, dict)
                or response.get("observation_sequence") != observation_sequence
                or response.get("camera") != camera
                or not isinstance(response.get("detections"), list)):
            raise ValueError("invalid_modal_sam3_response")
        detections = []
        for item in response["detections"]:
            if not isinstance(item, dict) or item.get("label") not in SEMANTIC_LABELS:
                raise ValueError("invalid_modal_sam3_detection")
            mask_png = item.get("mask_png")
            if not isinstance(mask_png, bytes) or not mask_png or len(mask_png) > self.max_image_bytes:
                raise ValueError("invalid_modal_sam3_mask")
            with Image.open(BytesIO(mask_png)) as mask_image:
                if mask_image.mode not in ("1", "L"):
                    mask_image = mask_image.convert("L")
                mask = np.asarray(mask_image) > 0
            detections.append(Detection(item["label"], item["confidence"], mask))
        return DetectionBatch(observation_sequence, camera, tuple(detections))


def create_detector():
    """Factory for ``run_skills.py --perception-factory``; connects to Modal."""
    return ModalSam3Detector()
