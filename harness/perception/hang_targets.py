"""Episode-local rack frame cache and grasp-aware analytic hang anchors."""

from hashlib import sha256

import numpy as np

from .anchors import Anchor
from .geometry import pose7
from .opus_frame import ground_public_object_frame
from .pose_grounding import tcp_targets_for_mug_alignment


class StaticRackFrameCache:
    """Cache one grounded rack frame and resolve it through later occlusion.

    The cache is created per episode. A frame is admitted only from a same-
    observation RGB keypoint report, a same-camera SAM mask and matching public
    RGB-D. Later lookups require the same camera image shape and intrinsics;
    the camera extrinsic may change as the robot moves. The cached pose is
    already in world coordinates, so later rack visibility is unnecessary.
    """

    def __init__(self, *, minimum_confidence=0.55,
                 max_mask_gap_px=8.0, max_depth_gap_m=0.06):
        limits = (minimum_confidence, max_mask_gap_px, max_depth_gap_m)
        if (not np.isfinite(limits).all() or not 0 <= minimum_confidence <= 1
                or max_mask_gap_px < 0 or max_depth_gap_m < 0):
            raise ValueError("invalid_rack_frame_cache_limits")
        self.minimum_confidence = float(minimum_confidence)
        self.max_mask_gap_px = float(max_mask_gap_px)
        self.max_depth_gap_m = float(max_depth_gap_m)
        self._reference = None

    def cache(self, observation, report, mask):
        if not isinstance(report, dict) or report.get("target") != "rack_peg":
            raise ValueError("rack_frame_report_required")
        pose = ground_public_object_frame(observation, report, mask,
            minimum_confidence=self.minimum_confidence,
            max_mask_gap_px=self.max_mask_gap_px,
            max_depth_gap_m=self.max_depth_gap_m)
        camera_name = report["camera"]
        image = np.asarray(observation.images[camera_name])
        camera = (observation.camera_frames or {})[camera_name]
        self._reference = {
            "pose": tuple(pose7(pose)),
            "camera": camera_name,
            "sequence": observation.sequence,
            "image_shape": tuple(image.shape[:2]),
            "intrinsic_sha256": _intrinsic_hash(camera),
            "calibration_sha256": _calibration_hash(camera),
            "confidence": float(report["confidence"]),
        }
        return self.metadata()

    def resolve(self, observation):
        reference = self._reference
        if reference is None:
            raise ValueError("rack_frame_not_cached")
        camera_name = reference["camera"]
        image = observation.images.get(camera_name)
        camera = (observation.camera_frames or {}).get(camera_name)
        if image is None or camera is None:
            raise ValueError("rack_frame_camera_unavailable")
        if (tuple(np.asarray(image).shape[:2]) != reference["image_shape"]
                or _intrinsic_hash(camera) != reference["intrinsic_sha256"]):
            raise ValueError("rack_frame_camera_changed")
        return np.asarray(reference["pose"], dtype=float)

    def clear(self):
        self._reference = None

    def metadata(self):
        if self._reference is None:
            return {"available": False}
        reference = self._reference
        return {"available": True, "camera": reference["camera"],
                "observation_sequence": reference["sequence"],
                "confidence": reference["confidence"],
                "image_shape_px": list(reference["image_shape"]),
                "camera_calibration_sha256": reference["calibration_sha256"]}


def bind_hang_targets(cache, observation, mug_report, mug_mask, *, tcp_pose,
                      anchors, arm, approach_offset_m=0.05,
                      max_target_distance_m=0.65):
    """Ground a current mug handle and bind measured-grasp-aware move targets.

    The rack is episode-static; the mug handle must be grounded in the current
    observation because its pose follows the actual grasp. The resulting
    symbolic anchors are ``rack_approach`` (outside the peg tip) and
    ``rack_seat`` (handle aligned with the peg seat). This function computes
    geometry only; callers still sequence MOVE_TO and RELEASE through the
    existing skill vocabulary and release guard.
    """
    if arm not in ("left", "right"):
        raise ValueError("hang_target_arm_required")
    if (isinstance(max_target_distance_m, bool)
            or not np.isfinite(max_target_distance_m) or max_target_distance_m <= 0):
        raise ValueError("invalid_hang_target_distance")
    if not isinstance(mug_report, dict) or mug_report.get("target") != "mug_handle":
        raise ValueError("mug_handle_report_required")
    rack_pose = cache.resolve(observation)
    mug_pose = ground_public_object_frame(observation, mug_report, mug_mask)
    measured_tcp = pose7(tcp_pose)
    prehang_tcp, seat_tcp = tcp_targets_for_mug_alignment(
        measured_tcp, mug_pose, rack_pose,
        approach_offset_m=approach_offset_m)
    for name, target in (("rack_approach", prehang_tcp), ("rack_seat", seat_tcp)):
        target = pose7(target)
        distance = float(np.linalg.norm(target[:3] - measured_tcp[:3]))
        if distance > max_target_distance_m:
            raise ValueError("hang_target_out_of_reach")
        anchors.bind(name, arm, Anchor(tuple(target), observation.sequence,
                                       source="perception"))
    return {"rack_approach": anchors.resolve("rack_approach", arm, observation),
            "rack_seat": anchors.resolve("rack_seat", arm, observation)}


def _calibration_hash(camera):
    digest = sha256()
    digest.update(np.asarray(camera.intrinsic, dtype=np.float64).tobytes())
    digest.update(np.asarray(camera.world_to_camera, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _intrinsic_hash(camera):
    return sha256(np.asarray(camera.intrinsic, dtype=np.float64).tobytes()).hexdigest()
