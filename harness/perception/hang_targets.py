"""Episode-local rack frame cache and grasp-aware analytic hang anchors."""

import numpy as np

from .anchors import Anchor
from .geometry import pose7
from .opus_frame import ground_public_object_frame_rgbd
from .pose_grounding import tcp_targets_for_mug_alignment
from .rgbd import calibration_sha256, intrinsic_sha256


class StaticRackFrameCache:
    """Cache one grounded rack frame and resolve it through later occlusion.

    The cache is created per episode. A frame is admitted only from a same-
    observation RGB keypoint report and matching public RGB-D with locally
    continuous depth at each point. No segmentation model is used for pose
    calculation. Later lookups require the same camera image shape and intrinsics;
    the camera extrinsic may change as the robot moves. The cached pose is
    already in world coordinates, so later rack visibility is unnecessary.
    """

    def __init__(self, *, minimum_confidence=0.55, max_depth_gap_m=0.06):
        limits = (minimum_confidence, max_depth_gap_m)
        if (not np.isfinite(limits).all() or not 0 <= minimum_confidence <= 1
                or max_depth_gap_m < 0):
            raise ValueError("invalid_rack_frame_cache_limits")
        self.minimum_confidence = float(minimum_confidence)
        self.max_depth_gap_m = float(max_depth_gap_m)
        self._reference = None

    def cache(self, observation, report):
        if not isinstance(report, dict) or report.get("target") != "rack_peg":
            raise ValueError("rack_frame_report_required")
        pose = ground_public_object_frame_rgbd(observation, report,
            minimum_confidence=self.minimum_confidence,
            max_depth_gap_m=self.max_depth_gap_m)
        camera_name = report["camera"]
        image = np.asarray(observation.images[camera_name])
        camera = (observation.camera_frames or {})[camera_name]
        self._reference = {
            "pose": tuple(pose7(pose)),
            "camera": camera_name,
            "sequence": observation.sequence,
            "image_shape": tuple(image.shape[:2]),
            "intrinsic_sha256": intrinsic_sha256(camera),
            "calibration_sha256": calibration_sha256(camera),
            "confidence": float(report["confidence"]),
        }
        return self.metadata()

    def resolve(self, observation):
        reference = self._reference
        if reference is None:
            raise ValueError("rack_frame_not_cached")
        if observation.sequence < reference["sequence"]:
            raise ValueError("rack_frame_observation_before_reference")
        camera_name = reference["camera"]
        image = observation.images.get(camera_name)
        camera = (observation.camera_frames or {}).get(camera_name)
        if image is None or camera is None:
            raise ValueError("rack_frame_camera_unavailable")
        if (tuple(np.asarray(image).shape[:2]) != reference["image_shape"]
                or intrinsic_sha256(camera) != reference["intrinsic_sha256"]):
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


def bind_hang_targets(cache, observation, mug_report, *, tcp_pose,
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
    mug_pose = ground_public_object_frame_rgbd(observation, mug_report)
    measured_tcp = pose7(tcp_pose)
    prehang_tcp, seat_tcp = tcp_targets_for_mug_alignment(
        measured_tcp, mug_pose, rack_pose,
        approach_offset_m=approach_offset_m)
    prepared = {}
    for name, target in (("rack_approach", prehang_tcp), ("rack_seat", seat_tcp)):
        target = pose7(target)
        distance = float(np.linalg.norm(target[:3] - measured_tcp[:3]))
        if distance > max_target_distance_m:
            raise ValueError("hang_target_out_of_reach")
        prepared[name] = Anchor(tuple(target), observation.sequence, source="perception")
    # Reject the complete target pair before publishing either anchor.
    for name, anchor in prepared.items():
        anchors.bind(name, arm, anchor)
    return {"rack_approach": anchors.resolve("rack_approach", arm, observation),
            "rack_seat": anchors.resolve("rack_seat", arm, observation)}
