"""Validated public RGB-D camera data and metric pixel unprojection.

This module has no simulator dependency. Camera extrinsics use the OpenCV
world-to-camera convention; RoboTwin's depth raster is converted from mm to m
when its observation payload is adapted.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass(frozen=True)
class RGBDCamera:
    depth_m: np.ndarray
    intrinsic: np.ndarray
    world_to_camera: np.ndarray

    def __post_init__(self):
        depth = np.asarray(self.depth_m, dtype=np.float32)
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        extrinsic = np.asarray(self.world_to_camera, dtype=np.float64)
        if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("invalid_depth_image")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("invalid_camera_intrinsic")
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or abs(np.linalg.det(intrinsic)) < 1e-9
                or not np.allclose(intrinsic[2], (0, 0, 1), atol=1e-7)):
            raise ValueError("invalid_camera_intrinsic")
        if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
            raise ValueError("invalid_camera_extrinsic")
        rotation = extrinsic[:3, :3]
        if (not np.allclose(extrinsic[3], (0, 0, 0, 1), atol=1e-7)
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
                or np.linalg.det(rotation) < 0.999):
            raise ValueError("invalid_camera_extrinsic")
        for name, value in (("depth_m", depth), ("intrinsic", intrinsic),
                            ("world_to_camera", extrinsic)):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def copy(self):
        return RGBDCamera(self.depth_m, self.intrinsic, self.world_to_camera)

    def sample_depth(self, u, v, *, radius=0):
        """Return the local median metric depth, ignoring invalid zero pixels."""
        try:
            pixel = np.asarray((u, v), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("invalid_pixel") from None
        if (pixel.shape != (2,) or not np.isfinite(pixel).all()
                or type(radius) is not int or not 0 <= radius <= 8):
            raise ValueError("invalid_pixel")
        u, v = map(float, pixel)
        x, y = int(round(u)), int(round(v))
        height, width = self.depth_m.shape
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("pixel_out_of_bounds")
        roi = self.depth_m[max(0, y-radius):min(height, y+radius+1),
                           max(0, x-radius):min(width, x+radius+1)]
        valid = roi[(roi > 0) & np.isfinite(roi)]
        if not valid.size:
            raise ValueError("depth_unavailable")
        return float(np.median(valid))

    def local_depth_gap(self, u, v, *, search_radius=4, point_radius=1):
        """Measure how far the picked surface lies behind a nearby foreground.

        A large positive gap often means a pixel sees through an opening onto a
        farther surface. Callers should fail closed rather than ground that
        point as the requested object's surface.
        """
        if (type(search_radius) is not int or not 0 <= search_radius <= 16
                or type(point_radius) is not int or not 0 <= point_radius <= 8):
            raise ValueError("invalid_depth_neighborhood")
        selected_depth = self.sample_depth(u, v, radius=point_radius)
        try:
            pixel = np.asarray((u, v), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("invalid_pixel") from None
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("invalid_pixel")
        x, y = map(int, np.rint(pixel))
        height, width = self.depth_m.shape
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("pixel_out_of_bounds")
        roi = self.depth_m[max(0, y-search_radius):min(height, y+search_radius+1),
                           max(0, x-search_radius):min(width, x+search_radius+1)]
        valid = roi[(roi > 0) & np.isfinite(roi)]
        if not valid.size:
            raise ValueError("depth_unavailable")
        return max(0.0, selected_depth - float(np.min(valid)))

    def unproject(self, u, v, *, radius=0):
        """Return a world XYZ point from pixel coordinates and local median depth."""
        try:
            pixel = np.asarray((u, v), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("invalid_pixel") from None
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("invalid_pixel")
        u, v = map(float, pixel)
        depth = self.sample_depth(u, v, radius=radius)
        pixel = np.array([float(u), float(v), 1.0])
        camera_xyz = np.linalg.solve(self.intrinsic, pixel) * depth
        camera_h = np.append(camera_xyz, 1.0)
        world = np.linalg.solve(self.world_to_camera, camera_h)
        if not np.isfinite(world).all() or abs(world[3]) < 1e-9:
            raise ValueError("invalid_unprojected_point")
        return tuple(float(value) for value in world[:3] / world[3])


def intrinsic_sha256(camera) -> str:
    """Identify a camera's lens model; unchanged while the robot moves the camera."""
    return sha256(np.asarray(camera.intrinsic, dtype=np.float64).tobytes()).hexdigest()


def calibration_sha256(camera) -> str:
    """Identify a camera's full calibration (intrinsic plus world-to-camera pose)."""
    digest = sha256()
    digest.update(np.asarray(camera.intrinsic, dtype=np.float64).tobytes())
    digest.update(np.asarray(camera.world_to_camera, dtype=np.float64).tobytes())
    return digest.hexdigest()


def camera_frames_from_payload(payload, images):
    """Decode optional RoboTwin depth-mm/calibration fields into public frames."""
    frames = {}
    for name, image in images.items():
        keys = (f"{name}__depth_mm", f"{name}__intrinsic_cv", f"{name}__extrinsic_cv")
        present = [key in payload for key in keys]
        if not any(present):
            continue
        if not all(present):
            raise ValueError("incomplete_rgbd_camera_payload")
        depth_mm = np.asarray(payload[keys[0]])
        if depth_mm.shape != image.shape[:2]:
            raise ValueError("depth_rgb_shape_mismatch")
        depth_m = np.asarray(depth_mm, dtype=np.float32) / 1000.0
        extrinsic = np.asarray(payload[keys[2]], dtype=np.float64)
        # RoboTwin/SAPIEN returns the OpenCV world-to-camera matrix as 3x4.
        # Internally keep the equivalent homogeneous 4x4 form for inversion.
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack((extrinsic, (0.0, 0.0, 0.0, 1.0)))
        frames[name] = RGBDCamera(depth_m, payload[keys[1]], extrinsic)
    return frames


def require_camera_frames(observation, required_cameras):
    """Fail clearly when an RGB-D-enabled run lacks a required camera frame."""
    if not isinstance(required_cameras, (tuple, list, set, frozenset)):
        raise ValueError("invalid_required_rgbd_cameras")
    if any(not isinstance(name, str) or not name for name in required_cameras):
        raise ValueError("invalid_required_rgbd_cameras")
    frames = observation.camera_frames or {}
    missing = sorted(set(required_cameras) - set(frames))
    if missing:
        raise ValueError("required_rgbd_camera_frames_missing:" + ",".join(missing))
    return observation


def camera_archive_values(frames):
    """Return NPZ-safe metric sensor fields for replayable public observations."""
    values = {}
    for camera, frame in (frames or {}).items():
        values[f"{camera}__depth_m"] = np.asarray(frame.depth_m, dtype=np.float32)
        values[f"{camera}__intrinsic"] = np.asarray(frame.intrinsic, dtype=np.float64)
        values[f"{camera}__world_to_camera"] = np.asarray(frame.world_to_camera, dtype=np.float64)
    return values


def camera_frames_from_archive(archive, images):
    frames = {}
    for camera, image in images.items():
        keys = (f"{camera}__depth_m", f"{camera}__intrinsic", f"{camera}__world_to_camera")
        present = [key in archive for key in keys]
        if not any(present):
            continue
        if not all(present):
            raise ValueError("incomplete_rgbd_archive")
        if np.asarray(archive[keys[0]]).shape != image.shape[:2]:
            raise ValueError("depth_rgb_shape_mismatch")
        frames[camera] = RGBDCamera(archive[keys[0]], archive[keys[1]], archive[keys[2]])
    return frames
