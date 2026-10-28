"""Fixed cameras added to the RoboTwin scene for perception only; the policy never sees them.

They are added after ``setup_demo`` has placed the mug and rack, so they draw no
random numbers and every seed keeps its exact scene (cameras listed in RoboTwin's
``static_camera_list`` draw four random numbers each and change the cup a seed
produces). Pose maths is plain NumPy; SAPIEN is imported only when a camera is
added to a live scene.
"""

from dataclasses import dataclass

import numpy as np

from .layout import PERCEPTION_CAMERAS

# RoboTwin's own camera clipping planes (envs/camera/camera.py).
NEAR_M, FAR_M = 0.1, 100.0
WORLD_UP = np.array((0.0, 0.0, 1.0))


@dataclass(frozen=True)
class FixedCamera:
    name: str
    position_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    fovy_deg: float
    width: int = 320
    height: int = 240

    def __post_init__(self):
        position, target = np.asarray(self.position_m, float), np.asarray(self.target_m, float)
        if self.name not in PERCEPTION_CAMERAS:
            raise ValueError("unknown_perception_camera")
        if position.shape != (3,) or target.shape != (3,) or not np.isfinite([*position, *target]).all():
            raise ValueError("invalid_perception_camera_pose")
        if not 10 <= self.fovy_deg <= 120 or not 32 <= self.width <= 1920 or not 32 <= self.height <= 1080:
            raise ValueError("invalid_perception_camera_lens")
        camera_to_world(position, target)  # rejects a target on top of the camera or straight below it


def parse_camera_spec(text):
    """``"default"`` or ``"x,y,z/tx,ty,tz/fovy_deg"`` (metres, degrees) -> ``FixedCamera``.

    One string so the same value works on every capture command line, Modal included.
    Pass it as ``--perception-camera=SPEC`` when it starts with a minus sign, or
    argparse reads it as an option.
    """
    if text == "default":
        return DEFAULT_PERCEPTION_CAMERA
    try:
        position, target, fovy = text.split("/")
        position, target = (tuple(float(value) for value in part.split(",")) for part in (position, target))
        return FixedCamera(DEFAULT_PERCEPTION_CAMERA.name, position, target, float(fovy))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"invalid_perception_camera_spec: expected 'default' or 'x,y,z/tx,ty,tz/fovy', "
                         f"got {text!r}") from error


def describe(spec):
    """JSON-ready settings for manifests and reports."""
    return {"name": spec.name, "position_m": list(spec.position_m), "target_m": list(spec.target_m),
            "fovy_deg": spec.fovy_deg, "image_wh": [spec.width, spec.height]}


def camera_to_world(position_m, target_m):
    """4x4 SAPIEN camera pose looking from ``position_m`` at ``target_m``.

    SAPIEN camera axes are x forward, y left, z up (the same columns RoboTwin
    builds in ``create_camera``).
    """
    position, target = np.asarray(position_m, float), np.asarray(target_m, float)
    forward = target - position
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("invalid_perception_camera_pose")
    forward /= np.linalg.norm(forward)
    left = np.cross(WORLD_UP, forward)
    if np.linalg.norm(left) < 1e-6:
        raise ValueError("invalid_perception_camera_pose")
    left /= np.linalg.norm(left)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    pose[:3, 3] = position
    return pose


# Far side of the table and to the left, looking diagonally back at the arms.
# The head camera looks down from behind the arms, so the grippers hide a held
# mug from it. Chosen with development.capture.scene_views on seeds 0, 2 and 3:
# the mug, both grippers, the handoff area and the rack are all in view. A
# camera behind the rack, (0.35, 0.40, 1.15), let the rack fill the foreground.
DEFAULT_PERCEPTION_CAMERA = FixedCamera("perception_camera", position_m=(-0.30, 0.45, 1.20),
                                        target_m=(0.0, -0.10, 0.80), fovy_deg=60.0)


def add_fixed_camera(scene, spec):
    """Add ``spec`` to a live SAPIEN scene and return the camera handle."""
    import sapien.core as sapien

    camera = scene.add_camera(name=spec.name, width=spec.width, height=spec.height,
                              fovy=np.deg2rad(spec.fovy_deg), near=NEAR_M, far=FAR_M)
    camera.entity.set_pose(sapien.Pose(camera_to_world(spec.position_m, spec.target_m)))
    return camera


def fixed_camera_payload(name, camera):
    """Render one fixed camera into the observation payload format of ``observation_of``.

    Call after ``task.get_obs()``, which syncs the renderer with the physics state.
    Depth follows RoboTwin's convention: millimetres along the camera axis.
    """
    camera.take_picture()
    rgba = np.asarray(camera.get_picture("Color"))
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    depth_mm = -np.asarray(camera.get_picture("Position"))[..., 2] * 1000.0
    return {name: np.ascontiguousarray(rgb),
            f"{name}__depth_mm": np.ascontiguousarray(depth_mm, dtype=np.float32),
            f"{name}__intrinsic_cv": np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64),
            f"{name}__extrinsic_cv": np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64)}
