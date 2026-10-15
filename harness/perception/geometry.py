import numpy as np


def pose7(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or abs(np.linalg.norm(pose[3:]) - 1.0) > 0.01):
        raise ValueError("invalid_pose")
    pose = pose.copy()
    pose[3:] /= np.linalg.norm(pose[3:])
    return pose


def world_yaw(pose, degrees):
    pose = pose7(pose)
    half = np.deg2rad(degrees) / 2
    c, s = np.cos(half), np.sin(half)
    w, x, y, z = pose[3:]
    pose[3:] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
    return pose


def pose_error(actual, requested):
    actual, requested = pose7(actual), pose7(requested)
    return (float(np.linalg.norm(actual[:3]-requested[:3])),
            float(2*np.arccos(np.clip(abs(actual[3:] @ requested[3:]), 0, 1))))
