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


def bounded_pose_waypoints(start, target, *, max_segment_m=0.20,
                           max_rotation_rad=np.deg2rad(45), max_total_m=0.65,
                           max_waypoints=4):
    """Interpolate a bounded Cartesian pose move from the measured start.

    Positions use linear interpolation and rotations use shortest-path SLERP.
    Callers must plan each returned pose from newly measured robot joints.
    """
    start, target = pose7(start), pose7(target)
    limits = (max_segment_m, max_rotation_rad, max_total_m)
    if (not np.isfinite(limits).all() or min(limits) <= 0
            or type(max_waypoints) is not int or max_waypoints < 1):
        raise ValueError("invalid_waypoint_limits")
    distance = float(np.linalg.norm(target[:3]-start[:3]))
    if distance > max_total_m + 1e-9:
        raise ValueError("pose_waypoint_limits")

    q0, q1 = start[3:].copy(), target[3:].copy()
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * np.arccos(dot)
    segments = max(1, int(np.ceil(distance/max_segment_m - 1e-12)),
                   int(np.ceil(angle/max_rotation_rad - 1e-12)))
    if segments > max_waypoints:
        raise ValueError("pose_waypoint_limits")

    waypoints = []
    for index in range(1, segments+1):
        fraction = index / segments
        pose = np.empty(7, dtype=float)
        pose[:3] = start[:3] + fraction*(target[:3]-start[:3])
        if dot > 0.9995:
            quaternion = (1-fraction)*q0 + fraction*q1
            quaternion /= np.linalg.norm(quaternion)
        else:
            theta = np.arccos(dot)
            denominator = np.sin(theta)
            quaternion = (np.sin((1-fraction)*theta)/denominator)*q0
            quaternion += (np.sin(fraction*theta)/denominator)*q1
        pose[3:] = quaternion
        waypoints.append(pose)
    return tuple(waypoints)
