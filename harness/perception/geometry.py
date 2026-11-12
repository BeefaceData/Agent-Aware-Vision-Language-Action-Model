import numpy as np


def pose7(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or abs(np.linalg.norm(pose[3:]) - 1.0) > 0.01):
        raise ValueError("invalid_pose")
    pose = pose.copy()
    pose[3:] /= np.linalg.norm(pose[3:])
    return pose


def rotation_to_quaternion(rotation):
    """Convert a right-handed 3x3 rotation matrix to a wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid_rotation")
    if (not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
            or np.linalg.det(matrix) < 0.999):
        raise ValueError("rotation_not_right_handed")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([(matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([(matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([(matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def pose_matrix(value):
    """Return a homogeneous transform for a world-frame wxyz pose."""
    pose = pose7(value)
    w, x, y, z = pose[3:]
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, pose[:3]
    return transform


def matrix_pose(transform):
    """Return a world-frame wxyz pose from a homogeneous transform."""
    transform = np.asarray(transform, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-7)):
        raise ValueError("invalid_transform")
    quaternion = rotation_to_quaternion(transform[:3, :3])
    return np.concatenate((transform[:3, 3], quaternion))


def planning_pose_for_tcp_target(planning_pose, tcp_pose, target_tcp_pose):
    """Convert a desired TCP pose to the pose expected by RoboTwin planning.

    RoboTwin plans a gripper control frame returned by ``get_*_ee_pose`` while
    contact predicates measure ``get_*_tcp_pose``. Preserve their measured
    rigid transform so mug-relative targets place the physical TCP, not the
    control frame, at the requested world pose.
    """
    planning = pose_matrix(planning_pose)
    tcp = pose_matrix(tcp_pose)
    target_tcp = pose_matrix(target_tcp_pose)
    planning_to_tcp = np.linalg.inv(planning) @ tcp
    return matrix_pose(target_tcp @ np.linalg.inv(planning_to_tcp))


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


def reverse_approach_waypoints(history, current_pose, retreat_distance_m, *,
                              sample_spacing_m=0.02, max_path_distance_m=0.20,
                              max_waypoints=12, start_tolerance_m=0.02):
    """Retrace history until it yields real clearance from the current TCP pose."""
    current = pose7(current_pose)
    poses = tuple(pose7(value) for value in history)
    limits = (retreat_distance_m, sample_spacing_m, max_path_distance_m,
              start_tolerance_m)
    if (not np.isfinite(limits).all() or min(limits) <= 0
            or type(max_waypoints) is not int or max_waypoints < 1):
        raise ValueError("invalid_reverse_approach_limits")
    if len(poses) < 2 or np.linalg.norm(poses[-1][:3] - current[:3]) > start_tolerance_m:
        raise ValueError("approach_history_unavailable")

    points = [current[:3]]
    points.extend(pose[:3] for pose in reversed(poses[:-1]))
    segments = []
    total_distance = 0.0
    clearance_path_distance = None
    for start, end in zip(points, points[1:]):
        distance = float(np.linalg.norm(end - start))
        if distance > 1e-6:
            segments.append((start, end, total_distance, distance))
            start_clearance = float(np.linalg.norm(start - current[:3]))
            end_clearance = float(np.linalg.norm(end - current[:3]))
            if start_clearance < retreat_distance_m <= end_clearance:
                low, high = 0.0, 1.0
                for _ in range(30):
                    fraction = (low + high) / 2
                    point = start + fraction * (end - start)
                    if np.linalg.norm(point - current[:3]) >= retreat_distance_m:
                        high = fraction
                    else:
                        low = fraction
                clearance_path_distance = total_distance + high * distance
                break
            total_distance += distance
        if total_distance >= max_path_distance_m:
            break
    if clearance_path_distance is None:
        raise ValueError("approach_history_clearance_unavailable")

    sample_count = int(np.ceil(clearance_path_distance / sample_spacing_m))
    if sample_count > max_waypoints:
        raise ValueError("reverse_approach_waypoint_limit")
    waypoints = []
    for sample in range(1, sample_count + 1):
        distance = min(sample * sample_spacing_m, clearance_path_distance)
        for start, end, distance_before, segment_length in segments:
            if distance <= distance_before + segment_length + 1e-9:
                fraction = (distance - distance_before) / segment_length
                waypoint = current.copy()
                waypoint[:3] = start + fraction * (end - start)
                waypoints.append(waypoint)
                break
    return tuple(waypoints)
