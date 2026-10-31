"""RGB-D keypoint frames and mug-to-gripper hang-target geometry.

This module deliberately accepts only public camera measurements and measured
TCP poses. It has no simulator imports and does not decide when a hang is safe.
"""

import numpy as np

from .geometry import pose7
from .rgbd import RGBDCamera


def _rotation_to_quaternion(rotation):
    """Convert a right-handed 3x3 rotation matrix to RoboTwin's wxyz format."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid_rotation")
    if (not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
            or np.linalg.det(matrix) < 0.999):
        raise ValueError("rotation_not_right_handed")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * scale,
                      (matrix[2, 1] - matrix[1, 2]) / scale,
                      (matrix[0, 2] - matrix[2, 0]) / scale,
                      (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                          (matrix[0, 1] + matrix[1, 0]) / scale,
                          (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.array([(matrix[0, 2] - matrix[2, 0]) / scale,
                          (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.array([(matrix[1, 0] - matrix[0, 1]) / scale,
                          (matrix[0, 2] + matrix[2, 0]) / scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    q /= np.linalg.norm(q)
    return q


def _pose_matrix(value):
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


def _matrix_pose(transform):
    transform = np.asarray(transform, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-7)):
        raise ValueError("invalid_transform")
    q = _rotation_to_quaternion(transform[:3, :3])
    return np.concatenate((transform[:3, 3], q))


def grounded_frame_from_pixels(camera, points_uv, *, max_depth_gap_m=0.06,
                               min_axis_length_m=0.008, max_axis_length_m=0.30):
    """Build a 3D frame from ordered ``origin, x-axis, xy-plane`` RGB pixels.

    The input order defines frame semantics. Each keypoint must lie on a locally
    consistent depth surface; absent, through-hole, or degenerate points fail
    closed. Pixel coordinates use full-resolution ``(u, v)`` values.
    """
    if not isinstance(camera, RGBDCamera):
        raise ValueError("rgbd_camera_required")
    limits = (max_depth_gap_m, min_axis_length_m, max_axis_length_m)
    if (not np.isfinite(limits).all() or max_depth_gap_m < 0
            or min_axis_length_m <= 0 or max_axis_length_m <= min_axis_length_m):
        raise ValueError("invalid_frame_limits")
    pixels = np.asarray(points_uv, dtype=float)
    if pixels.shape != (3, 2) or not np.isfinite(pixels).all():
        raise ValueError("three_ordered_keypoints_required")
    points = []
    for u, v in pixels:
        if camera.local_depth_gap(u, v, search_radius=8) > max_depth_gap_m:
            raise ValueError("keypoint_depth_discontinuity")
        # Unprojection uses this 3x3 median. A foreground median alone can hide
        # background pixels at a thin handle/peg edge and fabricate continuity.
        x, y = map(int, np.rint((u, v)))
        height, width = camera.depth_m.shape
        patch = camera.depth_m[max(0, y-1):min(height, y+2),
                               max(0, x-1):min(width, x+2)]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size and float(valid.max() - valid.min()) > max_depth_gap_m:
            raise ValueError("keypoint_depth_discontinuity")
        points.append(np.asarray(camera.unproject(u, v, radius=1), dtype=float))
    origin, x_point, plane_point = points
    x_axis = x_point - origin
    x_length = float(np.linalg.norm(x_axis))
    if not min_axis_length_m <= x_length <= max_axis_length_m:
        raise ValueError("x_axis_baseline_out_of_bounds")
    x_axis /= x_length
    y_axis = plane_point - origin
    y_axis -= x_axis * float(y_axis @ x_axis)
    y_length = float(np.linalg.norm(y_axis))
    if not min_axis_length_m <= y_length <= max_axis_length_m:
        raise ValueError("plane_axis_baseline_out_of_bounds")
    y_axis /= y_length
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return _matrix_pose(np.block([
        [np.column_stack((x_axis, y_axis, z_axis)), origin.reshape(3, 1)],
        [np.zeros((1, 3)), np.ones((1, 1))],
    ]))


def tcp_targets_for_mug_alignment(tcp_pose, mug_function_pose, rack_function_pose,
                                  *, approach_offset_m=0.05):
    """Return prehang and aligned TCP poses from measured/grounded 6D frames.

    ``mug_function_pose`` is the visible mug handle's functional frame and
    ``rack_function_pose`` is the selected hanger peg's mating frame. The
    current TCP-to-mug transform is preserved, so the solution adapts to the
    actual grasp rather than assuming a fixed wrist/object offset.
    """
    if (isinstance(approach_offset_m, bool) or not np.isfinite(approach_offset_m)
            or not 0.01 <= approach_offset_m <= 0.10):
        raise ValueError("approach_offset_out_of_bounds")
    tcp, mug, rack = map(_pose_matrix, (tcp_pose, mug_function_pose, rack_function_pose))
    tcp_to_mug = np.linalg.inv(tcp) @ mug
    target_mug = rack.copy()
    # The rack frame's local +X axis points from the peg seat toward its free
    # tip. Start beyond that tip, then move back along the peg to seat the
    # handle. No world-space rack coordinate is introduced.
    prehang_mug = rack.copy()
    prehang_mug[:3, 3] += rack[:3, 0] * float(approach_offset_m)
    prehang_tcp = prehang_mug @ np.linalg.inv(tcp_to_mug)
    target_tcp = target_mug @ np.linalg.inv(tcp_to_mug)
    return _matrix_pose(prehang_tcp), _matrix_pose(target_tcp)
