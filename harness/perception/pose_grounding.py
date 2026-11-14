"""RGB-D keypoint frames and mug-to-gripper hang-target geometry.

This module deliberately accepts only public camera measurements and measured
TCP poses. It has no simulator imports and does not decide when a hang is safe.
"""

import numpy as np

from .geometry import (matrix_pose as _matrix_pose, pose7,
                       pose_matrix as _pose_matrix,
                       rotation_to_quaternion as _rotation_to_quaternion)
from .rgbd import RGBDCamera


def measured_peg_centerline(camera, points_uv, width_uv, *, max_depth_gap_m=.06):
    """Register a visible shaft using opposing mid-shaft silhouette samples.

    Pixel semantics remain caller evidence. No CAD pose or hidden endpoint is
    inferred. The three axis points establish orientation; the opposing width
    samples establish a centerline offset on that same observed shaft.
    """
    frame = _pose_matrix(grounded_frame_from_pixels(camera, points_uv,
        max_depth_gap_m=max_depth_gap_m))
    width_uv = np.asarray(width_uv, dtype=float)
    if width_uv.shape != (2, 2) or not np.isfinite(width_uv).all():
        raise ValueError("visible_peg_cross_section_required")
    surfaces, errors = [], []
    for u, v in width_uv:
        x, y = np.rint((u, v)).astype(int)
        h, w = camera.depth_m.shape
        if not (1 <= x < w-1 and 1 <= y < h-1):
            raise ValueError("surface_neighborhood_out_of_bounds")
        patch = camera.depth_m[y-1:y+2, x-1:x+2]
        if np.any(patch <= 0) or np.ptp(patch) > max_depth_gap_m:
            raise ValueError("surface_depth_discontinuous")
        point = np.asarray(camera.unproject(u, v, radius=1))
        error = max(np.linalg.norm(np.asarray(camera.unproject(u+du, v+dv, radius=1))-point)
                    for du, dv in ((1,0),(-1,0),(0,1),(0,-1))) + float(np.ptp(patch))
        surfaces.append(point)
        errors.append(float(error))
    axis = frame[:3, 0]
    width_vector = surfaces[1]-surfaces[0]
    axial_width = float(width_vector @ axis)
    radius = float(np.linalg.norm(width_vector-axis*axial_width)/2)
    radius_error = sum(errors)/2
    if radius <= radius_error or abs(axial_width) > sum(errors):
        raise ValueError("peg_radius_subpixel_or_wrong_cross_section")
    midpoint = (surfaces[0]+surfaces[1])/2
    origin = frame[:3, 3]
    baseline = np.linalg.norm(np.asarray(camera.unproject(*points_uv[1], radius=1))-origin)
    along = float((midpoint-origin) @ axis)
    if not radius <= along <= baseline-radius:
        raise ValueError("peg_cross_section_not_midshaft")
    correction = midpoint-origin-axis*along
    if np.linalg.norm(correction) > radius+sum(errors):
        raise ValueError("peg_cross_section_axis_inconsistent")
    frame[:3, 3] += correction
    return dict(pose_world_wxyz=_matrix_pose(frame), radius_m=radius,
        radius_uncertainty_m=radius_error, centerline_offset_m=float(np.linalg.norm(correction)),
        position_sensitivity_m=sum(errors),
        basis="current public opposing mid-shaft surfaces; no CAD registration",
        systematic_calibration_error="unknown")


def frame_from_axes(origin, x_direction, plane_direction):
    """A right-handed frame at ``origin`` as a pose: +x along ``x_direction``,
    +y towards ``plane_direction`` within their plane, +z = x cross y."""
    x_axis = np.asarray(x_direction, dtype=float)
    y_axis = np.asarray(plane_direction, dtype=float)
    if np.linalg.norm(x_axis) < 1e-9:
        raise ValueError("degenerate_frame_axes")
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis - x_axis * float(y_axis @ x_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        raise ValueError("degenerate_frame_axes")
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return _matrix_pose(np.block([
        [np.column_stack((x_axis, y_axis, z_axis)), np.asarray(origin, dtype=float).reshape(3, 1)],
        [np.zeros((1, 3)), np.ones((1, 1))],
    ]))


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
    return frame_from_axes(origin, x_axis, y_axis)


def tcp_targets_for_mug_alignment(tcp_pose, mug_function_pose, rack_function_pose,
                                  *, approach_offset_m=0.05, seat_offset_m=0.05):
    """Return prehang and aligned TCP poses from measured/grounded 6D frames.

    ``mug_function_pose`` is the visible mug handle's functional frame and
    ``rack_function_pose`` is the selected hanger peg's mating frame. RoboTwin's
    ``hanging_mug`` task uses ``pre_dis=0.05`` and ``dis=-0.05`` along the
    functional frame's local +Z axis. Reproduce those offsets: approach at
    ``-approach_offset_m`` and seat at ``+seat_offset_m``. The current TCP-to-
    mug transform is preserved, so the solution adapts to the actual grasp.
    """
    offsets = (approach_offset_m, seat_offset_m)
    if (any(isinstance(value, bool) for value in offsets)
            or not np.isfinite(offsets).all()
            or not 0.01 <= approach_offset_m <= 0.10
            or not 0.01 <= seat_offset_m <= 0.10):
        raise ValueError("hang_frame_offset_out_of_bounds")
    tcp, mug, rack = map(_pose_matrix, (tcp_pose, mug_function_pose, rack_function_pose))
    tcp_to_mug = np.linalg.inv(tcp) @ mug
    rack_z = rack[:3, 2]
    target_mug = rack.copy()
    target_mug[:3, 3] += rack_z * float(seat_offset_m)
    prehang_mug = rack.copy()
    prehang_mug[:3, 3] -= rack_z * float(approach_offset_m)
    prehang_tcp = prehang_mug @ np.linalg.inv(tcp_to_mug)
    target_tcp = target_mug @ np.linalg.inv(tcp_to_mug)
    return _matrix_pose(prehang_tcp), _matrix_pose(target_tcp)
