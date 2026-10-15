"""Offline rack diagnostics and release geometry. Never supervisor observations."""

import numpy as np

from harness.trace import vector14


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 0.01:
        raise ValueError("finite normalized position/quaternion pose required")
    w, x, y, z = pose[3:] / np.linalg.norm(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = pose[:3]
    return matrix


def mug_contacts(task):
    names = set()
    for contact in task.scene.get_contacts():
        pair = [body.entity.name for body in contact.bodies]
        if task.mug.actor.name in pair and contact.points:
            names.update(name for name in pair if name != task.mug.actor.name)
    return names


def rack_diagnostics(task):
    mug_fp = np.asarray(task.mug.get_functional_point(0), dtype=float)
    rack_fp = np.asarray(task.rack.get_functional_point(0), dtype=float)
    mug_transform, rack_transform = pose_matrix(mug_fp), pose_matrix(rack_fp)
    rack_position = np.asarray(task.rack.get_pose().p, dtype=float)
    midpoint = (rack_position + rack_fp[:3]) / 2
    # Functional frames should align, as in hanging_mug.play_once's constrain='align'.
    angle = float(np.arccos(np.clip((np.trace(rack_transform[:3, :3].T @ mug_transform[:3, :3])-1)/2, -1, 1)))
    contacts = mug_contacts(task)
    return {
        "mug_position": list(map(float, task.mug.get_pose().p)),
        "mug_functional_pose": mug_fp.tolist(), "rack_functional_pose": rack_fp.tolist(),
        "rack_midpoint": midpoint.tolist(),
        "handle_rack_distance_m": float(np.linalg.norm(mug_fp[:3] - midpoint)),
        "handle_xy_error_m": float(np.max(np.abs(mug_fp[:2] - midpoint[:2]))),
        "handle_frame_error_rad": angle,
        "right_gripper": float(task.robot.get_right_gripper_val()),
        "right_bilateral_contact": {"fr_link7", "fr_link8"} <= contacts,
        "left_contact": any(name.startswith("fl_") for name in contacts),
        "mug_contact_entities": sorted(contacts), "task_success": bool(task.check_success()),
    }


def premature_release_candidate(diagnostics, state, proposed_action, *, minimum_distance=0.10):
    state, action = vector14(state), vector14(proposed_action)
    return bool(
        diagnostics["right_bilateral_contact"] and not diagnostics["left_contact"]
        and diagnostics["mug_position"][2] > 0.78 and not diagnostics["task_success"]
        and diagnostics["handle_rack_distance_m"] > minimum_distance
        and state[13] <= 0.25 and action[13] > max(0.1, state[13] + 0.05)
    )


def release_geometry_ready(diagnostics, *, xy_tolerance=0.018, distance_tolerance=0.06,
                           angle_tolerance=0.25):
    tip_distance = float(np.linalg.norm(np.asarray(diagnostics["mug_functional_pose"][:3])
                                       - diagnostics["rack_functional_pose"][:3]))
    values = [diagnostics["handle_xy_error_m"], tip_distance,
              diagnostics["handle_frame_error_rad"], diagnostics["mug_functional_pose"][2]]
    return bool(np.isfinite(values).all() and values[0] <= xy_tolerance
                and values[1] <= distance_tolerance and values[2] <= angle_tolerance
                and values[3] > 0.86)


def supported_release_ready(diagnostics, rack_name):
    """Allow gravity seating along the rack only after release was authorized.

    Entry still uses release_geometry_ready. During opening, distance to the tip
    grows as the handle seats; require actual rack contact and a bounded segment
    corridor, plus the original lateral/orientation/height constraints.
    """
    contacts = diagnostics["mug_contact_entities"]
    if rack_name not in contacts or "table" in contacts or diagnostics["left_contact"]:
        return False
    point = np.asarray(diagnostics["mug_functional_pose"][:3], dtype=float)
    tip = np.asarray(diagnostics["rack_functional_pose"][:3], dtype=float)
    midpoint = np.asarray(diagnostics["rack_midpoint"], dtype=float)
    segment = midpoint - tip
    length_squared = float(segment @ segment)
    if not np.isfinite([*point, *tip, *midpoint]).all() or not 1e-6 < length_squared <= 0.04:
        return False
    fraction = float((point - tip) @ segment / length_squared)
    lateral = float(np.linalg.norm(point - (tip + fraction * segment)))
    return bool(0 <= fraction <= 1 and lateral <= 0.018
                and diagnostics["handle_xy_error_m"] <= 0.018
                and diagnostics["handle_frame_error_rad"] <= 0.25 and point[2] > 0.86)
