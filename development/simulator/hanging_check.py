"""Passive, privileged hanging check; never writes actuator commands."""

from collections import deque
import numpy as np

from development.simulator.rack_geometry import rack_diagnostics


def observe_hanging(task, *, log, frame_callback=None, ticks=750, window=250,
                    maximum_motion_m=0.01):
    if (type(ticks) is not int or type(window) is not int
            or not 1 <= window <= ticks <= 2500
            or not np.isfinite(maximum_motion_m) or not 0 < maximum_motion_m <= 0.02):
        raise ValueError("invalid bounded hanging-check settings")
    samples = deque(maxlen=window)
    rack_name = task.rack.actor.name
    minimum_height = float("inf")
    for tick in range(1, ticks + 1):
        task.scene.step()
        d = rack_diagnostics(task)
        names = d["mug_contact_entities"]
        robot_contact = any(n.startswith(("fl_", "fr_", "left_", "right_")) for n in names)
        supported = (d["task_success"] and d["right_gripper"] >= 0.999
                     and rack_name in names and "table" not in names and not robot_contact)
        samples.append((supported, d["mug_position"]))
        minimum_height = min(minimum_height, d["mug_position"][2])
        log({"event": "hanging_sample", "tick": tick, "supported_without_robot": supported,
             "privileged": d})
        if frame_callback and tick % 25 == 0:
            frame_callback(tick)
    positions = np.asarray([p for _, p in samples], dtype=float)
    motion = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    stable = all(s for s, _ in samples) and np.isfinite(positions).all() and motion <= maximum_motion_m
    return {"stable_hanging": bool(stable), "observed_ticks": ticks, "window_ticks": window,
            "maximum_motion_m": motion, "motion_limit_m": maximum_motion_m,
            "supported_window_samples": sum(s for s, _ in samples),
            "minimum_mug_height_m": minimum_height, "final_privileged": d,
            "actuator_commands_written": False}
