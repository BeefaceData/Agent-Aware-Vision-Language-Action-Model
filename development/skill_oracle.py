"""Explicit opt-in oracle for development predicates; never installed by default."""

import numpy as np
from harness.predicates.hanging_mug import HangingMugPredicates, MugEvidence
from harness.predicates.registry import PredicateRegistry
from .rack_geometry import mug_contacts


def oracle_anchor_preparer(task, anchors, *, log):
    """Re-ground a supported symbol at the current frame; never a planner tool."""
    from harness.perception.anchors import Anchor

    def prepare(command, observation):
        if command.action != "move_to" or command.parameters.target == "current_ee":
            return
        if command.parameters.target != "mug_pregrasp":
            raise ValueError("unsupported_development_anchor")
        pose = tuple(map(float, task.get_grasp_pose(task.mug, command.arm,
                                                  contact_point_id=0, pre_dis=0.05)))
        log({"event_type": "anchor_oracle_evidence", "name": "mug_pregrasp",
             "arm": command.arm, "observation_sequence": observation.sequence, "privileged_pose": pose})
        anchors.bind("mug_pregrasp", command.arm,
                     Anchor(pose, observation.sequence, source="development_oracle"))
    return prepare


def oracle_predicates(task, *, log):
    def sample(observation):
        names = mug_contacts(task)
        robot = task.robot
        data = MugEvidence(
            observation_sequence=observation.sequence,
            mug_position=tuple(map(float, task.mug.get_pose().p)),
            left_tcp=tuple(map(float, robot.get_left_tcp_pose()[:3])),
            right_tcp=tuple(map(float, robot.get_right_tcp_pose()[:3])),
            left_contact=any(n.startswith("fl_") for n in names),
            right_contact=any(n.startswith("fr_") for n in names),
            left_bilateral={"fl_link7", "fl_link8"} <= names,
            right_bilateral={"fr_link7", "fr_link8"} <= names,
            left_gripper=float(robot.get_left_gripper_val()),
            right_gripper=float(robot.get_right_gripper_val()),
            table_contact="table" in names, rack_contact=task.rack.actor.name in names,
            rack_distance_m=float(np.linalg.norm(np.asarray(task.mug.get_functional_point(0)[:3])
                                                 - task.rack.get_functional_point(0)[:3])),
        )
        log({"event_type": "predicate_oracle_evidence", "privileged": data.model_dump(mode="json")})
        return data
    return PredicateRegistry(HangingMugPredicates(sample, source="development_oracle"))
