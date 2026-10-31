"""CPU-only right-arm hanging scenario using real runtime components.

The fake arm and rigid grasp are test assumptions, not a simulator checkpoint
or evidence that a physical mug is supported. No VLA or semantic service runs.
"""

import numpy as np

from harness.perception.anchors import Anchor, AnchorRegistry
from harness.perception.hang_targets import StaticRackFrameCache
from harness.perception.opus_frame import ground_public_object_frame_rgbd
from harness.perception.pose_grounding import tcp_targets_for_mug_alignment
from harness.planner.base import PlannerSelection
from harness.predicates.base import PredicateResult
from harness.predicates.rack_seat_gate import RackSeatReleaseGate
from harness.predicates.registry import PredicateRegistry
from harness.runtime.orchestrator import SkillOrchestrator
from harness.skills import SkillExecutor
from harness.skills.schemas import SkillCommand
from tests.test_analytic_skills import AnalyticEnvironment
from tests.test_hang_targets import make_observation, report, RACK_POINTS, MUG_POINTS


class NoPolicy:
    def reset_history(self):
        raise AssertionError("hanging scenario must not invoke VLA")

    def infer(self, *_args):
        raise AssertionError("hanging scenario must not invoke VLA")


class UnknownSupport:
    source = "perception"

    def planner_summary(self, observation):
        return {"source": "public_rgbd", "observation_sequence": observation.sequence,
                "status": {"support": "unknown"}, "synthetic": True}

    def evaluate(self, name, observation, *, arm="auto"):
        return PredicateResult(name=name, value=None, source="perception",
            observation_sequence=observation.sequence, reason="synthetic_support_not_assessed")


class HangingScenario:
    def __init__(self, *, grounding_error=None, drift_before_release=False,
                 seat_plan_fails=False, start_x=-0.2):
        self.environment = AnalyticEnvironment(x=start_x)
        self.environment.arms["right"]["gripper"] = 0.0
        self.anchors = AnchorRegistry()
        self.gate = RackSeatReleaseGate(self.environment)
        self.predicates = PredicateRegistry(perception=UnknownSupport(), rack_seat_gate=self.gate)
        self.events, self.assessments, self.contexts = [], [], []
        self.executor = SkillExecutor(self.environment, NoPolicy(), log=self.events.append,
                                      predicates=self.predicates, anchors=self.anchors)
        self.grounding_error = grounding_error
        self.drift_before_release = drift_before_release
        self.seat_plan_fails = seat_plan_fails
        self.prepared = []
        initial = make_observation()
        self.rack = StaticRackFrameCache()
        self.rack.cache(initial, report("rack_peg", 0, RACK_POINTS))
        self.initial_mug = ground_public_object_frame_rgbd(initial,
            report("mug_handle", 0, MUG_POINTS))
        self.initial_tcp = self.environment.arm_state("right")["pose"]
        self.commands = [
            SkillCommand(action="move_to", arm="right", parameters={"target": "rack_approach"}),
            SkillCommand(action="move_to", arm="right", parameters={"target": "rack_seat"}),
            SkillCommand(action="release", arm="right",
                         parameters={"authorization": "rack_seat_alignment", "ramp_ticks": 50}),
            SkillCommand(action="move_to", arm="right",
                         parameters={"target": "current_ee", "offset_xyz": (0., 0., .05)}),
        ]

    def prepare(self, command, observation):
        if command.action.value != "move_to" or command.parameters.target == "current_ee":
            return None
        if self.grounding_error:
            raise ValueError(self.grounding_error)
        # This no-rotation scenario assumes the mug moves rigidly with measured
        # TCP translation. Recompose from current synthetic mug/TCP each turn;
        # real mug-frame freshness/depth validation is covered in target tests.
        measured = self.environment.arm_state("right")["pose"]
        mug = self.initial_mug.copy()
        mug[:3] += measured[:3] - self.initial_tcp[:3]
        approach, seat = tcp_targets_for_mug_alignment(
            measured, mug, self.rack.resolve(make_observation(observation.sequence)))
        target = approach if command.parameters.target == "rack_approach" else seat
        anchor = Anchor(tuple(target), observation.sequence, source="perception")
        self.anchors.bind(command.parameters.target, "right", anchor)
        self.prepared.append((command.parameters.target, observation.sequence, target.copy()))
        return anchor

    def select(self, context):
        self.contexts.append(context)
        if context.previous is not None and not context.previous.completed:
            return PlannerSelection(SkillCommand(action="abort", reason="scenario_skill_failed"), "HANG")
        if context.grounding_feedback:
            return PlannerSelection(SkillCommand(action="abort", reason="scenario_grounding_rejected"), "HANG")
        index = len(self.contexts)-1
        if index >= len(self.commands):
            return PlannerSelection(SkillCommand(action="abort", reason="scenario_complete"), "HANG")
        if index == 2 and self.drift_before_release:
            self.environment.arms["right"]["pose"][0] += .04
        if index == 1 and self.seat_plan_fails:
            self.environment.plan_status = "Failure"
        return PlannerSelection(self.commands[index], "HANG")

    def run(self):
        return SkillOrchestrator(self.executor, self, log=self.assessments.append,
            save_observation=lambda observation: {"sequence": observation.sequence,
                "path": f"synthetic/{observation.sequence}.npz"},
            episode_id="synthetic-right-hang", seed=3, max_turns=6,
            completion_predicate="task_success", prepare=self.prepare).run()
