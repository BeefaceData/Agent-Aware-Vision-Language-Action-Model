"""One development-only left-arm missed-grasp recovery; never runtime eligible."""

from pathlib import Path
from typing import Annotated, Literal
from pydantic import Field, StrictFloat

from .base import RecoverySkill
from harness.legacy.schemas import FailureType, TaskPhase, SkillResult
from harness.schemas import Model
from harness.skills.motion import SkillStopped


class RetryGraspParameters(Model):
    arm: Literal["left"] = "left"
    backoff_scale: Annotated[StrictFloat, Field(ge=0.01, le=0.08)] = 0.05
    lift_distance: Annotated[StrictFloat, Field(ge=0.01, le=0.05)] = 0.02


class RetryGrasp(RecoverySkill):
    skill_id = "retry_grasp"
    version = 2
    description = "Development-only left-arm regrasp of a table-supported mug after missed acquisition."
    supported_phases = frozenset({TaskPhase.GRASP_MUG})
    supported_failures = frozenset({FailureType.MISSED_GRASP})
    parameter_model = RetryGraspParameters
    max_steps = 3000  # Physics ticks, at 250 Hz; distinct from policy target calls.
    interruption_conditions = ("task_complete", "safety_violation", "step_budget", "planning_failure", "cancelled")
    candidate_only = True
    runtime_observation_compatible = False  # Uses mug contact geometry via offline context.
    returns_to_policy = False  # Manual experiment ends; policy resume belongs to Phase 7.
    _root = Path(__file__).resolve().parents[2]
    dependency_files = (
        _root / "skills/motion.py", _root.parent / "development/recovery/retry_grasp_context.py",
        _root.parent / "RoboTwin/envs/_base_task.py",
        _root.parent / "RoboTwin/envs/robot/robot.py",
        _root.parent / "RoboTwin/envs/robot/planner.py",
        _root.parent / "RoboTwin/envs/utils/actor_utils.py",
        _root.parent / "RoboTwin/envs/utils/transforms.py",
        _root.parent / "RoboTwin/assets/embodiments/aloha-agilex/config.yml",
        _root.parent / "RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
    )

    def execute(self, context, parameters):
        parameters = self.validate_parameters(parameters)
        if context is None or not getattr(context, "development_only", False):
            raise ValueError("retry_grasp requires an explicit development context")
        context.begin(self.max_steps)
        reason, error_type = "completed", None
        try:
            context.preconditions()
            context.gripper("open", 1.0)
            context.offset("retreat", parameters["backoff_scale"])
            pregrasp, grasp = context.grasp_targets(parameters["backoff_scale"])
            context.move("pregrasp", pregrasp)
            context.move("approach", grasp)
            context.gripper("close", 0.0)
            context.offset("lift", parameters["lift_distance"])
            context.settle()
        except SkillStopped as exc:
            reason = exc.reason
        except Exception as exc:
            reason, error_type = "execution_error", type(exc).__name__
        finally:
            # Zero velocity targets without advancing physics; no resume on failure.
            context.stop()
        return SkillResult(
            skill_id=self.skill_id, executed_steps=context.executed_ticks,
            success_hint=None, termination_reason=reason, final_state=context.command_state(),
            metadata={"step_unit": "physics_tick", "development_only": True,
                      "requires_policy_refresh": True, "error_type": error_type or "none"},
        )
