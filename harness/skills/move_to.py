import numpy as np

from harness.perception.anchors import MUG_RECOVERY_MIN_RISE_M
from harness.perception.geometry import (
    bounded_pose_waypoints, pose7, reverse_approach_waypoints,
)
from .analytic import AnalyticSkill
from .motion import SkillStopped


class MoveTo(AnalyticSkill):
    def prepare(self, context, command, initial):
        parameters = command.parameters
        if parameters.path_mode == "reverse_approach":
            return self._prepare_native_preapproach_return(context, command, initial)

        if parameters.target == "current_ee":
            target = initial["pose"].copy()
        else:
            anchor = context.anchors.resolve(
                parameters.target, command.arm, context.observation,
                allow_oracle=context.allow_oracle)
            context.oracle_used |= anchor.source == "development_oracle"
            target = np.asarray(anchor.pose, dtype=float)
        target[:3] += parameters.offset_xyz
        if parameters.position_mode == "preserve_current":
            target[:3] = initial["pose"][:3]
        if parameters.orientation_mode == "preserve_current":
            target[3:] = initial["pose"][3:]
        if parameters.target == "mug_recovery_clearance":
            return self._prepare_grounded_clearance(context, command, initial, target)
        return target, None

    def _prepare_grounded_clearance(self, context, command, initial, target):
        """Rise, move above the grounded mug, then descend to pre-approach."""
        start = pose7(initial["pose"])
        target = pose7(target)
        # The grounded endpoint already includes its safe TCP clearance.
        # Require a rise from the measured start, but do not add that rise a
        # second time above an endpoint that is already higher.
        clearance_z = max(start[2] + MUG_RECOVERY_MIN_RISE_M, target[2])

        vertical = start.copy()
        vertical[2] = clearance_z
        overhead = target.copy()
        overhead[2] = clearance_z
        try:
            lateral = bounded_pose_waypoints(
                vertical, overhead, max_segment_m=0.12,
                max_total_m=1.30, max_waypoints=12)
            descent = bounded_pose_waypoints(
                overhead, target, max_segment_m=0.12,
                max_total_m=0.65, max_waypoints=6)
        except ValueError as exc:
            raise SkillStopped(str(exc)) from None
        waypoints = self._deduplicate_waypoints((vertical, *lateral, *descent))
        context.motion_waypoints = waypoints
        context.log({
            "event_type": "grounded_recovery_path_prepared",
            "arm": command.arm,
            "minimum_vertical_rise_m": MUG_RECOVERY_MIN_RISE_M,
            "waypoint_count": len(waypoints),
            "vertical_clearance_pose": vertical.tolist(),
            "grounded_preapproach_pose": target.tolist(),
        })
        return target, None

    @staticmethod
    def _deduplicate_waypoints(waypoints):
        result = []
        for waypoint in waypoints:
            pose = pose7(waypoint)
            if not result or not np.allclose(result[-1], pose, atol=1e-9, rtol=0.0):
                result.append(pose)
        return tuple(result)

    def _prepare_native_preapproach_return(self, context, command, initial):
        """Resolve a retreat path; every waypoint is subsequently native-planned."""
        parameters = command.parameters
        context.minimum_retreat_clearance_m = parameters.retreat_distance_m
        history = context.measured_tcp_history or {}
        approach_history = tuple(history.get(command.arm, ()))
        try:
            translation_waypoints = reverse_approach_waypoints(
                approach_history, initial["pose"],
                parameters.retreat_distance_m)
            source = "measured_tcp_history"
        except ValueError as exc:
            if str(exc) not in ("approach_history_unavailable",
                                "approach_history_clearance_unavailable"):
                raise SkillStopped(str(exc)) from None
            translation_waypoints, source = self._fresh_preapproach_waypoints(
                context, command, initial, history_error=str(exc))

        context.reverse_retreat_translation_waypoints = len(translation_waypoints)
        context.motion_waypoints = tuple(pose7(point) for point in translation_waypoints)
        if not context.motion_waypoints:
            raise SkillStopped("preapproach_path_unavailable")
        context.log({"event_type": "native_preapproach_return_prepared",
                     "arm": command.arm,
                     "source": source,
                     "requested_clearance_m": parameters.retreat_distance_m,
                     "waypoint_count": len(context.motion_waypoints),
                     "target_pose": context.motion_waypoints[-1].tolist()})
        return context.motion_waypoints[-1], None

    def _fresh_preapproach_waypoints(self, context, command, initial, *, history_error):
        """Use a freshly grounded preapproach pose when measured history is absent."""
        try:
            anchor = context.anchors.resolve(
                "mug_pregrasp", command.arm, context.observation,
                allow_oracle=context.allow_oracle)
        except ValueError as exc:
            context.log({"event_type": "native_preapproach_unavailable",
                         "arm": command.arm,
                         "history_error": history_error,
                         "anchor_error": str(exc)[:100]})
            raise SkillStopped("preapproach_target_unavailable") from None

        target = pose7(anchor.pose)
        distance = float(np.linalg.norm(target[:3] - initial["pose"][:3]))
        if distance + 1e-9 < command.parameters.retreat_distance_m:
            raise SkillStopped("preapproach_clearance_unavailable")
        context.oracle_used |= anchor.source == "development_oracle"
        translation_target = target.copy()
        translation_target[3:] = initial["pose"][3:]
        context.log({"event_type": "fresh_preapproach_selected",
                     "arm": command.arm,
                     "source": anchor.source,
                     "history_error": history_error,
                     "distance_m": distance,
                     "target_pose": target.tolist()})
        return (translation_target,), anchor.source
