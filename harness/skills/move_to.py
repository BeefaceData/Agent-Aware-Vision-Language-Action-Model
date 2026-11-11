import numpy as np
from harness.perception.geometry import pose7, reverse_approach_waypoints
from .analytic import AnalyticSkill
from .motion import SkillStopped


RETREAT_ENDPOINT_MARGIN_M = 0.02
MIN_FALLBACK_DIRECTION_DISTANCE_M = 0.02
MIN_SAFE_FALLBACK_CLEARANCE_M = 0.05
VERTICAL_FALLBACK_CLEARANCE_M = (
    MIN_SAFE_FALLBACK_CLEARANCE_M + RETREAT_ENDPOINT_MARGIN_M)


class MoveTo(AnalyticSkill):
    def prepare(self, context, command, initial):
        parameters = command.parameters
        if parameters.path_mode == "reverse_approach":
            context.reverse_retreat_fallback_level = 0
            context.last_reverse_retreat_fallback_pose = None
            context.minimum_retreat_clearance_m = parameters.retreat_distance_m
            history = context.measured_tcp_history or {}
            try:
                context.motion_waypoints = reverse_approach_waypoints(
                    history.get(command.arm, ()), initial["pose"],
                    parameters.retreat_distance_m + RETREAT_ENDPOINT_MARGIN_M)
            except ValueError as exc:
                if str(exc) not in ("approach_history_unavailable",
                                    "approach_history_clearance_unavailable"):
                    raise SkillStopped(str(exc)) from None
                target = self._retreat_fallback_target(
                    context, command, initial["pose"], str(exc))
                if target is None:
                    raise SkillStopped(str(exc)) from None
                context.reverse_retreat_fallback_level = 1
                context.last_reverse_retreat_fallback_pose = target.copy()
                context.log({"event_type": "reverse_approach_fallback",
                             "arm": command.arm,
                             "fallback": "short_retreat_toward_fresh_pregrasp",
                             "reason": str(exc),
                             "target_pose": target.tolist()})
                return target, None
            return context.motion_waypoints[-1], None
        if parameters.target == "current_ee":
            target = initial["pose"].copy()
        else:
            anchor = context.anchors.resolve(parameters.target, command.arm, context.observation,
                                             allow_oracle=context.allow_oracle)
            context.oracle_used |= anchor.source == "development_oracle"
            target = np.asarray(anchor.pose, dtype=float)
        target[:3] += parameters.offset_xyz
        if parameters.position_mode == "preserve_current":
            target[:3] = initial["pose"][:3]
        if parameters.orientation_mode == "preserve_current":
            target[3:] = initial["pose"][3:]
        return target, None

    def prepare_planning_fallback(self, context, command, measured,
                                  remaining_waypoints):
        if command.parameters.path_mode != "reverse_approach":
            return None
        fallback_level = context.reverse_retreat_fallback_level
        if fallback_level >= 2:
            return None
        remaining_waypoints = tuple(remaining_waypoints)
        if not remaining_waypoints:
            return None
        step_size_m = 0.01 if fallback_level == 0 else 0.005
        fallback_waypoints = self._subdivide_retreat_path(
            measured["pose"], remaining_waypoints, step_size_m)
        if not fallback_waypoints:
            return None
        first_target = fallback_waypoints[0]
        previous_target = context.last_reverse_retreat_fallback_pose
        context.reverse_retreat_fallback_level = fallback_level + 1
        if (previous_target is not None
                and np.linalg.norm(first_target[:3] - previous_target[:3]) < 1e-6
                and np.linalg.norm(first_target[3:] - previous_target[3:]) < 1e-6):
            context.log({"event_type": "reverse_approach_fallback_repeated_target",
                         "arm": command.arm,
                         "fallback_level": fallback_level + 1,
                         "target_pose": first_target.tolist()})
            return None
        context.last_reverse_retreat_fallback_pose = first_target.copy()
        target = np.asarray(remaining_waypoints[-1], dtype=float)
        context.minimum_retreat_clearance_m = command.parameters.retreat_distance_m
        context.log({"event_type": "reverse_approach_fallback_prepared",
                     "arm": command.arm,
                     "fallback_source": "measured_approach_path_subdivision",
                     "step_size_m": step_size_m,
                     "remaining_path_waypoints": len(remaining_waypoints),
                     "fallback_waypoint_count": len(fallback_waypoints),
                     "minimum_clearance_m": command.parameters.retreat_distance_m,
                     "target_pose": target.tolist()})
        return target, fallback_waypoints

    @staticmethod
    def _subdivide_retreat_path(start_pose, waypoints, max_segment_m):
        """Densify the remaining path while holding the measured wrist pose."""
        current = np.asarray(start_pose, dtype=float)
        fixed_orientation = current[3:].copy()
        result = []
        for target in waypoints:
            target = np.asarray(target, dtype=float)
            distance = float(np.linalg.norm(target[:3] - current[:3]))
            if distance < 1e-6:
                continue
            segment_count = int(np.ceil(distance / max_segment_m))
            for index in range(1, segment_count + 1):
                fraction = index / segment_count
                point = current + fraction * (target - current)
                point[3:] = fixed_orientation
                result.append(pose7(point))
            current = target
        return tuple(result)

    def _retreat_fallback_target(self, context, command, measured_pose, reason,
                                 *, requested_clearance_m=None):
        target = self._pregrasp_retreat_target(context, command, measured_pose,
                                               requested_clearance_m=requested_clearance_m)
        if target is not None:
            return target

        current = np.asarray(measured_pose, dtype=float)
        target = current.copy()
        target[2] += VERTICAL_FALLBACK_CLEARANCE_M
        if target[2] > 1.40:
            context.log({"event_type": "vertical_retreat_fallback_unavailable",
                         "arm": command.arm,
                         "reason": "workspace_height_limit",
                         "requested_clearance_m": VERTICAL_FALLBACK_CLEARANCE_M,
                         "current_z_m": float(current[2]),
                         "target_z_m": float(target[2])})
            return None

        context.minimum_retreat_clearance_m = MIN_SAFE_FALLBACK_CLEARANCE_M
        context.log({"event_type": "vertical_retreat_fallback_prepared",
                     "arm": command.arm,
                     "reason": reason,
                     "requested_clearance_m": VERTICAL_FALLBACK_CLEARANCE_M,
                     "minimum_clearance_m": MIN_SAFE_FALLBACK_CLEARANCE_M,
                     "orientation_mode": "preserve_current",
                     "target_pose": target.tolist()})
        return target

    def _pregrasp_retreat_target(self, context, command, measured_pose,
                                 *, requested_clearance_m=None):
        try:
            anchor = context.anchors.resolve(
                "mug_pregrasp", command.arm, context.observation,
                allow_oracle=context.allow_oracle)
        except ValueError as exc:
            context.log({"event_type": "reverse_approach_fallback_unavailable",
                         "arm": command.arm,
                         "reason": "pregrasp_anchor_unavailable",
                         "anchor_error": str(exc)[:100]})
            return None
        current_pose = np.asarray(measured_pose, dtype=float)
        pregrasp_pose = np.asarray(anchor.pose, dtype=float)
        toward_pregrasp = pregrasp_pose[:3] - current_pose[:3]
        anchor_distance = float(np.linalg.norm(toward_pregrasp))
        requested_clearance = (command.parameters.retreat_distance_m
                               if requested_clearance_m is None
                               else requested_clearance_m)
        target_clearance = min(requested_clearance + RETREAT_ENDPOINT_MARGIN_M,
                               anchor_distance)
        achievable_clearance = max(
            MIN_SAFE_FALLBACK_CLEARANCE_M,
            target_clearance - RETREAT_ENDPOINT_MARGIN_M)
        minimum_clearance = min(requested_clearance, achievable_clearance)
        if (anchor_distance < MIN_FALLBACK_DIRECTION_DISTANCE_M
                or anchor_distance + 1e-9 < minimum_clearance):
            context.log({"event_type": "reverse_approach_fallback_unavailable",
                         "arm": command.arm,
                         "reason": "pregrasp_direction_too_short",
                         "anchor_distance_m": anchor_distance,
                         "requested_clearance_m": requested_clearance,
                         "minimum_clearance_m": minimum_clearance,
                         "endpoint_margin_m": RETREAT_ENDPOINT_MARGIN_M,
                         "target_pose": pregrasp_pose.tolist()})
            return None
        target = current_pose.copy()
        target[:3] += toward_pregrasp / anchor_distance * target_clearance
        context.oracle_used |= anchor.source == "development_oracle"
        context.minimum_retreat_clearance_m = minimum_clearance
        context.log({"event_type": "reverse_approach_fallback_prepared",
                     "arm": command.arm,
                     "fallback_source": anchor.source,
                     "anchor_distance_m": anchor_distance,
                     "requested_clearance_m": requested_clearance,
                     "minimum_clearance_m": minimum_clearance,
                     "clearance_m": target_clearance,
                     "endpoint_margin_m": RETREAT_ENDPOINT_MARGIN_M,
                     "target_pose": target.tolist()})
        return target
