"""R6 contracts with injected provider; no credentials, network, or simulator."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.environment.base import Observation
from harness.planner.anthropic import AnthropicPlanner, AnthropicSettings
from harness.planner.base import PlannerContext
from harness.planner.prompt_builder import user_content, public_result
from harness.planner.response_parser import parse_response, InvalidPlannerResponse
from harness.predicates.base import PredicateResult
from harness.skill_orchestrator import SkillOrchestrator
from harness.skills.schemas import TrackingError
from tests.helpers import images
from tests.test_skills import setup_runtime, COMMAND


def response(name="vla_act", **arguments):
    defaults = (dict(arm="both", stop="chunk_budget_exhausted", reason="Try bounded acquisition",
                     parameters={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 4})
                if name == "vla_act" else {})
    arguments = {**defaults, **arguments}
    arguments.setdefault("task_phase", "grasp")
    arguments.setdefault("phase_evidence", "The current run ends at mug acquisition.")
    return {"stop_reason": "tool_use", "content": [dict(type="tool_use", id="call1", name=name, input=arguments)],
            "usage": {"input_tokens": 100, "output_tokens": 30}}


class Provider:
    def __init__(self, replies):
        self.replies, self.requests = iter(replies), []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result


def context(turn=1, previous=None):
    return PlannerContext(turn, Observation(turn, images(0), (0.0,)*14), (
        PredicateResult(name="mug_lifted", value=True, source="development_oracle",
                        observation_sequence=turn, reason="PRIVATE_SENTINEL"),), previous,
        remaining_budget={"planner_turns": 10})


class AnthropicTests(unittest.TestCase):
    def planner(self, replies, **settings):
        provider = Provider(replies)
        require_pixel_grounding = settings.pop("require_pixel_grounding", False)
        return AnthropicPlanner(AnthropicSettings(model="claude-test-model", **settings),
                                client=provider,
                                require_pixel_grounding=require_pixel_grounding), provider

    def test_one_call_images_limits_and_no_oracle_in_request(self):
        planner, provider = self.planner([response()])
        selected = planner.select(context())
        self.assertEqual(selected.command.action, "vla_act")
        self.assertEqual(len(provider.requests), 1)
        request = provider.requests[0]
        self.assertTrue(request["tool_choice"]["disable_parallel_tool_use"])
        self.assertEqual({tool["name"] for tool in request["tools"]},
                         {"vla_act", "abort"})
        vla_schema = next(tool["input_schema"]["properties"]["parameters"]
                          for tool in request["tools"] if tool["name"] == "vla_act")
        self.assertTrue({"max_chunks", "chunk_size", "max_policy_actions"}
                        <= set(vla_schema["required"]))
        self.assertTrue({"lock_arm", "failure_stop", "failure_arm"}.isdisjoint(
            vla_schema["properties"]))
        self.assertEqual(len([b for b in request["messages"][0]["content"] if b["type"] == "image"]), 3)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(request))
        self.assertNotIn("development_oracle", json.dumps(request))
        self.assertEqual(selected.metadata["usage"]["input_tokens"], 100)
        self.assertEqual(len(selected.metadata["prompt_sha256"]), 64)
        self.assertEqual(selected.metadata["task_phase"], "grasp")
        self.assertEqual(selected.metadata["phase_basis"], "planner_assessment")
        self.assertEqual(selected.metadata["phase_evidence"],
                         "The current run ends at mug acquisition.")

    def test_mug_lifted_stop_is_offered_only_with_public_perception(self):
        from dataclasses import replace

        public = replace(context(), predicates=(PredicateResult(
            name="mug_lifted", value=False, source="perception",
            observation_sequence=1, reason="public visual evidence"),))
        planner, provider = self.planner([response(stop="mug_lifted")])
        selected = planner.select(public)
        self.assertEqual(selected.command.stop, "mug_lifted")
        stops = next(tool["input_schema"]["properties"]["stop"]["enum"]
                     for tool in provider.requests[0]["tools"] if tool["name"] == "vla_act")
        self.assertEqual(stops, ["mug_lifted", "chunk_budget_exhausted"])

        planner, provider = self.planner([response(stop="mug_lifted")])
        selected = planner.select(context())
        self.assertEqual(selected.command.action.value, "abort")
        stops = next(tool["input_schema"]["properties"]["stop"]["enum"]
                     for tool in provider.requests[0]["tools"] if tool["name"] == "vla_act")
        self.assertEqual(stops, ["chunk_budget_exhausted"])

    def test_planner_accepts_only_pre_motion_move_plan_failure_for_reselection(self):
        from harness.skills.schemas import SkillResult, is_recoverable_executed_motion_failure

        state = {"command_state": [0.0] * 14}
        rejected_plan = SkillResult.model_validate({
            "call_id": 1, "skill": "move_to", "completed": False,
            "stop_reason": "planning_failure", "executed_policy_actions": 0,
            "vla_chunks_requested": 0, "discarded_actions": 0,
            "executed_physics_ticks": 0, "state_before": state, "state_after": state,
            "observation_before": 1, "observation_after": 1, "episode_terminal": False,
            "task_success": False, "controlled_arms": "left", "information_source": "public",
            "elapsed_seconds": 0.1,
        })
        planner, provider = self.planner([response()])
        selected = planner.select(context(2, previous=rejected_plan))
        self.assertEqual(selected.command.action.value, "vla_act")
        self.assertEqual(len(provider.requests), 1)
        self.assertIn('"stop_reason": "planning_failure"',
                      provider.requests[0]["messages"][0]["content"][-1]["text"])

        partial = rejected_plan.model_copy(update={
            "stop_reason": "target_not_reached", "executed_physics_ticks": 282,
            "observation_after": 2, "position_error_m": 0.034,
            "orientation_error_rad": 0.091,
            "state_after": {"command_state": [0.1] * 14},
        })
        self.assertTrue(is_recoverable_executed_motion_failure(partial))
        planner, provider = self.planner([response()])
        selected = planner.select(context(3, previous=partial))
        self.assertEqual(selected.command.action.value, "vla_act")
        result_text = provider.requests[0]["messages"][0]["content"][-1]["text"]
        self.assertIn('"stop_reason": "target_not_reached"', result_text)
        self.assertIn('"position_error_m": 0.034', result_text)
        self.assertIn('"orientation_error_rad": 0.091', result_text)

        uncertain = rejected_plan.model_copy(update={"execution_uncertain": True})
        planner, provider = self.planner([response()])
        selected = planner.select(context(2, previous=uncertain))
        self.assertEqual(selected.command.action.value, "abort")
        self.assertEqual(provider.requests, [])

        uncertain_partial = partial.model_copy(update={"execution_uncertain": True})
        planner, provider = self.planner([response()])
        self.assertEqual(planner.select(context(4, previous=uncertain_partial)).command.action.value,
                         "abort")
        self.assertEqual(provider.requests, [])

    def test_planner_reassesses_fresh_perception_gap_instead_of_auto_aborting(self):
        from harness.skills.schemas import SkillResult

        state = {"command_state": [0.0] * 14}
        gap = SkillResult.model_validate({
            "call_id": 1, "skill": "vla_act", "completed": False,
            "stop_reason": "predicate_unavailable", "executed_policy_actions": 2,
            "vla_chunks_requested": 1, "discarded_actions": 2,
            "state_before": state, "state_after": state,
            "observation_before": 1, "observation_after": 3,
            "episode_terminal": False, "task_success": False,
            "controlled_arms": "both", "information_source": "public",
            "predicate": {"name": "mug_lifted", "value": None,
                "source": "perception", "observation_sequence": 3,
                "reason": "mug_geometry_unavailable"},
            "elapsed_seconds": 0.1,
        })
        planner, provider = self.planner([response(stop="chunk_budget_exhausted")])

        selected = planner.select(context(4, previous=gap))

        self.assertEqual(selected.command.action.value, "vla_act")
        self.assertEqual(selected.command.stop, "chunk_budget_exhausted")
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual({tool["name"] for tool in provider.requests[0]["tools"]},
                         {"vla_act", "abort"})
        payload = provider.requests[0]["messages"][0]["content"][-1]["text"]
        self.assertIn('"stop_reason": "predicate_unavailable"', payload)
        self.assertIn('"value": null', payload)

    def test_unknown_milestone_abort_limit_counts_vla_feedback_only(self):
        from dataclasses import replace
        from harness.skills.schemas import SkillResult

        unknown = lambda turn, previous=None: replace(context(turn, previous=previous),
            predicates=(PredicateResult(name="mug_lifted", value=None,
                source="perception", observation_sequence=turn,
                reason="mug_geometry_unavailable"),))
        state = {"command_state": [0.0] * 14}
        def feedback(call_id):
            return SkillResult.model_validate({
                "call_id": call_id, "skill": "vla_act", "completed": True,
                "stop_reason": "chunk_budget_exhausted", "executed_policy_actions": 8,
                "vla_chunks_requested": 1, "discarded_actions": 0,
                "state_before": state, "state_after": state,
                "observation_before": call_id-1, "observation_after": call_id,
                "episode_terminal": False, "task_success": False,
                "controlled_arms": "both", "information_source": "public",
                "elapsed_seconds": 0.1})

        planner, provider = self.planner([response(), response()],
            unknown_milestone_policy="abort_after_limit",
            max_consecutive_unknown_vla_feedback=2)
        self.assertEqual(planner.select(unknown(1)).command.action.value, "vla_act")
        self.assertEqual(planner.select(unknown(2, feedback(1))).command.action.value,
                         "vla_act")
        stopped = planner.select(unknown(3, feedback(2)))
        self.assertEqual(stopped.command.action.value, "abort")
        self.assertEqual(stopped.command.reason, "public_milestone_unknown_limit")
        self.assertEqual(stopped.metadata["consecutive_unknown_vla_feedback"], 2)
        self.assertEqual(stopped.metadata["unknown_milestones"], ["mug_lifted"])
        self.assertEqual(len(provider.requests), 2)

        planner, _ = self.planner([response(), response(), response()],
            unknown_milestone_policy="abort_after_limit",
            max_consecutive_unknown_vla_feedback=2)
        planner.select(unknown(1))
        planner.select(unknown(2, feedback(1)))
        known_false = replace(context(3, previous=feedback(2)), predicates=(
            PredicateResult(name="mug_lifted", value=False, source="perception",
                observation_sequence=3, reason="visible mug still on table"),))
        self.assertEqual(planner.select(known_false).command.action.value, "vla_act")
        self.assertEqual(planner.unknown_vla_feedback, 0)

    def test_unknown_milestone_episode_cap_mode_retains_vla(self):
        from dataclasses import replace
        from harness.skills.schemas import SkillResult

        state = {"command_state": [0.0] * 14}
        previous = SkillResult.model_validate({
            "call_id": 1, "skill": "vla_act", "completed": True,
            "stop_reason": "chunk_budget_exhausted", "executed_policy_actions": 8,
            "vla_chunks_requested": 1, "discarded_actions": 0,
            "state_before": state, "state_after": state,
            "observation_before": 0, "observation_after": 1,
            "episode_terminal": False, "task_success": False,
            "controlled_arms": "both", "information_source": "public",
            "elapsed_seconds": 0.1})
        planner, provider = self.planner([response(), response(), response()],
            unknown_milestone_policy="continue_to_episode_cap",
            max_consecutive_unknown_vla_feedback=2)
        for turn in range(1, 4):
            missing = replace(context(turn, previous=previous if turn > 1 else None),
                predicates=(PredicateResult(name="mug_lifted", value=None,
                    source="perception", observation_sequence=turn,
                    reason="mug_geometry_unavailable"),))
            self.assertEqual(planner.select(missing).command.action.value, "vla_act")
        self.assertEqual(planner.unknown_vla_feedback, 2)
        self.assertEqual({tool["name"] for tool in provider.requests[2]["tools"]},
                         {"vla_act", "abort"})

    def test_planner_reassesses_mid_release_perception_dropout(self):
        from harness.skills.schemas import SkillResult

        before = {"command_state": [0.0] * 14}
        after = {"command_state": [0.0] * 13 + [0.25]}
        gap = SkillResult.model_validate({
            "call_id": 2, "skill": "release", "completed": False,
            "stop_reason": "release_support_lost", "executed_policy_actions": 0,
            "vla_chunks_requested": 0, "discarded_actions": 0,
            "executed_physics_ticks": 25,
            "state_before": before, "state_after": after,
            "observation_before": 1, "observation_after": 2,
            "episode_terminal": False, "task_success": False,
            "controlled_arms": "left", "information_source": "public",
            "predicate": {"name": "release_ready", "value": None,
                "source": "perception", "observation_sequence": 2,
                "reason": "mug_or_table_geometry_unavailable"},
            "elapsed_seconds": 0.1,
        })
        planner, provider = self.planner([response(stop="chunk_budget_exhausted")])
        selected = planner.select(context(3, previous=gap))

        self.assertEqual(selected.command.action.value, "vla_act")
        self.assertEqual(len(provider.requests), 1)
        payload = provider.requests[0]["messages"][0]["content"][-1]["text"]
        self.assertIn('"stop_reason": "release_support_lost"', payload)
        self.assertIn('"value": null', payload)

    def test_inter_arm_contact_requires_bounded_right_retreat_then_reenables_vla(self):
        from harness.skills.schemas import SkillResult

        state = {"command_state": [0.0] * 14}
        contact = SkillResult.model_validate({
            "call_id": 1, "skill": "vla_act", "completed": False,
            "stop_reason": "arm_overlap_detected", "executed_policy_actions": 1,
            "vla_chunks_requested": 1, "discarded_actions": 3,
            "state_before": state, "state_after": state,
            "observation_before": 1, "observation_after": 2,
            "episode_terminal": False, "arm_overlap_detected": True,
            "task_success": False, "controlled_arms": "both",
            "elapsed_seconds": 0.1,
        })
        retreat = {"arm": "right", "parameters": {"target": "current_ee",
            "offset_xyz": (0.05, 0.0, 0.0)}, "reason": "Retreat away from the left arm",
            "task_phase": "handoff", "phase_evidence": "The arms are in contact."}
        planner, provider = self.planner([response("move_to", **retreat)], task_scope="full_task")
        selected = planner.select(context(2, previous=contact))
        self.assertEqual(selected.command.action.value, "move_to")
        self.assertEqual(selected.command.arm, "right")
        self.assertEqual(len(provider.requests), 1)
        planner, provider = self.planner([response()], task_scope="full_task")
        selected = planner.select(context(2, previous=contact))
        self.assertEqual(selected.command.action.value, "abort")
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual({tool["name"] for tool in provider.requests[0]["tools"]},
                         {"move_to", "abort"})

        separated = contact.model_copy(update={"call_id": 2, "skill": __import__(
            "harness.skills.schemas", fromlist=["SkillType"]).SkillType.MOVE_TO,
            "completed": True, "stop_reason": "completed", "arm_overlap_detected": False,
            "executed_policy_actions": 0, "vla_chunks_requested": 0,
            "discarded_actions": 0, "observation_before": 2, "observation_after": 3})
        planner, provider = self.planner([response("move_to", **retreat), response(task_phase="handoff",
            phase_evidence="The right arm has separated and the cup remains at handoff.")],
            task_scope="full_task")
        retreat_selected = planner.select(context(2, previous=contact))
        self.assertEqual(retreat_selected.command.action.value, "move_to")
        selected = planner.select(context(3, previous=separated))
        self.assertEqual(selected.command.action.value, "vla_act")
        self.assertEqual(len(provider.requests), 2)

    def test_prompt_accounts_for_orientation_preserving_analytic_staging(self):
        planner, provider = self.planner([response()])
        planner.select(context())
        system = provider.requests[0]["system"]
        self.assertIn("MOVE_TO preserves the measured gripper orientation", system)
        self.assertIn("Start with VLA_ACT from the current pose", system)
        self.assertIn("approach and contact phase", system)
        self.assertIn("ROTATE_WRIST only changes world-Z", system)
        self.assertIn("not treat it as a general 3D grasp-pose correction", system)
        self.assertIn("zero analytic ticks does not", system)

    def test_prompt_requires_public_release_authorization(self):
        planner, provider = self.planner([response()], task_scope="full_task")
        planner.select(context())
        system = provider.requests[0]["system"]
        self.assertIn("prompt-visible `release_ready=true`", system)
        self.assertIn("Do not infer", system)
        self.assertIn("empty hand or support", system)

    def test_release_tools_require_public_ready_predicate(self):
        from dataclasses import replace

        ready = replace(context(), predicates=(PredicateResult(
            name="release_ready", value=True, source="perception",
            observation_sequence=1, reason="public support evidence"),
            PredicateResult(name="grasp_attempt_failed", value=True,
                source="perception", observation_sequence=1,
                reason="visible failed grasp")))
        planner, provider = self.planner([response()])
        planner.select(ready)
        tools = {tool["name"]: tool for tool in provider.requests[0]["tools"]}
        self.assertIn("release", tools)
        self.assertEqual(tools["set_gripper"]["input_schema"]["properties"]
                         ["parameters"]["properties"]["value"]["maximum"], 1.0)

        unauthorized_open = response("release", arm="right", parameters={}, reason="Try opening")
        planner, _ = self.planner([unauthorized_open])
        self.assertEqual(planner.select(context()).command.action.value, "abort")

        authorized_open = response("release", arm="right", parameters={}, reason="Release with support")
        planner, _ = self.planner([response(), authorized_open])
        planner.select(context())
        self.assertEqual(planner.select(ready).command.action.value, "release")

    def test_grounded_rack_seat_offers_only_explicit_alignment_release(self):
        from dataclasses import replace

        context_with_alignment = replace(context(), predicates=(PredicateResult(
            name="rack_seat_aligned", value=True, source="runtime",
            observation_sequence=1, reason="measured target alignment"),
            PredicateResult(name="left_transfer_complete", value=True,
                source="perception", observation_sequence=1,
                reason="public transfer evidence")))
        aligned_release = response("release", arm="right",
            parameters={"authorization": "rack_seat_alignment"},
            reason="Open only after reaching the grounded rack seat.",
            task_phase="hang", phase_evidence="The right wrist reached the grounded peg pose.")
        planner, provider = self.planner([aligned_release], task_scope="full_task")
        planner.active_phase = "hang"

        selected = planner.select(context_with_alignment)

        self.assertEqual(selected.command.action.value, "release")
        self.assertEqual(selected.command.parameters.authorization, "rack_seat_alignment")
        tool = next(item for item in provider.requests[0]["tools"]
                    if item["name"] == "release")
        self.assertEqual(tool["input_schema"]["properties"]["parameters"]
                         ["properties"]["authorization"]["enum"],
                         ["rack_seat_alignment"])

    def test_full_task_locks_right_only_during_initial_left_grasp(self):
        grasp = response(task_phase="grasp", phase_evidence="The left arm is beginning mug acquisition.",
            stop="chunk_budget_exhausted")
        hang = response(task_phase="hang", phase_evidence="The cup is on the table and the left arm has cleared.",
            stop="task_success")
        planner, provider = self.planner([grasp, hang], task_scope="full_task")
        first = planner.select(context())
        self.assertEqual(first.phase, "GRASP")
        self.assertEqual(first.command.stop, "chunk_budget_exhausted")
        self.assertEqual(first.command.parameters.lock_arm, "right")
        self.assertIn("strict rack hanging", provider.requests[0]["system"])
        self.assertEqual(next(tool for tool in provider.requests[0]["tools"]
                              if tool["name"] == "vla_act")["input_schema"]["properties"]["stop"]["enum"],
                         ["chunk_budget_exhausted"])

        from dataclasses import replace
        transferred = replace(context(2), predicates=(
            PredicateResult(name="left_transfer_complete", value=True, source="perception",
                observation_sequence=2, reason="public visual evidence"),))
        second = planner.select(transferred)
        self.assertEqual(second.phase, "HANG")
        self.assertEqual(second.command.stop, "task_success")
        self.assertIsNone(second.command.parameters.lock_arm)

    def test_full_task_normal_turn_hides_recovery_skills(self):
        planner, provider = self.planner([response()], task_scope="full_task")
        selected = planner.select(context())
        self.assertEqual(selected.command.action.value, "vla_act")
        self.assertEqual({tool["name"] for tool in provider.requests[0]["tools"]},
                         {"vla_act", "abort"})
        payload = json.loads(provider.requests[0]["messages"][0]["content"][-1]["text"])
        self.assertFalse(payload["runtime_limits"]["analytic_skills_available"])
        self.assertIsNone(payload["runtime_limits"]["recovery_basis"])
        self.assertEqual(payload["available_targets"], [])

        invalid = response("move_to", arm="left", parameters={"target": "current_ee"},
                           reason="Move before recovery is eligible", task_phase="grasp",
                           phase_evidence="The cup is untouched.")
        planner, _ = self.planner([invalid], task_scope="full_task")
        rejected = planner.select(context())
        self.assertEqual(rejected.command.action.value, "abort")
        self.assertEqual(rejected.metadata["validation_error"],
                         "analytic_skill_not_recovery_eligible")

    def test_acquisition_scope_cannot_claim_later_phase(self):
        planner, provider = self.planner([response(task_phase="handoff",
            phase_evidence="The right arm appears ready.")])
        selection = planner.select(context())
        self.assertEqual(selection.command.action.value, "abort")
        self.assertEqual(selection.metadata["validation_error"], "phase_outside_task_scope")
        tool = next(item for item in provider.requests[0]["tools"] if item["name"] == "vla_act")
        self.assertEqual(tool["input_schema"]["properties"]["task_phase"]["enum"], ["grasp"])

    def test_full_task_targets_are_only_exposed_when_public_rgbd_grounded(self):
        from dataclasses import replace
        visual = {"source": "public_rgbd", "status": {"mug": "grounded", "rack": "grounded",
                    "table": "not_detected"}}
        transferred = (PredicateResult(name="left_transfer_complete", value=True,
            source="perception", observation_sequence=1, reason="public transfer evidence"),)
        grounded_context = replace(context(), visual_context=visual, predicates=transferred)
        payload = json.loads(user_content(grounded_context,
            limits={"task_scope": "full_task"}, history=[])[-1]["text"])
        self.assertEqual(payload["available_targets"],
                         ["current_ee", "mug_pregrasp", "rack_approach"])
        cached_visual = {**visual, "rack_static_reference": {
            "world_anchor_available": True, "current_visibility": "not_detected"}}
        cached_context = replace(context(), visual_context=cached_visual)
        cached_payload = json.loads(user_content(cached_context,
            limits={"task_scope": "full_task", "vla_phase": "hang"},
            history=[])[-1]["text"])
        self.assertIn("rack_seat", cached_payload["available_targets"])
        pixel_payload = json.loads(user_content(context(),
            limits={"task_scope": "full_task"}, history=[], pixel_grounding=True)[-1]["text"])
        self.assertIn("rack_approach", pixel_payload["available_targets"])
        self.assertIn("support bar", pixel_payload["pixel_coordinate_convention"])
        move = response("move_to", arm="right", parameters={"target": "rack_approach"},
            reason="Move the held mug to the grounded rack approach", task_phase="hang",
            phase_evidence="The left arm has cleared the scene.")
        planner, _ = self.planner([move], task_scope="full_task")
        self.assertEqual(planner.select(grounded_context).command.parameters.target, "rack_approach")

        planner, _ = self.planner([move], task_scope="full_task")
        rejected = planner.select(replace(context(), predicates=transferred))
        self.assertEqual(rejected.command.action.value, "abort")
        self.assertEqual(rejected.metadata["validation_error"], "unsupported_anchor")
        self.assertTrue(planner._target_available("rack_seat", cached_context,
                                                   arm="right", phase="hang"))

    def test_handoff_is_bimanual_and_staging_alone_advances_to_hang(self):
        from dataclasses import replace

        for is_staged, claimed_phase, expected_phase in (
                (False, "handoff", "HANDOFF"),
                (False, "hang", "HANDOFF"),
                (True, "hang", "HANG")):
            planner, _ = self.planner([response(task_phase="hang", stop="task_success",
                phase_evidence="The mug is staged and the VLA coordinates both arms.")],
                task_scope="full_task")
            planner.active_phase = "handoff"
            predicates = []
            if is_staged:
                predicates.append(PredicateResult(name="mug_staged_on_table", value=True,
                    source="perception", observation_sequence=1,
                    reason="public lift then table support"))
            ctx = replace(context(), predicates=tuple(predicates))
            selected = planner.select(ctx)
            self.assertEqual(selected.phase, expected_phase)
            self.assertIsNone(selected.command.parameters.lock_arm)

    def test_public_staged_mug_alone_advances_phase_without_oracle(self):
        from dataclasses import replace
        predicates = (PredicateResult(name="mug_staged_on_table", value=True,
            source="perception", observation_sequence=1,
            reason="public lift then table support"),)
        ctx = replace(context(), predicates=predicates)
        self.assertEqual(AnthropicPlanner._public_phase_floor(ctx, "full_task"), "hang")
        self.assertTrue(AnthropicPlanner._handoff_complete_confirmed(ctx))

    def test_planner_payload_contains_public_transfer_evidence_but_no_oracle_predicates(self):
        from dataclasses import replace
        predicates = (
            PredicateResult(name="mug_staged_on_table", value=True,
                source="perception", observation_sequence=1,
                reason="public_lift_then_table_support"),
            PredicateResult(name="left_transfer_complete", value=True,
                source="development_oracle", observation_sequence=1,
                reason="PRIVATE_TRANSFER_SENTINEL"),
        )
        ctx = replace(context(), predicates=predicates)
        payload = json.loads(user_content(ctx, limits={"task_scope": "full_task"},
                                          history=[])[-1]["text"])
        projected = {item["name"]: item for item in payload["predicates"]}
        self.assertTrue(projected["mug_staged_on_table"]["value"])
        self.assertNotIn("left_arm_clear", projected)
        self.assertNotIn("left_transfer_complete", projected)
        self.assertNotIn("PRIVATE_TRANSFER_SENTINEL", json.dumps(payload))

    def test_handoff_prompt_preserves_coordinated_two_arm_motion(self):
        from dataclasses import replace
        planner, provider = self.planner([response(task_phase="handoff",
            stop="chunk_budget_exhausted",
            phase_evidence="The left arm is withdrawing as the right starts approaching." )],
            task_scope="full_task")
        handoff_context = replace(context(), predicates=(PredicateResult(
            name="mug_lifted", value=True, source="perception",
            observation_sequence=1, reason="public lift evidence"),))
        selected = planner.select(handoff_context)
        self.assertEqual(selected.phase, "HANDOFF")
        self.assertIsNone(selected.command.parameters.lock_arm)
        prompt = provider.requests[0]["system"]
        self.assertIn("both arms without per-arm locking", prompt)
        self.assertIn("initial rack approach", prompt)
        self.assertIn("RoboTwin reports direct", prompt)
        self.assertIn("left-right arm contact", prompt)

    def test_left_home_clearance_recovery_is_not_exposed(self):
        from dataclasses import replace

        ctx = replace(context(), predicates=(PredicateResult(
            name="left_arm_clear", value=False, source="runtime",
            observation_sequence=1, reason="measured reset-pose distance"),
            PredicateResult(name="grasp_attempt_failed", value=True,
                source="perception", observation_sequence=1,
                reason="public retry evidence")))
        planner, _ = self.planner([response("move_to", arm="left",
            parameters={"target": "left_home"}, reason="Clear the workspace",
            task_phase="hang", phase_evidence="The mug is supported and the left arm is not clear.")],
            task_scope="full_task")
        planner.active_phase = "hang"
        self.assertFalse(planner._target_available("left_home", ctx, arm="left", phase="hang"))
        self.assertFalse(planner._target_available("left_home", ctx, arm="left", phase="handoff"))
        self.assertFalse(planner._target_available("left_home", context(), arm="left", phase="hang"))
        payload = json.loads(user_content(ctx,
            limits={"task_scope": "full_task", "vla_phase": "hang"}, history=[])[-1]["text"])
        self.assertNotIn("left_home", payload["available_targets"])
        selected = planner.select(ctx)
        self.assertEqual(selected.command.action.value, "abort")
        self.assertEqual(selected.metadata["validation_error"], "unsupported_anchor")

    def test_hang_starts_with_vla_but_recovery_skills_are_available(self):
        from dataclasses import replace
        staged = (PredicateResult(name="mug_staged_on_table", value=True,
            source="perception", observation_sequence=1,
            reason="public lift then table support"),)
        first_reply = response(task_phase="hang", stop="chunk_budget_exhausted",
            phase_evidence="The right arm begins pickup and rack approach.")
        move = response("move_to", arm="right", parameters={"target": "rack_approach"},
            reason="Correct the observed approach pose", task_phase="hang",
            phase_evidence="After one VLA chunk, the held mug is visibly off the grounded rack approach.")
        planner, provider = self.planner([first_reply, move], task_scope="full_task")
        visual = {"source": "public_rgbd", "status": {"mug": "grounded", "rack": "grounded"},
                  "rack_static_reference": {"world_anchor_available": True}}
        first = planner.select(replace(context(), predicates=staged, visual_context=visual))
        self.assertEqual(first.command.action.value, "vla_act")
        self.assertIsNone(first.command.parameters.lock_arm)
        tools = {tool["name"] for tool in provider.requests[0]["tools"]}
        self.assertIn("vla_act", tools)
        self.assertIn("move_to", tools)

        previous = SimpleNamespace(episode_terminal=False, execution_uncertain=False,
            completed=True, call_id=1, skill=SimpleNamespace(value="vla_act"),
            executed_policy_actions=8, vla_chunks_requested=1, executed_physics_ticks=0,
            discarded_actions=0, controlled_arms="both", policy_history_reset=False,
            information_source="public", stop_reason="chunk_budget_exhausted",
            objective_satisfied=False, position_error_m=None, orientation_error_rad=None,
            tracking_error=None, predicate=None)
        second = planner.select(replace(context(2, previous=previous), predicates=staged,
                                        visual_context=visual))
        self.assertEqual(second.command.action.value, "move_to")
        self.assertEqual(planner._recovery_basis(
            replace(context(3), predicates=staged, visual_context=visual), "hang"),
            "hang_phase_recovery_available")

    def test_full_task_vla_stop_must_match_selected_phase(self):
        invalid = response(task_phase="hang", stop="mug_lifted",
            phase_evidence="The mug is on the rack and task is complete.")
        planner, _ = self.planner([invalid], task_scope="full_task")
        from dataclasses import replace
        completed_transfer = replace(context(), predicates=(
            PredicateResult(name="left_transfer_complete", value=True, source="perception",
                observation_sequence=1, reason="visually verified transfer"),))
        selected = planner.select(completed_transfer)
        self.assertEqual(selected.command.action.value, "abort")
        self.assertEqual(selected.metadata["validation_error"],
                         "vla_call_outside_task_phase_limits")

    def test_pixel_grounding_uses_existing_opus_call_and_rejects_missing_point(self):
        grounded = response("move_to", arm="left", reason="Stage at visible mug surface",
            parameters={"target": "mug_pregrasp", "grounding": {
                "camera": "head_camera", "uv": [0.45, 0.62]}})
        planner, provider = self.planner([grounded], require_pixel_grounding=True)
        failed_grasp = __import__("dataclasses").replace(context(), predicates=(
            PredicateResult(name="grasp_attempt_failed", value=True,
                source="perception", observation_sequence=1,
                reason="visible failed grasp"),))
        selection = planner.select(failed_grasp)
        self.assertEqual(selection.command.parameters.grounding.camera, "head_camera")
        self.assertEqual(len(provider.requests), 1)
        request_text = provider.requests[0]["messages"][0]["content"][-1]["text"]
        self.assertTrue(json.loads(request_text)["pixel_grounding_required"])
        self.assertIn("image_sizes_px", json.loads(request_text))

        missing = response("move_to", arm="left", reason="Stage", parameters={"target": "mug_pregrasp"})
        planner, _ = self.planner([missing], require_pixel_grounding=True)
        rejected = planner.select(failed_grasp)
        self.assertEqual(rejected.command.action, "abort")
        self.assertEqual(rejected.metadata["validation_error"], "missing_grounding_point")

    def test_full_task_rack_pixel_is_required_and_accepted_without_a_sam_mask(self):
        selected = response("move_to", arm="right", reason="Stage at the visible rack bar",
            parameters={"target": "rack_approach", "grounding": {
                "camera": "right_camera", "uv": [0.52, 0.38]}},
            task_phase="hang", phase_evidence="The transfer is complete and the rack is visible.")
        planner, _ = self.planner([selected], task_scope="full_task",
                                  require_pixel_grounding=True)
        transferred = __import__("dataclasses").replace(context(), predicates=(
            PredicateResult(name="left_transfer_complete", value=True,
                source="perception", observation_sequence=1,
                reason="public transfer evidence"),))
        result = planner.select(transferred)
        self.assertEqual(result.command.parameters.target, "rack_approach")
        self.assertEqual(result.command.parameters.grounding.camera, "right_camera")

        missing = response("move_to", arm="right", reason="Stage at rack",
            parameters={"target": "rack_approach"}, task_phase="hang",
            phase_evidence="The rack is visible.")
        planner, _ = self.planner([missing], task_scope="full_task",
                                  require_pixel_grounding=True)
        rejected = planner.select(transferred)
        self.assertEqual(rejected.command.action.value, "abort")
        self.assertEqual(rejected.metadata["validation_error"], "missing_grounding_point")

    def test_cached_static_rack_anchor_allows_symbolic_staging_during_hang(self):
        from dataclasses import replace
        visual = {"source": "public_rgbd", "status": {"rack": "not_detected"},
                  "rack_static_reference": {"initialized": True,
                      "world_anchor_available": True, "current_visibility": "not_detected"}}
        transferred = replace(context(), visual_context=visual, predicates=(
            PredicateResult(name="left_transfer_complete", value=True, source="perception",
                observation_sequence=1, reason="public transfer evidence"),))
        stage = response("move_to", arm="right", parameters={"target": "rack_approach"},
            task_phase="hang", phase_evidence="The transfer is complete; stage for hanging.",
            reason="Move to the runtime cached rack staging reference.")
        planner, provider = self.planner([stage], task_scope="full_task",
                                         require_pixel_grounding=True)

        selected = planner.select(transferred)

        self.assertEqual(selected.command.action.value, "move_to")
        self.assertIsNone(selected.command.parameters.grounding)
        payload = json.loads(user_content(transferred,
            limits={"task_scope": "full_task", "vla_phase": "hang"}, history=[],
            pixel_grounding=True)[-1]["text"])
        self.assertIn("rack_approach", payload["available_targets"])
        self.assertTrue(payload["rack_pixel_grounding_exempt"])
        self.assertIn("world_anchor_available", json.dumps(provider.requests[0]["messages"]))

    def test_grounding_reselection_feedback_is_structured_and_contains_no_coordinates(self):
        from dataclasses import replace
        text = user_content(replace(context(), grounding_feedback=(
            "The previous mug pixel failed a depth consistency check. Select a visibly solid point."
        ), grounding_camera_to_avoid="head_camera"), limits={}, history=[])[-1]["text"]
        payload = json.loads(text)
        self.assertEqual(payload["grounding_feedback"],
            "The previous mug pixel failed a depth consistency check. Select a visibly solid point.")
        self.assertNotIn("uv", payload["grounding_feedback"])
        self.assertEqual(payload["grounding_camera_to_avoid"], "head_camera")

    def test_reselection_prompt_drops_rejected_pixel_from_recent_command_history(self):
        from dataclasses import replace
        first = response("move_to", arm="left", reason="Stage",
            parameters={"target": "mug_pregrasp", "grounding": {
                "camera": "head_camera", "uv": [0.27, 0.52]}})
        planner, provider = self.planner([first, response("abort", arm="auto", parameters={},
                                                           reason="No reliable point")],
                                          require_pixel_grounding=True)
        failed_grasp = replace(context(), predicates=(PredicateResult(
            name="grasp_attempt_failed", value=True, source="perception",
            observation_sequence=1, reason="visible failed grasp"),))
        planner.select(failed_grasp)
        feedback = "Select a solid point on the outer mug wall."
        planner.select(PlannerContext(2, context(2).observation, context(2).predicates, None,
                                      grounding_feedback=feedback))
        payload = json.loads(provider.requests[1]["messages"][0]["content"][-1]["text"])
        self.assertEqual(payload["grounding_feedback"], feedback)
        self.assertEqual(payload["recent_commands"], [])
        self.assertNotIn("0.27", json.dumps(payload))

    def test_parser_rejects_multiple_unknown_raw_actions_and_truncation(self):
        variants = []
        value = response(); value["content"] *= 2; variants.append(value)
        value = response(); value["stop_reason"] = "max_tokens"; variants.append(value)
        variants.append(response("teleport"))
        value = response(); value["content"][0]["input"]["parameters"]["joint_targets"] = [0]*14; variants.append(value)
        value = response(); value["content"][0]["input"]["action"] = "abort"; variants.append(value)
        variants.append({"stop_reason": "end_turn", "content": [{"type": "text", "text": "move_to"}]})
        for value in variants:
            with self.subTest(value=value), self.assertRaises(InvalidPlannerResponse):
                parse_response(value)
        for bad_phase in ("unknown", "hang" * 100):
            value = response(task_phase=bad_phase)
            with self.subTest(bad_phase=bad_phase), self.assertRaises(InvalidPlannerResponse):
                parse_response(value)

    def test_overlong_reason_is_clipped_without_discarding_valid_skill(self):
        from harness.planner.response_parser import parse_planner_response

        value = response(reason="A" * 1400)
        parsed = parse_planner_response(value)
        self.assertEqual(parsed.command.reason, "A" * 1000)
        self.assertTrue(parsed.reason_truncated)
        self.assertEqual(len(value["content"][0]["input"]["reason"]), 1400)
        planner, _ = self.planner([value])
        selected = planner.select(context())
        self.assertEqual(selected.command.action.value, "vla_act")
        self.assertTrue(selected.metadata["reason_truncated_to_schema_limit"])

    def test_provider_error_is_sanitized_abort_without_retry(self):
        planner, provider = self.planner([TimeoutError("SECRET_SENTINEL")])
        selected = planner.select(context())
        self.assertEqual(selected.command.action, "abort")
        self.assertNotIn("SECRET_SENTINEL", str(selected))
        self.assertEqual(planner.select(context(2)).command.action, "abort")
        self.assertEqual(len(provider.requests), 1)

    def test_rejected_response_retains_only_safe_diagnostics(self):
        value = response()
        value["stop_reason"] = "max_tokens"
        value["content"] = [{"type": "text", "text": "SECRET_SENTINEL"}]
        planner, provider = self.planner([value])
        selected = planner.select(context())
        self.assertEqual(selected.command.action, "abort")
        self.assertEqual(selected.metadata["provider_stop_reason"], "max_tokens")
        self.assertEqual(selected.metadata["validation_error"], "expected_completed_tool_call")
        self.assertEqual(selected.metadata["usage"]["output_tokens"], 30)
        self.assertNotIn("SECRET_SENTINEL", str(selected))
        self.assertEqual(len(provider.requests), 1)

    def test_diagnostic_codes_and_stop_reason_cannot_leak_provider_text(self):
        value = response(); value["stop_reason"] = "SECRET_SENTINEL"
        for rejected in (value, InvalidPlannerResponse("SECRET_SENTINEL")):
            with self.subTest(rejected=type(rejected).__name__):
                planner, _ = self.planner([rejected])
                selected = planner.select(context())
                self.assertEqual(selected.command.action, "abort")
                self.assertNotIn("SECRET_SENTINEL", str(selected))

    def test_parameter_errors_have_safe_prefixed_paths(self):
        bad = response("set_gripper", arm="left", parameters={"value": "SECRET_SENTINEL"}, reason="open")
        planner, _ = self.planner([bad])
        selected = planner.select(context())
        self.assertEqual(selected.command.action, "abort")
        self.assertEqual(selected.metadata["validation_issues"],
                         [{"path": ["parameters", "value"], "code": "float_type"}])
        self.assertNotIn("SECRET_SENTINEL", str(selected))

    def test_validation_diagnostics_bound_and_redact_unknown_fields(self):
        bad = response("move_to", arm="left", parameters={
            **{f"SECRET_SENTINEL_{i}": "PRIVATE_VALUE" for i in range(20)}}, reason="stage")
        planner, _ = self.planner([bad])
        selected = planner.select(context())
        self.assertEqual(len(selected.metadata["validation_issues"]), 8)
        self.assertTrue(all(issue == {"path": ["parameters", "unknown_field"], "code": "extra_forbidden"}
                            for issue in selected.metadata["validation_issues"]))
        self.assertNotIn("SECRET_SENTINEL", str(selected))
        self.assertNotIn("PRIVATE_VALUE", str(selected))
        bad = response("move_to", arm="left", parameters={"offset_xyz": [0.2, 0.2, 0.]}, reason="stage")
        planner, _ = self.planner([bad])
        selected = planner.select(context())
        self.assertEqual(selected.metadata["validation_issues"],
                         [{"path": ["parameters"], "code": "value_error"}])

    def test_command_field_validation_and_malformed_tool_name(self):
        bad = response("move_to", arm="SECRET_SENTINEL", parameters={}, reason="stage")
        planner, _ = self.planner([bad])
        selected = planner.select(context())
        self.assertEqual(selected.metadata["validation_issues"],
                         [{"path": ["arm"], "code": "literal_error"}])
        self.assertNotIn("SECRET_SENTINEL", str(selected))
        bad = response(); bad["content"][0]["name"] = {"SECRET_SENTINEL": 1}
        with self.assertRaises(InvalidPlannerResponse): parse_response(bad)

    def test_all_six_valid_tool_contracts_remain_accepted(self):
        cases = [response(), response("abort", arm="auto", parameters={}, reason="stop"),
                 response("move_to", arm="left", parameters={"offset_xyz": [0., 0., 0.05]}, reason="stage"),
                 response("rotate_wrist", arm="right", parameters={"delta_yaw_deg": 10.}, reason="rotate"),
                 response("set_gripper", arm="left", parameters={"value": 1.}, reason="open"),
                 response("release", arm="right", parameters={"ramp_ticks": 200}, reason="release")]
        from harness.skills.schemas import SkillCommand
        for value in cases:
            call = value["content"][0]
            with self.subTest(skill=call["name"]):
                expected = SkillCommand.model_validate({"action": call["name"], **{
                    key: value for key, value in call["input"].items()
                    if key not in ("task_phase", "phase_evidence")}})
                self.assertEqual(parse_response(value), expected)

    def test_pixel_grounding_tool_schema_is_inline_and_normalized(self):
        from harness.planner.response_parser import tool_schemas
        schema = next(item["input_schema"] for item in tool_schemas()
                      if item["name"] == "move_to")["properties"]["parameters"]
        self.assertNotIn("$defs", schema)
        self.assertNotIn("$ref", json.dumps(schema))
        pixel = schema["properties"]["grounding"]
        self.assertEqual(pixel["type"], "object")
        self.assertEqual(pixel["properties"]["camera"]["enum"],
                         ["head_camera", "left_camera", "right_camera"])
        uv = pixel["properties"]["uv"]
        self.assertEqual(uv["minItems"], 2)
        self.assertEqual(uv["maxItems"], 2)
        self.assertEqual(uv["prefixItems"][0]["minimum"], 0.)
        self.assertEqual(uv["prefixItems"][1]["maximum"], 1.)

    def test_planner_call_cap_and_vla_parameter_limits(self):
        planner, provider = self.planner([response(), response()])
        self.assertEqual(planner.select(context()).command.action, "vla_act")
        self.assertEqual(planner.select(context(2)).command.action, "vla_act")
        self.assertIn("vla_act", [tool["name"] for tool in provider.requests[1]["tools"]])
        planner, provider = self.planner([response()], max_calls=1)
        planner.select(context()); planner.select(context(2))
        self.assertEqual(len(provider.requests), 1)
        value = response(); value["content"][0]["input"]["parameters"]["max_chunks"] = 32
        planner, _ = self.planner([value])
        self.assertEqual(planner.select(context()).command.action, "abort")
        value = response("move_to", arm="left", parameters={"target": "invented_anchor"}, reason="stage")
        planner, _ = self.planner([value])
        self.assertEqual(planner.select(context()).command.action, "abort")

    def test_vla_has_no_per_phase_attempt_cap(self):
        from dataclasses import replace

        planner, provider = self.planner(
            [response(), response(), response()], task_scope="full_task")
        self.assertEqual(planner.select(context(1)).command.action.value, "vla_act")
        self.assertEqual(planner.select(context(2)).command.action.value, "vla_act")
        handoff = replace(context(3), predicates=(PredicateResult(
            name="mug_lifted", value=True, source="perception",
            observation_sequence=3, reason="visual evidence"),))
        self.assertEqual(planner.select(handoff).command.action.value, "vla_act")
        limits = json.loads(provider.requests[2]["messages"][0]["content"][-1]["text"])
        self.assertEqual(limits["runtime_limits"]["vla_phase"], "handoff")
        self.assertNotIn("vla_attempts_remaining", limits["runtime_limits"])
        self.assertNotIn("vla_phase_attempts_remaining", limits["runtime_limits"])

    def test_planner_phase_advances_from_one_tool_call_and_never_regresses(self):
        first_grasp = response(task_phase="grasp", stop="chunk_budget_exhausted",
            phase_evidence="The left arm is acquiring the mug.")
        later = response(task_phase="handoff", stop="chunk_budget_exhausted",
            phase_evidence="The left arm is placing the lifted mug on the table.")
        next_phase = response(task_phase="hang", stop="task_success",
            phase_evidence="The left transfer is complete and the right arm is active.")
        earlier = response("abort", arm="auto", parameters={},
            reason="Stop after uncertain handoff", task_phase="grasp",
            phase_evidence="The right arm is now active.")
        from types import SimpleNamespace
        previous = SimpleNamespace(
            episode_terminal=False, execution_uncertain=False, completed=True,
            call_id=1, skill=SimpleNamespace(value="vla_act"),
            executed_policy_actions=4, vla_chunks_requested=1,
            executed_physics_ticks=0, discarded_actions=0,
            controlled_arms="both", policy_history_reset=False,
            information_source="public", stop_reason="chunk_budget_exhausted",
            objective_satisfied=False, position_error_m=None,
            orientation_error_rad=None, tracking_error=None, predicate=None)
        planner, provider = self.planner([first_grasp, later, next_phase, earlier],
            task_scope="full_task")
        first = planner.select(context())
        second = planner.select(context(2, previous=previous))
        third = planner.select(context(3, previous=previous))
        fourth = planner.select(context(4, previous=previous))
        self.assertEqual(first.metadata["task_phase"], "grasp")
        self.assertEqual(first.metadata["phase_basis"], "planner_assessment")
        self.assertEqual(second.metadata["task_phase"], "handoff")
        self.assertEqual(third.metadata["task_phase"], "handoff")
        self.assertEqual(fourth.metadata["task_phase"], "handoff")
        self.assertTrue(fourth.metadata["phase_claim_corrected"])
        self.assertEqual(len(provider.requests), 4)

    def test_public_milestone_is_phase_floor_even_if_planner_claims_earlier_phase(self):
        move = response("vla_act", task_phase="grasp", stop="chunk_budget_exhausted",
            phase_evidence="RGB-D confirms the left transfer is complete.")
        planner, _ = self.planner([move], task_scope="full_task")
        handoff = __import__("dataclasses").replace(context(), predicates=(
            PredicateResult(name="left_transfer_complete", value=True, source="perception",
                observation_sequence=1, reason="visually verified transfer"),))
        selected = planner.select(handoff)
        self.assertEqual(selected.command.action.value, "vla_act")
        self.assertEqual(selected.metadata["task_phase"], "hang")
        self.assertEqual(selected.metadata["phase_basis"], "public_milestone")
        self.assertTrue(selected.metadata["phase_claim_corrected"])

    def test_acquisition_returns_at_feedback_boundary_and_opus_can_select_recovery(self):
        from harness.skills.schemas import SkillResult

        first = response()
        fallback = response("set_gripper", arm="left", parameters={"value": 1.0},
                            reason="Open the supported mug before re-staging")
        retry = response()
        planner, provider = self.planner([first, fallback, retry, fallback])

        initial = planner.select(context())
        self.assertEqual(initial.command.parameters.max_chunks, 1)
        self.assertIsNone(initial.command.parameters.failure_stop)
        limits = json.loads(provider.requests[0]["messages"][0]["content"][-1]["text"])
        self.assertEqual(limits["runtime_limits"]["vla"]["max_chunks"], 1)
        self.assertNotIn("failure_stop", limits["runtime_limits"]["vla"])

        state = {"command_state": [0.0] * 14}
        failed_attempt = SkillResult.model_validate({
            "call_id": 1, "skill": "vla_act", "completed": True,
            "stop_reason": "chunk_budget_exhausted", "executed_policy_actions": 8,
            "vla_chunks_requested": 1, "discarded_actions": 0,
            "state_before": state, "state_after": state, "observation_before": 0,
            "observation_after": 8, "episode_terminal": False, "task_success": False,
            "controlled_arms": "both", "information_source": "public",
            "elapsed_seconds": 1.0,
        })
        release_ready = __import__("dataclasses").replace(context(41, previous=failed_attempt),
            predicates=(PredicateResult(name="release_ready", value=True, source="perception",
                observation_sequence=41, reason="public support evidence"),
                PredicateResult(name="grasp_attempt_failed", value=True,
                    source="perception", observation_sequence=41,
                    reason="visible failed grasp")))
        selected_fallback = planner.select(release_ready)
        self.assertEqual(selected_fallback.command.action.value, "set_gripper")
        self.assertEqual(selected_fallback.command.parameters.value, 1.0)
        request_payload = json.dumps(provider.requests[1])
        self.assertNotIn("development_oracle", request_payload)
        self.assertNotIn("failure_predicate_satisfied", request_payload)

        opened = SkillResult.model_validate({**failed_attempt.model_dump(), "call_id": 2,
            "skill": "set_gripper",
            "stop_reason": "completed", "executed_policy_actions": 0,
            "vla_chunks_requested": 0, "discarded_actions": 0,
            "controlled_arms": "left", "information_source": "public"})
        selected_retry = planner.select(context(42, previous=opened))
        self.assertEqual(selected_retry.command.action.value, "vla_act")
        self.assertIsNone(selected_retry.command.parameters.failure_stop)
        self.assertEqual(planner.attempts, 2)
        self.assertIn("vla_act", [tool["name"] for tool in provider.requests[2]["tools"]])
        self.assertIn("fresh images", provider.requests[0]["system"])

    def test_planner_can_select_grounded_analytic_recovery_after_grasp_failure(self):
        from harness.skills.schemas import SkillResult

        staged_pixel = response("move_to", arm="left", reason="Use a fresh visible mug point",
            parameters={"target": "mug_pregrasp", "grounding": {
                "camera": "head_camera", "uv": [.46, .61]}})
        close_gripper = response("set_gripper", arm="left", reason="Close after visual alignment",
            parameters={"value": 0.0})
        lift = response("move_to", arm="left", reason="Lift the aligned grasp",
            parameters={"target": "current_ee", "offset_xyz": [0.0, 0.0, .06]})
        planner, provider = self.planner([response(), staged_pixel, close_gripper, lift],
            require_pixel_grounding=True)

        state = {"command_state": [0.0] * 14}
        def result(call_id, skill, sequence):
            return SkillResult.model_validate({
                "call_id": call_id, "skill": skill, "completed": True,
                "stop_reason": "chunk_budget_exhausted" if skill == "vla_act" else "completed",
                "executed_policy_actions": 4 if skill == "vla_act" else 0,
                "vla_chunks_requested": 1 if skill == "vla_act" else 0,
                "discarded_actions": 0, "state_before": state, "state_after": state,
                "observation_before": sequence-1, "observation_after": sequence,
                "episode_terminal": False, "task_success": False,
                "controlled_arms": "both" if skill == "vla_act" else "left",
                "information_source": "public", "elapsed_seconds": 0.01})

        first = planner.select(context(1))
        self.assertEqual(first.command.action.value, "vla_act")
        failed_grasp = __import__("dataclasses").replace(
            context(2, previous=result(1, "vla_act", 2)), predicates=(
                PredicateResult(name="grasp_attempt_failed", value=True,
                    source="perception", observation_sequence=2,
                    reason="visible failed grasp"),))
        second = planner.select(failed_grasp)
        self.assertEqual(second.command.action.value, "move_to")
        self.assertEqual(second.command.parameters.target, "mug_pregrasp")
        self.assertEqual(second.command.parameters.grounding.uv, (.46, .61))
        third = planner.select(context(3, previous=result(2, "move_to", 3)))
        self.assertEqual(third.command.action.value, "set_gripper")
        self.assertEqual(third.command.parameters.value, 0.0)
        fourth = planner.select(context(4, previous=result(3, "set_gripper", 4)))
        self.assertEqual(fourth.command.action.value, "move_to")
        self.assertEqual(fourth.command.parameters.offset_xyz, (0.0, 0.0, .06))
        self.assertEqual(planner.attempts, 1)
        self.assertEqual(len(provider.requests), 4)
        self.assertTrue(all(len([block for block in request["messages"][0]["content"]
                                if block["type"] == "image"]) == 3
                            for request in provider.requests))

    def test_private_result_is_projected_not_copied_to_prompt(self):
        _, executor, _ = setup_runtime()
        result = executor.execute(COMMAND).model_copy(update={
            "information_source": "development_oracle", "objective_satisfied": True,
            "stop_reason": "PRIVATE_SENTINEL", "predicate": context().predicates[0],
            "position_error_m": 0.034, "orientation_error_rad": 0.091})
        data = public_result(result)
        self.assertIsNone(data["objective_satisfied"])
        self.assertIsNone(data["predicate"])
        self.assertIsNone(data["position_error_m"])
        self.assertIsNone(data["orientation_error_rad"])
        self.assertEqual(data["motion_accounting"]["physics_ticks_scope"], "analytic_skills_only")
        self.assertFalse(data["motion_accounting"]["zero_physics_ticks_means_no_vla_motion"])
        self.assertEqual(data["motion_accounting"]["policy_actions_confirmed"],
                         result.executed_policy_actions)
        content = user_content(context(previous=result), limits={}, history=[])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(content))
        self.assertEqual(result.stop_reason, "PRIVATE_SENTINEL")

    def test_tracking_error_diagnostic_is_available_to_planner(self):
        _, executor, _ = setup_runtime()
        diagnostic = TrackingError(
            arm="left", comparison="controlled_arm", joint_index=2,
            measured_position_rad=3.08, commanded_position_rad=2.83,
            residual_rad=0.25, threshold_rad=0.25)
        result = executor.execute(COMMAND).model_copy(update={"tracking_error": diagnostic})
        projected = public_result(result)
        self.assertEqual(projected["tracking_error"]["joint_index"], 2)
        self.assertAlmostEqual(projected["tracking_error"]["residual_rad"], 0.25)

    def test_invalid_model_call_becomes_only_abort_in_real_executor(self):
        runtime, executor, events = setup_runtime()
        bad = response(); bad["content"] *= 2
        planner, _ = self.planner([bad])
        logs = []
        controller = SkillOrchestrator(executor, planner, log=logs.append,
            save_observation=lambda obs: {"sequence": obs.sequence}, episode_id="test", seed=3)
        summary = controller.run()
        self.assertEqual(summary["outcome"], "planner_request_or_validation_failed")
        self.assertEqual((runtime.chunks, runtime.actions), (0, []))
        self.assertEqual(events[-1]["result"]["skill"], "abort")

    def test_sdk_receives_key_only_from_environment_with_bounded_transport_retries(self):
        factory = unittest.mock.Mock(return_value=Provider([]))
        with patch.dict("sys.modules", {"anthropic": SimpleNamespace(Anthropic=factory)}), \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "FAKE_TEST_ONLY"}, clear=True):
            planner = AnthropicPlanner(AnthropicSettings(model="claude-test-model"))
        self.assertEqual(factory.call_args.kwargs["api_key"], "FAKE_TEST_ONLY")
        self.assertEqual(factory.call_args.kwargs["max_retries"], 3)
        self.assertEqual(AnthropicSettings(model="claude-test-model", transport_retries=0).transport_retries, 0)
        with self.assertRaises(ValueError):
            AnthropicSettings(model="claude-test-model", transport_retries=6)
        self.assertNotIn("FAKE_TEST_ONLY", planner.settings.model_dump_json())
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            AnthropicPlanner(AnthropicSettings(model="claude-test-model"))

    def test_private_latched_predicate_does_not_cause_zero_action_vla_handback(self):
        from harness.predicates.hanging_mug import HangingMugPredicates
        from harness.predicates.registry import PredicateRegistry
        from tests.test_analytic_skills import executor as make_executor
        from tests.test_skill_orchestrator import AcquisitionEnvironment
        environment = AcquisitionEnvironment(succeeds_after=1)
        gates = PredicateRegistry(HangingMugPredicates(environment.evidence, source="development_oracle"))
        executor, worker, events = make_executor(environment, predicates=gates, oracle=True)
        planner, provider = self.planner([
            response(), response("abort", arm="auto", parameters={}, reason="End test episode")])
        logs = []
        summary = SkillOrchestrator(executor, planner, log=logs.append,
            save_observation=lambda obs: {"sequence": obs.sequence}, episode_id="test", seed=3).run()
        # The development oracle's already-latched lift value is private to the
        # planner; it must not stop VLA before any policy action is executed.
        self.assertEqual(summary["outcome"], "End test episode")
        self.assertFalse(summary["task_success"])
        self.assertEqual((len(provider.requests), worker.chunks), (2, 1))
        self.assertEqual(summary["policy_actions"], 4)
        self.assertNotIn("development_oracle", json.dumps(provider.requests))
