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
from tests.helpers import images
from tests.test_skills import setup_runtime, COMMAND


def response(name="vla_act", **arguments):
    if not arguments:
        arguments = dict(arm="both", stop="mug_lifted", reason="Try bounded acquisition",
                         parameters={"max_chunks": 1, "chunk_size": 4, "max_policy_actions": 4})
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
        return AnthropicPlanner(AnthropicSettings(model="claude-test-model", **settings), client=provider), provider

    def test_one_call_images_limits_and_no_oracle_in_request(self):
        planner, provider = self.planner([response()])
        selected = planner.select(context())
        self.assertEqual(selected.command.action, "vla_act")
        self.assertEqual(len(provider.requests), 1)
        request = provider.requests[0]
        self.assertTrue(request["tool_choice"]["disable_parallel_tool_use"])
        self.assertEqual(len(request["tools"]), 6)
        self.assertEqual(len([b for b in request["messages"][0]["content"] if b["type"] == "image"]), 3)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(request))
        self.assertNotIn("development_oracle", json.dumps(request))
        self.assertEqual(selected.metadata["usage"]["input_tokens"], 100)
        self.assertEqual(len(selected.metadata["prompt_sha256"]), 64)

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
                expected = SkillCommand.model_validate({"action": call["name"], **call["input"]})
                self.assertEqual(parse_response(value), expected)

    def test_request_and_vla_attempt_caps_and_parameter_limits(self):
        planner, provider = self.planner([response(), response()], max_vla_attempts=1)
        self.assertEqual(planner.select(context()).command.action, "vla_act")
        self.assertEqual(planner.select(context(2)).command.action, "abort")
        planner, provider = self.planner([response()], max_calls=1)
        planner.select(context()); planner.select(context(2))
        self.assertEqual(len(provider.requests), 1)
        value = response(); value["content"][0]["input"]["parameters"]["max_chunks"] = 32
        planner, _ = self.planner([value])
        self.assertEqual(planner.select(context()).command.action, "abort")
        value = response("move_to", arm="left", parameters={"target": "invented_anchor"}, reason="stage")
        planner, _ = self.planner([value])
        self.assertEqual(planner.select(context()).command.action, "abort")

    def test_private_result_is_projected_not_copied_to_prompt(self):
        _, executor, _ = setup_runtime()
        result = executor.execute(COMMAND).model_copy(update={
            "information_source": "development_oracle", "objective_satisfied": True,
            "stop_reason": "PRIVATE_SENTINEL", "predicate": context().predicates[0]})
        data = public_result(result)
        self.assertIsNone(data["objective_satisfied"])
        self.assertIsNone(data["predicate"])
        content = user_content(context(previous=result), limits={}, history=[])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(content))
        self.assertEqual(result.stop_reason, "PRIVATE_SENTINEL")

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

    def test_sdk_receives_key_only_from_environment_with_retries_disabled(self):
        factory = unittest.mock.Mock(return_value=Provider([]))
        with patch.dict("sys.modules", {"anthropic": SimpleNamespace(Anthropic=factory)}), \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "FAKE_TEST_ONLY"}, clear=True):
            planner = AnthropicPlanner(AnthropicSettings(model="claude-test-model"))
        self.assertEqual(factory.call_args.kwargs["api_key"], "FAKE_TEST_ONLY")
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertNotIn("FAKE_TEST_ONLY", planner.settings.model_dump_json())
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            AnthropicPlanner(AnthropicSettings(model="claude-test-model"))

    def test_two_provider_turns_execute_move_then_bounded_acquisition(self):
        from harness.predicates.hanging_mug import HangingMugPredicates
        from harness.predicates.registry import PredicateRegistry
        from tests.test_analytic_skills import executor as make_executor
        from tests.test_skill_orchestrator import AcquisitionEnvironment
        environment = AcquisitionEnvironment(succeeds_after=1)
        gates = PredicateRegistry(HangingMugPredicates(environment.evidence, source="development_oracle"))
        executor, worker, events = make_executor(environment, predicates=gates, oracle=True)
        planner, provider = self.planner([
            response("move_to", arm="left", parameters={"offset_xyz": [0.0, 0.0, 0.02]}, reason="Stage"),
            response()])
        logs = []
        summary = SkillOrchestrator(executor, planner, log=logs.append,
            save_observation=lambda obs: {"sequence": obs.sequence}, episode_id="test", seed=3).run()
        self.assertEqual(summary["outcome"], "acquisition_confirmed")
        self.assertFalse(summary["task_success"])
        self.assertEqual((len(provider.requests), worker.chunks), (2, 1))
        self.assertEqual(summary["policy_actions"], 2)
        self.assertNotIn("development_oracle", json.dumps(provider.requests))
