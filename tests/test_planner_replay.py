import json
from pathlib import Path
import tempfile
import unittest

from development.planner_replay import load_turn, replay_turn, main
from harness.perception.observations import ObservationStore
from harness.planner.anthropic import AnthropicSettings
from harness.skill_config import RunSettings
from harness.skills.schemas import SkillCommand, SkillType
from tests.test_anthropic_planner import Provider, response
from tests.test_skills import setup_runtime, COMMAND


class PlannerReplayTests(unittest.TestCase):
    def fixture(self, directory):
        episode = Path(directory)
        _, executor, _ = setup_runtime()
        result = executor.execute(COMMAND)
        observation = executor.environment.observe()
        ref = ObservationStore(episode / "observations").save(observation)
        settings = RunSettings()
        manifest = dict(mode="anthropic_development", planner="anthropic",
            settings=settings.model_dump(mode="json"),
            anthropic=AnthropicSettings(model="claude-test-model").model_dump(mode="json"),
            prompt_sha256="historical-prompt", memory_snapshot_sha256=None)
        (episode / "manifest.json").write_text(json.dumps(manifest))
        rows = [dict(event_type="planner_selection", planner_turn=1,
                     command=SkillCommand.model_validate(COMMAND).model_dump(mode="json")),
                dict(event_type="planner_turn_result", planner_turn=1,
                     result=result.model_dump(mode="json"), phase="GRASP_MUG"),
                dict(event_type="planner_selection", planner_turn=2,
                     observation_before=ref, predicates_before=[])]
        (episode / "assessments.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return rows, result

    def test_one_request_reconstructs_history_budgets_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, previous = self.fixture(tmp)
            provider = Provider([response()])
            report = replay_turn(tmp, 2, client=provider)
            self.assertEqual(report["provider_calls_made"], 1)
            self.assertFalse(report["robot_motion"])
            self.assertFalse(report["simulator_started"])
            self.assertEqual(len(provider.requests), 1)
            data = json.loads(provider.requests[0]["messages"][0]["content"][-1]["text"])
            self.assertEqual(len(data["recent_commands"]), 1)
            self.assertEqual(data["remaining_budget"]["policy_actions"],
                             RunSettings().budget.max_policy_actions-previous.executed_policy_actions)
            self.assertNotIn("vla_attempts_remaining", data["runtime_limits"])
            self.assertEqual(len(report["source_hashes"]), 3)

    def test_replays_next_turn_after_recorded_safe_pre_motion_plan_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, original = self.fixture(tmp)
            ref = rows[-1]["observation_before"]
            state = original.state_before
            failure = original.model_copy(update={
                "skill": SkillType.MOVE_TO, "completed": False, "stop_reason": "planning_failure",
                "executed_policy_actions": 0, "vla_chunks_requested": 0,
                "executed_physics_ticks": 0, "state_before": state, "state_after": state,
                "observation_before": ref["sequence"], "observation_after": ref["sequence"],
                "episode_terminal": False, "task_success": False, "objective_satisfied": None,
                "controlled_arms": "left", "information_source": "public", "predicate": None,
                "execution_uncertain": False,
            })
            rows[1] = dict(event_type="planner_turn_result", planner_turn=1,
                result=failure.model_dump(mode="json"), phase="RETRY_RETREAT",
                observation_after=ref, post_condition=[{
                    "name": "mug_lifted", "value": False, "source": "runtime",
                    "observation_sequence": ref["sequence"], "reason": "replay_fixture"}])
            (Path(tmp)/"assessments.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:2]))
            provider = Provider([response()])
            report = replay_turn(tmp, 2, client=provider)
            data = json.loads(provider.requests[0]["messages"][0]["content"][-1]["text"])
            self.assertEqual(report["command"]["action"], "vla_act")
            self.assertFalse(report["robot_motion"])
            self.assertEqual(data["previous_result"]["stop_reason"], "planning_failure")
            self.assertEqual(data["previous_result"]["skill"], "move_to")
            self.assertNotIn("vla_attempts_remaining", data["runtime_limits"])

    def test_next_turn_replay_rejects_partial_or_uncertain_plan_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, original = self.fixture(tmp)
            ref = rows[-1]["observation_before"]
            base = original.model_copy(update={
                "skill": SkillType.MOVE_TO, "completed": False, "stop_reason": "planning_failure",
                "executed_policy_actions": 0, "vla_chunks_requested": 0,
                "executed_physics_ticks": 0, "state_after": original.state_before,
                "observation_after": ref["sequence"], "episode_terminal": False,
                "task_success": False, "controlled_arms": "left", "information_source": "public",
            })
            for unsafe in (base.model_copy(update={"execution_uncertain": True}),
                           base.model_copy(update={"executed_physics_ticks": 1})):
                with self.subTest(unsafe=unsafe.execution_uncertain, ticks=unsafe.executed_physics_ticks):
                    rows[1] = dict(event_type="planner_turn_result", planner_turn=1,
                        result=unsafe.model_dump(mode="json"), phase="RETRY_RETREAT",
                        observation_after=ref, post_condition=[])
                    (Path(tmp)/"assessments.jsonl").write_text(
                        "\n".join(json.dumps(r) for r in rows[:2]))
                    provider = Provider([response()])
                    with self.assertRaises(ValueError): replay_turn(tmp, 2, client=provider)
                    self.assertEqual(provider.requests, [])

    def test_invalid_source_fails_before_provider_call(self):
        for issue in ("traversal", "sequence", "terminal", "duplicate"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as tmp:
                rows, _ = self.fixture(tmp)
                if issue == "traversal": rows[-1]["observation_before"]["path"] = "../outside.npz"
                if issue == "sequence": rows[-1]["observation_before"]["sequence"] += 1
                if issue == "terminal": rows[1]["result"]["episode_terminal"] = True
                if issue == "duplicate": rows.append(rows[-1])
                (Path(tmp)/"assessments.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
                provider = Provider([])
                with self.assertRaises(ValueError): replay_turn(tmp, 2, client=provider)
                self.assertEqual(provider.requests, [])

    def test_does_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"report.json"
            output.write_text("original")
            with self.assertRaises(FileExistsError):
                main(["--episode", tmp, "--turn", "2", "--output", str(output)])
            self.assertEqual(output.read_text(), "original")
