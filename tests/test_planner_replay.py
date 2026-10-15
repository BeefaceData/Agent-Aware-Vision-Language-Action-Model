import json
from pathlib import Path
import tempfile
import unittest

from development.planner_replay import load_turn, replay_turn, main
from harness.perception.observations import ObservationStore
from harness.planner.anthropic import AnthropicSettings
from harness.skill_config import RunSettings
from harness.skills.schemas import SkillCommand
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
            self.assertEqual(data["runtime_limits"]["vla_attempts_remaining"], 2)
            self.assertEqual(len(report["source_hashes"]), 3)

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
