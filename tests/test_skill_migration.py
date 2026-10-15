"""Terminology migration preserves old evidence without weakening validation."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from harness.skills.schemas import EpisodeBudget, SkillResult
from harness.memory import MemoryCatalog, MemorySnapshot
from harness.memory.snapshot import canonical
from harness.exploration.candidates import candidates_from_attempt
from tests.test_memory import rule
from tests.test_skills import setup_runtime, COMMAND
from tests.test_exploration import FakeAttemptRunner


class SkillMigrationTests(unittest.TestCase):
    def test_legacy_budget_and_result_keys_read_but_new_keys_written(self):
        budget = EpisodeBudget.model_validate({"max_primitive_calls": 3})
        self.assertEqual(budget.max_skill_calls, 3)
        self.assertNotIn("max_primitive_calls", budget.model_dump())
        with self.assertRaises(ValueError):
            EpisodeBudget.model_validate({"max_skill_calls": 2, "max_primitive_calls": 3})
        _, executor, events = setup_runtime()
        data = executor.execute(COMMAND).model_dump(mode="json")
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(events[0]["event_type"], "skill_requested")
        data["primitive"] = data.pop("skill")
        data["schema_version"] = 1
        result = SkillResult.model_validate(data)
        self.assertEqual(result.skill, "vla_act")
        self.assertNotIn("primitive", result.model_dump())

    def test_old_snapshot_verified_then_migrated_to_new_file_without_promotion(self):
        catalog = MemoryCatalog(global_memory=(rule().model_copy(update={"planner_reviewed": False}),)).model_dump(mode="json")
        entry = catalog["global_memory"][0]
        entry["primitive"] = entry.pop("skill")
        envelope = dict(sha256=sha256(canonical(catalog)).hexdigest(), catalog=catalog)
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp)/"old.json", Path(tmp)/"new.json"
            original = canonical(envelope); old.write_bytes(original)
            snapshot = MemorySnapshot.migrate_legacy(old, new)
            self.assertEqual(old.read_bytes(), original)
            MemorySnapshot.load(new, expected_sha256=snapshot.sha256)
            self.assertNotIn('"primitive"', new.read_text())
            self.assertFalse(snapshot.catalog().global_memory[0].planner_reviewed)
            with self.assertRaises(FileExistsError): MemorySnapshot.migrate_legacy(old, new)
            envelope["catalog"]["global_memory"][0]["statement"] = "tampered"
            old.write_bytes(canonical(envelope))
            with self.assertRaises(ValueError): MemorySnapshot.migrate_legacy(old, Path(tmp)/"bad.json")

    def test_old_turn_journals_still_support_candidate_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"attempt_0001"
            FakeAttemptRunner()(seed=3, config=None, output=output, timeout=10)
            path = output/"assessments.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for event in records:
                event["schema_version"] = 1
                if "result" in event:
                    event["result"]["primitive"] = event["result"].pop("skill")
                    event["result"]["schema_version"] = 1
                if "skill_calls" in event:
                    event["primitive_calls"] = event.pop("skill_calls")
            path.write_text("\n".join(json.dumps(event) for event in records)+"\n")
            catalog = candidates_from_attempt(output, attempt_id=output.name, seed=3, execution_kind="synthetic")
            self.assertEqual(len(catalog.task_memory), 1)
            self.assertFalse(catalog.task_memory[0].planner_reviewed)
