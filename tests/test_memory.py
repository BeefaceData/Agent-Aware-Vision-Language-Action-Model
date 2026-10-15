"""R7 provenance, immutable snapshots, retrieval, and provider-boundary checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from harness.memory import MemoryCatalog, MemorySnapshot, TaskMemory, GlobalMemory
from harness.memory.snapshot import MemorySelection
from harness.planner.prompt_builder import user_content
from tests.test_anthropic_planner import context, Provider, response


def rule(id="rule", **overrides):
    return GlobalMemory.model_validate(dict(id=id, phases=["all"], source="public",
        planner_reviewed=True, category="failure", statement="Do not equate chunk completion with grasp.", **overrides))


def recipe(**overrides):
    value = dict(id="acquisition", phases=["APPROACH_MUG"], source="public", planner_reviewed=True,
        objective="mug_lifted", description="Candidate stage and acquire recipe", trace=[
            {"phase": "APPROACH_MUG", "command": {"action": "move_to", "arm": "left",
                                                  "parameters": {"target": "mug_pregrasp"}}}])
    value.update(overrides)
    return TaskMemory.model_validate(value)


def evidence(seed=1, **overrides):
    value = dict(episode=f"episode{seed}", seed=seed, artifact="PRIVATE_ARTIFACT_REFERENCE",
                 metric="mug_lifted", outcome="success", source="public", execution="live")
    value.update(overrides)
    return value


class MemoryTests(unittest.TestCase):
    def test_claim_levels_and_success_metrics_cannot_be_promoted_accidentally(self):
        for values in (
            dict(evidence_level="single_case"),
            dict(evidence_level="single_case", evidence=[evidence(execution="synthetic")]),
            dict(evidence_level="repeated", evidence=[evidence(), evidence()]),
            dict(evidence_level="single_case", objective="strict_rack_success",
                 evidence=[evidence(metric="native_task_success")]),
            dict(evidence=[evidence(source="development_oracle")]),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                recipe(**values)
        value = recipe(evidence_level="repeated", evidence=[evidence(1), evidence(2)])
        self.assertEqual(value.evidence_level, "repeated")

    def test_snapshot_roundtrip_hash_pin_and_no_overwrite(self):
        snapshot = MemorySnapshot(MemoryCatalog(global_memory=(rule(),)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            snapshot.export(path)
            loaded = MemorySnapshot.load(path, expected_sha256=snapshot.sha256)
            with self.assertRaises(FileExistsError): snapshot.export(path)
            with self.assertRaises(ValueError): MemorySnapshot.load(path, expected_sha256="0"*64)
            data = json.loads(path.read_text()); data["catalog"]["global_memory"][0]["statement"] = "tampered"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): MemorySnapshot.load(path)
            selected = loaded.retrieve(task="hanging_mug", phase="APPROACH_MUG")
            self.assertEqual(selected.global_memory[0].statement, rule().statement)
            with self.assertRaises(AttributeError): loaded.sha256 = "0"*64

    def test_private_unreviewed_wrong_scope_and_unsupported_anchor_excluded(self):
        private = rule().model_copy(update={"id": "private", "source": "development_oracle", "statement": "PRIVATE_SENTINEL"})
        unreviewed = rule().model_copy(update={"id": "candidate", "planner_reviewed": False})
        unrelated = rule().model_copy(update={"id": "other", "phases": ("HANDOFF",)})
        bad_recipe = recipe(trace=[{"phase": "APPROACH_MUG", "command": {"action": "move_to", "arm": "left",
                                "parameters": {"target": "rack_precontact"}}}])
        snapshot = MemorySnapshot(MemoryCatalog(task_memory=(bad_recipe,), global_memory=(rule(), private, unreviewed, unrelated)))
        selection = snapshot.retrieve(task="hanging_mug", phase="APPROACH_MUG")
        self.assertEqual([r["id"] for r in selection.planner_payload()], ["rule"])
        content = user_content(replace(context(), memory=selection), limits={}, history=[])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(content))
        self.assertIn(snapshot.sha256, json.dumps(content))
        # Injection of a well-typed private entry at the planner boundary is filtered too.
        injected = MemorySelection(snapshot_sha256=snapshot.sha256, global_memory=(private,))
        self.assertEqual(injected.planner_payload(), [])

    def test_retrieval_is_deterministic_bounded_and_never_exposes_evidence_paths(self):
        records = tuple(rule(f"r{i}") for i in reversed(range(10)))
        task = recipe(evidence_level="single_case", evidence=[evidence()])
        snapshot = MemorySnapshot(MemoryCatalog(task_memory=(task,), global_memory=records))
        selection = snapshot.retrieve(task="hanging_mug", phase="APPROACH_MUG")
        self.assertEqual([r.id for r in selection.global_memory], ["r0", "r1", "r2", "r3"])
        self.assertNotIn("PRIVATE_ARTIFACT_REFERENCE", json.dumps(selection.planner_payload()))
        self.assertEqual(selection.references()[0], {"kind": "task", "id": "acquisition", "revision": 1})
        with self.assertRaises(ValueError): MemoryCatalog(global_memory=(rule(), rule()))

    def test_orchestrator_records_snapshot_and_provider_receives_memory(self):
        from harness.planner.anthropic import AnthropicPlanner, AnthropicSettings
        from harness.skill_orchestrator import SkillOrchestrator
        from tests.test_skills import setup_runtime
        runtime, executor, _ = setup_runtime()
        provider = Provider([response("abort", arm="auto", parameters={}, reason="End test")])
        planner = AnthropicPlanner(AnthropicSettings(model="claude-test-model"), client=provider)
        snapshot = MemorySnapshot(MemoryCatalog(global_memory=(rule(),)))
        logs = []
        summary = SkillOrchestrator(executor, planner, log=logs.append,
            save_observation=lambda obs: {"sequence": obs.sequence}, episode_id="test", seed=3, memory=snapshot).run()
        self.assertEqual(summary["memory_snapshot_sha256"], snapshot.sha256)
        self.assertEqual(logs[0]["memory_retrieved"], [{"kind": "global", "id": "rule", "revision": 1}])
        request_data = json.loads(provider.requests[0]["messages"][0]["content"][-1]["text"])
        self.assertEqual(request_data["memory_retrieved"][0]["id"], "rule")
        self.assertEqual(runtime.actions, [])

    def test_recipe_rejects_new_skills_and_absolute_pose_actions(self):
        for command in ({"action": "teleport"}, {"action": "move_to", "arm": "left",
                        "parameters": {"pose": [0.0]*7}}):
            with self.assertRaises(ValueError):
                recipe(trace=[{"phase": "APPROACH_MUG", "command": command}])
