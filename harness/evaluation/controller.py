"""Run each frozen evaluation seed once; never write exploration candidates."""

import json
from pathlib import Path
import time
import yaml

from harness.jsonl import append_jsonl
from harness.exploration.controller import write_json
from harness.memory import MemorySnapshot
from .manifest import FrozenEvaluation


class EvaluationController:
    def __init__(self, frozen, *, root, output, runner, clock=time.monotonic):
        self.frozen = FrozenEvaluation.model_validate(frozen.model_dump())
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.runner, self.clock, self.started = runner, clock, False

    def run(self):
        if self.started:
            raise RuntimeError("evaluation_is_single_use")
        self.started = True
        self.frozen.verify(self.root)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "episodes").mkdir()
        frozen_path = self.output / "frozen.json"
        self.frozen.save(frozen_path)
        memory_path = self.output / "memory_snapshot.json"
        memory = MemorySnapshot(self.frozen.memory)
        memory.export(memory_path)
        config = self.output / "episode.yaml"
        with config.open("x") as stream:
            yaml.safe_dump(self.frozen.settings.episode.model_dump(mode="json"), stream)
        self.runner.memory = (memory_path, memory.sha256)
        self.runner.evaluation = (frozen_path, self.frozen.digest)
        records = []
        started = self.clock()
        reason, error_type = "completed", None

        def check():
            FrozenEvaluation.load(frozen_path, expected_sha256=self.frozen.digest)
            self.frozen.verify(self.root)
            MemorySnapshot.load(memory_path, expected_sha256=memory.sha256)
            with config.open() as stream:
                if yaml.safe_load(stream) != self.frozen.settings.episode.model_dump(mode="json"):
                    raise ValueError("evaluation_config_changed")

        try:
            for index, seed in enumerate(self.frozen.settings.seeds):
                check()
                remaining = self.frozen.settings.campaign_timeout_seconds - (self.clock()-started)
                if remaining <= 0:
                    reason = "campaign_timeout"
                    break
                directory = self.output / "episodes" / f"episode_{index+1:04d}"
                record = dict(seed=seed, episode=directory.name, outcome="unfinished",
                              acquisition_confirmed=False, native_task_success=False)
                records.append(record)
                append_jsonl(self.output / "events.jsonl", {"event_type": "evaluation_episode_started", **record})
                result = self.runner(seed=seed, config=config, output=directory,
                    timeout=min(remaining, self.frozen.settings.episode_timeout_seconds))
                check()
                record.update(returncode=result.returncode, timed_out=result.timed_out)
                if result.timed_out:
                    record["outcome"] = reason = "episode_timeout"
                    break
                with (directory / "summary.json").open("rb") as stream:
                    raw = stream.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise ValueError("episode_summary_too_large")
                summary = json.loads(raw)
                if (summary.get("mode") != "eval" or summary.get("evaluation_sha256") != self.frozen.digest
                    or summary.get("condition") != self.frozen.settings.condition):
                    raise ValueError("episode_missing_frozen_evaluation_identity")
                if (summary.get("seed") != seed or summary.get("episode") != directory.name
                    or summary.get("completion_predicate") != "mug_lifted"):
                    raise ValueError("evaluation_episode_identity_mismatch")
                record["outcome"] = summary.get("outcome")
                if (result.returncode not in (0, 1) or summary.get("error_type")
                    or any(k.endswith("cleanup_error_type") for k in summary)
                    or record["outcome"] in ("runner_error", "controller_error", "planner_request_or_validation_failed")):
                    reason = "episode_infrastructure_failure"
                    break
                record["acquisition_confirmed"] = (summary.get("outcome") == "acquisition_confirmed"
                                                    and summary.get("milestone_reached") is True)
                record["native_task_success"] = summary.get("task_success") is True
                record["metrics"] = {k: summary.get(k) for k in ("planner_turns", "skill_calls", "vla_chunks",
                    "policy_actions", "analytic_ticks", "retries", "elapsed_seconds")}
                append_jsonl(self.output / "events.jsonl", {"event_type": "evaluation_episode_finished", **record})
            check()
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as exc:
            reason, error_type = "evaluation_invalid", type(exc).__name__
        complete = reason == "completed" and len(records) == len(self.frozen.settings.seeds)
        count = sum(r["acquisition_confirmed"] for r in records)
        summary = dict(schema_version=1, mode="eval", condition=self.frozen.settings.condition,
            execution_kind=self.runner.execution_kind, evaluation_sha256=self.frozen.digest,
            stop_reason=reason, error_type=error_type, scheduled_count=len(self.frozen.settings.seeds),
            started_count=len(records), episodes=records, protocol_completed=complete,
            acquisition_count=count, acquisition_rate=count/len(records) if complete else None,
            strict_hanging_success=None, remote_policy_weights_verified=False,
            elapsed_seconds=self.clock()-started)
        write_json(self.output / "summary.json", summary)
        return summary
