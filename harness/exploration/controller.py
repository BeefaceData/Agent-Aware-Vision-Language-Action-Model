"""Bounded outer exploration schedule; all writes stay under a new output root."""

import json
from pathlib import Path
import time
import yaml

from harness.assessment_log import append_jsonl
from harness.memory import MemoryCatalog
from .config import ExploreSettings
from .candidates import candidates_from_attempt


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


class ExplorationController:
    def __init__(self, settings, *, runner, output, metadata=None, clock=time.monotonic, memory=None):
        self.settings = ExploreSettings.model_validate(settings.model_dump())
        self.runner, self.output, self.clock = runner, Path(output).resolve(), clock
        self.metadata = metadata or {}
        self.memory = memory
        self.started = False

    def run(self):
        if self.started:
            raise RuntimeError("exploration_is_single_use")
        self.started = True
        self.output.mkdir(parents=True, exist_ok=False)
        for name in ("attempts", "configs", "candidates"):
            (self.output / name).mkdir()
        if self.memory:
            path = self.output / "input_memory.json"
            self.memory.export(path)
            # Give every process the same archived, hash-pinned input. Candidate
            # writing never mutates that snapshot or the caller's source file.
            self.runner.memory = (path, self.memory.sha256)
        write_json(self.output / "manifest.json", dict(schema_version=1, mode="explore",
            settings=self.settings.model_dump(mode="json"), execution_kind=self.runner.execution_kind,
            metadata=self.metadata, candidate_auto_promotion=False))
        started = self.clock()
        attempts, task_memory, global_memory = [], [], []
        reason = "schedule_complete"
        try:
            for schedule_index, (seed, repeat, variant) in enumerate(self.settings.schedule()):
                if schedule_index < self.settings.schedule_offset:
                    continue
                if len(attempts) >= self.settings.max_attempts:
                    reason = "attempt_budget_exhausted"
                    break
                remaining = self.settings.campaign_timeout_seconds - (self.clock()-started)
                if remaining <= 0:
                    reason = "campaign_timeout"
                    break
                attempt_id = f"attempt_{len(attempts)+1:04d}"
                output = self.output / "attempts" / attempt_id
                config = self.output / "configs" / f"{attempt_id}.yaml"
                with config.open("x") as stream:
                    yaml.safe_dump(variant.settings.model_dump(mode="json"), stream)
                record = dict(attempt=attempt_id, seed=seed, repeat=repeat, variant=variant.name,
                              config=f"configs/{config.name}", output=f"attempts/{attempt_id}")
                append_jsonl(self.output / "events.jsonl", {"event_type": "attempt_started", **record})
                # Append before starting: interruption/failure must remain in denominator.
                attempts.append(record)
                exit_status = self.runner(seed=seed, config=config, output=output,
                    timeout=min(remaining, self.settings.attempt_timeout_seconds))
                record.update(returncode=exit_status.returncode, timed_out=exit_status.timed_out)
                if exit_status.timed_out:
                    record["outcome"] = reason = "attempt_timeout"
                elif not (output / "summary.json").exists():
                    record["outcome"] = reason = "missing_attempt_summary"
                else:
                    with (output / "summary.json").open("rb") as stream:
                        raw_summary = stream.read(1_000_001)
                    if len(raw_summary) > 1_000_000:
                        raise ValueError("attempt_summary_too_large")
                    summary = json.loads(raw_summary)
                    # Never copy raw failure strings/provider text to campaign logs.
                    record["outcome"] = summary.get("outcome")
                    record["acquisition_confirmed"] = (summary.get("acquisition_confirmed") is True or
                        (summary.get("outcome") == "acquisition_confirmed" and summary.get("milestone_reached") is True)
                        or summary.get("outcome") == "task_success")
                    record["handoff_confirmed"] = (summary.get("outcome") == "handoff_confirmed"
                                                   and summary.get("milestone_reached") is True)
                    record["full_task_success"] = (summary.get("outcome") == "task_success"
                                                   and summary.get("task_success") is True)
                    record["native_task_success"] = summary.get("task_success") is True
                    if (exit_status.returncode not in (0, 1) or summary.get("error_type")
                        or any(key.endswith("cleanup_error_type") for key in summary)
                        or record["outcome"] in ("runner_error", "controller_error", "planner_request_or_validation_failed")):
                        reason = "attempt_infrastructure_failure"
                    else:
                        candidates = candidates_from_attempt(output, attempt_id=attempt_id,
                            seed=seed, execution_kind=self.runner.execution_kind)
                        write_json(self.output / "candidates" / f"{attempt_id}.json", candidates.model_dump(mode="json"))
                        task_memory.extend(candidates.task_memory)
                        global_memory.extend(candidates.global_memory)
                append_jsonl(self.output / "events.jsonl", {"event_type": "attempt_finished", **record})
                if reason != "schedule_complete":
                    break
                if record.get("acquisition_confirmed") and self.settings.stop_on_acquisition:
                    reason = "acquisition_stop_requested"
                    break
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as exc:
            reason = "exploration_error"
            append_jsonl(self.output / "events.jsonl", {"event_type": "exploration_error", "error_type": type(exc).__name__})
        catalog = MemoryCatalog(task_memory=tuple(task_memory), global_memory=tuple(global_memory))
        write_json(self.output / "candidate_catalog.json", catalog.model_dump(mode="json"))
        summary = dict(schema_version=1, mode="explore", stop_reason=reason, attempts=attempts,
            attempt_count=len(attempts), acquisition_count=sum(a.get("acquisition_confirmed", False) for a in attempts),
            handoff_count=sum(a.get("handoff_confirmed", False) for a in attempts),
            full_task_success_count=sum(a.get("full_task_success", False) for a in attempts),
            explored_seeds=sorted({a["seed"] for a in attempts}), elapsed_seconds=self.clock()-started,
            schedule_offset=self.settings.schedule_offset,
            task_candidates=len(task_memory), global_candidates=len(global_memory), evaluation_admitted=False)
        write_json(self.output / "summary.json", summary)
        return summary
