"""Report saved supervised-episode timings without a GPU, keys, or network.

python -m development.replays.latency --episode data/runs/NAME --output data/latency/NAME
"""

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics


@dataclass
class Stage:
    name: str
    scope: str
    note: str
    samples: list[float] = field(default_factory=list)
    invalid_samples: int = 0

    def add(self, value, scale=1.0):
        if value is None:
            return
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            self.invalid_samples += 1
            return
        self.samples.append(float(value) * scale)

    def report(self):
        values = sorted(self.samples)
        # Linear interpolation, equivalent to numpy.percentile's default method.
        position = (len(values) - 1) * 0.9
        low, high = math.floor(position), math.ceil(position)
        p90 = (values[low] + (values[high] - values[low]) * (position - low)
               if values else None)
        return dict(name=self.name, scope=self.scope, note=self.note, count=len(values),
                    invalid_samples=self.invalid_samples,
                    median_seconds=statistics.median(values) if values else None,
                    p90_seconds=p90, total_seconds=math.fsum(values) if values else None)


STAGES = (
    ("planner_selection", "controller sibling", "Client selection including prompt preparation, retries and validation; not pure Opus inference."),
    ("skill_execution", "controller sibling", "Inclusive skill wall time; includes policy, simulator and in-skill predicate work."),
    ("vla_request", "inside skill", "Client infer() including observation copy, packing, Modal RPC and response conversion."),
    ("policy_pack", "inside VLA request", "Client RGB/state compression; not network transfer."),
    ("policy_rpc", "inside VLA request", "Client-observed Modal round trip, including server work and possible cold start/queueing."),
    ("policy_server_compute", "inside policy RPC", "Server-reported compute duration."),
    ("policy_server_preprocess", "inside server compute", "Server-reported preprocessing."),
    ("policy_server_inference", "inside server compute", "Server-reported frozen-policy inference."),
    ("policy_server_postprocess", "inside server compute", "Server-reported postprocessing."),
    ("policy_rpc_residual", "inside policy RPC", "RPC minus reported compute; includes transport, queueing and cold start, not a network-only measurement."),
    ("policy_reset", "inside skill", "Client policy-history reset round trip."),
    ("action_round_trip", "inside skill", "Client environment step including response observation and any attached video capture."),
    ("simulator_reset", "outside controller", "Reset phase only, excluding the reset observation."),
    ("simulator_task_action", "inside action round trip", "RoboTwin take_action and success check; not individual analytic ticks."),
    ("simulator_observation", "mixed reset/action scope", "Only observations from LocalSimulator requests; direct LocalEnvironmentClient.observe calls are absent."),
    ("simulator_request", "mixed reset/action scope", "Inclusive /reset and /step request durations; overlaps simulator phase rows."),
    ("perception", "unmeasured", "No dedicated saved wall duration for SAM 3/Opus perception in these logs."),
    ("predicate_evaluation", "unmeasured", "Predicate results record values, not isolated durations; may include perception."),
    ("analytic_ticks", "unmeasured", "Physics tick counts are not wall-clock durations."),
    ("artifact_writing", "unmeasured", "No isolated observation/JSONL/video write durations; some writes are inside inclusive spans."),
)

POLICY_FIELDS = {
    "pack_ms": "policy_pack", "modal_rpc_ms": "policy_rpc",
    "modal_compute_ms": "policy_server_compute", "modal_overhead_ms": "policy_rpc_residual",
    "cloud_preprocess_ms": "policy_server_preprocess",
    "cloud_inference_ms": "policy_server_inference",
    "cloud_postprocess_ms": "policy_server_postprocess",
}


def load_sources(episode):
    """Read only known timing sources; retain hashes, never arbitrary payloads."""
    documents, hashes, warnings = {}, {}, []
    for name in ("manifest.json", "summary.json", "assessments.jsonl",
                 "skill_events.jsonl", "simulator_timing.jsonl"):
        path = episode / name
        if not path.exists():
            warnings.append(f"Missing {name}; associated measurements may be unavailable.")
            documents[name] = [] if name.endswith(".jsonl") else {}
            continue
        raw = path.read_bytes()
        hashes[name] = sha256(raw).hexdigest()
        if name.endswith(".json"):
            try:
                document = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ValueError(f"Invalid JSON in {name}") from None
            if not isinstance(document, dict):
                raise ValueError(f"Expected an object in {name}")
            documents[name] = document
            continue
        rows = []
        lines = raw.splitlines()
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                # An interrupted append can leave one unfinished trailing record.
                if index == len(lines) and not raw.endswith(b"\n"):
                    warnings.append(f"Ignored incomplete final record in {name}; timing coverage is partial.")
                    break
                raise ValueError(f"Invalid JSON in {name} at line {index}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {name} at line {index}")
            rows.append(row)
        documents[name] = rows
    if not hashes:
        raise ValueError("No recognized episode files found")
    return documents, hashes, warnings


def collect_stages(documents, warnings):
    stages = {name: Stage(name, scope, note) for name, scope, note in STAGES}
    seen = {"planner_selection": set(), "skill_result": set()}
    duplicates = False
    for source in ("assessments.jsonl", "skill_events.jsonl"):
        for row in documents[source]:
            event = row.get("event_type")
            if event in seen:
                identity = row.get("planner_turn" if event == "planner_selection" else "call_id")
                if type(identity) is not int or identity < 1:
                    warnings.append(f"{event} lacks a valid identity; reconciliation is disabled.")
                    duplicates = True
                elif identity in seen[event]:
                    warnings.append(f"Duplicate {event} identity; duplicate skipped and reconciliation disabled.")
                    duplicates = True
                    continue
                else:
                    seen[event].add(identity)
            if event == "planner_selection" and source == "assessments.jsonl":
                stages["planner_selection"].add(row.get("planner_latency_seconds"))
            elif event == "skill_result" and source == "skill_events.jsonl":
                result = row.get("result")
                if isinstance(result, dict):
                    stages["skill_execution"].add(result.get("elapsed_seconds"))
            elif event == "chunk_received" and source == "skill_events.jsonl":
                stages["vla_request"].add(row.get("inference_ms"), 0.001)
                timing = row.get("policy_timing", {})
                if isinstance(timing, dict):
                    for key, stage in POLICY_FIELDS.items():
                        stages[stage].add(timing.get(key), 0.001)
            elif event == "action_confirmed" and source == "skill_events.jsonl":
                stages["action_round_trip"].add(row.get("environment_step_ms"), 0.001)
            elif event == "policy_history_reset" and source == "skill_events.jsonl":
                stages["policy_reset"].add(row.get("reset_ms"), 0.001)
    phase_stages = {"reset": "simulator_reset", "task_action": "simulator_task_action",
                    "observation": "simulator_observation"}
    for row in documents["simulator_timing.jsonl"]:
        if row.get("event_type") == "simulator_phase_completed":
            name = phase_stages.get(row.get("phase"))
            if name:
                stages[name].add(row.get("elapsed_ms"), 0.001)
        elif row.get("event_type") == "simulator_request_completed":
            stages["simulator_request"].add(row.get("elapsed_ms"), 0.001)
    return stages, duplicates


def reconcile(summary, stages, ambiguous):
    wall = summary.get("elapsed_seconds")
    valid_wall = type(wall) in (int, float) and math.isfinite(wall) and wall >= 0
    selected = [stages["planner_selection"], stages["skill_execution"]]
    measured = math.fsum(value for stage in selected for value in stage.samples)
    reason = None
    if not valid_wall:
        reason = "Controller elapsed_seconds unavailable; full process duration cannot be inferred."
    elif ambiguous or any(stage.invalid_samples for stage in selected):
        reason = "Ambiguous or invalid sibling timings; totals cannot be reconciled safely."
    elif measured > wall:
        reason = "Sibling totals exceed controller elapsed time; check trace consistency."
    elif not any(stage.samples for stage in selected):
        reason = "No measured controller sibling spans."
    return dict(scope="SkillOrchestrator.run only; excludes simulator setup/reset and cleanup",
                controller_elapsed_seconds=float(wall) if valid_wall else None,
                measured_sibling_seconds=measured,
                unexplained_seconds=float(wall) - measured if reason is None else None,
                measured_fraction=measured / wall if reason is None and wall > 0 else None,
                status="partial coverage" if reason is None else "unavailable",
                reason=reason,
                formula="controller elapsed = planner selection + inclusive skills + unexplained",
                full_process_elapsed_seconds=None)


def analyze_episode(episode):
    episode = Path(episode)
    if not episode.is_dir():
        raise ValueError("Episode directory does not exist")
    documents, hashes, warnings = load_sources(episode)
    stages, ambiguous = collect_stages(documents, warnings)
    manifest, summary = documents["manifest.json"], documents["summary.json"]
    settings = manifest.get("settings")
    fingerprint = (sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()
                   if isinstance(settings, dict) else None)
    provenance = manifest.get("latency_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    metadata = {key: manifest.get(key) for key in (
        "seed", "planner", "mode", "development_oracle", "perception_factory", "prompt_sha256")}
    metadata.update(outcome=summary.get("outcome"), settings_sha256=fingerprint,
                    source_commit=provenance.get("source_commit"),
                    source_dirty=provenance.get("source_dirty"),
                    config_name=provenance.get("config_name"),
                    config_sha256=provenance.get("config_sha256"),
                    synthetic=manifest.get("synthetic") is True)
    if not metadata["source_commit"]:
        warnings.append("Source commit unavailable; do not infer it from the analyzer's checkout.")
    if metadata["source_dirty"]:
        warnings.append("Source checkout was dirty; commit alone does not reproduce the run.")
    if metadata["synthetic"]:
        warnings.append("SYNTHETIC fixture: these durations validate the analyzer, not robot performance.")
    for name in ("policy_actions", "analytic_ticks", "vla_chunks", "planner_turns", "skill_calls"):
        metadata[name] = summary.get(name)
    for stage in stages.values():
        if stage.invalid_samples:
            warnings.append(f"{stage.name}: rejected {stage.invalid_samples} invalid duration(s).")
    if not stages["skill_execution"].samples:
        warnings.append("No canonical skill_result durations; planner_turn_result copies are deliberately not counted.")
    for counter, stage in (("planner_turns", "planner_selection"), ("skill_calls", "skill_execution")):
        expected = summary.get(counter)
        count = len(stages[stage].samples)
        if type(expected) is int and expected != count:
            warnings.append(f"Summary declares {expected} {counter}, but {count} {stage} durations were recorded.")
    skill_rows = documents["skill_events.jsonl"]
    for requested, completed in (("skill_requested", "skill_result"),
                                 ("chunk_requested", "chunk_received"),
                                 ("action_requested", "action_confirmed")):
        started = sum(row.get("event_type") == requested for row in skill_rows)
        finished = sum(row.get("event_type") == completed for row in skill_rows)
        if started > finished:
            warnings.append(f"{started - finished} {requested} event(s) have no completion; durations are partial.")
    reconciliation = reconcile(summary, stages, ambiguous)
    warnings.append("Completed spans only: failed/in-flight calls and all uninstrumented work remain unexplained.")
    return dict(schema_version=1, metadata=metadata, sources_sha256=hashes,
                stages=[stage.report() for stage in stages.values()],
                reconciliation=reconciliation, warnings=warnings,
                next_experiment=("Collect a bounded local smoke run and locate the largest measured sibling. "
                    "Inspect its nested rows; instrument missing perception/predicate or write durations "
                    "before assigning residual time. Compare one change using identical config, seed, "
                    "evidence mode, policy deployment and warm/cold conditions. "
                    "Expected metric: lower matched-stage median/p90 and controller elapsed; "
                    "tradeoff: instrumentation adds overhead. Check action counts/outcomes and baseline compatibility."))


def render_markdown(report):
    def seconds(value):
        return f"{value:.6f}" if value is not None else "unmeasured"

    metadata = report["metadata"]
    lines = ["# Episode latency report", "",
             "SYNTHETIC VALIDATION — not live performance data." if metadata["synthetic"]
             else "Saved episode analysis; conclusions are limited to recorded timings.", "",
             "## Run selection", ""]
    lines.extend(f"- {key}: `{value if value is not None else 'unknown'}`"
                 for key, value in metadata.items())
    lines += ["", "## Stage durations", "",
              "Seconds. Rows are inclusive and overlap; do not sum this table. Missing samples are not zero.", "",
              "| Stage | Scope | Count | Median | p90 | Total |",
              "|---|---|---:|---:|---:|---:|"]
    for stage in report["stages"]:
        lines.append(f"| {stage['name']} | {stage['scope']} | {stage['count']} | "
                     f"{seconds(stage['median_seconds'])} | {seconds(stage['p90_seconds'])} | "
                     f"{seconds(stage['total_seconds'])} |")
    lines += ["", "p90 uses linear interpolation between sorted samples. Counts are events, not independent episodes.",
              "", "## Wall-time reconciliation", ""]
    reconciliation = report["reconciliation"]
    lines += [reconciliation["scope"] + ".", "", reconciliation["formula"] + ".", "",
              f"- Controller elapsed: {seconds(reconciliation['controller_elapsed_seconds'])} s",
              f"- Measured sibling spans: {seconds(reconciliation['measured_sibling_seconds'])} s",
              f"- Unexplained: {seconds(reconciliation['unexplained_seconds'])} s",
              f"- Status: {reconciliation['status']}",
              "- Full process wall time: unmeasured."]
    if reconciliation["reason"]:
        lines.append("- " + reconciliation["reason"])
    lines += ["", "## Measurement boundaries", ""]
    lines.extend(f"- {stage['name']}: {stage['note']}" for stage in report["stages"])
    lines += ["", "## Limitations", ""]
    lines.extend("- " + warning for warning in report["warnings"])
    lines += ["", "## Next experiment", "", report["next_experiment"], "",
              "## Source hashes", ""]
    lines.extend(f"- {name}: `{digest}`" for name, digest in report["sources_sha256"].items())
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="Saved run directory (read only)")
    parser.add_argument("--output", type=Path, required=True, help="New directory for report.json and report.md")
    args = parser.parse_args(argv)
    try:
        episode, output = args.episode.resolve(), args.output.resolve()
        if output == episode or output.is_relative_to(episode):
            raise ValueError("Report output must be outside the source episode")
        report = analyze_episode(episode)
        args.output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (output / "report.md").write_text(render_markdown(report))
    except ValueError as exc:
        # Validation messages contain only source names, never raw input records.
        parser.exit(2, f"Cannot create latency report: {exc}\n")
    except OSError as exc:
        parser.exit(2, f"Cannot create latency report: {type(exc).__name__}; check paths and use a new output directory.\n")
    print(f"Wrote {output / 'report.md'} and {output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
