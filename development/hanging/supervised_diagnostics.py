"""Timeline diagnostics for supervised hanging runs: what the supervisor believed, and when.

Reads a supervised run folder (``report.json`` and ``supervisor_events.jsonl``)
and lines up, per observation:

- the public visual report: stage, right_held, mug_off_table, task-state confidence;
- the lift-corroboration status and reason;
- the planner's selection, task phase and its written phase evidence;
- supervisor decisions (grace period, hand-back, takeover);
- right-gripper command transitions.

Optional ground truth (the observation at which the mug was actually lost,
from frame review) gives detection latency. Uses no simulator state.

    python -m development.hanging.supervised_diagnostics RUN_FOLDER [RUN_FOLDER ...] --output DIR
"""

import argparse
import json
from pathlib import Path


def _events(folder):
    rows = []
    for line in (Path(folder)/"supervisor_events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _gripper_transitions(events, index=13):
    out, state = [], None
    for event in events:
        if event.get("event_type") != "action_confirmed" or "command_state" not in event:
            continue
        value = event["command_state"][index]
        label = "closed" if value < .05 else "open" if value > .5 else "moving"
        if label != state:
            out.append((event.get("observation"), label))
            state = label
    return out


def timeline(folder):
    """Ordered rows of what was perceived and decided, keyed by observation sequence."""
    rows = []
    for event in _events(folder):
        kind = event.get("event_type")
        if kind == "public_hanging_proposal":
            report = event.get("report") or {}
            lift = event.get("lift_corroboration") or {}
            rows.append(dict(observation=event.get("observation_sequence"), source="perception",
                stage=report.get("stage"), right_held=report.get("right_held"),
                mug_off_table=report.get("mug_off_table"),
                task_state_confidence=report.get("task_state_confidence"),
                lift_status=lift.get("status"), lift_reason=lift.get("reason"),
                corroborated=event.get("corroborated_lift_count"),
                note=(report.get("evidence") or "")[:220]))
        elif kind == "planner_selection":
            meta = event.get("planner_metadata") or {}
            command = event.get("command") or {}
            rows.append(dict(observation=(event.get("observation_before") or {}).get("sequence")
                             if isinstance(event.get("observation_before"), dict) else None,
                source="planner", turn=event.get("planner_turn"), action=command.get("action"),
                phase=event.get("phase"), task_phase=meta.get("task_phase"),
                error=meta.get("validation_error") or meta.get("error_type"),
                note=(meta.get("phase_evidence") or "")[:220]))
        elif kind == "hanging_supervisor_decision":
            rows.append(dict(observation=event.get("observation_sequence"), source="supervisor",
                decision=event.get("decision"), reason=event.get("reason")))
        elif kind == "public_grounding_failure":
            rows.append(dict(observation=event.get("observation_sequence"), source="perception",
                note="vision request failed: " + json.dumps(event.get("failure"))[:160]))
    # Planner rows without an observation inherit the latest perception observation.
    last = None
    for row in rows:
        if row.get("observation") is None:
            row["observation"] = last
        else:
            last = row["observation"]
    return rows


def summarize(folder, *, lost_at=None):
    folder = Path(folder)
    report = json.loads((folder/"report.json").read_text())
    summary = report.get("summary") or {}
    rows = timeline(folder)
    perception = [r for r in rows if r["source"] == "perception" and "stage" in r]
    planner = [r for r in rows if r["source"] == "planner"]
    decisions = [r for r in rows if r["source"] == "supervisor"]

    def first(items, test):
        return next((r["observation"] for r in items if test(r)), None)

    held_seen = first(perception, lambda r: r.get("right_held") is True)
    lift_confirmed = first(perception, lambda r: r.get("lift_status") == "confirmed")
    entry = first(decisions, lambda r: r.get("decision") in ("policy_grace_period", "take_over_hanging",
                                                               "continue_policy_and_reacquire"))
    lapse = first(decisions, lambda r: r.get("reason") in ("right_grasp_evidence_expired", "right_grasp_lost"))
    handback = first(decisions, lambda r: r.get("decision") == "hand_back_to_policy")
    first_abort = first(planner, lambda r: r.get("action") == "abort")
    not_held = first(perception, lambda r: lost_at is not None and r["observation"] is not None
                     and r["observation"] >= lost_at and r.get("right_held") is False)
    noticed = [x for x in (lapse, first_abort, not_held) if x is not None and lost_at is not None and x >= lost_at]
    phases = []
    for r in planner:
        if not phases or phases[-1][1] != r.get("task_phase"):
            phases.append((r["observation"], r.get("task_phase")))
    confidences = [r["task_state_confidence"] for r in perception if r.get("task_state_confidence") is not None]
    return dict(run=folder.as_posix(), outcome=report.get("status"), success=summary.get("task_success") is True,
        policy_actions=summary.get("policy_actions"), claude_requests=report.get("model_requests"),
        right_gripper=_gripper_transitions(_events(folder)),
        first_right_held_report=held_seen, first_lift_confirmed=lift_confirmed, supervisor_entry=entry,
        lift_evidence_lapsed=lapse, handed_back=handback, first_planner_abort=first_abort,
        planner_aborts=sum(r.get("action") == "abort" for r in planner),
        planner_phases=phases, perception_reports=len(perception),
        median_task_state_confidence=(sorted(confidences)[len(confidences)//2] if confidences else None),
        vision_failures=sum("vision request failed" in (r.get("note") or "") for r in rows),
        mug_lost_at=lost_at, detection_latency_actions=(min(noticed)-lost_at if noticed else None),
        timeline=rows)


def write_markdown(result, path):
    lines = [f"# Supervised run diagnostics: {result['run']}", "",
             f"Outcome **{result['outcome']}**, success {result['success']}, {result['policy_actions']} policy "
             f"actions, {result['claude_requests']} Claude requests.", "",
             "| Signal | Observation |", "|---|---|"]
    for key in ("mug_lost_at", "first_right_held_report", "first_lift_confirmed", "supervisor_entry",
                "lift_evidence_lapsed", "handed_back", "first_planner_abort", "detection_latency_actions"):
        lines.append(f"| {key.replace('_', ' ')} | {result[key] if result[key] is not None else '-'} |")
    lines += ["", f"Planner aborts: {result['planner_aborts']}. Vision failures: {result['vision_failures']}. "
              f"Median task-state confidence: {result['median_task_state_confidence']}.", "",
              "Right gripper command: " + ", ".join(f"{o}:{s}" for o, s in result["right_gripper"]), "",
              "Planner task phase: " + ", ".join(f"{o}:{p}" for o, p in result["planner_phases"]), "",
              "## Timeline", "", "| Obs | Source | What | Detail |", "|---|---|---|---|"]
    for row in result["timeline"]:
        if row["source"] == "perception" and "stage" in row:
            what = (f"held={row.get('right_held')} off_table={row.get('mug_off_table')} "
                    f"conf={row.get('task_state_confidence')} lift={row.get('lift_status')}/{row.get('lift_reason')}")
        elif row["source"] == "planner":
            what = f"{row.get('action')} phase={row.get('task_phase')}" + (f" error={row['error']}" if row.get("error") else "")
        elif row["source"] == "supervisor":
            what = f"{row.get('decision')}: {row.get('reason')}"
        else:
            what = ""
        detail = (row.get("note") or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['observation']} | {row['source']} | {what} | {detail} |")
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lost-at", default="", help="comma list run_name=observation from frame review")
    args = parser.parse_args(argv)
    lost = dict(item.split("=") for item in args.lost_at.split(",") if item)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for run in args.runs:
        name = run.parent.name if run.name in ("supervised", "policy_only") else run.name
        result = summarize(run, lost_at=int(lost[name]) if name in lost else None)
        write_markdown(result, args.output/f"{name}.md")
        results.append({k: v for k, v in result.items() if k != "timeline"})
    (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
    for r in results:
        print(json.dumps({k: r[k] for k in ("run", "outcome", "mug_lost_at", "first_right_held_report",
            "first_lift_confirmed", "lift_evidence_lapsed", "first_planner_abort", "detection_latency_actions",
            "planner_aborts", "median_task_state_confidence")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
