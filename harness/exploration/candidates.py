"""Convert completed, matched turn evidence into unreviewed memory candidates.

No inference of causal failure rules, automatic admission, or strict-hang claims.
Candidate extraction never reads raw oracle logs or evaluates arbitrary code.
"""

import json
from pathlib import Path
from harness.memory import MemoryCatalog, TaskMemory, GlobalMemory
from harness.memory.base import Evidence
from harness.memory.task_memory import RecipeStep
from harness.skills.schemas import SkillResult, SkillCommand
from harness.predicates.base import PredicateResult


def candidates_from_attempt(directory, *, attempt_id, seed, execution_kind):
    directory = Path(directory)
    if execution_kind not in ("live", "synthetic"):
        raise ValueError("unknown_execution_kind")
    path = directory / "assessments.jsonl"
    if not path.exists():
        return MemoryCatalog()
    with path.open("rb") as stream:
        raw = stream.read(10_000_001)
    if len(raw) > 10_000_000:
        raise ValueError("attempt_assessments_too_large")
    pending, trace, results, summaries, seen_turns = {}, [], [], [], set()
    private = False
    lifted = False
    for line in raw.splitlines():
        event = json.loads(line)
        if event.get("seed") != seed or event.get("episode") != attempt_id:
            raise ValueError("attempt_evidence_identity_mismatch")
        kind = event.get("event_type")
        if kind == "planner_selection":
            turn = event["planner_turn"]
            if turn in seen_turns:
                raise ValueError("duplicate_planner_turn")
            seen_turns.add(turn)
            command = SkillCommand.model_validate(event["command"])
            pending[turn] = RecipeStep(phase=event["phase"], command=command)
            private |= any(p.get("source") == "development_oracle" for p in event.get("predicates_before", []))
        elif kind == "planner_turn_result":
            turn = event["planner_turn"]
            step = pending.pop(turn, None)
            result = SkillResult.model_validate(event["result"])
            if step is None or step.command.action != result.skill or result.call_id != len(results)+1:
                raise ValueError("unmatched_skill_result")
            results.append(result)
            # ABORT is not a useful staging/acquisition recipe step.
            if result.skill != "abort":
                trace.append(step)
            predicates = [PredicateResult.model_validate(p) for p in event.get("post_condition", [])]
            private |= result.information_source == "development_oracle" or any(
                p.source == "development_oracle" for p in predicates)
            lifted = any(p.name == "mug_lifted" and p.value is True and p.source != "unavailable"
                         and p.observation_sequence == result.observation_after for p in predicates)
        elif kind == "episode_summary":
            summaries.append(event)
            private |= event.get("development_oracle") is True
    if len(summaries) != 1 or not trace:
        return MemoryCatalog()
    summary = summaries[0]
    successful = (summary.get("outcome") == "acquisition_confirmed" and
                  summary.get("milestone_reached") is True and lifted and not pending and
                  all(r.completed and not r.execution_uncertain for r in results))
    # A failed/ambiguous rollout cannot turn a partial command list into a success.
    outcome = "success" if successful else "unknown" if any(r.execution_uncertain for r in results) else "failure"
    source = "development_oracle" if private else "public"
    evidence = Evidence(episode=attempt_id, seed=seed,
        artifact=f"attempts/{attempt_id}/assessments.jsonl", metric="mug_lifted",
        outcome=outcome, source=source, execution=execution_kind)
    level = "single_case" if successful and execution_kind == "live" else "hypothesis"
    phases = tuple(dict.fromkeys(s.phase for s in trace))
    task = ()
    if len(trace) <= 32:
        task = (TaskMemory(id=f"{attempt_id}.sequence", phases=phases, source=source,
            evidence_level=level, planner_reviewed=False, evidence=(evidence,), objective="mug_lifted",
            description="Acquisition attempt candidate; inspect evidence before reuse.", trace=tuple(trace)),)
    rule = GlobalMemory(id=f"{attempt_id}.observation", task="hanging_mug", phases=phases,
        source=source, planner_reviewed=False, evidence_level="hypothesis", evidence=(evidence,),
        category="success" if successful else "failure",
        statement=("This parameterized attempt reached the acquisition gate." if successful else
                   "This attempt did not establish successful acquisition. Inspect the linked turn evidence before retrying."),
        operating_conditions=("One observed attempt, not a causal explanation or a validated operating range.",))
    return MemoryCatalog(task_memory=task, global_memory=(rule,))
