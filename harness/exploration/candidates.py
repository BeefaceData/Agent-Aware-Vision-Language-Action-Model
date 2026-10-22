"""Convert completed, matched turn evidence into unreviewed memory candidates.

No inference of causal failure rules, automatic admission, or strict-hang claims.
Candidate extraction never reads raw oracle logs or evaluates arbitrary code.
"""

import json
import math
from pathlib import Path
from harness.memory import MemoryCatalog, TaskMemory, GlobalMemory
from harness.memory.base import Evidence
from harness.memory.task_memory import RecipeStep
from harness.skills.schemas import SkillResult, SkillCommand
from harness.predicates.base import PredicateResult


def _mug_descriptor_condition(visual_context):
    """Keep a compact, view-dependent public descriptor in review candidates."""
    if not isinstance(visual_context, dict):
        return None
    mug = visual_context.get("objects", {}).get("mug")
    if not isinstance(mug, dict):
        return None
    descriptor = {}
    for name, size, low, high in (
        ("normalized_bbox", 4, 0.0, 1.0),
        ("median_rgb_0_255", 3, 0.0, 255.0),
        ("visible_surface_extent_m", 3, 0.0, 10.0),
    ):
        values = mug.get(name)
        if (isinstance(values, (list, tuple)) and len(values) == size
                and all(type(value) in (int, float) and math.isfinite(value)
                        and low <= value <= high for value in values)):
            descriptor[name] = [round(float(value), 4) for value in values]
    if not descriptor:
        return None
    if mug.get("camera") in ("head_camera", "left_camera", "right_camera"):
        descriptor["camera"] = mug["camera"]
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return "Public mug visual descriptor (view-dependent; not a diagnosis): " + encoded


def _memory_safe_step(step):
    """Keep symbolic targets and relative offsets; never replay saved image pixels."""
    command = step.command
    updates = {"reason": None} if command.reason is not None else {}
    if command.action.value == "move_to" and command.parameters.grounding is not None:
        parameters = command.parameters.model_copy(update={"grounding": None})
        updates["parameters"] = parameters
    if updates:
        command = command.model_copy(update=updates)
    return RecipeStep(phase=step.phase, command=command)


def _append_compact_step(trace, step):
    """Collapse adjacent identical skill calls; the source trace retains counts."""
    if not trace or trace[-1].model_dump(mode="json") != step.model_dump(mode="json"):
        trace.append(step)


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
    visual_conditions = []
    private = False
    milestones_confirmed = set()
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
            pending[turn] = _memory_safe_step(RecipeStep(phase=event["phase"], command=command))
            condition = _mug_descriptor_condition(event.get("visual_context"))
            if condition and condition not in visual_conditions and len(visual_conditions) < 3:
                visual_conditions.append(condition)
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
                _append_compact_step(trace, step)
            predicates = [PredicateResult.model_validate(p) for p in event.get("post_condition", [])]
            private |= result.information_source == "development_oracle" or any(
                p.source == "development_oracle" for p in predicates)
            milestones_confirmed.update(p.name for p in predicates
                if p.value is True and p.source != "unavailable"
                and p.observation_sequence == result.observation_after
                and p.name in ("mug_lifted", "handoff_confirmed", "task_success"))
        elif kind == "episode_summary":
            summaries.append(event)
            private |= event.get("development_oracle") is True
    if len(summaries) != 1 or not trace:
        return MemoryCatalog()
    summary = summaries[0]
    objective = summary.get("completion_predicate", "mug_lifted")
    expected = {"mug_lifted": "acquisition_confirmed", "handoff_confirmed": "handoff_confirmed",
                "task_success": "task_success"}
    if objective not in expected:
        return MemoryCatalog()
    successful = (summary.get("outcome") == expected[objective] and
                  summary.get("milestone_reached") is True and objective in milestones_confirmed and not pending and
                  all(r.completed and not r.execution_uncertain for r in results))
    # A failed/ambiguous rollout cannot turn a partial command list into a success.
    outcome = "success" if successful else "unknown" if any(r.execution_uncertain for r in results) else "failure"
    source = "development_oracle" if private else "public"
    artifact = (f"attempts/{attempt_id}/assessments.jsonl"
                if directory.parent.name == "attempts" else "assessments.jsonl")
    evidence = []
    for metric in ("mug_lifted", "handoff_confirmed", "task_success"):
        if metric in milestones_confirmed and not (metric == objective and outcome != "success"):
            evidence.append(Evidence(episode=attempt_id, seed=seed, artifact=artifact,
                metric=metric, outcome="success", source=source, execution=execution_kind))
    if not any(item.metric == objective and item.outcome == outcome for item in evidence):
        evidence.append(Evidence(episode=attempt_id, seed=seed, artifact=artifact,
            metric=objective, outcome=outcome, source=source, execution=execution_kind))
    evidence = tuple(evidence)
    level = "single_case" if successful and execution_kind == "live" else "hypothesis"
    phases = tuple(dict.fromkeys(s.phase for s in trace))
    task = ()
    if len(trace) <= 32:
        task = (TaskMemory(id=f"{attempt_id}.sequence", phases=phases, source=source,
            evidence_level=level, planner_reviewed=False, evidence=evidence, objective=objective,
            description=("Compacted skill-composition candidate; adjacent repeated calls and free-text "
                         "rationales are omitted. Re-ground symbolic targets from current RGB-D and "
                         "inspect the linked full turn trace before reuse."), trace=tuple(trace)),)
    rule = GlobalMemory(id=f"{attempt_id}.observation", task="hanging_mug", phases=phases,
        source=source, planner_reviewed=False, evidence_level="hypothesis", evidence=evidence,
        category="success" if successful else "failure",
        statement=(f"This parameterized attempt reached {objective}." if successful else
                   (f"This attempt confirmed {', '.join(sorted(milestones_confirmed))} but did not establish {objective}. "
                    if milestones_confirmed else f"This attempt did not establish {objective}. ")
                   + "Inspect the linked turn evidence before retrying."),
        operating_conditions=("One observed attempt, not a causal explanation or a validated operating range.",
                              *visual_conditions))
    return MemoryCatalog(task_memory=task, global_memory=(rule,))
