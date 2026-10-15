"""Diagnose one saved planner turn without importing or starting a simulator.

python -m development.planner_replay --episode EPISODE --turn 4 --output NEW.json
Makes at most one fresh provider call. It never executes the returned skill.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from harness.environment.base import Observation
from harness.memory import MemorySnapshot
from harness.planner.anthropic import AnthropicPlanner, AnthropicSettings
from harness.planner.base import PlannerContext
from harness.predicates.base import PredicateResult
from harness.skill_config import RunSettings
from harness.skills.schemas import SkillCommand, SkillResult
from harness.trace import CAMERAS


def load_turn(episode, turn):
    episode = Path(episode).resolve()
    if type(turn) is not int or turn < 1:
        raise ValueError("invalid_replay_turn")
    sources = {}

    def read(relative):
        path = (episode / relative).resolve()
        if not path.is_relative_to(episode):
            raise ValueError("replay_source_outside_episode")
        raw = path.read_bytes()
        sources[path.relative_to(episode).as_posix()] = sha256(raw).hexdigest()
        return raw

    manifest = json.loads(read("manifest.json"))
    if manifest.get("planner") != "anthropic" or manifest.get("mode") != "anthropic_development":
        raise ValueError("replay_requires_anthropic_development_episode")
    settings = RunSettings.model_validate(manifest["settings"])
    planner_settings = AnthropicSettings.model_validate(manifest["anthropic"])
    rows = [json.loads(line) for line in read("assessments.jsonl").splitlines()]

    def row(kind, number):
        matches = [r for r in rows if r.get("event_type") == kind and r.get("planner_turn") == number]
        if len(matches) != 1:
            raise ValueError("replay_requires_unique_contiguous_turns")
        return matches[0]

    selection = row("planner_selection", turn)
    history, results = [], []
    for number in range(1, turn):
        history.append(SkillCommand.model_validate(row("planner_selection", number)["command"]).model_dump(mode="json"))
        result = SkillResult.model_validate(row("planner_turn_result", number)["result"])
        if not result.completed or result.episode_terminal or result.execution_uncertain:
            raise ValueError("cannot_replay_after_terminal_or_uncertain_result")
        results.append(result)
    reference = selection["observation_before"]
    # Validate containment and bind the bytes used for loading to the report.
    from io import BytesIO
    with np.load(BytesIO(read(reference["path"])), allow_pickle=False) as stored:
        observation = Observation(int(stored["sequence"]),
            {camera: stored[camera].copy() for camera in CAMERAS},
            tuple(stored["command_state"])).copy()
    if observation.sequence != reference["sequence"]:
        raise ValueError("replay_observation_sequence_mismatch")
    budget = settings.budget
    remaining = dict(planner_turns=settings.max_turns-turn+1,
        skill_calls=budget.max_skill_calls-len(results),
        policy_actions=budget.max_policy_actions-sum(r.executed_policy_actions for r in results),
        vla_chunks=budget.max_vla_chunks-sum(r.vla_chunks_requested for r in results),
        analytic_ticks=budget.max_analytic_ticks-sum(r.executed_physics_ticks for r in results))
    if min(remaining.values()) < 0 or remaining["planner_turns"] == 0:
        raise ValueError("invalid_replay_budget")
    memory = None
    if manifest.get("memory_snapshot_sha256"):
        read("memory_snapshot.json")
        snapshot = MemorySnapshot.load(episode / "memory_snapshot.json",
                                       expected_sha256=manifest["memory_snapshot_sha256"])
        phase = row("planner_turn_result", turn-1)["phase"] if turn > 1 else "APPROACH_MUG"
        memory = snapshot.retrieve(task="hanging_mug", phase=phase)
    context = PlannerContext(turn, observation,
        tuple(PredicateResult.model_validate(p) for p in selection["predicates_before"]),
        results[-1] if results else None, remaining_budget=remaining, memory=memory)
    return context, planner_settings, history, sources, manifest["prompt_sha256"]


def replay_turn(episode, turn, *, client=None):
    context, settings, history, sources, original_prompt = load_turn(episode, turn)
    planner = AnthropicPlanner(settings, client=client)
    planner.calls = turn-1
    planner.attempts = sum(command["action"] == "vla_act" for command in history)
    planner.history = history[-8:]
    try:
        selected = planner.select(context)
        return dict(schema_version=1, mode="planner_diagnostic", robot_motion=False,
            simulator_started=False, source_turn=turn, source_hashes=sources,
            original_prompt_sha256=original_prompt, current_prompt_sha256=planner.prompt_hash,
            settings=settings.model_dump(mode="json"),
            provider_calls_made=planner.calls-(turn-1),
            command=selected.command.model_dump(mode="json"), metadata=selected.metadata,
            original_response_recovered=False)
    finally:
        planner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Reserve a new artifact before spending a provider call; never overwrite evidence.
    with args.output.open("x") as stream:
        try:
            report = replay_turn(args.episode, args.turn)
        except Exception as exc:
            report = dict(mode="planner_diagnostic", robot_motion=False, simulator_started=False,
                          error_type=type(exc).__name__)
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 1 if "error_type" in report or report.get("metadata", {}).get("error_type") else 0


if __name__ == "__main__":
    raise SystemExit(main())
