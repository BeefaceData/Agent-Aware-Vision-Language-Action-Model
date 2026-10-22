"""Replay a verified lifted entry, then measure one right-arm pregrasp move.

Development-only diagnostic. It makes no Modal/Anthropic calls, never resets
after replay, and does not close the right gripper or attempt a handoff.
"""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from development.acquisition_entry import AcquisitionEntry, replay_entry
from development.checkpoint_bank import write_json_atomic
from development.episode_video import EpisodeVideo, RecordingEnvironment
from development.skill_oracle import oracle_anchor_preparer, oracle_predicates
from development.rack_geometry import mug_contacts
from harness.assessment_log import append_jsonl
from harness.evaluation.manifest import runtime_identity
from harness.perception.observations import ObservationStore
from harness.skills import SkillExecutor
from harness.skills.schemas import EpisodeBudget, MoveParameters, SkillCommand


def geometry(task):
    mug = np.asarray(task.mug.get_pose().p, dtype=float)
    robot = task.robot
    left = np.asarray(robot.get_left_tcp_pose()[:3], dtype=float)
    right = np.asarray(robot.get_right_tcp_pose()[:3], dtype=float)
    contacts = mug_contacts(task)
    return dict(mug_position=mug.tolist(), left_tcp=left.tolist(), right_tcp=right.tolist(),
        left_tcp_to_mug_center_m=float(np.linalg.norm(left-mug)),
        right_tcp_to_mug_center_m=float(np.linalg.norm(right-mug)),
        left_contact=any(n.startswith("fl_") for n in contacts),
        right_contact=any(n.startswith("fr_") for n in contacts),
        left_bilateral={"fl_link7", "fl_link8"} <= contacts,
        right_bilateral={"fr_link7", "fr_link8"} <= contacts,
        left_gripper=float(robot.get_left_gripper_val()),
        right_gripper=float(robot.get_right_gripper_val()),
        table_contact="table" in contacts,
        rack_contact=task.rack.actor.name in contacts,
        contacts=sorted(contacts))


class NoVLA:
    def infer(self, *_args, **_kwargs):
        raise RuntimeError("vla_not_available_in_right_stage_probe")


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-physics-ticks", type=int, default=1200)
    args = parser.parse_args(argv)
    if not 100 <= args.max_physics_ticks <= 3000:
        parser.error("--max-physics-ticks must be between 100 and 3000")

    # Validate all source artifacts and provenance before simulator startup.
    entry = AcquisitionEntry(args.episode, args.source_runtime, root=root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output/"manifest.json", dict(schema_version=1,
        condition="verified_replay_then_one_right_mug_pregrasp_move", seed=entry.seed,
        source_episode=str(entry.episode), source_hashes=entry.hashes,
        policy_calls=0, analytic_skill="move_to", arm="right", target="mug_pregrasp",
        max_physics_ticks=args.max_physics_ticks, gripper_action="none",
        full_state_restored=False, remote_random_state_restored=False))
    write_json_atomic(output/"runtime_identity.json", runtime_identity(root))

    # Native simulator imports happen only after the replay artifacts are validated.
    from development.local_simulator import LocalSimulator

    simulator = video = None
    summary = dict(outcome="entry_rejected", stage_completed=False, vla_calls=0)
    try:
        video = EpisodeVideo(output)
        simulator = LocalSimulator(root/"RoboTwin")
        environment = RecordingEnvironment(simulator, video)
        environment.policy_kind = "replay_acquisition"
        oracle_log = partial(append_jsonl, output/"oracle.jsonl")
        predicates = oracle_predicates(simulator.task, log=oracle_log)
        restored, _history = replay_entry(entry, environment, predicates,
            log=partial(append_jsonl, output/"replay_events.jsonl"))
        write_json_atomic(output/"restore.json", restored)
        if restored["accepted"]:
            environment.policy_kind = "analytic_right_stage"
            before = geometry(simulator.task)
            observation = environment.observe()
            append_jsonl(output/"stage_metrics.jsonl", dict(event="before", sequence=observation.sequence,
                geometry=before))
            executor = SkillExecutor(environment, NoVLA(), predicates=predicates,
                anchors=None, allow_development_oracle=True,
                log=partial(append_jsonl, output/"skill_events.jsonl"),
                budget=EpisodeBudget(max_skill_calls=1, max_policy_actions=1,
                    max_vla_chunks=1, max_analytic_ticks=args.max_physics_ticks))
            prepare = oracle_anchor_preparer(simulator.task, executor.anchors, log=oracle_log)
            command = SkillCommand(action="move_to", arm="right",
                parameters=MoveParameters(target="mug_pregrasp", max_physics_ticks=args.max_physics_ticks))
            prepare(command, observation)
            result = executor.execute(command)
            after_observation = environment.observe()
            after = geometry(simulator.task)
            append_jsonl(output/"stage_metrics.jsonl", dict(event="after",
                sequence=after_observation.sequence, geometry=after,
                right_tcp_motion_m=float(np.linalg.norm(np.asarray(after["right_tcp"])-before["right_tcp"])),
                right_mug_center_distance_change_m=(after["right_tcp_to_mug_center_m"]
                    - before["right_tcp_to_mug_center_m"])))
            summary = dict(outcome="right_pregrasp_completed" if result.completed else "right_pregrasp_failed",
                stage_completed=result.completed, stage_stop_reason=result.stop_reason,
                stage_error_type=result.error_type, position_error_m=result.position_error_m,
                orientation_error_rad=result.orientation_error_rad,
                analytic_ticks=result.executed_physics_ticks, policy_actions=result.executed_policy_actions,
                vla_calls=0, restore=restored, before=before, after=after,
                right_tcp_motion_m=float(np.linalg.norm(np.asarray(after["right_tcp"])-before["right_tcp"])),
                right_mug_center_distance_change_m=(after["right_tcp_to_mug_center_m"]
                    - before["right_tcp_to_mug_center_m"]),
                right_contact_after=after["right_contact"],
                right_bilateral_after=after["right_bilateral"],
                gripper_unchanged=abs(after["right_gripper"]-before["right_gripper"]) <= 1e-6)
    except Exception as exc:
        summary.update(outcome="runner_error", error_type=type(exc).__name__)
    finally:
        if video is not None:
            try:
                video.close()
            except Exception as exc:
                summary["video_cleanup_error_type"] = type(exc).__name__
            summary["video"] = video.metadata()
        if simulator is not None:
            try:
                simulator.close()
            except Exception as exc:
                summary["cleanup_error_type"] = type(exc).__name__
        write_json_atomic(output/"summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary.get("outcome") == "right_pregrasp_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
