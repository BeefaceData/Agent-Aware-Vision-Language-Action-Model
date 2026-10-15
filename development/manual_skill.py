"""Manually invoke an allowlisted development candidate from a labeled checkpoint.

Owns a local simulator process; do not run alongside a GPU simulator server.
This is an experiment command, not registry admission or an evaluation runner.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from harness.assessment_log import append_jsonl
from harness.legacy.recovery_skills.retry_grasp import RetryGrasp
from harness.legacy.recovery_skills.backoff_reapproach_rack import BackoffReapproachRack
from .checkpoint_bank import CheckpointBank, load_prefix, write_json_atomic
from .replay import ReplayEnvironment
from .retry_grasp_context import RetryGraspContext
from .rack_approach_context import RackApproachContext


MANUAL_SKILLS = {
    "retry_grasp": (RetryGrasp, RetryGraspContext),
    "backoff_reapproach_rack": (BackoffReapproachRack, RackApproachContext),
}


def check_request(bank, checkpoint, parameters, skill_id="retry_grasp"):
    if skill_id not in MANUAL_SKILLS:
        raise ValueError("unknown manual skill")
    skill = MANUAL_SKILLS[skill_id][0]()
    parameters = skill.validate_parameters(parameters)
    if checkpoint.phase not in skill.supported_phases or checkpoint.failure_type not in skill.supported_failures:
        raise ValueError("manual skill requires a reviewed checkpoint with supported phase/failure labels")
    bank.verify_artifacts(checkpoint)
    load_prefix(bank, checkpoint)
    return skill, parameters


def run_manual(bank, checkpoint, simulator, output, *, parameters=None, max_ticks=3000,
               cancel=lambda: False, skill_id="retry_grasp", video=False):
    skill, parameters = check_request(bank, checkpoint, parameters or {}, skill_id)
    if video and skill_id != "backoff_reapproach_rack":
        raise ValueError("recovery video currently supported for rack candidate only")
    if type(max_ticks) is not int or not 1 <= max_ticks <= skill.max_steps:
        raise ValueError("max_ticks outside candidate execution limit")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_label_revision": checkpoint.label_revision,
        "checkpoint_artifact_hashes": checkpoint.artifact_hashes,
        "skill_id": skill.skill_id, "skill_version": skill.version, "code_hash": skill.code_hash(),
        "parameters": parameters, "max_physics_ticks": max_ticks,
        "development_only": True, "runtime_eligible": False,
        "policy_resumed": False, "recovery_outcome": "unknown", "status": "error",
    }
    write_json_atomic(output / "request.json", report)
    context = None
    video_writer = None
    try:
        replay = checkpoint.restore(ReplayEnvironment(bank, simulator))
        report["replay"] = replay.model_dump(mode="json")
        if replay.status != "matched":
            report["status"] = "restore_rejected"
            return report
        from run_inference_chunked import observation_of, CAMERAS
        options = {}
        if video:
            import imageio.v2 as imageio
            video_writer = imageio.get_writer(output / "recovery.mp4", fps=10)
            options["frame_callback"] = lambda: video_writer.append_data(
                observation_of(simulator.task, simulator.blank, False)["head_camera"])
        context = MANUAL_SKILLS[skill_id][1](
            simulator.task, log=lambda event: append_jsonl(output / "execution.jsonl", event),
            cancel=cancel, max_ticks=max_ticks, **options)
        report["before_privileged"] = context.diagnostics()
        before = observation_of(simulator.task, simulator.blank, False)
        if video_writer is not None:
            video_writer.append_data(before["head_camera"])
        np.savez_compressed(output / "before.npz", **{name: before[name] for name in CAMERAS})
        result = skill.execute(context, parameters)
        report["result"] = result.model_dump(mode="json")
        report["after_privileged"] = context.diagnostics()
        after = observation_of(simulator.task, simulator.blank, simulator.task.check_success())
        if video_writer is not None:
            video_writer.append_data(after["head_camera"])
        np.savez_compressed(output / "after.npz", **{name: after[name] for name in CAMERAS})
        report["status"] = "invoked" if result.termination_reason in ("completed", "task_complete") else "stopped"
        return report
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["confirmed_physics_ticks"] = context.executed_ticks if context else 0
        return report
    finally:
        # A failed invocation never falls back to policy or starts another motion.
        try:
            if video_writer is not None:
                video_writer.close()
        finally:
            write_json_atomic(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_id")
    parser.add_argument("--skill", choices=tuple(MANUAL_SKILLS), default="retry_grasp")
    parser.add_argument("--video", action="store_true", help="record rack recovery at 10 frames per simulated second")
    parser.add_argument("--path", default="data/checkpoints")
    parser.add_argument("--output", required=True, help="new directory for this one manual attempt")
    parser.add_argument("--robotwin-root", default="RoboTwin")
    parser.add_argument("--backoff-scale", type=float)
    parser.add_argument("--lift-distance", type=float)
    parser.add_argument("--max-ticks", type=int, default=3000)
    args = parser.parse_args()
    bank = CheckpointBank(args.path)
    output, root = Path(args.output).resolve(), Path(args.robotwin_root).resolve()
    try:
        checkpoint = bank.get(args.checkpoint_id)
        parameters = {key: value for key, value in {"backoff_scale": args.backoff_scale,
                                                  "lift_distance": args.lift_distance}.items() if value is not None}
        skill, parameters = check_request(bank, checkpoint, parameters, args.skill)
        if args.video and args.skill != "backoff_reapproach_rack":
            raise ValueError("video is only implemented for rack recovery")
        if not 1 <= args.max_ticks <= skill.max_steps or output.exists():
            raise ValueError("invalid tick budget or output directory already exists")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"manual request rejected: {type(exc).__name__}\n")
    from .local_simulator import LocalSimulator
    simulator = LocalSimulator(root)
    try:
        report = run_manual(bank, checkpoint, simulator, output, parameters=parameters, max_ticks=args.max_ticks,
                            skill_id=args.skill, video=args.video)
    finally:
        simulator.close()
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "invoked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
