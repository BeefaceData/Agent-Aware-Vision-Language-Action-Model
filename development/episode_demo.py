"""Record reset-to-hanging replay with a manually selected development recovery.

No fresh policy inference, autonomous failure detection, or runtime admission.
The video labels recorded policy playback separately from live recovery physics.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from harness.assessment_log import append_jsonl
from .checkpoint_bank import CheckpointBank, write_json_atomic, sha256
from .manual_skill import check_request
from .replay import ReplayEnvironment
from .rack_approach_context import RackApproachContext
from .hanging_check import observe_hanging


class DemoVideo:
    def __init__(self, path):
        import imageio.v2 as imageio
        self.writer = imageio.get_writer(path, fps=30)
        self.frames = 0

    def append(self, rgb, title, detail, repeat=1):
        from PIL import Image, ImageDraw
        picture = Image.new("RGB", (320, 272))
        picture.paste(Image.fromarray(rgb).resize((320, 240)), (0, 0))
        draw = ImageDraw.Draw(picture)
        draw.text((4, 242), title, fill="white")
        draw.text((4, 257), detail, fill="white")
        for _ in range(repeat):
            self.writer.append_data(np.asarray(picture))
            self.frames += 1

    def close(self):
        self.writer.close()


def finish_demo(simulator, skill, parameters, output, video, report):
    from run_inference_chunked import observation_of, CAMERAS
    def physics_frame(title, detail):
        observation = observation_of(simulator.task, simulator.blank, False)
        video.append(observation["head_camera"], title, detail, repeat=3)
    context = RackApproachContext(simulator.task,
        log=lambda e: append_jsonl(output / "execution.jsonl", e),
        frame_callback=lambda: physics_frame("DETERMINISTIC RECOVERY", f"Physics tick {context.executed_ticks} | 250 Hz"))
    report["before_privileged"] = context.diagnostics()
    observation = observation_of(simulator.task, simulator.blank, False)
    np.savez_compressed(output / "before_recovery.npz", **{k: observation[k] for k in CAMERAS})
    result = skill.execute(context, parameters)
    report["result"] = result.model_dump(mode="json")
    report["after_recovery_privileged"] = context.diagnostics()
    if result.termination_reason != "completed" or not result.metadata["release_authorized"]:
        report["status"] = "recovery_stopped"
        return report
    report["hanging_check"] = observe_hanging(simulator.task,
        log=lambda e: append_jsonl(output / "hanging_check.jsonl", e),
        frame_callback=lambda tick: physics_frame("PASSIVE HANGING CHECK", f"Physics tick {tick}/750 | no arm motion"))
    report["status"] = "stable_hanging" if report["hanging_check"]["stable_hanging"] else "hanging_not_established"
    observation = observation_of(simulator.task, simulator.blank, False)
    np.savez_compressed(output / "after.npz", **{k: observation[k] for k in CAMERAS})
    video.append(observation["head_camera"], "STABLE HANGING" if report["status"] == "stable_hanging" else "CHECK DID NOT PASS",
                 "One development case | skill unverified", repeat=60)
    return report


def run_demo(bank, checkpoint, simulator, output):
    from run_inference_chunked import observation_of, CAMERAS
    skill, parameters = check_request(bank, checkpoint, {}, "backoff_reapproach_rack")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "checkpoint_id": checkpoint.checkpoint_id,
              "seed": checkpoint.seed, "skill_id": skill.skill_id, "skill_version": skill.version,
              "code_hash": skill.code_hash(), "parameters": parameters,
              "development_only": True, "runtime_eligible": False,
              "policy_source": "recorded frozen-policy action replay", "modal_calls": 0,
              "anthropic_calls": 0, "automatic_detection": False, "status": "error",
              "video_timing": "30 recorded policy target frames/s; recovery and dwell at simulated speed"}
    write_json_atomic(output / "request.json", report)
    video = DemoVideo(output / "complete_episode.mp4")
    step = 0
    def transport(host, port, path, payload):
        nonlocal step
        observation, timing = simulator(host, port, path, payload)
        step = 0 if path == "/reset" else step + 1
        video.append(observation["head_camera"], "RECORDED POLICY REPLAY", f"Action {step}/{checkpoint.step} | 30 targets/s")
        return observation, timing
    try:
        replay = checkpoint.restore(ReplayEnvironment(bank, transport))
        report["replay"] = replay.model_dump(mode="json")
        if replay.status != "matched":
            report["status"] = "restore_rejected"
            return report
        return finish_demo(simulator, skill, parameters, output, video, report)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        return report
    finally:
        video.close()
        report["video_frames"] = video.frames
        report["implementation_hashes"] = {name: sha256(Path(__file__).with_name(name))
                                            for name in ("episode_demo.py", "hanging_check.py")}
        write_json_atomic(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_id")
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robotwin-root", default="RoboTwin")
    args = parser.parse_args()
    bank = CheckpointBank(args.path)
    checkpoint = bank.get(args.checkpoint_id)
    check_request(bank, checkpoint, {}, "backoff_reapproach_rack")
    output, root = Path(args.output).resolve(), Path(args.robotwin_root).resolve()
    if output.exists(): parser.error("output directory already exists")
    from .local_simulator import LocalSimulator
    simulator = LocalSimulator(root)
    try:
        report = run_demo(bank, checkpoint, simulator, output)
    finally:
        simulator.close()
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "stable_hanging": raise SystemExit(1)


if __name__ == "__main__":
    main()
