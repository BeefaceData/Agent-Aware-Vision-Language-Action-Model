"""Fresh Modal episode plus an explicitly privileged development rack intervention."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from harness.jsonl import append_jsonl
from harness.legacy.pass_through import PassThroughEpisode
from harness.legacy.recovery_skills.backoff_reapproach_rack import BackoffReapproachRack
from development.capture.rack_failure import RackCaptureHarness
from development.capture.failures import CaptureRecorder
from development.checkpoints.bank import CheckpointBank, write_json_atomic, sha256
from development.demos.episode_demo import DemoVideo, finish_demo, implementation_hashes
from development.checkpoints.provenance import local_provenance
from development.simulator.rack_geometry import rack_diagnostics


def demonstration_boundary(d):
    """Select a pre-contact approach opportunity, not a proven policy failure."""
    return (d["right_bilateral_contact"] and not d["left_contact"]
            and d["mug_position"][2] > 0.78 and d["right_gripper"] <= 0.25
            and 0.02 < d["handle_xy_error_m"] <= 0.16
            and not d["task_success"]
            and set(d["mug_contact_entities"]) == {"fr_link7", "fr_link8"})


class DemoEpisode(PassThroughEpisode):
    def before_action(self, step, images, state, action, **kwargs):
        d = rack_diagnostics(self.harness.task)
        record = {"step": step, "privileged": d}
        append_jsonl(self.harness.oracle_path, record)
        if demonstration_boundary(d):
            self.harness.candidate = record
            self.request_abort("Development demonstration: stop policy before rack recovery")
        return super().before_action(step, images, state, action, **kwargs)


class DemoHarness(RackCaptureHarness):
    def episode(self, *, seed, max_steps, chunk_size):
        if self._active: raise RuntimeError("episode already active")
        self._episode_index += 1
        return DemoEpisode(self, f"fresh_demo_{self._episode_index}_seed_{seed}", seed, max_steps, chunk_size)


class CountedWorker:
    def __init__(self, worker):
        self.worker, self.reset_calls, self.chunk_calls = worker, 0, 0
    def reset(self):
        self.reset_calls += 1
        return self.worker.reset()
    def act_chunk(self, *args):
        self.chunk_calls += 1
        return self.worker.act_chunk(*args)


def run_fresh(worker, simulator, output, *, seed=3, steps=1200, chunk_size=8):
    from run_inference_chunked import run_episode
    if type(seed) is not int or seed < 0 or min(steps, chunk_size) < 1:
        raise ValueError("invalid fresh episode settings")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    identity = f"fresh_{uuid4().hex[:12]}_seed_{seed}"
    bank = CheckpointBank(output / "bank")
    recorder = CaptureRecorder(bank, identity, identity, seed, simulator, image_interval=16)
    harness = DemoHarness(simulator.task, output)
    skill = BackoffReapproachRack(); parameters = skill.validate_parameters({})
    worker = CountedWorker(worker)
    report = {"schema_version": 1, "seed": seed, "skill_id": skill.skill_id,
              "skill_version": skill.version, "code_hash": skill.code_hash(), "parameters": parameters,
              "development_only": True, "runtime_eligible": False, "status": "error",
              "policy_source": "fresh Modal inference", "anthropic_calls": 0,
              "intervention_selector": "privileged development approach opportunity; not a failure diagnosis",
              "video_timing": "30 policy target frames/s; recovery and dwell at simulated speed"}
    write_json_atomic(output / "request.json", report)
    video = DemoVideo(output / "complete_episode.mp4")
    def transport(*args):
        observation, timing = recorder(*args)
        video.append(observation["head_camera"], "LIVE MODAL POLICY", f"Action {recorder.step} | playback 30 targets/s")
        return observation, timing
    provenance = {"schema_version": 1, "local": local_provenance(),
                  "capture": {"purpose": "fresh full-episode demonstration", "seed": seed,
                              "max_steps": steps, "chunk_size": chunk_size},
                  "policy": {"frozen": True, "remote_weights_attested": False}}
    try:
        success = run_episode(worker, seed, steps, chunk_size, None, "local", 0,
                              output / "timing", harness=harness, sim_client=transport)
        report["policy_native_success"] = bool(success)
        cp = recorder.finish(reason="development_intervention" if harness.candidate else "policy_ended",
                             provenance=provenance)
        report["checkpoint_id"] = cp.checkpoint_id if cp else None
        report["trigger"] = harness.candidate
        report["confirmed_policy_actions"] = recorder.step
        if not harness.candidate:
            report["status"] = "policy_success_without_recovery" if success else "no_intervention_opportunity"
            return report
        # Existing runner ABORT has discarded the proposal and chunk remainder.
        # Simulator is not reset; authored recovery continues the same episode.
        return finish_demo(simulator, skill, parameters, output, video, report)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        if not recorder.finalized:
            recorder.finish(reason="error", provenance=provenance, error_type=type(exc).__name__)
        return report
    finally:
        video.close()
        report["video_frames"] = video.frames
        report["modal_reset_calls"], report["modal_chunk_calls"] = worker.reset_calls, worker.chunk_calls
        report["implementation_hashes"] = implementation_hashes(
            "demos/fresh_episode_demo.py", "demos/episode_demo.py", "simulator/hanging_check.py")
        report["artifact_hashes"] = {p.name: sha256(p) for p in output.iterdir()
                                    if p.is_file() and p.name != "report.json"}
        write_json_atomic(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--robotwin-root", default="RoboTwin")
    args = parser.parse_args()
    output, root = Path(args.output).resolve(), Path(args.robotwin_root).resolve()
    if output.exists() or args.seed < 0 or args.steps < 1: parser.error("invalid output, seed or horizon")
    from development.simulator.local_simulator import LocalSimulator
    from harness.vla.modal_client import ModalPolicyClient
    simulator = LocalSimulator(root)
    try:
        report = run_fresh(ModalPolicyClient("robotwin-policy", "PolicyWorker"), simulator,
                           output, seed=args.seed, steps=args.steps)
    finally:
        simulator.close()
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "stable_hanging": raise SystemExit(1)


if __name__ == "__main__":
    main()
