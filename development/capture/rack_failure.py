"""Capture before a frozen-policy release proposal far from the rack (offline only).

The privileged trigger selects candidates, not verified failure labels. Owns one
local simulator; do not run alongside another GPU simulator process.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from harness.jsonl import append_jsonl
from harness.legacy.config import HarnessConfig
from harness.legacy.pass_through import PassThroughHarness, PassThroughEpisode
from development.capture.failures import CaptureRecorder
from development.checkpoints.bank import CheckpointBank, write_json_atomic
from development.checkpoints.provenance import local_provenance
from development.simulator.rack_geometry import rack_diagnostics, premature_release_candidate


class RackCaptureEpisode(PassThroughEpisode):
    def before_action(self, step, images, state, action, **kwargs):
        diagnostics = rack_diagnostics(self.harness.task)
        record = {"step": step, "privileged": diagnostics, "proposed_action": list(map(float, action))}
        append_jsonl(self.harness.oracle_path, record)
        if premature_release_candidate(diagnostics, state, action):
            self.harness.candidate = record
            self.request_abort("Offline capture: opening proposal while held mug is far from rack")
        return super().before_action(step, images, state, action, **kwargs)


class RackCaptureHarness(PassThroughHarness):
    def __init__(self, task, output):
        config = HarnessConfig.model_validate({
            "harness": {"enabled": True},
            "logging": {"trace_dir": output / "traces", "assessment_dir": output / "assessments",
                        "evaluation_dir": output / "evaluation"},
        })
        super().__init__(config)
        self.task, self.oracle_path, self.candidate = task, output / "oracle.jsonl", None

    def episode(self, *, seed, max_steps, chunk_size):
        if self._active:
            raise RuntimeError("capture episode already active")
        self._episode_index += 1
        return RackCaptureEpisode(self, f"rack_capture_{self._episode_index}_seed_{seed}", seed, max_steps, chunk_size)


def capture(worker, simulator, bank, *, seed=0, steps=600, chunk_size=8, video=True):
    import run_inference_chunked as runner
    if seed < 0 or min(steps, chunk_size) < 1:
        raise ValueError("invalid seed or horizon")
    identity = f"rack_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    recorder = CaptureRecorder(bank, identity, f"{identity}_seed_{seed}", seed, simulator, image_interval=16)
    harness = RackCaptureHarness(simulator.task, recorder.directory)
    provenance = {"schema_version": 1, "local": local_provenance(),
                  "capture": {"seed": seed, "horizon": steps, "chunk_size": chunk_size,
                              "purpose": "natural frozen-policy premature-release candidate"},
                  "policy": {"weights_modified": False, "remote_weights_attested": False},
                  "trigger": "held right mug >10cm from rack midpoint, rising gripper opening proposal"}
    reason, error_type = "error", None
    try:
        success = runner.run_episode(worker, seed, steps, chunk_size, recorder.directory / "rollout.mp4" if video else None,
                                     "local", 0, recorder.directory / "timing", harness=harness, sim_client=recorder)
        reason = "premature_release_candidate" if harness.candidate else ("success" if success else "horizon")
    except Exception as exc:
        error_type = type(exc).__name__
    checkpoint = recorder.finish(reason=reason, provenance=provenance, error_type=error_type)
    result = {"reason": reason, "error_type": error_type, "confirmed_actions": recorder.step,
              "checkpoint_id": checkpoint.checkpoint_id if checkpoint else None,
              "candidate": harness.candidate, "supervisor_privileged_inputs": False,
              "trigger_is_development_only": True}
    write_json_atomic(recorder.directory / "capture_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/checkpoints/rack")
    parser.add_argument("--robotwin-root", default="RoboTwin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or min(args.steps, args.chunk_size) < 1:
        parser.error("invalid seed or horizon")
    from development.simulator.local_simulator import LocalSimulator
    from harness.vla.modal_client import ModalPolicyClient
    bank = CheckpointBank(args.output)
    simulator = LocalSimulator(Path(args.robotwin_root).resolve())
    try:
        result = capture(ModalPolicyClient("robotwin-policy", "PolicyWorker"), simulator, bank,
                         seed=args.seed, steps=args.steps, chunk_size=args.chunk_size, video=not args.no_video)
    finally:
        simulator.close()
    print(json.dumps(result, indent=2))
    if result["error_type"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
