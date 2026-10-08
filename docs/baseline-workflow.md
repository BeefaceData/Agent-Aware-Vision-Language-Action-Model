# Baseline to report: a contributor walkthrough

Run commands from the repository root. A replay needs Python 3.10 and this
checkout, but no LeRobot, model download, simulator, or network connection. A
live baseline episode additionally needs the [declared setup](../README.md#setup),
the [startup preflight](baseline-startup.md), and the resource and data-use
allowance in PRD #1 and issue #55. The replay exercises the same public
`run_episode` interface with scripted adapters; its outcome is not a LIBERO
measurement.

## 1. Check the environment and select an initial state

For a new installation, follow the Python 3.10 setup and exact preflight commands
in [baseline startup](baseline-startup.md). That check records installed versions,
the pinned policy repository and revision, and the selected LIBERO catalog state.
The example selects `libero_10`, task `0`, initial-state index `0`, and seed `0`.
The state index selects a catalog entry; the seed does not select a different
entry. The preflight verifies the catalog digest, while only an authorized
episode reset can verify application and readback in the simulator.

The corresponding episode command, **only after the live-run gates are met**, is:

```bash
python run_smolvla_episode.py --suite libero_10 --task-id 0 \
  --initial-state-id 0 --seed 0 --device cuda \
  --output-dir outputs/task0_state0_attempt1
```

Choose a new output directory for each attempt. The runner refuses an existing
directory and uses the frozen `HuggingFaceVLA/smolvla_libero` policy at revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Its effective action horizon
is read from the environment and recorded, not inferred from a command default.
See [asset resolution](pinned-baseline-assets.md) for the other pinned files.

## 2. Inspect a completed attempt

In the following commands, replace the path with the output directory printed by
the runner. Keep the entire directory, including the videos, together. Run the
bundle check **before** using its result as evidence:

```bash
python artifact_bundle.py outputs/task0_state0_attempt1 --replay
python episode_timeline.py outputs/task0_state0_attempt1/replay task0_timeline.html
```

The first command validates the final `bundle.json` inventory, SHA-256 checksums,
attempt identity, policy assets, frame index, and sealed replay, then executes a
model-free replay. The second writes a local HTML timeline to a **new** file;
open it to follow observations, proposed and executed actions, dispositions,
camera availability, and terminal evidence. It contains raw observation and
evaluator fields, so retain it under the same permissions as the original bundle.

Read `environment.json` for the actual interpreter, packages, render backend,
controller-source hashes, and resolved device; `policy-assets.json` for verified
model files; and `attempt.json` for the task instruction, state digests, seeds,
effective configuration, and pre-action identity. `result.json` has the final
`status`, `artifact_status`, `success`, `steps`, `stop_reason`, and attempt digest.
The replay's `manifest.json` records its `config` and outcome; `steps.jsonl`,
`frames.jsonl`, and the video files support action and camera inspection.
Compare the episode ID and configuration across these records. The bundle
verifier performs these identity checks, but neither hashes nor replay prove the
scientific correctness of the task outcome.

`status: completed` with `success: false` is an unsuccessful attempt. For example,
`stop_reason: step_limit` means the action horizon was exhausted. A failed
preflight, policy-asset resolution, reset, recorder, or bundle seal is an
infrastructure or evidence failure, not a policy task failure. Such an attempt
may have only partial files and no `bundle.json`; retain them for diagnosis and
do not count them as verified completed outcomes. If a process stopped after
actions, `python execution_progress.py <attempt>/replay/execution.jsonl` can
report the acknowledged prefix, but cannot establish completion or authorize
resumption.

## 3. Reproduce the inspection without live resources

The repository includes a sealed, **synthetic unsuccessful** episode. These
commands work from a plain Python 3.10 environment and do not run a live episode:

```bash
python recorded_replay.py tests/fixtures/recorded_episode
python -c "from recorded_replay import load_recorded_replay; from replay_adapters import ReplayRecorder; r=load_recorded_replay('tests/fixtures/recorded_episode'); sink=ReplayRecorder(); o=r.run(sink); print(o.success, o.steps, o.stop_reason); print([(s[3].action_record.proposed_action, s[2]) for s in sink.steps])"
python episode_timeline.py tests/fixtures/recorded_episode baseline_fixture_timeline.html
```

Expect `success: false`, `steps: 2`, `stop_reason: terminated`, with proposed to
executed actions `[0.1]` to `[0.5]` and `[0.2]` to `[0.2]`. In the timeline,
decision 1 is an override, decision 2 is a pass, and observation 1 has an
unavailable wrist view. The replay manifest's recorded configuration is
`{"seed": 17, "max_steps": 3}`; its source episode ID differs from the fresh
replay episode ID. This fixture has a **trace seal only**, so use
`recorded_replay.py`, not `artifact_bundle.py`: it has no enclosing
`attempt.json`, videos, or final `bundle.json`.

For a successful public-interface comparison, run this independent in-memory
fixture. It has no disk artifacts or model inference:

```bash
python -c "from episode_harness import run_episode; from replay_adapters import successful_replay; f=successful_replay(); o=run_episode(f.config,f.policy,f.environment,f.recorder); print(o.success,o.steps,o.stop_reason); print(f.environment.actions)"
```

Expect `True`, `2`, `success`, and `[('reach', 0.25), ('place', 0.75)]`.
These two fixtures explain success and failure paths in the harness; neither
estimates a baseline success rate. The preliminary one-attempt-per-task
observations in the [README](../README.md#initial-baseline-results) are historical
and separate from these fixtures and any sealed evaluation campaign.
