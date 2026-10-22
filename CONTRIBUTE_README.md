# Contributing to the RoboTwin runtime harness

This repository supervises a **frozen Neotix vision-language-action policy** on RoboTwin's `hanging_mug` task. RoboTwin and motion execution run on the local machine; policy inference stays on Modal. Anthropic Opus can select one bounded, typed skill at a time. The planner does not emit low-level joint commands. The baseline inference runner remains available independently, and the policy is not retrained here.

This guide is for contributors extending the runtime, especially public perception. `data/` and `docs/` are local-only and ignored by Git; generated validation reports, videos, observations, checkpoints, implementation notes, and API credentials should not be committed.

## Runtime path

```text
RoboTwin RGB-D + camera calibration + 14-DoF state
          │
          ├── RGB-only image → Modal SAM 3 masks ─┐
          └── public observations → Opus grounding│
                                                  ▼
                         local RGB-D geometry and anchors
                                                  │
                         predicates + public evidence
                                                  │
                         Opus / scripted planner
                                                  │ one validated skill
                   ┌──────────────────────────────┴────────────────────┐
                   │                                                   │
       analytic bounded skills                         VLA_ACT → Modal Neotix
       measured-start waypoints                         frozen policy chunks
                   └──────────────────────────────┬────────────────────┘
                                                  ▼
                                local RoboTwin action + next observation
```

`run_skills.py` wires the supervised loop: load settings, choose evidence providers, initialize the simulator, build the planner/orchestrator, record structured events, and write the episode summary. Each planner turn receives the current public observation and predicate state; its response is schema-validated before execution. The skill executor owns all simulator actions. Skills return a structured result and fresh observation for the next turn.

`VLA_ACT` is a bounded call to the existing frozen policy client in `harness/vla/modal_client.py`. Policy actions retain the fixed 14-DoF interface. Chunks are open-loop between feedback checks, and unused stale actions are discarded when a stop condition or intervention ends a chunk. Analytic motion skills plan waypoints from the measured current arm state; their parameters are bounded. See `harness/skills/` and `harness/skill_orchestrator.py`.

The independent baseline path is `run_inference_chunked.py`: its simulator server runs RoboTwin locally, while its policy client sends RGB and state to the existing Modal policy worker. Keep changes to supervision isolated from baseline behavior unless the task explicitly requires a shared-interface update.

## Perception boundary

The perception code is under `harness/perception/`:

- `observations.py` and `rgbd.py` define public camera observations and RGB-D handling. Depth, intrinsics, and extrinsics are obtained from RoboTwin through `harness/environment/`; RGB images alone are sent to semantic detectors.
- `modal_sam3.py` is the lightweight client for the separate `modal_sam3.py` Modal app. The app uses a T4 and receives encoded RGB; it returns semantic masks. SAM 3 is not loaded beside the local simulator.
- `grounding.py`, `geometry.py`, `pose_grounding.py`, and `anchors.py` turn public masks or reported image points into depth-grounded 3D references, validate geometry, and cache episode anchors.
- `opus_frame.py`, `opus_rack.py`, and `opus_mug_state.py` contain bounded Opus visual-grounding/assessment calls. Their schemas and prompt text live in `harness/prompts/` so perception requests can be reviewed separately from controller logic.
- `table_plane.py` estimates the reset-frame table plane; `hang_targets.py` derives rack-relative approach targets. `evaluation.py` and `replay.py` support offline scoring and saved-observation analysis.
- `predicates/` converts evidence into task milestones. Keep unavailable or ambiguous evidence as `unknown`; never use simulator-only state as deployed planner input.

The public data path is: RoboTwin observation → RGB-only detector/grounder → local same-frame depth and calibration → typed evidence → predicate evaluation → planner context. `harness/skill_orchestrator.py` keeps development-oracle inputs separate from the public evidence passed to Opus. When changing camera selection, masks, grounding, or predicates, update both the provider contract and its replay/scenario tests.

## Where to work

| Area | Main implementation | Useful entry points |
|---|---|---|
| Perception and geometry | `harness/perception/` | `harness/perception/grounding.py`, `table_plane.py`, `opus_rack.py`, `opus_mug_state.py` |
| Planner and prompt contract | `harness/planner/`, `harness/prompts/` | `harness/planner/anthropic.py`, `prompt_builder.py`, `response_parser.py` |
| Skills and runtime loop | `harness/skills/`, `harness/skill_orchestrator.py` | `run_skills.py` |
| Environment RPC and local simulator | `harness/environment/`, `development/local_simulator.py` | `run_skills.py` |
| Frozen baseline and remote VLA | `harness/vla/` | `run_inference_chunked.py`, `modal_policy_chunked.py` |
| Offline replays and recording | `development/` | `development/*replay.py`, `development/episode_video.py` |
| Scenarios and unit tests | `tests/` | `tests/fixtures/hanging_mug_scenarios.json`, `tests/scenario_replay.py` |
| Runtime presets | `configs/` | `skill_runtime_opus_full_task_rgbd_table_lift.yaml`, `explore_opus_acquisition.yaml` |

For perception work, prefer this sequence: define or update the typed public-evidence result; test image/depth geometry with a saved or synthetic scenario; add a replay/evaluation case; then wire the provider into the existing perception factory. Avoid adding a model dependency to the local simulator environment. Use the existing Modal SAM 3 app for segmentation workloads, and keep learned-policy inference on its existing Modal worker.

## Development and validation

Install the simulation/planner dependencies from `requirements-sim.txt` and `requirements-planner.txt`; perception worker dependencies are isolated in `requirements-perception.txt`. Follow `README.md` for the pinned RoboTwin/CuRobo setup and Modal workers. Set `ANTHROPIC_API_KEY` in the local environment or a secret manager; do not put it in source, configs, logs, or reports. The SAM 3 Modal app uses the Modal secret named `huggingface` with `HF_TOKEN`.

Run CPU-side checks before any simulator episode:

```bash
python -m unittest discover -s tests -q
python -m development.perception_replay --help
python -m development.table_plane_lift_replay --help
```

Recorded RGB-D inputs and run outputs belong under ignored `data/`. Do not run concurrent live jobs on this workstation; local RoboTwin/CuRobo runs share limited GPU VRAM. A live validation should state its seed, config, evidence mode (public perception or development oracle), Modal workers, timeout, output directory, and completion predicate. Keep evaluation changes separate from exploratory presets, and do not treat simulator-native success alone as verified visual rack contact.

## Collaboration notes

Keep pull requests focused on one subsystem and include the contract change, one or more CPU/replay checks, and any effect on the public evidence boundary. Runtime prompt files under `harness/prompts/` are executable inputs and should be versioned with the parser/schema they serve. `README.md` is the short setup guide; this file is the technical map. Generated notes and validation reports in `docs/` remain local by repository policy.
