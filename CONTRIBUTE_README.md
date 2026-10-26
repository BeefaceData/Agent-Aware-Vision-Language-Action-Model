# Contributing to the RoboTwin runtime harness

This repository supervises a **frozen Neotix vision-language-action policy** on RoboTwin's `hanging_mug` task. RoboTwin and motion execution run on the local machine; policy inference stays on Modal. Anthropic Opus can select one bounded, typed skill at a time. The planner does not emit low-level joint commands. The baseline inference runner remains available independently, and the policy is not retrained here.

This guide is for contributors extending the runtime, especially public perception. The reviewed project guide in `docs/v1/` and task-specific collaboration notes in `docs/Collaboration/` are versioned. Generated validation reports, videos, observations, checkpoints, older implementation notes, and API credentials remain local and should not be committed.

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

`run_skills.py` is the CLI: it validates options, owns the simulator for one reset, and writes the manifest and summary. The parts it composes are built by named functions in `harness/runtime/assembly.py` (planner, perception provider, reset-frame rack and table grounding, evidence mode), and `harness/runtime/orchestrator.py` runs the turn loop. Each planner turn receives the current public observation and predicate state; its response is schema-validated before execution. The skill executor owns all simulator actions. Skills return a structured result and fresh observation for the next turn.

`VLA_ACT` is a bounded call to the existing frozen policy client in `harness/vla/modal_client.py`. Policy actions retain the fixed 14-DoF interface. Chunks are open-loop between feedback checks, and unused stale actions are discarded when a stop condition or intervention ends a chunk. Analytic motion skills plan waypoints from the measured current arm state; their parameters are bounded. See `harness/skills/` and `harness/runtime/orchestrator.py`.

The independent baseline path is `run_inference_chunked.py`: its simulator server runs RoboTwin locally, while its policy client sends RGB and state to the existing Modal policy worker. The pieces it shares with the runtime live in the harness: RoboTwin task construction, observation payloads and the simulator HTTP client in `harness/environment/robotwin.py`, and the Modal policy transport in `harness/vla/modal_client.py`. Keep changes to supervision isolated from baseline behavior unless the task explicitly requires a shared-interface update.

## Perception boundary

The perception code is under `harness/perception/`:

- `observations.py` and `rgbd.py` define public camera observations and RGB-D handling. Depth, intrinsics, and extrinsics are obtained from RoboTwin through `harness/environment/`; RGB images alone are sent to semantic detectors.
- `modal_sam3.py` is the lightweight client for the separate `modal_apps/sam3.py` Modal app. The app uses a T4 and receives encoded RGB; it returns semantic masks. SAM 3 is not loaded beside the local simulator.
- `grounding.py`, `geometry.py`, `pose_grounding.py`, and `anchors.py` turn public masks or reported image points into depth-grounded 3D references, validate geometry, and cache episode anchors. Camera identity checks use `intrinsic_sha256`/`calibration_sha256` from `rgbd.py`.
- `opus_frame.py`, `opus_rack.py`, `opus_mug_state.py`, and the support grounder in `table_plane.py` contain bounded Opus visual-grounding/assessment calls. They share request plumbing (`vision_request.py`: image encoding, the forced single tool call, confidence/text checks). Their prompt text lives in `harness/prompts/` (loaded with `read_prompt`) so perception requests can be reviewed separately from controller logic.
- `table_plane.py` estimates the reset-frame table plane; `hang_targets.py` derives rack-relative approach targets. `evaluation.py` and `replay.py` support offline scoring and saved-observation analysis.
- `predicates/` converts evidence into task milestones. Keep unavailable or ambiguous evidence as `unknown`; never use simulator-only state as deployed planner input.

The public data path is: RoboTwin observation → RGB-only detector/grounder → local same-frame depth and calibration → typed evidence → predicate evaluation → planner context. `harness/runtime/orchestrator.py` keeps development-oracle inputs separate from the public evidence passed to Opus; oracle wiring itself stays in `run_skills.py` and `development/`, outside the `harness` package. When changing camera selection, masks, grounding, or predicates, update both the provider contract and its replay/scenario tests.

## Repository layout

```text
run_skills.py              supervised episode CLI (one reset, one episode)
run_exploration.py         bounded exploration campaigns (one subprocess per attempt)
run_evaluation.py          freeze and run a frozen evaluation contract
run_inference_chunked.py   frozen-policy baseline (simulator server + policy client)
harness/
  runtime/                 settings, component assembly, the turn loop
  planner/                 Anthropic and scripted planners, prompt projection, parsing
  skills/                  six bounded skills, executor, measured-start motion helpers
  perception/              RGB-D grounding, Opus/SAM 3 clients, table/rack caches
  predicates/              evidence -> tri-state task milestones, rack-seat release gate
  environment/             Observation contract, RoboTwin adapter, local/HTTP clients
  vla/                     Modal policy transport and VLAClient adapter
  memory/ exploration/ evaluation/   snapshots, campaigns, frozen contracts
  legacy/                  pass-through supervisor for the baseline --harness-config hook
  prompts/                 versioned prompt text (executable inputs)
  schemas.py, jsonl.py     shared base types and the append-only JSONL writer
modal_apps/                Modal deployments: policy.py (Neotix), sam3.py (SAM 3)
development/               offline developer tools; never runtime evidence
  simulator/               in-process RoboTwin, the opt-in development oracle, rack geometry
  replays/                 perception/planner replays over saved observations (no simulator)
  capture/                 record policy failures and RGB-D frames; annotate masks
  checkpoints/             failure checkpoint bank, seeded action replay, source provenance
  recovery/                adapters for the legacy development-only recovery skills
  demos/                   recorded demonstrations and sampled episode video
configs/  requirements/  tests/  patches/
```

## Plugging in a component

Each extension point is a small protocol plus one builder, so a replacement does not touch the turn loop:

- **Planner**: implement `select(context) -> PlannerSelection` (`harness/planner/base.py`) and register it in `assembly.build_planner`. The orchestrator only sees the protocol.
- **Semantic detector**: provide a factory returning an object with `detect(images, *, observation_sequence)` that yields `DetectionBatch`es (`harness/perception/grounding.py`), then pass it as `--perception-factory package.module:factory`. `assembly.load_perception_provider` validates it before the simulator starts.
- **Evidence mode**: return `(predicates, prepare)` from a builder like `assembly.public_perception_evidence`; `run_skills.build_evidence` selects one per run.
- **Policy backend**: implement `VLAClient` (`reset_history`, `infer`) from `harness/vla/base.py`.
- **Opus perception request**: build it from `harness/perception/vision_request.py` and a prompt in `harness/prompts/`.

## Where to work

| Area | Main implementation | Useful entry points |
|---|---|---|
| Perception and geometry | `harness/perception/` | `harness/perception/grounding.py`, `table_plane.py`, `opus_rack.py`, `opus_mug_state.py` |
| Planner and prompt contract | `harness/planner/`, `harness/prompts/` | `harness/planner/anthropic.py`, `prompt_builder.py`, `response_parser.py` |
| Skills and runtime loop | `harness/skills/`, `harness/runtime/` | `run_skills.py`, `harness/runtime/assembly.py` |
| Environment RPC and local simulator | `harness/environment/`, `development/simulator/local_simulator.py` | `harness/environment/robotwin.py` |
| Frozen baseline and remote VLA | `harness/vla/`, `modal_apps/` | `run_inference_chunked.py`, `modal_apps/policy.py` |
| Legacy pass-through supervisor | `harness/legacy/` | `run_inference_chunked.py --harness-config` |
| Offline replays and recording | `development/replays/`, `development/demos/` | `python -m development.replays.<tool> --help`, `development/demos/episode_video.py` |
| Scenarios and unit tests | `tests/` | `tests/fixtures/hanging_mug_scenarios.json`, `tests/scenario_replay.py` |
| Runtime presets | `configs/` | `skill_runtime_opus_full_task_rgbd_table_lift.yaml`, `explore_opus_acquisition.yaml` |

For perception work, prefer this sequence: define or update the typed public-evidence result; test image/depth geometry with a saved or synthetic scenario; add a replay/evaluation case; then wire the provider into the existing perception factory. Avoid adding a model dependency to the local simulator environment. Use the existing Modal SAM 3 app for segmentation workloads, and keep learned-policy inference on its existing Modal worker.

## Development and validation

Install the simulation/planner dependencies from `requirements/sim.txt` and `requirements/planner.txt`; perception worker dependencies are isolated in `requirements/perception.txt`. Follow `README.md` for the pinned RoboTwin/CuRobo setup and Modal workers. Set `ANTHROPIC_API_KEY` in the local environment or a secret manager; do not put it in source, configs, logs, or reports. The SAM 3 Modal app uses the Modal secret named `huggingface` with `HF_TOKEN`.

Run CPU-side checks before any simulator episode:

```bash
python -m unittest discover -s tests -q
python -m development.replays.perception --help
python -m development.replays.table_plane_lift --help
```

Recorded RGB-D inputs and run outputs belong under ignored `data/`. Do not run concurrent live jobs on this workstation; local RoboTwin/CuRobo runs share limited GPU VRAM. A live validation should state its seed, config, evidence mode (public perception or development oracle), Modal workers, timeout, output directory, and completion predicate. Keep evaluation changes separate from exploratory presets, and do not treat simulator-native success alone as verified visual rack contact.

## Collaboration notes

Keep pull requests focused on one subsystem and include the contract change, one or more CPU/replay checks, and any effect on the public evidence boundary. Runtime prompt files under `harness/prompts/` are executable inputs and should be versioned with the parser/schema they serve. `README.md` is the short setup guide; this file is the technical map. Start with the task-specific guide for your assigned work:

- [Olusola — perception and object grounding](docs/Collaboration/Olusola.md)
- [Boniface — hanging phase](docs/Collaboration/Boniface.md)
- [Thabo — grasping phase](docs/Collaboration/Thabo.md)
- [Almond — efficient observation sampling](docs/Collaboration/Almond.md)
- [Gabriel — latency analysis](docs/Collaboration/Gabriel.md)

The guides share the same branch, test, and PR workflow. Historical run reports and implementation notes remain excluded; only reviewed team documentation is explicitly unignored in `.gitignore`.
