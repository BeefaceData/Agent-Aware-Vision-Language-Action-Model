# Neotix capstone PRD discovery

Status: high-level discovery complete on 2026-09-29, including Q20-Q22; the team delegated engineering details within the settled scope. The canonical specification is [GitHub issue #1](https://github.com/BeefaceData/Agent-Aware-Vision-Language-Action-Model/issues/1), published through to-spec with the ready-for-agent label. This file preserves discovery provenance; implementation requirements and readiness gates live in that issue.

## Sources and authority

- [VLA Neotix_3.pdf](../neotix-docs/VLA%20Neotix_3.pdf): the designated client-context source. Distinguish its requirements, recommendations, and explicitly open client questions.
- [VLA Neotix_2.pdf](../neotix-docs/VLA%20Neotix_2.pdf): draft ideas and architecture; page 13 introduces an additional offline harness-improvement loop. Draft proposals are not automatically accepted decisions.
- Team answers in the ongoing grill-with-docs session (2026-09-27 and 2026-09-29): authority for decisions below; later clarifications supersede earlier assumptions.
- README and runner: implementation evidence. Historical results in the README are reported observations and have not been independently reproduced during discovery.

## Product and objective

- Primary users: Neotix research engineers.
- Deliver a working inference harness around a frozen VLA, with a frozen VLM supervisor requesting bounded corrections and with outcome-linked memory. Documentation supports installation, operation, evaluation, and handover.
- Minimum outcome: **+10 percentage points in task success over the relevant frozen-policy baseline**. The target applies to simulation and the client application, each measured on its own defined tasks and baseline. The primary client task is bimanual towel folding.
- Example: 50 successful folds per 100 attempts becomes at least 60 per 100 under comparable conditions. This is a 10-percentage-point gain, not a 10% relative gain.
- Q22 refines the initial intervention scope to include both predefined bounded recovery tools and bounded numerical action adjustments in v1, together with outcome-linked memory. Supervisor changes to the VLA instruction are deferred to v2 alongside automatic harness modification; see [ADR-0001](../adr/0001-defer-automatic-harness-modification.md) and [ADR-0005](../adr/0005-bounded-recovery-tools-and-action-adjustments.md).
- Both VLA and supervisor VLM learned parameters remain frozen. Memory may accumulate experience in its designated development/adaptation mode; see [ADR-0002](../adr/0002-improve-through-memory-with-frozen-models.md).
- The user reaffirmed that the immediate objective is the +10-point gain. Further autonomous harness improvement is a later ambition. The examples of 90-95% success from a 70% baseline express desired headroom; 100% within three minutes is illustrative, not an accepted requirement.
- [ENPIRE](https://research.nvidia.com/labs/gear/enpire/) and [ASPIRE](https://research.nvidia.com/labs/gear/aspire/) are related work and engineering inspiration. Direct benchmarking against them is not required. No claim of outperforming them is accepted. See the [source check](../research/enpire-aspire-fact-check.md).

## Selected development baseline

- Develop first on **LIBERO-10**, with all ten tasks contributing equally to the simulation acceptance metric. Use one or two informative tasks during development; final evaluation includes predefined initial states and repeated trials across the suite, with per-task results reported.
- Selected policy: **`HuggingFaceVLA/smolvla_libero`**. This resolves the earlier `lerobot/smolvla_base` link in favor of the existing LIBERO-compatible checkpoint.
- Planning revision: **`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`**, obtained from the [Hugging Face model metadata](https://huggingface.co/api/models/HuggingFaceVLA/smolvla_libero) on 2026-09-27. Record and enforce this revision in the implementation and experiment manifests before evaluation. The runner has not yet been changed to enforce the pin, and this does not recover the provenance of earlier reported runs.
- Preserve the selected checkpoint and processors across comparisons. Adopting an already task-adapted checkpoint and then freezing it is consistent with the no-training experiment.
- Existing integration uses LeRobot 0.4.3. The selected [checkpoint configuration](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json) declares an 8-value state, 7-value action, two image inputs, `chunk_size=50`, and `n_action_steps=1`.
- In [LeRobot v0.4.3](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/smolvla/modeling_smolvla.py#L293), the checkpoint's cadence predicts a chunk but queues one action, then recomputes on the next normal call. A plain request to replan is not an identified additional recovery capability; a useful intervention must alter execution, conditioning, or a defined sampling/selection procedure.
- The installed simulator/controller configuration, initial-state selection, complete dependencies, compute configuration, and quantitative trial allocation still need to be captured before experiments are comparable.

## Portability and client integration

- LIBERO is the current one-arm setting; the client robot uses two arms. Develop independently of the pending client interface; see [ADR-0003](../adr/0003-develop-in-libero-before-client-integration.md).
- Accepted portability scope: reuse the harness, experiment recording, supervisor interface, and memory machinery. Policy/environment adapters change; task descriptions, correction tools, limits, and relevant memory may differ.
- Q20 accepts an **unchanged-harness transfer probe** before task-specific tuning: freeze harness logic and supervisor prompt templates, permit necessary adapters, client task instruction and robot configuration, and start with empty cross-episode memory. Simulation-memory transfer is a separate optional experiment. Transfer performance remains an empirical result.
- Client-side action representation, frames, arm coordination, timing, and controller limits remain unknown. A schema accepting two arms does not demonstrate working bimanual supervision.
- **No client demonstrations are expected.** This supersedes the earlier discussion about using future demonstrations. The project concerns inference with an already trained client policy; development and evaluation will produce their own rollout traces. Client demonstrations are not a training or memory-seeding dependency.

## Evaluation agreements

- Minimum acceptance combines an observed improvement of at least 10 percentage points and an uncertainty interval supporting a positive improvement, under a protocol fixed before final testing. The PRD specifies a 95% stratified paired cluster-bootstrap interval as the engineering default; final trial allocation is a pre-evaluation readiness gate using development throughput and statistical feasibility.
- The uncertainty interval concerns the estimated difference in success rates. It is not a guarantee that future deployments or a new task distribution retain at least +10 points. Aiming above the threshold provides a development goal, not a statistical guarantee. See [NIST's confidence-interval explanation](https://www.itl.nist.gov/div898/handbook/prc/section1/prc14.htm).
- Fixed-memory comparison: (A) frozen VLA baseline; (B) supervisor with cross-episode memory disabled; (C) same supervisor with a frozen development-memory snapshot.
- Separate adaptation experiment: begin from a declared snapshot, permit memory updates, and report performance against episodes/experience consumed. See [ADR-0004](../adr/0004-separate-fixed-memory-and-adaptation-evaluation.md).
- The primary comparison uses the same action horizon for baseline and supervised configurations. Recovery actions consume that budget; resets begin a new attempt. Report elapsed time, intervention counts, and assistance alongside success (Q24, 2026-09-29).
- An optional **uncapped-action-horizon ablation** may remove the standard task action limit to inspect additional recovery. It is separate from the primary +10-point acceptance result. Its operational stopping rule and cost/time reporting must be specified before a run; no run has been requested or launched.
- The PRD defines memory admission, failure-label development, reporting and failure-handling rules. Exact held-out initial-state splits, repetitions, operating limits and the client folding-success rubric must be fixed at their named readiness gates before evaluation. Simulation and physical outcomes are reported separately.

## Initial runtime decision

- The first simulation PoC may pause synchronously while the supervisor decides (Q23, 2026-09-29). Record all waiting time.
- Use measured latency and client controller requirements to choose the physical execution strategy later. A synchronous simulation result does not establish timely real-robot supervision.
- Q21 accepts chronological main/wrist RGB, task instruction, available robot/end-effector/gripper state, proposed/recent executed actions and timestamps. Privileged simulator object poses and hidden task-success predicates remain outside supervisor correction decisions. Q22 includes recovery tools and numerical adjustments in v1; the user explicitly requires robust failure detection. The PRD makes temporal evidence, uncertain diagnoses, calibration and false-intervention reporting concrete.

## Resources and calendar

- The team clarified on 2026-09-29 that the client setup is readily available, but the client requires a demonstrated **+10-percentage-point simulation improvement before proceeding to physical integration**. This is an explicit client gate, not simply an unspecified hardware procurement delay. The concrete client interface is still absent from this workspace.
- The end-of-October milestone **includes client-interface integration and actual physical-task results**, following a SmolVLA/LIBERO PoC. It is not silently replaced by simulation-only acceptance.
- The end-of-October physical milestone remains in force after the simulation gate. The user also accepted requesting a minimal interface contract and runnable example during PoC development; Boniface coordinates those client questions. The client gate's review turnaround and named signoff contact remain to be recorded.
- Passing the simulation gate supports advancing to real-world integration. It does not establish the physical +10-point improvement; evaluate that independently against the client's frozen-policy baseline.
- The original 2026-09-27 planning snapshot gave 34 days to the provisional date 2026-10-31. As of 2026-09-29, 32 days remain: about 4.6 weeks and 274 nominal person-hours at 60 hours/week before overhead. The official final deadline remains unspecified.
- Robot access/reset arrangements, interface handoff timing after the simulation gate, compute/API budgets, and data-use permissions are not yet documented. Read-only workspace and GitHub inspection found no additional client-interface pointers; this does not establish that resources are unavailable elsewhere.

## Team and workflow

| Member | Weekly availability | Agreed responsibility |
| --- | --- | --- |
| Thabo | 12 hours | PRD allocation: policy/environment adapters and reproducible baseline |
| Boniface | 12 hours | Coordinate client questions; PRD allocation: evaluation, recording and reporting |
| Almond | 12 hours | Review code; PRD allocation: harness and intervention execution |
| Jean Gabriel | 12 hours | PRD allocation: supervisor, failure detection and VLM/resource measurements |
| Samuel Olusola | 12 hours | PRD allocation: memory, snapshots and adaptation support |

Total: 60 person-hours/week. All members contribute using coding agents. The PRD now assigns a primary owner and reviewer per module under the team's delegated engineering decision; review is shared rather than falling solely on Almond. Human time covers understanding, review, integration, debugging, and experiments as well as code production.

The local AFK loop for assigned GitHub issues remains planned **after this grilling session**. It has not been configured or launched.

## Current implementation and control evidence

- Tracked implementation: one selected-task SmolVLA/LIBERO runner with main-camera video, executed actions, outcomes, and aggregate timing. It does not implement supervision, interventions, memory, or a client robot adapter.
- Historical README results use one episode per task; raw artifacts are absent from the tracked repository. Those observations do not establish bimanual towel-folding performance.
- Source inspection of upstream LIBERO and its specified robosuite 1.4.0 dependency identifies Panda `OSC_POSE` commands: six pose-control values followed by one gripper value. Commands are scaled controller inputs, not metres. Runtime configuration must be checked before fixing corrective scales. [LIBERO wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py), [controller](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py), [scaling](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/base_controller.py).
- Main/wrist RGB and proprioception are available signal categories in the simulation integration. Q21 accepts deployable observation/history inputs while reserving privileged object poses and task-success predicates for evaluation. [LeRobot interface](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/envs/libero.py).
- The runner has a seam before `env.step` for action selection/override. Final actions would need validation after modification. The selected recovery tools and numerical adjustments will use this seam; neither mechanism is implemented yet.
- Missing instrumentation includes proposed-versus-executed actions, wrist/proprioceptive history, per-step timestamps, observation age, supervisor latency, and intervention evidence.

## Design tree

| Decision branch | State |
| --- | --- |
| Product users, frozen models, initial memory mechanism | Settled |
| Related-work role | Inspiration; no mandatory direct comparison |
| Simulation checkpoint | Selected and revision recorded; enforcement pending implementation |
| Simulation acceptance population | All ten tasks, equal weighting; initial states and repetitions pending |
| Minimum acceptance evidence | Observed +10 points and interval supporting positive improvement; PRD specifies analysis and a pre-evaluation allocation gate |
| Evaluation modes | Fixed-memory comparison plus separate adaptation experiment |
| October milestone | Physical integration and results retained; client first requires demonstrated +10 points in simulation |
| Production portability | Shared machinery, replaceable adapters and task-specific configuration accepted |
| Unchanged-harness curiosity experiment | Q20 settled: frozen harness/prompt templates; necessary adapters/configuration; empty memory; measure before task tuning |
| Client demonstrations | Not supplied or required; generate inference rollout traces |
| Supervisor observation contract | Q21 settled: deployable temporal inputs; privileged state and success predicates evaluation-only |
| Failure detection and correction authority | Q22 settled: robust detection with bounded tools and numerical adjustments in v1; PRD defines initial failure hypotheses and eligibility gates |
| Simulation scheduling | Synchronous paused PoC accepted; measure waiting time |
| Physical scheduling, latency, and fallback | Pending measured latency and client contract |
| Action horizons | Matched primary budgets accepted; uncapped-horizon ablation optional and separately reported |
| Memory admission, retrieval, and outcome attribution | Specified in the PRD; fixed-memory and adaptation remain separate |
| VLM selection, compute/API budget, and deployment | PRD selection criteria and owned resource gate; actual provider and allowance require evidence before runs |
| Evaluation splits, throughput, trial allocation, and stopping rule | PRD protocol and owned freeze gate; measured allocation precedes held-out testing |
| Module interfaces and team ownership | PRD defines public contracts and primary/reviewer allocation |
| Research-engineer workflow and handover | Specified through PRD user stories and acceptance tests |
| AFK issue workflow | Future work after specification and task decomposition; not launched |

## Recording rules

Resolved terminology belongs in [CONTEXT.md](../../CONTEXT.md), consequential trade-offs in `docs/adr/`, and implementation work in GitHub issues when the team is ready. Preserve open decisions without converting recommendations into accepted requirements. The team explicitly requested the to-spec workflow and chose separate public module tests in addition to complete-episode tests. Specification publication does not launch experiments, model inference, physical robot operation, or an AFK campaign.
