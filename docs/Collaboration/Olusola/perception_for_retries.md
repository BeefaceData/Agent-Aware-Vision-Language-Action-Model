# Perception for retries: what exists and how to use it

This note is for whoever builds **retries that are driven by camera perception**: noticing
that a grasp, handoff or placement failed, and choosing a bounded recovery. It is written
to be handed to a coding agent as working context. It describes the state of branch
`perception/grounding-evaluation` (merged with `main` on 2026-10-29).

Read it together with:

- `docs/Collaboration/Olusola/perception_guide.pdf` (source `perception_guide.tex`): the
  illustrated explanation of predicates, providers, planners, the evaluation loop, every
  perception fix and its measured result. Appendix A is a plan for recovering a toppled mug.
- `CONTRIBUTE_README.md`: the technical map, including the scoring definition of each
  predicate (section on perception evaluation).

Everything below was checked against the code. File references are relative to the
repository root.

---

## 1. Ground rules (do not break these)

1. **No simulator truth at runtime.** Object poses, contacts and labels from RoboTwin
   must never reach the planner, the predicates or the anchors in a real run. The only
   exception is the explicit development oracle (`--development-oracle`), which is
   refused in evaluation. Measured robot state (joint positions, the arms' measured TCP
   poses from `get_{arm}_ee_pose`) is allowed: a real robot has it too.
2. **`unknown` is a valid answer.** Predicates are tri-state: `True`, `False`, `None`.
   When evidence is ambiguous, return `None` with a reason. Never turn `None` into a guess.
3. **Precision before recall.** A wrong `True` makes the supervisor act on something false
   (for example opening the gripper on a toppled mug). A wrong `None` only costs time.
   Every perception change is reported with precision, recall, unknown rate and the count
   of false trues, on the same frames, before and after.
4. **Do not lower a threshold just to make a check pass.** Tune thresholds on development
   seeds 0-4 only. Seeds 5-9 have already been used as a held-out test; confirm new rules
   on fresh seeds (10-14 or later).
5. **No heavy vision models in the local simulator environment.** SAM 3 runs on Modal
   (`modal_apps/sam3.py`). Geometry on depth runs locally on CPU.
6. **Do not change the policy** (weights, inputs, Modal policy behaviour) or unrelated
   prompts. Opus prompt/schema changes are a separate, versioned experiment.
7. **One GPU job at a time.** The deployed policy keeps one observation history; two
   episodes or captures at once mix their inputs.

---

## 2. Mental model in one minute

```
RoboTwin ──RGB-D images──► RGBDPerceptionProvider ──answers──► PredicateRegistry
                           (SAM 3 masks + depth)                     │
                                                                     ▼
            skills ◄── one SkillCommand per turn ◄── planner (scripted or Opus)
            (vla_act stops when its `stop` predicate is True;
             the gripper opens only if `release_ready` is True)
```

- **Provider** (`harness/perception/grounding.py`, class `RGBDPerceptionProvider`):
  turns one observation into facts. SAM 3 paints the mug, rack and table in each camera;
  same-frame depth turns the mask into world points; geometry decides the predicate.
- **Registry** (`harness/predicates/registry.py`, `PredicateRegistry.evaluate`): the one
  place to ask a predicate. It forwards to the provider. `task_success` and
  `chunk_budget_exhausted` are answered by the runtime itself.
- **Planner** (`harness/planner/`): `select(context) -> PlannerSelection` with one
  `SkillCommand`. `context.value("release_ready")` reads a predicate.
- **Skills** (`harness/skills/`): `vla_act`, `move_to`, `rotate_wrist`, `set_gripper`,
  `release`, `abort`. Skills check predicates through `context.check_predicate(name)`.
- **When predicates are computed:** at each planner turn
  (`harness/runtime/orchestrator.py`, `_predicates`) and when a skill checks its stop
  condition. Not after every action: one SAM 3 call per camera per observation is too
  expensive. Results are cached per observation sequence, so a planner turn and a skill's
  stop check never see different answers for the same frame.

---

## 3. Turning camera perception on

Perception is used only when you pass a detector factory and a config with RGB-D:

```bash
# Once per Modal account (the worker must know the perception camera and probe method):
modal deploy modal_apps/sam3.py

# A full-task episode on Modal with the Opus planner and camera perception:
modal run modal_apps/simulator.py::episode \
  --config configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml \
  --planner anthropic \
  --perception-factory harness.perception.modal_sam3:create_detector \
  --output data/runs/my_run --time-limit-minutes 75

# Scripted planner with camera perception, locally:
conda run -n robotwin python run_skills.py --config configs/skill_runtime_rgbd.yaml \
  --perception-factory harness.perception.modal_sam3:create_detector --seed 3
```

- `perception.rgbd_enabled: true` in the config is required (depth is off by default).
- Without `--perception-factory` (and without `--development-oracle`) every perception
  predicate is `None` with reason `no_approved_evidence_provider`.
- `configs/skill_runtime_opus_full_task_rgbd_table_lift.yaml` also enables
  `opus_mug_lift_grounding`: `mug_lifted` then comes from an Opus-chosen support pixel
  plus a reset-time table plane (`harness/perception/table_plane.py`), not from the SAM 3
  mug centroid. The offline evaluator scores the SAM 3 path.
- **The perception-only camera** (`harness/environment/perception_cameras.py`, a fixed
  RGB-D camera on the far side of the table) is currently available only in the
  development capture tools (`--perception-camera default`). There is **no run setting
  yet** to add it to `run_skills.py` episodes. It is a small planned follow-up; until
  then, live episodes use the head and wrist cameras only. The numbers in section 5 show
  what that costs.
- `render_once_per_step: true` (top level of a run config) skips a discarded pre-step
  render in local episodes: same observations, faster. Off by default.

---

## 4. What each predicate gives you with camera perception

### 4.1 Predicates the camera provider can answer

| Predicate | True when | Latches | Typical `False` / `None` reasons |
|---|---|---|---|
| `mug_lifted` | the mug has been seen above its reset height at some point | yes | `mug_geometry_unavailable` (None) |
| `release_ready` | the mug is upright, settled and supported by the table **now** | no | see 4.3 |
| `mug_staged_on_table` | `release_ready` after an earlier lift | no | same as `release_ready`, plus `mug_lift_not_previously_observed` |
| `right_mug_lifted` | the right gripper alone has held the mug up (two stable observations) | yes | see 4.4 |
| `mug_near_rack` | mug and rack points within 0.10 m | no | `mug_or_rack_geometry_unavailable` (None) |
| `mug_visible`, `rack_visible` | the object was grounded in some camera | no | detection status (4.5) |

`mug_visible` and `rack_visible` are available from `provider.predicate_evidence(...)`
but are not in `PREDICATE_NAMES`, so a planner cannot ask for them through the registry
yet.

### 4.2 Predicates the camera provider can NOT answer (important for retries)

For these the provider always returns `None` with reason
`not_provable_from_semantic_rgbd_alone`:

`mug_grasp_confirmed`, `grasp_attempt_failed`, `handoff_confirmed`, `mug_placed_on_table`,
`left_transfer_complete`, `rack_contact_detected`, `mug_released`, `rack_seat_aligned`
(the last one is answered by `RackSeatReleaseGate` when a rack frame is cached).

Consequences you will hit immediately:

- **`GraspRetryPlanner` (left-arm acquisition retry, `harness/planner/scripted.py`)
  aborts under camera perception** after its first grasp attempt with
  `grasp_evidence_unavailable`, because it branches on `mug_grasp_confirmed`.
  Its retry logic was built against the development oracle.
- **`HandoffPlanner` (`harness/planner/handoff.py`)** asks for `left_transfer_complete`
  after the left lift and aborts with `left_transfer_evidence_unavailable` when it is
  `None`.
- **`vla_act`'s `failure_stop="grasp_attempt_failed"`** relies on
  `begin_attempt("grasp_attempt_failed", ...)`, which only the oracle provider
  (`HangingMugPredicates`) implements. `RGBDPerceptionProvider` has no `begin_attempt`,
  so the early-failure stop never fires with camera perception.

So the first real task for perception-driven retries is to **define public versions of the
failure and success facts** your retry needs, composed from what the provider can see
(section 7), measured offline before any planner acts on them.

### 4.3 `release_ready` / `mug_staged_on_table` reasons (in evaluation order)

| Reason | Value | Meaning |
|---|---|---|
| `mug_or_table_geometry_unavailable` | None | mug or table not grounded in this frame |
| `table_support_not_judged_from_wrist_view` | None | the mug was grounded only by a wrist camera; support is judged only from the head or perception camera (end-on wrist views look upright) |
| `insufficient_support_surface_points`, `local_table_surface_unavailable` | None | too few mug edge points or table points near the mug to judge contact |
| `mug_table_support_not_established` | None | the mug's silhouette does not meet the local table surface (held, or unclear) |
| `mug_orientation_unavailable` | None | fewer than 300 usable surface-normal pixels |
| `mug_surface_faces_up_not_upright` | **False** | median vertical normal component > 0.25: the mug is lying on its side (seen side-on). The only `False` reason |
| `mug_above_initial_table_height` | None | support looked OK but the mug is more than 3 cm from its reset height (still held) |
| `mug_not_yet_settled` | None | first observation, or the mug moved > 2 cm since the previous one |
| `mug_silhouette_meets_local_table` | True | upright, settled, supported |
| `public_lift_then_table_support` | True | (staged only) supported after an earlier observed lift |
| `mug_lift_not_previously_observed` | False | (staged only) supported, but no lift was seen earlier |

Note what this means for retries: `release_ready` is almost never `False`. "Not
supported" shows up as `None`. Only a side-on toppled mug gives an explicit `False`. Do
not read `None` as "the mug is in the air".

### 4.4 `right_mug_lifted` reasons

| Reason | Value | Meaning |
|---|---|---|
| `right_gripper_lift_association_latched` | True | confirmed earlier in the episode (latched) |
| `stable_right_gripper_lift_association` | True | just confirmed: mug lifted, not on the table, nearest the right TCP for 2 observations |
| `right_gripper_lift_candidate_not_stable` | None | right association seen once so far |
| `mug_gripper_association_ambiguous` | None | neither arm clearly nearest (association limit 0.24 m, margin 0.06 m) |
| `mug_associated_with_left_gripper` | False | the left arm still has it |
| `mug_still_supported_by_table` | False | `release_ready` is True |
| `mug_not_lifted` | False | below reset height + 3 cm |
| `mug_geometry_unavailable` | None | mug hidden; does not reset the count |
| `initial_mug_height_unavailable`, `measured_arm_tcp_poses_unavailable` | None | missing reference data |

Why 0.24 m: the measured TCP is the arm's frame behind the fingers, and the policy grips
the rim or handle, so a right-held mug's centre is 0.20-0.22 m from it.

### 4.5 Detection status per object (`GroundingResult.status[label]`)

`grounded`, `not_detected`, `low_confidence` (below the score floor: mug 0.60 at the
provider, 0.40 at the SAM 3 worker), `ambiguous_instances` (two candidates within 0.10
score), `occluded_or_invalid_depth`, `occluded_or_empty_mask`, `stale_observation`,
`rgbd_shape_mismatch`, `mask_shape_mismatch`.

Camera order: head camera first, then the perception camera if present, then a wrist
camera. Which wrist is tried first depends on the planner phase passed through
`provider.set_task_phase(phase)`.

---

## 5. How good these answers are (measured)

All numbers: precision / recall over frames, perception provider replayed offline against
simulator labels. "Unknown" frames count against recall, not precision.

| Predicate | Seeds 0-4, head + wrists | Seeds 0-4, + perception camera | Seeds 5-9 held out, + camera |
|---|---|---|---|
| `mug_lifted` | 1.00 / 0.37 | 1.00 / 0.74 | 1.00 / 0.81 |
| `release_ready` | 1.00 / 0.63 | 1.00 / 0.81 | **0.81** / 0.83 |
| `mug_staged_on_table` | 1.00 / 0.47 | 1.00 / 0.82 | **0.75** / 0.81 |
| `right_mug_lifted` | 1.00 / 0.52 | 1.00 / 0.88 | 1.00 / 0.72 |

Unknown rate with the camera (seeds 0-4): `mug_lifted` 0.21, `release_ready` 0.48,
`right_mug_lifted` 0.21.

Mug detection rate by situation, seeds 0-4, without -> with the perception camera:
gripped on table 0.56 -> 0.90, held left 0.69 -> 1.00, **held right 0.10 -> 0.75**,
toppled 0.17 -> 0.97, touching rack 0.47 -> 0.73. Median mug position error when
grounded: 2-4 cm in x/y; the reported point sits about 5-6 cm above the mug's base origin
because it is the median of the visible surface.

### Known blind spots (design your retries around them)

1. **End-on toppled mug.** If the mug falls with its flat base facing the head camera,
   the visible surface faces sideways like an upright wall, and `release_ready` says
   True. In held-out seed 6 this one resting mug produced all 25 wrong trues. Side-on
   toppled mugs are caught (seed 7: 28 of 28). A fix (estimating the mug axis from depth)
   is planned. **Until then, never open a gripper on `release_ready` alone after an event
   that could topple the mug without an independent check.**
2. **Mugs turned on their side in a gripper or at the rack** are often not detected
   (`not_detected`): about 15 frames per 216 on seeds 0-4. No SAM 3 prompt or threshold
   fixed this without adding wrong groundings.
3. **Latency of facts.** One observation per policy chunk (16 actions). `release_ready`
   needs two observations (settle rule), so the first frame after a set-down is
   `mug_not_yet_settled`. `right_mug_lifted` confirms 1-3 observations after the lift.
4. **Wrist-only views never establish table support** (by design).
5. **A mug on the floor** is out of every view. In one frame a wrist camera grounded
   something else 0.74 m away; expect `mug_visible` to be unreliable when the mug has
   left the table.

---

## 6. Other perception you can use

- **`GroundedObject`** (`result.objects["mug"]` after `provider.observe(obs)`):
  `position` (world xyz, median of the mask's depth points), `confidence`, `camera`,
  `valid_depth_fraction`, `surface_points`, `boundary_points`, `normalized_bbox`,
  `median_rgb_0_255`, `visible_surface_extent_m`, `median_normal_z`.
- **`provider.planner_summary(obs)`**: per-object camera, confidence, bbox, colour and
  extent, with no world coordinates. This is what Opus may see.
- **`provider.detection_mask(label, obs, camera=...)`**: the SAM 3 mask itself.
- **Anchors for `move_to`** (`harness/perception/preparation.py`,
  `PerceptionAnchorPreparer`): `mug_pregrasp`, `rack_approach`, `rack_seat`,
  `table_approach`, plus `current_ee`. They are rebuilt from the current observation each
  time a `move_to` is requested; a stale or missing anchor fails the skill instead of
  using old data. `move_to` offsets are limited to 20 cm.
- **Measured TCP poses**: `observation.measured_tcp_poses["left"|"right"]` (local runs).
- **Table plane** (`StaticTablePlaneCache`) and **rack frame** (`StaticRackFrameCache`,
  `hang_targets.py`): cached once at reset from public RGB-D.
- **Physical action history**: `SkillResult` for the previous skill
  (`context.previous`) has `completed`, `execution_uncertain`, `objective_satisfied` and
  `stop_reason` (for example `chunk_budget_exhausted`, `failure_predicate_satisfied`).

---

## 7. The retry logic that exists today

### 7.1 Right-arm grasp retry (works with camera perception)

`HandoffPlanner._select_right_grasp_retry` in `harness/planner/handoff.py`:

1. After a right-arm `vla_act` attempt, if `right_mug_lifted` is `False` (not `None`),
   the attempt is retryable (`_retryable_right_grasp_failure`: the VLA call completed,
   was not uncertain, and stopped on a budget or failure predicate), retries remain
   (`recipe.max_retries`), **and `release_ready` is True** (so opening cannot drop it):
2. `set_gripper(right, open)` -> `move_to(right, current_ee, +5 cm z)` ->
   `move_to(right, mug_pregrasp)` (re-grounded from the current observation) ->
   `vla_act` with at most 176 policy actions and `stop="right_mug_lifted"`.
3. If `right_mug_lifted` is still not True: abort `right_grasp_retry_budget_exhausted`.

If `right_mug_lifted` is `None` at the decision point the planner aborts
(`right_lift_evidence_unavailable`) rather than guessing.

### 7.2 Left-arm acquisition retry (oracle only today)

`GraspRetryPlanner` (`harness/planner/scripted.py`): approach `mug_pregrasp`, `vla_act`
until `mug_lifted`; on failure open the gripper (only if `release_ready`), retreat 5 cm,
re-stage with an optional per-retry offset (`retry_stage_offsets`), retry up to
`max_retries`. It needs `mug_grasp_confirmed`, which camera perception does not provide
(section 4.2).

### 7.3 Legacy and development recovery tools

- `harness/legacy/recovery_skills/` (`retry_grasp.py`, `backoff_reapproach_rack.py`).
- `development/recovery/manual_skill.py`: run an allow-listed recovery skill from a saved
  checkpoint (`development/checkpoints/`), without running the policy to that state.
- `development/checkpoints/`: checkpoint bank, extraction and replay tools.

---

## 8. Suggested approach for perception-driven retries

1. **Write down each decision the retry needs as a fact.** For example:
   - "the grasp attempt failed and the mug is still where it was" -> a public
     `grasp_attempt_failed` / `mug_grasp_confirmed` replacement;
   - "the left hand put the mug down and let go" -> a public `left_transfer_complete`;
   - "safe to open" -> `release_ready` (mind blind spot 1).
2. **Compose them from what the provider already sees**, in the provider or a small
   wrapper, returning tri-state values with searchable reasons. Ideas that only use public
   evidence:
   - failed grasp: after a `vla_act`, the mug is `release_ready` (upright, settled, on the
     table), not lifted, and within a few cm of where it was before the attempt; the
     gripper's measured TCP is near it;
   - successful grasp: `mug_lifted` became True during the attempt, or the mug is above
     reset height and nearest the acting arm's TCP (the same association logic as
     `right_mug_lifted`, for the left arm);
   - left transfer complete: `mug_staged_on_table` True and the left TCP has moved away
     from the mug by more than the association distance.
3. **Score each new fact offline before any planner uses it** (section 9). Require zero
   false trues on development seeds, then confirm on fresh seeds.
4. **Wire it into the planner** as a bounded state machine like
   `_select_right_grasp_retry`: every branch checks `None` explicitly, every retry
   consumes a budget, and a non-retryable state aborts with a specific reason.
5. **Re-ground before every retry.** Use `move_to(target="mug_pregrasp")` so the anchor
   is rebuilt from the current observation; never reuse an old position.
6. **Prefer waiting to acting on `None`.** One more observation (one more chunk or a
   short no-op) often resolves `mug_not_yet_settled` or
   `right_gripper_lift_candidate_not_stable`.

---

## 9. Measuring a new or changed predicate offline (CPU, seconds)

Captured frames plus a SAM 3 cache let you re-score perception without the simulator or
GPU.

1. **Data** (not in git; ask Olusola for a copy or re-create it):
   - `data/perception/frames_cam/seed_0..4/`: development frames with the perception camera
   - `data/perception/frames_holdout/seed_5..9/`: held-out frames (already used once)
   - each folder: `observations/*.npz` (public RGB-D), `labels.jsonl` (simulator truth,
     offline only), `sam3_cache.json` (SAM 3 answers), `manifest.json`
2. **Add truth for your predicate** in `development/replays/perception_evaluation.py`:
   extend `EVALUATED_PREDICATES` and `truth_predicates(label, ...)` with what the
   predicate should answer, using label fields (`mug.pose`, `mug.tilt_deg`,
   `mug.height_change_m`, `contacts`, `grippers`, `lifted_before`, `state`). Document
   the definition in `CONTRIBUTE_README.md` next to the existing table.
3. **Run the evaluator**:
   ```bash
   conda run -n robotwin env PYTHONPATH=$PWD python -m development.replays.perception_evaluation \
     --frames data/perception/frames_cam/seed_0 --frames data/perception/frames_cam/seed_1 \
     --frames data/perception/frames_cam/seed_2 --frames data/perception/frames_cam/seed_3 \
     --frames data/perception/frames_cam/seed_4 \
     --output data/perception/reports/my_change
   # As if the perception camera did not exist (what live episodes see today):
   #   ... --output data/perception/reports/my_change_no_camera --ignore-camera perception_camera
   ```
   It writes `summary.md`, `summary.json` and `frames.jsonl` (every frame's answer,
   reason and truth). It refuses an existing `--output` folder.
4. **Look at the failures**, not just the totals: filter `frames.jsonl` for
   `public == True and truth == False`, then open those frames.
5. **New frames** (GPU): capture with
   `python -m development.capture.labeled_frames --seed N --output ... --perception-camera default`
   (local; only the policy uses Modal) or
   `SIM_GPU=L4,A10G,T4 modal run modal_apps/simulator.py::frames --seeds ... --perception-camera default --output-root ...`,
   then `python -m development.replays.sam3_cache --frames <folder>` per seed. Policy
   calls time out after 5 minutes; a failed seed's folder must be moved aside before it
   is captured again.

Label states you can filter on: `resting`, `gripped_on_table`, `held_left`, `held_right`,
`held_both`, `staged`, `touching_rack`, `hung`, `toppled`, `off_table`.

---

## 10. Testing patterns

- Run everything: `conda run -n robotwin python -m pytest -q tests` (CPU only, no Modal).
- Provider tests: `tests/test_perception_grounding.py` uses a `FakeDetector`, small
  synthetic `camera(...)` / `observation(...)` / `mask(...)` helpers and hand-built
  grounded objects; copy that style for new predicates.
- Planner and skill tests: `tests/test_skills.py`, `tests/test_analytic_skills.py` and
  `tests/test_anthropic_planner.py` build `PlannerContext` objects with chosen predicate
  values, so a retry state machine can be tested without the simulator.
- Evaluator tests: `tests/test_perception_scoring.py` (truth definitions and scoring).
- Every behaviour change needs a test that would have failed before it.

---

## 11. File map

| Area | Files |
|---|---|
| Provider and predicates | `harness/perception/grounding.py` (`RGBDPerceptionProvider.observe`, `predicate_evidence`, `evaluate`, `_table_support_evidence`) |
| Depth and geometry | `harness/perception/rgbd.py` (`RGBDCamera.unproject`, `vertical_normal_components`), `geometry.py`, `pose_grounding.py` |
| SAM 3 client / worker | `harness/perception/modal_sam3.py` (`create_detector`), `modal_apps/sam3.py` |
| Registry | `harness/predicates/registry.py` (`PREDICATE_NAMES`, `evaluate`) |
| Oracle provider (dev only) | `harness/predicates/hanging_mug.py`, `development/simulator/skill_oracle.py` |
| Wiring an episode | `run_skills.py` (`build_evidence`), `harness/runtime/assembly.py` (`perception_provider_for`, `public_perception_evidence`) |
| Turn loop | `harness/runtime/orchestrator.py` (`_predicates`, `BASE_PREDICATES`, `PERCEPTION_PREDICATES`) |
| Planners | `harness/planner/scripted.py`, `handoff.py`, `anthropic.py`, `base.py` |
| Skills | `harness/skills/schemas.py` (`SkillCommand`, parameters), `vla_act.py`, `analytic.py`, `executor.py`, `base.py` |
| Anchors | `harness/perception/preparation.py`, `anchors.py`, `table_plane.py`, `hang_targets.py` |
| Perception camera | `harness/environment/perception_cameras.py` |
| Evaluation tools | `development/capture/labeled_frames.py`, `development/replays/sam3_cache.py`, `perception_evaluation.py`, `sam3_mug_probe.py` |
| Checkpoints / recovery | `development/checkpoints/`, `development/recovery/` |

---

## 12. Checklist for the agent

- [ ] Read sections 1, 4.2 and 5 first; they decide what is possible today.
- [ ] Never read `labels.jsonl` or simulator object state from runtime code.
- [ ] For each retry decision, name the fact, its truth definition and its reasons.
- [ ] Implement it tri-state; handle `None` explicitly in every planner branch.
- [ ] Add a unit test per behaviour and a truth definition in the evaluator.
- [ ] Report precision / recall / unknown rate / false trues on seeds 0-4, with and
      without the perception camera; zero false trues before wiring into a planner.
- [ ] Confirm on fresh seeds before claiming a result.
- [ ] Keep retries bounded (count and policy-action budget) and abort with a specific reason.
