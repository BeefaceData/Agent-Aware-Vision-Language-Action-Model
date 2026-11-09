# Pose estimation for retries: what exists and how to use it

This note is for whoever builds **retries, recovery or other supervisor logic** on top of
perception. It is written to be handed to a coding agent as working context. It describes
branch `perception/pose-estimation` at commit `90d89bd`, and every claim was checked against
that code and the reports listed in section 8.

Read `perception_for_retries.md` (same folder) first. It covers predicates, the provider, and
the retry logic that exists. This note adds only what the pose work changed. For the
illustrated explanation, see `guides/pose_estimation/pose_estimation_guide.pdf`.

**One change to the older note:** its blind spot 1 (end-on toppled mug) is now reduced but not
gone. See section 6.

---

## 1. In one minute

From public RGB-D, SAM 3 masks and nothing else, the provider now fits:

- **the mug** as a cylinder: base centre, axis, tilt, radius, handle direction;
- **the rack** as a pole and a peg: pole foot, peg tip, and the seat where a hung handle rests.

Each answer is **trusted**, **untrusted**, or **absent with a reason**. No simulator state or
mesh is used at runtime.

What already consumes it:

| Consumer | Status |
|---|---|
| `release_ready` veto: a mug whose axis tilts over 45 deg is never safe to release | **On** whenever RGB-D perception runs |
| `mug_pregrasp`, `rack_approach`, the hang's rack frame | **Off** by default, behind `perception.depth_pose_anchors` (section 7) |
| Planner prompts, `planner_summary`, `PredicateRegistry` | **None.** Nothing exposes poses to them. Retries must read the provider directly |

No retry uses poses yet. Sections 4 and 5 say what they can support.

---

## 2. Reading a pose

```python
result = provider.observe(observation)   # cached per observation sequence: no extra SAM 3 call

result.mug_pose          # ObjectPose | None
result.mug_pose_reason   # str, always set
result.rack_pose         # ObjectPose | None, the pose kept for the episode
result.rack_pose_reason
provider.episode_rack_pose   # the kept rack pose, even if this frame's detector failed
```

`ObjectPose` is in `harness/perception/object_pose.py`. All coordinates are world frame, metres.

| Field | Mug | Rack |
|---|---|---|
| `position` | Base centre if `base_known`, otherwise the middle of the visible body | Pole foot on the table |
| `axis` | Unit vector along the body. **Points base to top only if `base_known`** | Up the pole |
| `base_known` | Whether `position` really is the base | Always true |
| `radius_m` | Body radius | `None` |
| `heading` | Unit vector from the axis towards the handle, or `None` | Along the peg |
| `key_points` | `"handle"` if seen | `"peg_tip"`, `"peg_seat"` |
| `quality` | Share of points the fit explains, 0 to 1 | Same idea |
| `trusted` | `quality >= 0.60` | Always true when present |
| `tilt_deg()` | 0 upright, 90 lying. Ignores which end is up | |
| `reason`, `cameras`, `observation_sequence` | Provenance | |

### The trust rule

- **Trusted:** may confirm something or place a target.
- **Untrusted:** may only **block** an action, for example "this reads as lying down, do not
  release". Never use one to allow an action or to aim the arm.
- **`None`:** read the reason. Look again, or fall back. Never guess.

### Reasons

| Reason | Meaning | Sensible response |
|---|---|---|
| `wall_fit_1_camera`, `wall_fit_2_cameras`, `end_cap_fit_...` | Trusted fit | Use it |
| `cylinder_fit_inliers_low` | A pose exists but is untrusted | Veto only |
| `mug_geometry_unavailable` | Mug not grounded this frame | Next observation |
| `object_not_seen`, `too_few_surface_points` | Too little visible | Wait for the arm to clear |
| `no_consistent_cylinder`, `cylinder_centre_unresolved`, `cylinder_radius_out_of_range`, `cylinder_longer_than_object` | The points do not look like a mug | Treat as unknown |
| `cameras_disagree` | Trusted cameras conflict | Treat as unknown |
| `pole_and_peg_fit_...`, `fitted_earlier_this_episode` | Rack pose available | Use it |
| `rack_fit_awaiting_confirmation` | One view seen, waiting for a second to agree | Next observation |
| `peg_reach_out_of_range`, `peg_not_seen`, `peg_not_found`, `pole_not_found`, `pole_not_upright`, `peg_parallel_to_pole` | Rack mask did not give a usable fit | Next observation |
| `table_height_unavailable` | No table mask this frame, so no rack fit | Next observation |

`release_ready` reports `mug_axis_tilted_not_upright` when the axis veto fires.

---

## 3. How good it is

Fresh seeds 10-14, frozen code, 290 frames, **head and wrist cameras only, which is what a live
run has today**. Errors are medians over trusted poses.

| Mug state | Frames | Trusted pose | Axis error | Base known | Base error |
|---|---|---|---|---|---|
| resting | 18 | 78% | 0.8 deg | 14 | 2 mm |
| staged | 71 | 89% | 0.5 deg | 63 | 2 mm |
| gripped on table | 63 | 59% | 1.6 deg | 36 | 3 mm |
| toppled | 71 | 99% | 1.1 deg | 48 | 1.6 cm |
| touching rack | 23 | 70% | 0.7 deg | 15 | 1.2 cm |
| held by left arm | 15 | 60% | 0.8 deg | 8 | 1.4 cm |
| held by right arm | 29 | **28%** | 3.8 deg | 6 | 1.2 cm |

| Other measure | Result |
|---|---|
| Upright/toppled mix-ups | 0 on all 15 seeds |
| Handle heading, median | 3.1 deg (118 frames) |
| Rack peg tip, median (worst) | 3.1 mm (3.6 mm), 5 of 5 episodes |
| Peg seat to the hang target, xy | 3.8 mm median |
| Worst base error on the table | 2.9 cm, so do not read differences under about 3 cm as movement |
| Cost | About 32 ms per observation for the mug; about 164 ms per rack try until kept |

**Confident wrong poses: 4 of 290**, against a target of 0. Section 6 explains them.

---

## 4. Facts you can build on

Each is a suggestion, not a tested predicate. **Score it offline before a planner acts on it**
(section 9 of `perception_for_retries.md`).

**Is the mug lying down?** `pose.trusted and pose.tilt_deg() > 45` caught 69 of 71 toppled frames.

> **Tilt alone does not mean toppled.** The same test is true on 23 frames where the mug was
> legitimately sideways, held or at the rack. Combine it with evidence that the mug is low on
> the table and no gripper is holding it. That combination has not been scored yet.

**Which way is up on a lying mug?** Use `position` and `axis` only if `base_known`. It was known
in 48 of 71 toppled frames. When false, the axis direction is arbitrary.

**Is the mug still where it was?** For a mug on the table, compare two trusted `base_known`
positions. The typical error is 2-3 mm, but allow for the 2.9 cm worst case. This is the
strongest new input for "the grasp failed and the mug did not move".

**Which side is the handle on?** `heading`, as a hint only, and only when the handle is likely
to be in view (section 6). It often exists on lying mugs, depending on how the mug landed: 0 of 29
frames on seeds 0-4, 28 of 55 on seeds 5-9, 43 of 71 on seeds 10-14, at 3-6 deg median error
with the perception camera.

**Where is the rack?** `provider.episode_rack_pose`. The rack does not move, so a kept pose
survives occlusion and detector failures. It may arrive late or not at all (seed 10: frame 112),
so handle `None`.

---

## 5. What it cannot tell you

- **Whether the gripper holds the mug.** Poses have no contact information, and held mugs are
  often unknown. The older note's section 4.2 gaps (`mug_grasp_confirmed`, `handoff_confirmed`,
  `left_transfer_complete` and others) are **still open**.
- **Whether the mug is hung.** The seat says where a handle should end up, not that it did.
- **The handle's exact position.** `key_points["handle"]` is 2.9 cm off (median), more than the
  2 cm hang tolerance. The hang's mug-handle frame still comes from Opus.
- **A mug off the table.** Treat it as unknown. The older note's blind spot 5 still applies.

---

## 6. Known failures

Found on the fresh seeds and deliberately **not patched**, so that those seeds stayed a clean
test. Neither affects a decision today, but both would affect a retry that relies on them.

| Failure | Frames | When | Consequence for you |
|---|---|---|---|
| **Rim read as a handle** | 3 | Handle hidden behind the mug. A band near the rim looks like a handle facing the camera (mug model 4) | `heading` 145-168 deg wrong while trusted. Do not choose a grip side from one frame without another check |
| **Upside-down mug near vertical** | 1 | True tilt 146 deg while tipping. The code assumes a near-vertical axis points up | `base_known` true, base 12 cm off, at the rim end |

Proposed fixes: require handle points to span part of the mug's height, and use the base-end
cues on tilted mugs too. Tune them on seeds 0-4.

**The end-on toppled mug:** the axis veto cut wrong `release_ready` answers on seeds 5-9 from 25
to 6. The 6 that remain are that same mug. The fresh seeds held no such case, so the gain is
unconfirmed. Keep the older note's advice: after an event that could topple the mug, do not
open a gripper on `release_ready` alone.

**Gripper points are not removed** from the fit, which is why held mugs are often untrusted.

---

## 7. Setup and constraints

**Cameras.** Live runs have **no perception camera**: only the capture tools add it. SAM 3 looks
at the head camera each observation, and tries another camera only if the head finds no confident
mug (the wrist for the phase), or, with the `explore_mug_rack` profile, no clear rack (the right
wrist). So most mug poses come from one view, and cross-camera agreement rarely applies.

With the perception camera, trusted poses rise from 75% to 86% on the fresh seeds, and right-hand
held frames from 28% to 59%. It adds coverage but barely changes SAM 3 calls.

**Aiming with poses.** Off by default, and never run in a live episode, so whether it improves
hanging is unknown. Rack results were measured with the `explore_mug_rack` prompts; the default
profile is untested.

```yaml
perception:
  rgbd_enabled: true
  depth_pose_anchors: true    # also needs --perception-factory
  rack_prompt_profile: explore_mug_rack
```

Each target falls back to today's behaviour when its pose is missing.

**Rules that still apply**

- Poses are world coordinates. `planner_summary`, which Opus may see, deliberately has none. If
  a planner needs a pose fact, give it a category ("mug lying down, base known"), not numbers.
- Do not lower `MIN_INLIER_FRACTION` or other thresholds to make a retry fire.
- Check every `None` explicitly, and give every retry a budget.

---

## 8. Measuring and files

Score on saved frames, on CPU, in under a minute per five seeds (`data/` is not in git, so
captures and SAM 3 caches must be made locally or shared):

```bash
conda run -n robotwin python -m development.replays.perception_evaluation \
  --frames data/perception/frames_cam/seed_0 ... --frames data/perception/frames_cam/seed_4 \
  --ignore-camera perception_camera \
  --output data/perception/reports/my_change
```

Omit `--ignore-camera` to score with the perception camera. Each row of `frames.jsonl` has a
`pose`, a `rack` and the mug's true `state`. `summary.json` has `pose_by_state`.

**Seeds.** 0-4 are for tuning. 5-9 have been inspected. Pose fixes must not be tuned on 10-14,
which have now been seen too: confirm pose changes on new seeds (15 onwards). A new retry fact
that has not looked at 10-14 can still use them once as a held-out test.

Reports behind this note, under `data/perception/reports/`: `fresh_without_camera`,
`fresh_with_camera`, `fresh_step9`, `step9_dev`, `step9_holdout`.

| File | What is there |
|---|---|
| `harness/perception/object_pose.py` | `ObjectPose`, specs, cylinder and rack fits |
| `harness/perception/grounding.py` | `observe`, `episode_rack_pose`, `bind_pose_approach_anchor`, the release veto |
| `harness/perception/hang_targets.py` | `rack_frame_from_pose`, `DepthRackFrame` |
| `harness/perception/preparation.py` | How `move_to` targets choose pose or fallback |
| `harness/perception/modal_sam3.py` | Which cameras SAM 3 queries |
| `development/replays/perception_evaluation.py` | Offline truth and pose metrics |
| `tests/test_object_pose.py` | Synthetic depth renderer, for testing without a simulator |
| `docs/Collaboration/Olusola/plans/pose_estimation_plan.md` | Decisions and step-by-step results |
