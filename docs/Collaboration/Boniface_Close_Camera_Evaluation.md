# Close perception cameras and seed-5 policy-only runs

Recorded 13 November 2026 on branch `hang/stage-evidence` (local commits, not
pushed). All runs are seed 5 on a Modal T4. Run folders are under ignored
`data/runs/`.

## Why the close cameras

Every live supervised seed-4/5 run that reached a right-hand lift stopped before
any hanging motion with `complete_visible_aperture_unavailable`. From the
default perception camera the held mug's handle opening is about 12 px across at
entry, too small for a complete outline or continuous handle depth.

`perception_view="close"` (`--perception-camera close` for baselines) moves the
fixed camera to about 0.54 m from the workspace at 1200x900, placed from the
recorded seed-4 and seed-5 entries: a 2 cm opening is about 33-41 px and the
held mug and whole rack, including the pole foot, stay in frame. A second close
camera (`complementary_view` with the close view, or `close_pair`) looks at the
handle from the opposite side for its back face.

## Runs

| Run | Cameras | Actions | Outcome |
|---|---|---|---|
| `baseline_seed5_20261113T073625Z_548700` | close | 480 (cap) | `action_limit`; opening fully visible 456-480 |
| `baseline_seed5_20261113T143152Z_259000` | close pair | 520 (cap) | `action_limit`; handle threaded on the peg by 512, gripper still closed at 520. First attempt hung at action 383 (`Runner heartbeat timeout: 900 seconds`) and Modal restarted it |
| `baseline_seed5_20261113T155810Z_413400` | close | 900 (limit) | **`native_success` at action 521**, no planner, perception or analytic motion |

In the 900-action run the frozen policy threads the handle onto the peg branch
by about action 496, then opens the right gripper from action 512 (command
0.11) to 521 (0.80). The mug leaves the fingers on the peg and RoboTwin's
checker latches. The baseline stops at the latch, so no frames after release
show whether the mug stays supported: sustained support is unverified.

**The policy is not repeatable on one seed.** The first two runs' commanded
joints differ by up to 0.53 rad at action 200 and 0.37 rad at action 480, so
entry timing and handle orientation vary between runs of the same seed.

## Offline aperture checks (saved frames, unreviewed Claude proposals)

`modal_apps/grounding660.py::saved_crops`, at most two Claude calls per frame,
no motion.

| Check | Frames | Result |
|---|---|---|
| `close_seed5_aperture_check_02` (run 1) | 416, 448-480 | Peg axis validated on 6/6 frames, peg width on 1/6. Handle rejected: Claude called body-bounded openings incomplete; outlines failed strict convexity by 0.5-0.8 px |
| `close_seed5_aperture_check_03` (run 1) | 464, 472, 480 | After the prompt and 1 px tolerance changes, **frame 480's opening passed the geometry check** (2.6 mm residual, 1.1 mm centre and 3.1 deg normal sensitivity, 13 mm inscribed radius). Claude confidence 0.7 |
| `close_pair_seed5_check_01` (run 2) | 472-512 | Only 472's opening passed; 496 disagreed with it (36 mm, 11.5 deg) because the peg was entering the opening. No handle back face reported: in this run the back camera saw the handle edge-on behind the rack post. No peg reference kept |

## What changed in code today

- Close front and back perception cameras, opt-in.
- Prompt: the mug body closing off the handle opening is part of its boundary,
  not an occlusion.
- Opening outlines within 1 px of their convex hull (same vertex order) are
  treated as convex; deeper dents still reject.
- Geometry scored below 0.85 by Claude (but at least 0.5) needs a second
  agreeing frame: the opening after the measured TCP motion within 5 mm and
  5 deg, the static peg's seat and tip within 5 mm. The first validated peg
  reference is kept; clearance uses the larger of measured and 5.5 mm physical
  peg radius.
- The supervisor keeps the policy running while geometry only awaits that
  agreement or a peg reference. Rejected geometry still stops the run.
- Earlier the same day: native success no longer steers staged recovery, a
  takeover needs this frame's own lift evidence, and the seat reuses the first
  validated fit.

## What this means

1. On seed 5 the frozen policy can complete the hang on its own by about
   action 520. Earlier supervised seed-5 runs ended between actions 416 and 480
   (`recovery_blocked`, `visibility_budget_exhausted`,
   `right_grasp_evidence_expired`), before the policy got there. In those runs
   the supervisor's stop decisions, not the policy, ended the attempt.
2. By the time the opening is clearly visible (about 456-480) the policy is
   already moving the handle onto the peg. A takeover window for the analytic
   hang exists but is short, and it depends on run-to-run variation.
3. The back-face camera was placed from one run's handle orientation and did
   not see the back face in the next run. Thickness from two opposite views is
   not yet reliable.

## Recommended next steps

- Compare like with like: several policy-only and several supervised seed-5
  runs, counting `native_success` over the same 900-action limit.
- When geometry is unavailable or rejected, hand control back to the policy
  instead of ending the episode, so supervision can never do worse than the
  policy alone. This changes supervisor behaviour and needs a decision.
- Add a short post-success observation window to the baseline, so a native
  success can be checked for sustained support.
- Rendering two 1200x900 cameras every action slowed the T4 from about 15 to 9
  actions per minute before one attempt hung. Use an L4 or render the
  second camera only at visibility checks.
