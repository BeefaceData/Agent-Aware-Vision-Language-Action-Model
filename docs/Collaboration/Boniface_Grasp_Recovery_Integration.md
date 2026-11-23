# Grasp recovery in supervised hanging runs

Branch `integrate/grasp-recovery`: Thabo's `grasp/retry-sequence` merged into
the supervised hanging work, with public-evidence grasp recovery added.

## Why

In the seed 15-19 evaluation, 8 of the 9 failed runs lost the mug before the
hang: 3 right-hand grasps missed at the handoff and 4 slipped while carrying.
In seed 16 supervised, frame review could not tell a miss from an immediate
slip. Nothing in the supervised runner acted on a lost grasp
(`data/evaluations/seeds15-19_20261113/analysis/`).

## What the merge needed

Thabo's per-action grasp monitor (`HangingMugPredicates`) reads gripper-mug
contacts and the mug pose from the simulator (development oracle). The
supervised runner uses public evidence only, so his planner's monitored grasp
call would stop at once with `failure_monitor_unavailable`. The planner now
takes `runtime_grasp_monitor`; the supervised runner sets it off and uses the
public monitor below instead.

## Detection (every policy-chunk boundary, right arm)

| Signal | Rule |
|---|---|
| Fingers | The right gripper was seen open, then commanded closed for two boundaries; the **measured** finger opening is at or below `empty_hand_opening` on two consecutive boundaries. |
| Camera | This frame's public report: right hand empty, mug on the table, task-state confidence at least 0.75, on two frames. Uncertain frames are skipped; a frame showing the mug held or lifted resets the count. |

The measured finger opening comes from the base finger joint position,
normalized with RoboTwin's gripper command scale
(`LocalEnvironmentClient.measured_gripper_opening`). RoboTwin's
`get_*_gripper_val` returns the last command, which is why it could not tell
an empty hand. Every `action_confirmed` event now logs both arms' measured
opening.

No extra Claude request: the camera signal reuses the report already requested
for the planner turn.

## Recovery (Thabo's sequence)

1. The mug must be visibly upright (`mug_upright` is now part of the public
   report). A toppled mug is never re-grasped; unclear frames wait.
2. Mug position: median visible mug-body surface point from the reported body
   polygon and perception-camera depth.
3. One `MOVE_TO mug_recovery_clearance`, 10 cm above that point, current
   wrist orientation. His skill rises first, travels, then descends.
4. Open the gripper, only if the fingers still measure empty after the move.
   `release_ready` answers from that measurement, without a vision request.
5. Hand control back to the frozen policy with a fresh history. The hanging
   supervisor starts over for the new grasp.

At most 2 re-grasps per episode (`--grasp-retries`). A failed move, a hand
that is not measured empty, or an exhausted budget stops recovery, and the
policy keeps control.

## Offline check on the saved runs (camera signal only)

| Seed (supervised) | Mug lost (frames) | Camera trigger |
|---|---|---|
| 15 (success) | never | none (no false alarm) |
| 16 | ~380 | 384 |
| 17 | ~430 | 496 |
| 18 | ~356 | 400 |
| 19 | ~440 | 496 |

The saved runs have neither finger measurements nor `mug_upright`, so the
finger signal and the recovery motion are covered by unit tests only
(`tests/test_grasp_recovery.py`).

## Finger calibration (policy-only seed 18, 14 Nov 2026)

`data/calibration/finger_opening_seed18`: one policy-only episode, 900 actions,
no Claude calls, with the measured opening logged on every action.

| State (checked in the video) | Measured opening |
|---|---|
| Left hand holding the mug (actions 100-183) | 0.25-0.26 |
| Right hand closed beside the mug on the table (actions 457-900) | 0.1818 on every action |
| Open | ~1.0 |

An empty closed hand stops at its joint limit, not at 0. The threshold is
**0.19**: just above the empty limit and well below a held mug. The first guess
(0.02) would never have fired. On this log the finger signal fires about two
chunks after the right hand closed on nothing.

Still open: a right-hand grasp of the mug has not been measured. If a
handle grasp reads close to 0.19, lower the threshold toward 0.185.

## Before relying on it

- The frozen policy has not yet been seen re-grasping mid-episode. The first
  live runs show whether a fresh history after the clearance move leads to a
  new grasp.

## Options

`modal run modal_apps/simulator.py::supervised_hanging ... --grasp-recovery/--no-grasp-recovery
--grasp-retries 2 --empty-hand-opening 0.19`
