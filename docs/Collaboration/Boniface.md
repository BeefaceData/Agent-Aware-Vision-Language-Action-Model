# Boniface — hanging phase

## Goal

Evaluate the final rack-hanging phase and improve safe, repeatable behavior. The frozen Neotix VLA remains responsible for most of the trajectory; bounded deterministic skills are available for recovery or the known out-of-distribution rack contact phase. Any analytic hang move must use a fresh, public, rack-grounded target and the measured current arm pose.

## Where to work

- `harness/skills/analytic.py`, `move_to.py`, `rotate_wrist.py`, `gripper.py`, and `release.py`: measured-start motion, wrist, gripper, and guarded release behavior.
- `harness/perception/hang_targets.py`, `pose_grounding.py`, `grounding.py`, and `anchors.py`: rack-relative targets and grounding checks.
- `harness/predicates/rack_seat_gate.py` and `predicates/registry.py`: release authorization.
- `harness/runtime/orchestrator.py` and `assembly.py`: planner feedback and target preparation.
- Tests: `tests/test_analytic_skills.py`, `test_hang_targets.py`, `test_rack_seat_gate.py`, `test_rack_recovery.py`, and `test_skill_orchestrator.py`.

## Test sequence

1. Use a saved checkpoint/replay or a scripted harness scenario to inspect the rack approach and release stages before a full episode.
2. Verify targets are relative to the cached rack frame and current end-effector pose. No hard-coded start joint state, absolute rack coordinates, or teleport-style pose changes.
3. Check approach, contact, release, and retreat as separate outcomes. A successful approach is not proof that the mug was released or hung.
4. Exercise stale/missing rack grounding, depth discontinuities, out-of-reach targets, tracking error, and loss of release support. The safe result should be a typed failure returned to the planner, not an assumed success.
5. Record video only for an authorized serial live run. Use a unique ignored output path and report the exact evidence that supports the outcome.

## Constraints

The task sequence includes left-arm acquisition and table handoff, then right-arm pickup and rack hanging. Do not gate the VLA’s two-arm handoff with an invented arm lock. Avoid changing the grasp/handoff trajectory while isolating the hanging test. The VLA can be called through `VLA_ACT`; it must remain the frozen Modal policy and must not emit actions outside the existing 14-DoF interface. Keep simulator oracle contact/pose checks in `development/` and offline evaluation only.

RoboTwin and CuRobo share the local GPU with the simulator. Do not start concurrent live episodes or planner tests that initialize the simulator. Run CPU/replay checks first and coordinate one bounded live run at a time.

## Branch and PR

Use a clean clone/worktree; the shared workstation’s `main` has uncommitted edits. Do not use `git reset`, `git stash`, or force-push against that checkout.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c hang/<short-topic>
python -m unittest tests.test_analytic_skills tests.test_hang_targets tests.test_rack_seat_gate -v
python -m unittest discover -s tests -q
git diff --check
git add <only-your-files>
git commit -m "Test grounded hang motion"
git push -u origin hang/<short-topic>
gh pr create --base main --head hang/<short-topic>
```

If your checkout is dirty, first create a separate clone/worktree. Keep the PR limited to hanging behavior and its tests. Include the seed/checkpoint, public evidence mode, video/report location, result, and remaining risks. Do not commit raw recordings, checkpoints, weights, or credentials.
