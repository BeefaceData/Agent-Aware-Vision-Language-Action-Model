# Thabo — grasping phase

## Goal

Test the acquisition/grasp portion of `hanging_mug`, especially whether the mug is acquired cleanly and lifted from its support. The VLA-first design leaves contact-rich grasping to the frozen Neotix policy when it is working; the planner may use bounded retries or verified recovery skills after clear failure evidence. Do not replace every imperfect grasp with a deterministic pose rule.

## Where to work

- `harness/skills/vla_act.py`: bounded Modal VLA chunk execution and stop/failure monitoring.
- `harness/perception/table_plane.py`, `opus_mug_state.py`, `grounding.py`, and `predicates/hanging_mug.py`: public evidence for mug support/lift and grasp attempts.
- `harness/planner/anthropic.py`, `harness/prompts/acquisition.md`, and `full_task.md`: planner decisions and phase contract.
- `harness/runtime/orchestrator.py` and `assembly.py`: planner feedback and evidence wiring.
- Tests: `tests/test_acquisition_entry.py`, `test_opus_mug_state.py`, `test_table_plane_lift.py`, `test_analytic_skills.py`, and `test_skill_orchestrator.py`.

## Evaluation approach

1. Separate visual approach quality from actual lift evidence. A gripper near or overlapping the mug is not proof of grasp.
2. Review public RGB/RGB-D observations over time. Preserve `unknown` when the mug base, support surface, or gripper relation cannot be judged.
3. Test retries against distinct failure cases: miss, hollow-space approach, weak/partial grasp, mug still supported, and a successful lift. A retry should use fresh observations and discard stale VLA chunk actions.
4. Report the phase, attempt count, step/chunk boundaries, stop reason, and evidence for the result. Use fixed seeds for replay comparisons and held-out seeds only for evaluation.
5. Validate CPU/replay tests first. Request a serial live episode only after the hypothesis is clear; never run parallel simulator jobs on this machine.

## Pose-estimation integration from PR #205

Read [the pose handover](Olusola/handover/pose_estimation_for_retries.md) and
[the perception handover](Olusola/handover/perception_for_retries.md). The depth
cylinder fit and rack pole/peg fit run inside `RGBDPerceptionProvider.observe()`
and reuse the observation's masks. Trusted mug axes now aim the grasp
pre-approach and measured reverse-retreat targets when
`perception.depth_pose_anchors` is on.
The measured wrist orientation is preserved. Missing or untrusted fits retain
the existing public semantic target; an unavailable public target can use the
logged development-oracle fallback only with `--development-oracle`.

The focused left/right retry presets enable this option:
`skill_runtime_oracle_left_retry.yaml`, `skill_runtime_oracle_right_grasp_48.yaml`,
`skill_runtime_opus_grasp_retry.yaml`, and `skill_runtime_opus_right_grasp_retry.yaml`.
The legacy `oracle` names do not disable public perception: those recorded runs
use public RGB-D first and query the oracle only where explicitly enabled and
public evidence is unknown. The fixed `perception_camera` is included in these
runs; the upstream pose handover's statement that live runs lack this camera
describes its earlier branch snapshot.

Depth rack targeting keeps an existing public rack cache as fallback until a
depth fit is confirmed. Rack-mask inference remains available until that fit is
confirmed, then stops. The mug-handle contact frame still needs Opus or the
explicit development oracle; the fitted handle point has not been validated for
precise hanging. A pose alone does not confirm grasp or hanging success.

The grasp-retry sequence is one measured reverse retreat while holding the
current wrist orientation. After clearance, the skill may return to the measured
orientation from the confirmed pre-approach state using a bounded shortest
rotation, then hands control to a fresh VLA call. It does not open the gripper,
choose a new grasp orientation, or add a separate wrist or re-staging skill.
Apply this sequence to either arm. If the retreat fails or execution is
uncertain, abort for review.

Use one T4 simulator job at a time. The latest recorded right-grasp run used two
retries but did not confirm a lift; inspect it before evaluating the simplified
sequence.

## Constraints

The policy weights, training demonstrations, and Modal worker are frozen. No retraining or policy-side modification. The planner never generates arbitrary joint actions; all VLA actions go through the existing `VLA_ACT` client and its schema. Keep privileged simulator state out of planner observations. Do not modify the handoff or hanging test while evaluating grasp unless the change is required for a clearly documented interface.

## Branch and PR

Start from the latest clean `main` in your own clone/worktree. The shared checkout contains uncommitted changes; do not reset or stash them.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c grasp/<short-topic>
python -m unittest tests.test_acquisition_entry tests.test_opus_mug_state tests.test_table_plane_lift -v
python -m unittest discover -s tests -q
git diff --check
git add <only-your-files>
git commit -m "Improve grasp assessment"
git push -u origin grasp/<short-topic>
gh pr create --base main --head grasp/<short-topic>
```

If `git status --short` is not empty before branching, use a separate clean checkout. Keep the change focused and include tests, evidence limitations, and any live-run artifact path in the PR description. Generated data, videos, weights, and secrets stay untracked.

For an existing shared contribution branch, commit your own work before syncing,
then run `git fetch origin main` followed by `git merge origin/main`. Resolve
overlaps by retaining both the public-perception contract and the recovery
behavior, run the CPU suite, and push normally. Repeat the fetch/merge check just
before opening a PR. Do not rebase a branch teammates already use or force-push
to `main`. A clean merge today cannot prevent conflicts from later edits to main.
