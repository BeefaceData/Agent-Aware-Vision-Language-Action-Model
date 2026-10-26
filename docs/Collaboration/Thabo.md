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
