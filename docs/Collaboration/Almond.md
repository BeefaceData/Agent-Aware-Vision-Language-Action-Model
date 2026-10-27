# Almond — efficient observation and feedback sampling

## Goal

Measure when the harness needs to inspect task progress and ask the planner for feedback, then design a phase-aware sampling schedule that reduces unnecessary calls without missing failures. Grasp timing is only one example; evaluate the relevant acquisition, handoff, and hanging milestones. Use episode traces to propose a schedule rather than hard-coding a guessed step count.

## Where to work

- `harness/skills/vla_act.py`: per-action stop checks inside a VLA chunk and stale-tail discard behavior.
- `harness/runtime/orchestrator.py`: planner turns, post-skill observations, predicate checks, and phase/milestone transitions.
- Runtime presets under `configs/`: `chunk_size`, skill/action limits, and `planner_feedback_chunks`.
- `harness/perception/observations.py`, `trace.py`, and `assessment_log.py`: observation history and evidence records.
- Tests: `tests/test_skill_orchestrator.py`, `tests/test_anthropic_planner.py`, `tests/test_run_skills_episode.py`, `tests/test_runner_evidence.py`, and scenario/replay tests.

## Experiment plan

1. Inventory saved runs and derive phase-specific simulator-step/action ranges for acquisition, left-to-right handoff, and approach/hanging. Separate seeds, configurations, and outcomes; do not infer timing from one episode.
2. Count planner calls, predicate evaluations, image/model requests, VLA chunks/actions, and wall-clock time by phase. Distinguish a new observation from repeated checks on the same observation.
3. Propose sampling schedules with an early-failure trigger and an uncertainty fallback. A milestone becoming possible later is not a reason to suppress clear safety/failure checks earlier.
4. Replay the same recorded scenarios through baseline and candidate schedules. Compare missed/late detections, unnecessary calls, latency, and decisions. Include failures that happen earlier or later than the typical range.
5. Add deterministic CPU tests for schedule boundaries, predicate `unknown`, failure triggers, and stale action discard. Only then request one serial live comparison.

The current runtime presets vary, and the repository’s common chunk size is not the whole sampling cadence. Read the selected YAML before every run. `chunk_size` controls how many actions the VLA returns; `planner_feedback_chunks` controls how many chunks elapse between planner turns in configurations that use it. Per-action stop predicates inside `VLA_ACT` are a separate mechanism. Do not change all three at once or assume a larger chunk automatically reduces all perception calls.

## Constraints

Preserve hard safety checks and the existing public-evidence boundary. Keep the frozen policy unchanged. Do not let “typical timing” convert missing/ambiguous perception into success. Use a single variable per replay ablation and report confidence/coverage, not only average call reduction. No concurrent GPU runs.

## Branch and PR

Use a clean clone/worktree based on current `main`; the shared working tree has uncommitted changes.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c sampling/<short-topic>
python -m unittest tests.test_skill_orchestrator tests.test_anthropic_planner -v
python -m unittest discover -s tests -q
git diff --check
git add <only-your-files>
git commit -m "Tune feedback sampling"
git push -u origin sampling/<short-topic>
gh pr create --base main --head sampling/<short-topic>
```

If the checkout has local modifications, do not reset, stash, or reuse it. The PR should state the replay set, schedule, per-phase call counts, false-negative/late-trigger cases, tests, and GPU run plan. Keep traces and raw observations under ignored `data/`.
