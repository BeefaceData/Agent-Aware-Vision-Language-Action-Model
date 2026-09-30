# Worker procedure

1. Read the supplied issue and numbered acceptance criteria. Inspect existing
   public interfaces, tests and applicable repository guidance. Identify the
   smallest complete change. If the issue needs missing client evidence, unavailable
   resources, a new architectural decision or a blocked dependency, report blocked.
2. Implement that change in this worktree. Keep tests at public module seams and
   include full-episode behavior when relevant. The controller executes approved
   checks; use permitted focused checks during development. Claude has file tools
   only: write tests and report their execution as pending the controller. Do not start unrelated tasks or delegate to other
   coding agents/models.
3. Inspect the diff and each new file; use `git diff` when execution is available,
   or read the changed files with file tools. Check error paths, concurrency and timing,
   API compatibility, provenance, secret exposure and evaluation leakage where
   applicable. Explain each acceptance criterion with concrete file/test evidence.
   For a repair attempt, address supplied findings and re-run checks;
   old reviews no longer apply after an edit.
4. Write `.ralph-run/worker.json` with the schema below, then stop. Every numbered
   criterion must appear exactly once for `ready_for_review`. Use `blocked` for
   incomplete criteria or missing evidence beyond the approved checks. Checks that
   await controller execution may be reported as pending with `ready_for_review`;
   never claim they passed. The report is a claim for the controller and
   independent reviewers to check, not permission to publish or close the issue.

```json
{
  "issue": 2,
  "status": "ready_for_review",
  "summary": "Concrete behavior changed and why.",
  "acceptance": [{"criterion": 1, "evidence": "public test and relevant path"}],
  "checks": [{"command": "python -m unittest ...", "result": "passed/failed/unavailable; actual evidence"}],
  "risks": [],
  "merge_danger": {"door": "two-way", "blast_radius": "harness", "reason": "Explain rollback and affected behavior."},
  "commit": {
    "subject": "feat(harness): :sparkles: add episode interface",
   "why": "Explain the problem and why this implementation resolves it."
  }
}
```

For blocked work, keep the same shape, set `status` to `blocked`, put the concrete
blocker and next human action in `summary`, and list uncovered criteria in
`risks`. Preserve partial edits and failures. Never invent an execution result.
