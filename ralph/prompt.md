# Neotix Ralph

Complete one open GitHub issue labeled `ready-for-agent` per iteration in
`BeefaceData/Agent-Aware-Vision-Language-Action-Model`.
Choose the lowest-numbered eligible `[V1-...]` issue with every listed blocker
closed. Do not skip an eligible issue because another looks easier.

Read `AGENTS.md`, `CONTEXT.md`, the chosen issue and its comments, and relevant
ADRs. Respect issue dependencies. Use GitHub Issues as the source of scope.
Continue unfinished work when the previous commits or failure breadcrumb identify it.

Implement the chosen issue, run its relevant checks, and commit the completed
work on the current branch. Follow `.agents/skills/commit-convention/SKILL.md`.
Keep commits meaningful. Once all acceptance criteria are satisfied and checks
pass, check each acceptance-criteria box in the GitHub issue body (`[x]`) using
`gh issue edit <number> --body-file <path>`. Preserve the rest of the issue body.
Re-read the issue and verify that every acceptance-criteria box is checked before
closing it with `gh issue close <number>`. If any criterion lacks evidence, leave
it unchecked and report `[BLOCKED]`. After closing, end with:
`<promise>ISSUE COMPLETED #123</promise>`
Replace `123` with the chosen issue number. Emit this only after completing and
closing that issue; partial progress does not count. Complete only one issue
per iteration. The loop counts distinct completed issues, regardless of commits.
Preserve unrelated local changes. Keep credentials and generated artifacts out
of commits. Report what changed and the checks actually run.

If no agent-ready tasks remain, output exactly:
`<promise>NO MORE TASKS</promise>`

If the task is blocked, output `[BLOCKED]` followed by the reason and stop.
Only emit those signals when they describe the result of this iteration.

## Scientific contract
The goal is a measured improvement of at least **10 percentage points** over our
own frozen-policy baseline, with the PRD's uncertainty and evaluation rules.
Both VLA and VLM weights remain frozen. V1 permits bounded recovery tools and
bounded numeric action adjustments, with temporal failure detection and abstention.
Automatic harness/prompt/tool-code modification and instruction rewriting are v2.

Preserve equal-weight LIBERO-10 reporting, baseline/supervisor/fixed-memory
comparisons, action-horizon fairness, recovery accounting and outcome-linked memory
provenance. Keep fixed-memory evaluation separate from adaptation experiments.
Privileged simulator state and ground-truth success are evaluator data, not
supervisor inputs. Keep holdout data out of development and memory selection.
Report every attempted trial and its disposition; distinguish harness faults from
policy failures. Never fabricate results, omit failures, tune on held-out outcomes,
or claim simulation proves physical bimanual towel-folding performance.

Issue readiness authorizes scoped implementation, not paid VLM trials, dataset
uploads, training, benchmark campaigns or robot actuation. Those require the PRD's
resource, data-use and client gates. Missing evidence or authorization means blocked.

For simulator controller scaling and frame evidence, read `docs/simulator-controller.md`.
