# Neotix issue loops

Three entry points: `codex-afk.sh`, `cursor-afk.sh`, `claude-afk.sh`.
There is no generic `afk.sh`. The internal Python controller shares lifecycle
rules so providers cannot drift into different completion/merge policies.

One invocation handles one explicitly selected GitHub issue, including at most
two fresh worker attempts. It stops at a reviewed PR for a **human merge**.
It never chooses the next issue, merges a PR or closes an issue directly.

```text
assigned issue -> live blocker checks -> atomic branch claim -> isolated worktree
  -> worker -> controller checks -> commit/push -> draft PR
  -> fresh Astra review -> ready for human
       revise -> one bounded repair attempt + fresh checks/review
       blocked/error -> retain evidence and stop
```

## Models and accounts

[policy.json](policy.json) is the reviewed configuration, not an environment
variable default that can silently fall back.

| Role | Client | Model | Effort |
| --- | --- | --- | --- |
| Coding | Codex | gpt-6-sol | xhigh |
| Coding | Cursor | grok-4.6 (CLI ID cursor-grok-4.6-xhigh) | xhigh |
| Coding | Claude Code | claude-fable-5-1 | xhigh |
| Independent review | Codex | gpt-6-astra | medium |
| Merge | GitHub | A human teammate | Required |

The first two coding pins are the team's requested versions. Claude coding uses
Fable at xhigh as the initial frontier choice; verify access on each teammate's
personal subscription. A model being newer does not authorize changing this policy.
The review model is the team's selected quality gate, not proof of a universal
cross-provider capability ranking.

**The client's Claude API key is reserved for VLA supervision.** There is no Claude
review stage. Agents and approved tests receive an environment stripped of API,
GitHub and cloud credential variables. Coding/review require personal CLI logins;
the controller alone uses `gh` for GitHub operations. Every machine running a full
loop therefore needs personal Codex access to Astra, even if its worker is Cursor
or Claude. An unavailable reviewer stops the run; it never substitutes a cheaper
model or self-review.

Subscriptions have usage limits; this setup does not promise unlimited/free usage,
student-pack entitlements or automatic top-ups. It does not provision API keys.
An inherited environment filter is not an OS vault: use a development account with
no client secrets in accessible files. The isolated worktree omits ignored root
`.env` files. Approved checks execute in a container from a preloaded, digest-pinned image, with
no network, no credentials, a read-only source mount and an unprivileged user.
Dependency changes needing a new image require supervised provisioning before the run.

Model/CLI references checked on 2026-09-30:
[Sol](https://developers.openai.com/api/docs/models/gpt-6-sol),
[Astra](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Grok 4.6](https://prod.cursor.com/docs/models/grok-4-6),
[Claude model IDs](https://platform.claude.com/docs/en/models/overview),
[Codex CLI](https://developers.openai.com/codex/cli/reference/),
[Cursor CLI](https://cursor.com/docs/cli/reference/parameters),
[Claude CLI](https://code.claude.com/docs/en/cli-reference).
Actual account access is checked separately.

## Bootstrap the team

1. Review and land this setup through a supervised PR before running real issues.
   The controller compares its required files with `origin/main`; unpublished local
   prompts must not silently become another teammate's production instructions.
   Include `ralph/`, the workflow, `.gitignore`, `.gitattributes`, `AGENTS.md`,
   `CONTEXT.md`, `docs/agents/`, relevant `docs/adr/` and the project commit skill.
   Keep raw client PDFs and `.scratch/` publication artifacts out of that PR.
2. Install Python 3.11+, Git 2.43+, Bash, Docker, `gh`, the chosen coding CLI and Codex **in the
   same OS environment**. Authenticate personally using `gh auth login`,
   `codex login`, `agent login`, or `claude auth login` as applicable.
3. Cursor AFK requires a Linux/macOS sandbox. On Windows use WSL with a Linux
   checkout, Linux CLI installation and its own login. The tested Windows Cursor
   CLI rejects sandbox mode; this launcher does not disable it. Codex uses
   workspace-write for coding and read-only for review, with the elevated Windows
   sandbox selected on Windows. Claude is an edit-only worker with Read/Edit/Write/Glob/Grep tools;
   the controller runs its tests. Bash and external-service tools are unavailable.
   No launcher uses a sandbox-bypass flag. Cursor permits unattended writes inside
   its required sandbox; it never switches to native Windows allow-all mode.
4. Run the doctor and small smoke from an ordinary teammate terminal:

```sh
bash ralph/codex-afk.sh --doctor
bash ralph/cursor-afk.sh --doctor
bash ralph/claude-afk.sh --doctor
python ralph/smoke.py codex
python ralph/smoke.py cursor
```

The smoke consumes a small amount of personal model usage, writes only synthetic
temperature-conversion fixtures under ignored `.ralph/smoke/`, and never touches
GitHub. Codex smoke also exercises Astra read-only review. The doctor checks login
and available Cursor model IDs, not coding capability or unrestricted entitlement.
A managed parent application's permissions can still prevent a nested CLI from
editing; do not weaken those permissions to make a smoke look successful.

## One issue from scrum

Assign exactly one implementation ticket to the teammate's GitHub account. It
must be open, have `capstone-v1` and `ready-for-agent`, have acceptance criteria,
and have no open native or textual blockers. Module contacts in issue bodies do
not substitute for GitHub assignment. Optional ablations also need
`--include-optional`; a triage label does not authorize paid trials or robot use.

Copy [checks.example.json](checks.example.json) to `.ralph/checks-ISSUE.json`.
Set the issue number, a preloaded Python-capable container image pinned by `@sha256:...`, and
the actual test commands appropriate to that ticket. Commands run inside the
container, so use `python`, not a host-specific Python path. Preload/build the image
with the approved development dependencies; AFK validation never pulls images.
These are argument arrays, not shell snippets. They are deliberately approved
before the agent writes code; the worker cannot replace them with `echo success`.
The example pins the Python 3.13 image used for the isolation smoke; provision any
additional approved dependencies into a new pinned image. It presumes a future `tests/` suite and is **not** sufficient acceptance
coverage for every issue. Add public module tests and episode-level checks as
needed. Preload the approved image and dependencies before AFK execution. The controller
checks image availability before claiming an issue.

```sh
bash ralph/codex-afk.sh --issue 2 --checks .ralph/checks-2.json --dry-run
bash ralph/codex-afk.sh --issue 2 --checks .ralph/checks-2.json
# Choose ONE coding launcher for the issue:
bash ralph/cursor-afk.sh --issue 2 --checks .ralph/checks-2.json
bash ralph/claude-afk.sh --issue 2 --checks .ralph/checks-2.json
```

PowerShell can use the same controller:
`python ralph/runner.py --backend codex --issue 2 --checks .ralph/checks-2.json`.
For Git Bash, `RALPH_PYTHON` can select an installed Python executable.

The dry run makes no remote mutation and calls no model. A real run creates the
atomic remote claim `ralph/issue-N`; an existing branch blocks another machine.
The working checkout is `.ralph/worktrees/issue-N`. The user's original checkout
is untouched. Every fresh worker receives **both** [prompt.md](prompt.md) and
[worker-prompt.md](worker-prompt.md), the live issue/PRD, issue discussion,
acceptance criteria, repository instructions, ADRs and the approved check plan.
Cursor and Claude receive a short positional instruction to read the complete
packet in `.ralph-run/worker-context.md`; the full packet is also retained in the
invocation's local evidence folder.

The controller rebuilds a private Git index from inspected paths and creates a
candidate commit without invoking hooks. Tests run against exact Git blobs (without archive transformations) copied with
verified Unix modes into a fresh Docker volume, excluding worker-created ignored
artifacts. Tests mount that volume read-only. It checks receipt shape,
changed paths, conventional commit structure, test exit codes and zero-test runs. It publishes a
draft only after checks pass. The independent reviewer gets the full changed files,
diff, scope and actual controller test outputs. A passing review must name the
exact base/head, cover all acceptance criteria and contain no blocking finding.
Malformed responses, stale reviews and large review packets stop publication
readiness. Issue, PRD or discussion changes also stop the run for reconciliation.

## Stops, repairs and handoff

Each run has a 90-minute wall-clock deadline, at most two worker invocations and at
most three review calls (one additional call allows recovery from a reviewer outage). Worker calls have a 30-minute timeout; reviews have 15 minutes.
These are time/call bounds, not monetary billing caps. No automatic model downgrade,
quota retry, history-rewriting push, rebase, claim stealing or issue rollover occurs.
Branch publication requires fast-forward ancestry and an exact expected-old-SHA
lease, so a deletion or concurrent branch change rejects the update.

Logs, prompts, receipts, staged diffs and state stay under
`.ralph/runs/issue-N/`. Treat them as private local artifacts; inspect before
sharing. The PR contains concise acceptance/validation/review evidence, not raw
transcripts. Failed checks leave work locally; failed reviews leave a draft PR.

After inspecting an interruption, `--resume` continues only the same machine's
run, owner, issue snapshot, model policy, check plan and base. It preserves the
original deadline and attempt counter. An already validated candidate resumes
publication/review directly, without inventing a new code change or worker attempt.
Validation calls are capped at four across the whole run; failures remain recorded. If main advanced, budgets expired, the
push outcome is uncertain, or another person changed the branch, perform a
supervised handoff. Preserve the old ledger, reconcile the branch, and repeat
checks and independent review for the final commit. Do not delete a lock or remote
claim merely because a process stopped printing. Confirm the recorded process is
gone and coordinate with the assignee before any manual cleanup.

Changes to prompts, loop code, CI, agent instructions or architecture decisions
require supervised review; workers cannot change their own rules incidentally.
Binary/model/data artifacts and possible credentials are also held for inspection.
The heuristic credential scan supplements review; it is not a complete secret detector.

## Enforce human merge on GitHub

A draft/ready flag and local model receipt are not a server-enforced approval.
A repository admin must protect `main` with:

- pull requests and at least one independent human approval;
- stale approvals dismissed and approval of the most recent push required;
- required CI checks, including both `Ralph checks (ubuntu-latest)` and
  `Ralph checks (windows-latest)`, plus the project's relevant validation jobs;
- resolved conversations, no force pushes/deletions, and the team's agreed bypass policy.

The human reviewer must inspect the AI review evidence and match the final head
and base before merging. Any new code commit needs fresh tests and Astra review.
The committed workflow tests controller behavior on unprivileged `pull_request`
runners. It carries no client secret and runs no paid model. It is not an AI-review
attestation service. GitHub approval policy remains an admin action; this setup
does not pretend a local JSON file is a tamper-proof server check.

Run local controller verification with:
`python -m unittest discover -s ralph/tests -v`.
See [VALIDATION.md](VALIDATION.md) for the actual setup verification and open limits.
