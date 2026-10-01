# Ralph shell loops

Inspired by `ralph-inspiration/ralph`: a target count, fresh agent runs, live output,
and local logs. Bash + GitHub CLI handle the loop; there is no Python controller.

From this repository in **PowerShell**:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ralph/codex-afk.sh 50
```

From Git Bash or Linux:

```bash
bash ralph/codex-afk.sh 50
```

The positive number (1–50; default 1) is the target number of issues completed
from the ordered simulation preset. Repeating the same command resumes saved
progress automatically. The existing Codex 50-issue queue is imported when it
has no legacy worker run; its original deadline and active issue are preserved.
Keep the same target/backend after starting a queue.

To check installation, login, repository access, the validation image, base
commit and Windows sandbox file access without launching models or changing GitHub:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ralph/codex-afk.sh 50 --doctor
```

## Providers

| Launcher | Coding model | Environment |
| --- | --- | --- |
| `codex-afk.sh` | `gpt-6-sol`, xhigh | Git Bash on Windows, or Linux |
| `cursor-afk.sh` | `cursor-grok-4.6-xhigh` | WSL/Linux with sandbox enabled |
| `claude-afk.sh` | `claude-fable-5-1`, xhigh | Personal Claude subscription, file tools only |

Every coding issue receives a separate `gpt-6-astra` / medium review using
personal Codex login. There is no Claude review stage or generic `afk.sh`.

Install Git, Bash, `gh`, `jq`, GNU `timeout`, Docker and the provider CLI.
Authenticate `gh auth login` and `codex login`; Cursor also needs `agent login`
inside Linux. Claude requires personal subscription authentication. Model
availability is a provider/account property; an unavailable pinned model stops
the run instead of silently substituting another model.

Docker must be running with the image pinned in
`queues/simulation-supervisor.json` already installed. Project checks currently
use Python's unittest inside that container; Python is the VLA project's language,
not the loop controller.

## One issue at a time

1. Read ready issues, ownership, explicit and native dependencies using `gh`.
2. Atomically claim an issue branch and create an isolated Git worktree.
3. Give a fresh worker `prompt.md`, `worker-prompt.md`, the issue, PRD and checks.
4. Commit a candidate after checking paths and its acceptance receipt. Run
   project tests against that commit in a read-only, network-disabled container.
5. Request independent Astra review of that exact base/head, full changed files
   and actual test evidence. One repair attempt is available for failing candidate
   tests or actionable review.
6. Push the issue branch, create its PR using `gh`, and wait for required CI.
7. Auto-merge only when the reviewed head, base and checks still match.
   A non-forced GitHub ref update fast-forwards main to the exact reviewed commit;
   branch protection and a concurrent main update can reject it.
8. Count completion only after GitHub confirms the PR merged and issue closed.
   Poll for up to two minutes while GitHub updates that metadata. If confirmation
   remains unavailable, preserve the merging phase for read-only reconciliation
   on resume; do not rerun the worker or send another merge request.

The user has authorized this auto-merge policy. Workers cannot publish or merge.
A new main commit from a teammate stops an outdated candidate for fresh review.
Local main is left untouched; each subsequent issue uses GitHub's current main.
Git manages local commits and worktrees. `gh` uploads Git objects and handles
issues, refs, PRs and checks, verifying every uploaded hash before moving a branch.
Existing local base objects avoid unnecessary fetches. Network failures show the command and
stderr instead of a generic Python timeout.

## Logs and stopping

State and transcripts live under `.ralph/shell/`; worktrees under
`.ralph/worktrees/`. They are ignored by Git. Each issue keeps receipts, review,
check output and its PR details. A stopped queue retains its branch and edits.

Limits: two worker attempts, three review calls, four validation runs, 90 minutes
per issue and 96 hours per queue. Resuming does not reset limits. Quota failures,
ambiguous remote claims, failed checks, changed bases and incomplete work stop
the loop. Resolve the recorded cause before rerunning. Do not delete state to
make a failed attempt disappear.

Client credentials are removed from child process environments. The loop makes
no paid supervisor calls; see [QUEUE.md](QUEUE.md). Sandboxes and worktrees are
not a vault against every same-user filesystem read: the prompts prohibit
reading client secrets and the root `.env` is not copied into worker worktrees.

Test the controller without models, GitHub mutations or paid API calls:

```bash
bash ralph/tests/test-shell.sh
bash ralph/test-codex-render.sh
```
