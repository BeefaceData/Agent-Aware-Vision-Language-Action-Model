# Serial simulation implementation queue

The approved preset contains 50 dependency-ordered issues. It builds observation-only
supervision (#63), VLM identity (#42), both bounded interventions in replay (#89)
and frozen-asset enforcement (#92). Active corrective simulation readiness (#94)
requires a larger dependency closure, calibration and evidence. This batch does
not establish the PRD's +10 percentage-point result or physical towel performance.

## Launch

First run `python ralph/smoke.py codex` from an ordinary authenticated terminal.
The managed parent used during setup blocked nested coding edits; do not assume
a login check proves that the worker can edit. Start Docker and preload the
image in [the preset](queues/simulation-supervisor.json). Code and replay tests
need no client API call. Every machine needs the personal Astra reviewer login.

From the repository root in **PowerShell**:

```powershell
python ralph/runner.py --backend codex --queue ralph/queues/simulation-supervisor.json --max-issues 50 --auto-merge --claim-unassigned --dry-run
python ralph/runner.py --backend codex --queue ralph/queues/simulation-supervisor.json --max-issues 50 --auto-merge --claim-unassigned
```

In **Bash** (Git Bash for Codex, Linux/WSL for Cursor):

```sh
bash ralph/codex-afk.sh --queue ralph/queues/simulation-supervisor.json --max-issues 50 --auto-merge --claim-unassigned
```

Choose `cursor-afk.sh` or `claude-afk.sh` instead only after its personal-login
smoke passes in that OS. There is no common `afk.sh`. Without `--auto-merge`, the
queue stops after the first reviewed PR and resumes after its human merge.
`--claim-unassigned` assigns eligible unowned issues to the logged-in GitHub user;
issues owned by another teammate are skipped. No branch claim is stolen.

`--max-issues` is a positive configurable ceiling, not an unlimited run. The
preset has exactly 50 issues. A larger reviewed manifest can name more issues.
`--queue-hours` defaults to 96 and accepts 1..168. Each issue retains its original
90-minute, two-worker-attempt limits. Runs stop on failures, exhausted quota,
changed scope/base, failed review/checks, or no eligible issue. They do not skip a
failed issue and keep spending. Live blockers and sole ownership are rechecked.

Resume with the **same command and options plus `--resume`**. The original count,
deadline, active issue and API ledger persist; resume never renews them. State is
in `.ralph/queues/simulation-supervisor-v1/state.json` and `.ralph/runs/issue-N/`.
Keep the machine awake and terminal open. Inspect stopped evidence before resuming.
Changing the plan or starting a new batch requires a reviewed handoff, not deletion
of the old state. Separate checkouts/clones require coordinated ownership.

## What auto-merge verifies

The controller requires passing independent GPT-6 Astra medium review of the exact
base/head, matching successful isolated validation, both configured GitHub Actions
jobs, all other reported statuses/checks and GitHub's clean mergeability state.
It fast-forwards the reviewed head to `main` with an exact old-SHA lease and an
ancestry check. A concurrent main update rejects that push; no merge combination
is invented after review. GitHub recognizes the reachable PR as indirectly merged.
The next issue starts only once GitHub reports the PR merged and issue closed.
Existing server protections remain binding; a rejection requires reconciliation,
never bypass. This local receipt is not a cryptographic GitHub attestation.

The preset validates public `unittest` tests under `tests/` in a digest-pinned,
network-disabled Python container. Missing/empty suites fail. Issue #2 establishes
the initial episode tests; later issues must add relevant module/episode coverage.
Heavy LIBERO/CUDA integration and measured trials need their own provisioned
environment and evidence. A replay pass is not a real simulator pass.

## Supervisor API authorization and interface

The user approved **Claude Sonnet 5.5**, key variable `ANTHROPIC_API_KEY`, and a
total **USD 50** development allowance. This applies to synthetic fixtures and
public LIBERO observations for simulation supervision. It does not authorize
client data upload, coding/review calls with that key, or benchmark campaigns.
AFK workers and CI use injected transports with zero API spend.

When implementing the live adapter, use this production entry point:

```python
from pathlib import Path
from ralph.supervisor_api import SupervisorAPI

api = SupervisorAPI.from_worktree(Path.cwd())
response = api.message(
    "run-id:episode-id:observation-id",
    [{"role": "user", "content": [{"type": "text", "text": "Observation context"}]}],
    system="The frozen supervisor task/prompt", max_tokens=1024,
)
```

Inject a fake transport for tests. Messages accept chronological text and embedded
base64 images, with no external tools, caching or URL fetches. The request pins
`claude-sonnet-5-5`, low effort, standard service and at most 2,048 output tokens.
Record the model ID and settings; this name is not proof of immutable provider weights.

The gateway resolves Git's common directory so all worktrees use the root's
`.ralph/supervisor-budget.sqlite3`. It reads the key only at request time from the
environment or ignored root `.env`. Keys and payload text are absent from the
ledger. Coding/reviewer process environments remove API keys; filesystem access
is not a secret vault, so never copy the root `.env` into a worktree or packet.

The ledger reserves a conservative full input-context charge plus bounded output
before each call, serializes concurrent reservations, and reconciles only confirmed
usage. Timeouts/unknown outcomes retain their reservation. Stable request IDs
reject automatic paid retries. An unexpected model/accounting response blocks
further paid calls. Restarts do not reset the budget. Near the cap a request can
be refused despite some remaining credit because its worst case will not fit.

```powershell
python ralph/supervisor_api.py --status
```

Pricing is pinned to the official Sonnet 5.5 standard API rates of $2 input / $10
output per million tokens, checked 2026-10-01, expiring 2026-10-08. The source is
in [supervisor-api-policy.json](supervisor-api-policy.json). Expired or changed
pricing requires reconciliation with the existing ledger; never delete it to
resume. This cap covers gateway token charges for this repository campaign, not
taxes, unrelated account use or separate clones. Use a provider account/project
spending limit as well if available. The gateway cannot control calls that bypass
it. No live API or simulator validation is implied by deterministic gateway tests.
