# Loop setup verification

## Queue extension — 2026-10-01

- The 64-test suite passes on WSL/Linux. Windows passes the original 44 tests
  (two Linux-only skips) and all 20 new queue/budget tests. The new tests exercise
  dependency ordering, assignment, dry-run, limits and resume, interrupted-merge
  reconciliation, exact reviewed-head publication and a real Git concurrent-main
  race. Simulated GitHub checks include missing, pending, failed and wrong-app jobs.
- Budget transport tests cover concurrent reservations, duplicate IDs, timeouts,
  changed policy, expired pricing, malformed content/accounting, chronological
  image payloads and one shared ledger across real Git worktrees. They use no
  real key or provider request. A Windows database-handle leak found by these
  tests was fixed before publication.
- Bash syntax passes for all provider launchers and the internal helper. Personal
  Codex/Sol and Astra login preflight passes; this is not an editing smoke.
- The approved production ledger initially reports USD 0 spent/reserved against
  USD 50. No live Anthropic request, simulator trial or capstone implementation
  issue was executed during this setup. See the setup PR for independent review
  and CI evidence attached to the final source head.

The user explicitly authorized auto-merge after independent review and checks
for this queue. That supersedes the earlier human-only preference below, which
records the original setup. Single-issue mode still stops for human merge.

## Original setup — 2026-09-30

This file records setup evidence, not capstone experiment results. No real
implementation ticket was claimed, executed or closed during these checks.

## Verified

- Windows: the 36-test controller suite passed with Python 3.13.12 (134.036
  seconds, one Linux-only filename test skipped). The three subsequently added
  publication regressions also passed (12.732 seconds), followed by the three
  instruction-path/Unicode review regressions (10.808 seconds) and the portable
  exact-byte/size-limit regression. Tests use real
  temporary Git repositories and simulated GitHub/model/container boundaries.
  They exercise issue ownership, blockers, atomic claims, invalid worker receipts,
  failed/empty checks, prohibited control edits, stale reviews, changed scope,
  retries, review-only resume, PR target/draft checks, ignored artifacts and exact
  Git blob export despite archive attributes, linked context paths, hidden index
  flags, unrelated GitHub closing directives, nested agent instructions and
  complete Unicode filenames in review packets.
- WSL/Linux: the complete final 44-test suite passed (10.533 seconds), including
  rejection of worker-created FIFOs before source reads and exact-byte preservation.
- GitHub Actions: Ubuntu and Windows checks passed on initial PR commit
  `1b73b6a` (39-test suite); subsequent commits require their own fresh checks.
- All three Bash entry points and the internal launcher pass `bash -n`.
- Personal-login preflight passed on Windows for Codex, Cursor and Claude.
  Cursor's Windows account advertised `cursor-grok-4.6-xhigh`. Authentication and
  catalog presence alone do not prove successful coding/model execution.
- A real Docker isolation smoke passed with
  `python@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b`:
  exact committed file contents and mode `0644`, UID `65534`, read-only source,
  credential variables absent, and an external network connection rejected.
- Astra medium was invoked through personal Codex login, with source inlined
  when the managed parent prevented read-only tools. Its reviews found substantive
  defects; the controller was revised to use an exact candidate commit, clean
  isolated validation, expected-SHA branch leases and phase-aware recovery.
  Later findings about instruction paths, review filenames and special source
  files were addressed with regression coverage. The last full-source assessment
  requested the special-file fix; its focused Astra follow-up passed after seeing
  the complete platform-guarded test and portable byte/size-limit test. That is a
  scoped source assessment, not blanket approval of provider execution or human
  acceptance. The setup PR remains a draft.

## Provider smoke limits

- Native Windows Cursor rejected sandbox mode: its CLI requires Linux/macOS for
  that mode. The approved team target is WSL/Linux; the loop stops on native
  Windows rather than disabling the sandbox. Linux Cursor 2026.09.28-64d2043 is
  installed, and WSL authentication/model discovery passed after the user's login.
  The sandboxed Grok coding smoke timed out at 240 seconds with no fixture edit,
  including after correcting prompt delivery to the documented positional form.
  Cursor coding remains unverified; do not treat login/model discovery as a pass.
- Nested Windows Codex coding was constrained to read-only access by the parent
  environment. The synthetic edit did not happen and all three fixture tests
  failed. This is an environment limitation, not a successful coding smoke.
  Run `python ralph/smoke.py codex` from an ordinary authenticated terminal.
- A native Linux Codex release download did not complete within the bounded
  installation attempt. WSL Codex/Astra review is not installed or authenticated.
  A full Cursor loop also needs that personal reviewer setup in the same OS;
  Windows review authentication does not automatically transfer into WSL.
- Claude had an authentication-only check. No Claude inference or review call
  was made, and the client's API key was not used.

## Adoption gates

The setup must be reviewed and merged through a supervised PR before production
loops can load it from `origin/main`. Each actual ticket also needs a sole GitHub
assignee, clear acceptance criteria, closed blockers and an approved, preloaded
validation image/check plan. Paid/GPU/robot experiments remain separate authorized
work; a passing software test is not experimental evidence.

No visible `main` protection/rulesets were found during setup inspection; the
current account has repository write access. A repository admin must apply the
human-approval/check requirements in the runbook. Local AI review receipts are
not a server-enforced GitHub approval or a cryptographic attestation.

Local evidence is retained under ignored `.ralph/`; raw prompts/transcripts and
personal authentication material are excluded from publication.
