# Simulation supervisor queue

`queues/simulation-supervisor.json` contains the 50 dependency-ordered issues.
Launch with `bash ralph/codex-afk.sh 50`; use Git Bash's explicit executable from
PowerShell as shown in [README.md](README.md).

This batch reaches an observation-only simulation supervisor, pinned VLM identity,
both bounded correction paths in replay and frozen-asset checks. It does not
complete active corrective simulation evaluation, establish a +10 percentage-point
result, or prove physical towel folding. Those require later issues and experiments.

Use deterministic public module tests and complete-episode replay. Keep heavy
simulator/model imports lazy and inject provider transports. The pinned validation
image runs the project's unittest suite without GPU, network or client credentials.

## Supervisor allowance

The client authorized at most **USD 50 total** for simulation supervision, with
`ANTHROPIC_API_KEY` and `claude-sonnet-5-5`. This allowance is for the VLA
supervisor, never coding or review. AFK workers and CI make **zero paid API calls**.

The former Python gateway was removed with the Python loop controller.
`supervisor-api-policy.json` preserves the authorization and model; it is not a
spend-enforcement implementation or a current pricing guarantee. The existing
ignored `.ralph/supervisor-budget.sqlite3` is preserved as historical accounting.
Before enabling live supervisor calls, the application adapter must enforce the
shared remaining allowance with durable reservation/accounting and current verified
pricing. Implement and test its contract through injected transports first;
a shell queue must never claim the removed gateway still enforces spend.

The old queue state is retained. On the first shell launch, its active issue and
original deadline are imported only if no legacy issue worker state exists.
