# Baseline and always-pass parity

Issue #9 is verified offline through `run_episode`: one path has no supervisor,
and the other returns an identity-bound `SupervisorPass` for every proposal.
Run the public contract check from the repository root:

```powershell
python -m unittest discover -s tests -p test_pinned_policy.py
```

The paired episodes use identical deterministic observations, seeds, effective
action horizons, pinned policy/backbone revisions, loader settings and processor
settings. Each path loads fresh adapters through `PinnedPolicyAssets.load`.
Non-identity preprocessing and postprocessing make bypassing either processor
observable. The environment rejects unexpected native actions; the policy rejects
unexpected processed observations. Fixture-only asset bytes/digests and injected
model/processor loaders keep the check offline. They do not represent downloaded
weights or actual LeRobot inference.

The check compares proposals, selected and executed actions exactly, all retained
observation payloads and capture timestamps, per-step reward/terminal flags,
action count, reward sum, task status, stop reason, terminal reference sequence,
and artifact finalization status. It covers success, unsuccessful termination,
truncation and exhaustion of the same two-action horizon. Every episode is sealed,
loaded and replayed with its expected terminal result asserted independently.
Unexpected action, observation, setting or outcome differences fail the test;
there is no numeric tolerance or action normalization.

Expected metadata differences remain explicit: independent attempts have distinct
episode/proposal IDs and artifact paths, and only the supervised trace contains
accepted pass decisions and supervisor-call accounting. Wall-clock timestamps,
latency and wait duration can differ because supervision adds work. These fields
remain in the traces and are outside the action/outcome equality assertion.
An action-limit stop has no environment-terminal reference, while the last
accepted observation is still retained and compared. No behavioral difference
was observed in these synthetic cases. This establishes the harness contract,
not live SmolVLA/LIBERO numerical parity, throughput or task performance.
