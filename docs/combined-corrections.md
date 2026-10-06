# Recovery followed by a numerical adjustment

Run the deterministic #89 scenario from the repository root:

```powershell
python -m unittest discover -s tests -p test_combined_corrections.py -v
```

The fixture uses the public `ReopenRetreatExecutor.resolve`,
`SingleActionAdjustment.resolve`, and `run_episode` interfaces. A trusted host
selector supplies synthetic requests and scene assessments. This exercises
execution contracts; it does not implement or certify perception, detector
readiness, live model selection, or robot safety.

The successful episode has a four-action horizon and permits two interventions,
including at most one `reopen_and_retreat` attempt:

| Action | Source observation | Execution | Intervention count |
| --- | --- | --- | --- |
| 1 | 0 | Replace the suspended proposal with gripper opening | 1 |
| 2 | 1, continuing the recovery requested at 0 | Execute the bounded retreat | 1 |
| 3 | 2 | Adjust the freshly resumed policy's translation | 2 |
| 4 | 3 | Execute the unchanged next policy proposal | 2 |

The selector and policy run only at observations 0, 2 and 3. Recovery owns both
replacement commands; the policy resumes with observation 2 after local recovery
completion, then with observation 3 after the single adjustment. Recovery and
adjustment requests have different decision and proposal identities. The numeric
residual changes only translation; rotation and gripper values remain unchanged.
The final action demonstrates that the residual does not carry forward.

Each case seals a temporary `TraceRecorder` bundle, validates it with
`load_recorded_replay`, and replays the complete outcome. Tests inspect the
retained recovery request and per-action assessments, distinct proposal/source
identities, proposed versus executed commands, and action/intervention counts.
The existing trace schema retains the recovery request, but retains only the
composed action and proposal evidence for the adjustment. The fixture separately
checks the original adjustment request's identity; replay is execution evidence,
not a reconstruction of model reasoning or the original adjustment response.
Temporary bundles are deleted after verification and no live artifacts are used.

A pass-through comparison uses the same four-action allowance. Shortened horizons
of two and three stop after recovery and adjustment respectively, with no extra
baseline action or policy resumption. Scripted success in the four-action cases
is a software test oracle, not measured improvement over a frozen-policy baseline.

Failure variants replay a stale adjustment proposal and an unsupported adjustment
frame after completed recovery. Both end with `proposal_rejected`, two acknowledged
actions and a readable rejection reason. A clearance failure after the opening
command ends with `recovery_aborted`, retains `clearance_unverified` and the
`stop_episode` path, and prevents both retreat and the later adjustment. That
local abort path establishes no physical stopping guarantee.
