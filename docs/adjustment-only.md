# Adjustment-only diagnostic configuration

Issue #91 extends the host-owned `EpisodeConfig.correction_mode` with
`adjustment_only`. Select it with
`dataclasses.replace(config, correction_mode='adjustment_only')`; retain the
same supervisor, frozen policy and instruction, controller evidence and runtime
limits as the default `combined` configuration. The complementary
[`recovery_only` mode](recovery-only.md) remains available.

The harness rejects recovery sequences before intervention admission or command
dispatch with `recovery disallowed in adjustment_only mode`. Without an explicit
`BaselineFallback`, this terminates the episode with `proposal_rejected`.
A configured fallback can execute only the unchanged baseline proposal after
its normal sensor-age, controller-health and native-action checks. The recorded
fallback cause retains the mode rejection. A disallowed recovery consumes no
intervention or recovery-tool attempt and grants no additional allowance.

Adjustment requests still pass through `SingleActionAdjustment` and its verified
converter: current proposal identity, supported target/frame/units, finite
bounded residuals and the final native action all remain validated. Each
adjustment applies to one action; subsequent baseline proposals are unchanged.
The harness retains intervention caps, action horizon, correction expiry,
supervisor-call allowance and episode deadline. The trusted host
`action_selector` remains the active execution boundary; `supervisor_decider`
remains observation-only. This setting neither changes provider settings nor
certifies a live supervisor integration.

`TraceRecorder.seal` stores the selected mode in `manifest.json` under `config`.
`load_recorded_replay` rejects recovery execution records in an adjustment-only
bundle. `RecordedReplay.report()` and the CLI include `correction_mode`,
`optional_ablation: true`, `primary_acceptance_evidence: false` and the replayed
outcome. Historical bundles without a mode retain combined behavior and their
existing annotation fingerprints.

```powershell
python -m unittest discover -s tests -p test_adjustment_only.py -v
python recorded_replay.py <sealed-trace-directory>
```

The complete synthetic episode requests recovery, then a bounded translation
adjustment, then pass-through execution. Adjustment-only dispatches an unchanged
baseline action, the adjusted action, and a fresh unchanged baseline action.
The combined counterpart executes the two-command recovery before the same
adjustment and pass-through, under identical configured limits. Both seal and
replay. Additional cases cover refusal without fallback, unhealthy fallback,
invalid and stale adjustments, intervention/action caps, and a falsely relabeled
combined manifest.

These checks establish software contracts only. The optional ablation cannot
satisfy or block primary improvement acceptance. Measured adjustment-only
experiments remain issue #148 and require the applicable evaluation and resource
gates; these tests perform no live inference or robot actuation.
