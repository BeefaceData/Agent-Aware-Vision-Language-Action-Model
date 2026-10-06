# Recovery-only diagnostic configuration

Issue #90 adds a host-owned `EpisodeConfig.correction_mode`. The default
`combined` preserves both correction mechanisms. Select the optional diagnostic
ablation with `dataclasses.replace(config, correction_mode='recovery_only')`.
Keep the same supervisor, frozen policy, instruction, scene evidence, and runtime
limits as the combined configuration; change only this setting.

The harness rejects every numerical `override` from the host selector, even an
identity override, before dispatch or intervention admission. The rejection is
`adjustment disallowed in recovery_only mode`. Without a configured fallback,
the episode stops with `proposal_rejected` and retains interruption evidence.
An explicitly configured `BaselineFallback` may admit the unchanged proposal
after its normal input-age, controller-health and native-action checks. Its
retained cause records the disallowed adjustment. Rejection cannot refund or
increase any runtime allowance.

Recovery requests still pass the public executor's request, scene, controller
and envelope checks. The harness still enforces action horizon, intervention
and tool-attempt caps, cooldown, correction expiry, episode deadline, local
completion/abort and fresh-observation resumption. This setting grants no new
execution authority. The current active execution boundary is the trusted host
`action_selector`; the separate `supervisor_decider` interface remains
observation-only. Selecting this ablation does not change provider identity or
settings or certify a live supervisor integration.

`TraceRecorder.seal` includes the mode in `manifest.json` under `config`.
`load_recorded_replay` rejects an adjusted-action trace labeled recovery-only.
`RecordedReplay.report()` replays the episode and returns its outcome, mode,
`optional_ablation` and `primary_acceptance_evidence: false`. The CLI emits that
same JSON report:

```powershell
python recorded_replay.py <sealed-trace-directory>
python -m unittest discover -s tests -p test_recovery_only.py -v
```

The full synthetic episode submits the same adjustment, recovery and pass
requests in combined and recovery-only configurations with identical limits.
Recovery-only executes the original baseline proposal, two recovery commands,
and a fresh unchanged baseline proposal. Only the recovery consumes an
intervention. Both episodes seal and replay. Additional cases verify explicit
rejection without fallback, invalid recovery evidence, exhausted intervention
allowance, recovery abort and a falsely relabeled manifest.

Historical bundles lacking this field retain combined behavior and their
existing annotation fingerprints. Replay reports establish software contracts
only. This optional ablation cannot satisfy or block the primary improvement
claim; measured diagnostic experiments remain issue #147 and require the
applicable evaluation and resource gates. No live inference or robot operation
is performed by these checks.
