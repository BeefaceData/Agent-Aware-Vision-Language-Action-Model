# Observation-only diagnosis

Use `run_episode(..., supervisor_decider=provider)` with a
`BoundedSupervisorProvider` and `SupervisorResponseDecoder(observation_only=True)`.
The chronological VLM adapter already requests pass/abstain temporal assessments.
This explicit decoder option additionally suppresses valid recovery or adjustment
requests if their caller-owned `recovery` / `adjustment` contracts are configured:

```python
decoder = SupervisorResponseDecoder(
    recovery=recovery_contract, adjustment=adjustment_contract,
    observation_only=True,
)
provider = BoundedSupervisorProvider(adapter, timeout_seconds=2, decoder=decoder)
trace = TraceRecorder(new_trace_directory, config, recorder)
outcome = run_episode(config, policy, environment, trace, supervisor_decider=provider)
trace.seal(outcome)
```

The contracts describe permitted response syntax and numeric bounds; they do not
enable execution. Valid correction responses become `SupervisorPass` records with
`suppressed_correction` set to `recovery` or `adjustment`. Their resolved temporal
diagnosis is retained. The marker records request suppression, not an executed
correction, and does not retain raw correction parameters. Conflict evidence still
forces unknown abstention. Unconfigured, malformed, stale or out-of-bounds requests
remain errors; this option does not bypass validation. Provider failures retain
the existing failure behavior and do not promise baseline completion.

The harness sends detached deployable inputs and executes the frozen policy's
proposal unchanged. No correction executor or action selector is attached. The
policy instruction is unchanged. Scheduling is synchronous and uses
`EpisodeConfig.supervisor_interval_actions`; model waiting changes elapsed time,
even when action and outcome parity hold in replay. This is a simulation/replay
workflow, not a physical hold/stop implementation. Live inference still requires
the project's resource and data-use gates.

Seal the episode with `TraceRecorder`. Inspect `load_recorded_replay(path).evidence()`:
each decision retains pass/abstention diagnosis, evidence, and the proposed and
executed actions. New decision rows also retain `timing`: latency in seconds is
`response_at - request_at`, and `cumulative_wait_seconds` totals waiting. These
are historical measurements, not timing measured by `replay.run()`. Legacy traces
without timing remain readable; do not infer missing latency as zero.

## Feed the offline detector metrics

```powershell
python observation_only.py path/to/sealed-trace --configuration-id frozen-config-id --provenance "Permitted development trace; recorded supervisor version and settings" > outputs.json
python detector_metrics.py path/to/sealed-trace path/to/review.json outputs.json
python -m unittest discover -s tests -p test_observation_only.py -v
```

`export_diagnosis_events` performs the same export without file writes. It validates
the sealed bundle and rejects any episode with overridden or rejected actions.
Declare configuration identity, provenance and matching tolerance before comparison.
The exporter binds the output to the same trace digest used by development reviews.
It coalesces consecutive same-family diagnoses into one detection at the first
sequence. Progress, unknown, missing diagnosis or a skipped assessment ends that
event; a later diagnosis starts another. Changing family also starts a new event.
This fixed rule describes assessment runs, not verified physical failure episodes
or persistence eligibility. Keep cadence and segmentation fixed across comparisons.

Unknown and abstention produce independent per-assessment records. Suppressed
requests produce no intervention records. Productive intervals are empty until
separately reviewed; zero false interventions here is a consequence of disabled
corrections, not evidence of active detector readiness. Retain the original trace,
export, configuration and [development annotation](development-annotations.md).
See [detector metrics](detector-metrics.md) for matching and coverage limitations.

The complete fixture includes repeated stall diagnoses, uncertainty, progress,
missed grasp, recovery/adjustment proposals, and an unsuccessful ending. It checks
action/outcome parity with an unsupervised baseline, sealed replay, controlled
latency, event segmentation, metrics, CLI export and source preservation. All
diagnoses and labels are synthetic; they establish contract behavior only.
