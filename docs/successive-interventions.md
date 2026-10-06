# Fresh evidence between interventions

Each new supervisor correction must bind to the current episode, observation
sequence and proposal ID. `SingleActionAdjustment` and `ReopenRetreatExecutor`
validate these identities and return a bound `ActionResolution`. The harness
checks that binding again before dispatch, so retaining a previously valid
resolution cannot authorize a later correction. Replacing an executor does not
make an old request match the current proposal.

After an adjustment or completed recovery, the harness resumes the policy from
the accepted resulting observation. Observation ingestion requires the next
sequence in the same episode; duplicate or out-of-order packets stop execution
before another proposal. A recovery's intermediate commands belong to the same
bounded intervention, not successive requests. Its original validity deadline
continues to apply throughout the sequence.

Recovery cooldown counts acknowledged baseline actions after completion. Its
expiry grants no permission to reuse an old request, resolution or scene
assessment. Recovery also requires current sensor references and scene evidence;
merely renumbering the enclosing packet cannot refresh those references. Fresh
identity is necessary, but does not replace sensor age, expiry, scene, readiness
or budget checks. Historical evidence may support a diagnosis; it cannot replace
the current scene assessment.

Trusted host selectors and recorded replay may still supply unbound choices.
That interface is not a supervisor response boundary: integrations must retain
the source identity returned by the correction executors and must not relabel
cached corrections as fresh.

The public regression matrix covers cached corrections after zero/two-action
cooldowns, all adjustment/recovery succession combinations, replaced executors,
old scene assessments and relabeled sensor evidence. Complete synthetic episodes
are sealed, loaded and replayed for both rejection and fresh acceptance:

```powershell
python -m unittest discover -s tests -p test_successive_interventions.py -v
```

These checks establish software identity and execution contracts only, not
perception quality, physical safety or measured task improvement.
