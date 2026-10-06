# Reopen-and-retreat eligibility

`recovery_eligibility.ReopenRetreatEligibility.check` is the executor scene
boundary for #71. It returns a `ResolvedRecovery` only when all checks pass;
otherwise it raises `ValueError` with an inspectable rejection reason. It has
no actuation path and does not alter the policy proposal.

Construct the gate from an executor-owned `CorrectionValidator`, a reviewed
local `envelope_id`, and a finite nonnegative sensor-age limit in seconds.
Only the static `reopen_and_retreat` contract with the `retreat_m` parameter
is supported. Required observation names use the deployable packet vocabulary:
`main`, `wrist`, and `robot_state`.

The trusted host supplies a `RecoverySceneAssessment` for the exact current
episode, observation sequence and proposal. Its evidence must cite current
observations and cover every source declared by the tool. Every cited scene
source must have available measurements, matching sequence metadata and fresh
sensor capture timestamps. Observation-return timestamps alone are insufficient.
Request evidence can refer to history only within the exact supplied bounded
`ObservationWindow`; the shared public response decoder resolves those references.
Foreign, missing, future and unavailable evidence fails closed.

An eligible scene requires all of the following:

- A host assessment of `suspected_missed_grasp`.
- `possible_held_payload` is exactly `False`. Possible, conflicting or uncertain
  holding evidence must map to `True` or `None` and is ineligible.
- `gripper_sweep_clear` is exactly `True`.
- Finite nonnegative `clear_retreat_m` covers the requested retreat, inclusively.
- The assessment's envelope identity matches the gate's reviewed envelope.

The envelope identity must bind the tool, target arm, frame, retreat direction,
opening sweep and local geometry-validation method in the host's reviewed
configuration. Clearance describes the full swept path in that envelope, not
an arbitrary nearest-object distance. Preserve the envelope/assessor provenance
with the run configuration and retain the assessment and cited observations.
An envelope identifier is a binding to host validation, not an implemented
collision detector or a calibration result. No default physical envelope exists.

The host derives this assessment from permitted camera/state evidence using its
reviewed perception or geometry adapter. It must never deserialize assessment
fields from VLM JSON or infer clearance from privileged simulator state. Model
diagnoses and open-gripper readings alone cannot establish that no payload is
held. If the host cannot establish these facts, use unknown values and reject.
The request schema rejects attempts to inject holding, clearance or envelope
overrides. Executor parameter/capability validation runs again on every check.

```python
gate = ReopenRetreatEligibility(validator, reviewed_envelope_id, sensor_age_limit)
selection = gate.check(
    recovery_request, current_proposal,
    scene=host_scene_assessment,
    now_monotonic=decision_time,
    window=active_observation_window,
)
```

Recheck immediately before future execution; do not cache a successful selection
as authorization. Recovery execution (#72), completion/abort checks (#73),
operational budgets and active readiness (#94) remain separate requirements.
This gate does not enable a physical controller or establish detector quality.

Run `python -m unittest discover -s tests -p test_recovery_eligibility.py -v`.
The synthetic fixtures cover eligible missed grasp, uncertain holding, gripper
and retreat clearance, evidence identity/availability/freshness, stricter host
bounds and a sealed complete episode that stops on rejection before recovery.
These tests establish contract behavior only, not measured robot safety or
task performance.
