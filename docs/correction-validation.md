# Executor correction parameter validation

`correction_validation.CorrectionValidator` is the public parameter boundary
for #66. Construct it before processing requests from host-owned configuration:

```python
from correction_validation import CorrectionValidator
from supervisor_adjustment import AdjustmentComponent, AdjustmentRequestDecoder

# registry and adapter_capabilities are trusted host/adapter declarations.
# These illustrative numerical bounds are synthetic, not robot calibration.
validator = CorrectionValidator(
    recovery=registry,
    capabilities=adapter_capabilities,
    adjustment=AdjustmentRequestDecoder('left_arm', 'world', (
        AdjustmentComponent('translation_x', 'm', -0.03, 0.03),
        AdjustmentComponent('translation_y', 'm', -0.03, 0.03),
        AdjustmentComponent('translation_z', 'm', -0.03, 0.03),
    )),
)
checked = validator.validate(request, current_proposal)
```

The input may be a correction wire dictionary or a decoded
`SupervisorAdjustmentRequest` / `SupervisorRecoveryRequest`. Even decoded
requests are checked again against the executor's own limits. A more permissive
supervisor decoder cannot widen them. Invalid requests raise `ValueError`.
Do not construct the validator, registry, capabilities or component contracts
from response fields.

Bounds are inclusive. Booleans, strings, nonfinite values, missing/extra
components, duplicate typed parameter names, unknown fields and stale proposal
identities fail. Adjustment dimensions are the exact named component set;
target, frame and units must match the configured contract. Recovery parameter
dimensions match the selected tool exactly. Recovery arm/frame/unit semantics
are fixed by the tool's host declaration and required adapter capabilities (use
arm-specific names when needed); a response cannot select a different target or
override a tool definition. An empty registry disables recovery and an absent
adjustment contract disables adjustments.

The output is immutable request data or `ResolvedRecovery`. It grants no
execution authority and leaves the policy proposal unchanged. The harness does
not yet dispatch these results as corrections. Native action conversion and
final composed-action validation, scene eligibility, evidence freshness,
readiness, cooldown and action/attempt budgets are separate gates. Neither
synthetic boundary tests nor this parameter check establish correction readiness.

Run `python -m unittest discover -s tests -p test_correction_validation.py -v`.
Tests cover both correction types, stricter executor bounds than upstream
decoders, immutable configuration, arm/component mismatches and complete
episodes terminating on rejection before an invalid correction can execute.
