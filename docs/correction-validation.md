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

## Declaring adjustment capabilities

The executor's `adjustment` contract is its capability allowlist: one exact
arm/group target, frame, component set, units and inclusive per-component bounds.
The `capabilities` string tuple on `CorrectionValidator` is used for recovery
tool compatibility; adding a string there never enables an adjustment.

The translation-only example above rejects rotation, gripper and coupled-arm
components explicitly, including zero-valued extras. Rejections identify the
unsupported component names or target/group. Missing components and incompatible
units or frames also fail; unsupported fields are never dropped or clipped.
Decoded requests from a more permissive contract receive the same checks.

A host can separately declare another capability without changing the validator
or assuming a seven-component, one-arm native action. For example, this synthetic
group contract validates all named residuals together:

```python
paired_contract = AdjustmentRequestDecoder('coordinated_pair', 'world', (
    AdjustmentComponent('left_translation_x', 'm', -0.01, 0.01),
    AdjustmentComponent('right_translation_x', 'm', -0.02, 0.02),
    AdjustmentComponent('relative_rotation_z', 'rad', -0.03, 0.03),
))
```

Only the trusted host may install this contract as `adjustment`. It must use
actual adapter semantics and reviewed limits; these example bounds are not a
client controller contract. Rotation or gripper components follow the same
declaration mechanism. A group label and successful parameter validation do not
establish atomic arm coordination or provide native conversion. The existing
LIBERO translation converter and single-action executor still reject these
other capabilities. Supporting their execution needs separately verified
adapter conversion and coordination contracts.

The output is immutable request data or `ResolvedRecovery`. It grants no
execution authority and leaves the policy proposal unchanged. Dispatch requires
a separate executor such as the [one-action translation executor](single-action-adjustment.md).
Native action conversion and final composed-action validation, scene eligibility, evidence freshness,
readiness, cooldown and action/attempt budgets are separate gates. Neither
synthetic boundary tests nor this parameter check establish correction readiness.

Run `python -m unittest discover -s tests -p test_correction_validation.py -v`.
Tests cover both correction types, stricter executor bounds than upstream
decoders, immutable configuration, arm/component mismatches and complete
episodes terminating on rejection before an invalid correction can execute.

Run `python -m unittest discover -s tests -p test_adjustment_capabilities.py -v`
for #70's capability checks: unsupported rotation/gripper/arm residuals, separate
host declarations, complete group validation, and sealed rejected episodes with
no dispatched actions. These synthetic fixtures establish contract behavior only.
