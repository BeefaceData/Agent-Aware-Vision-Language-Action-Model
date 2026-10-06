# Verified LIBERO translation residuals

`translation_conversion.LiberoTranslationConverter` implements #67 as a
data-only conversion boundary. It revalidates wire or decoded requests using
the executor-owned parameter validator from #66, then returns an immutable
`NativeTranslationResidual` containing the checked request, seven native
offsets and the SHA-256 of the supplied controller evidence.

The supported mapping is the single OnTheGroundPanda, fixed-impedance,
relative `OSC_POSE` controller with world-frame translation in metres, native
translation slots 0/1/2 and seven action components. Its symmetric native
range is [-1, 1]; translation output range is [-0.05, 0.05] metres. Thus
`native residual = metre residual / 0.05`: `(0.01, -0.02, 0.005)` metres becomes
`(0.2, -0.4, 0.1, 0, 0, 0, 0)`. These are additive offsets, not replacement
actions. The final four zero offsets preserve rotation and gripper commands;
they are not an adapter hold/stop command.

## Retained controller evidence

The local controller capture from 2026-10-05T10:09:21.168998+00:00 is
`docs/evidence/controller-wsl-20261005T100916Z.json`, SHA-256
`034b1c5c4274b0d6d6d9ffc3628eed7f8eff9e35e13c1d6b82ca76520475dd0d`.
It records LeRobot 0.4.3, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.3.7
and NumPy 1.26.4, with installed-source hashes. All six signed world-axis
probes map native +/-0.2 to goal offsets +/-0.01 metres and back. The capture
checked the end-effector position against the MuJoCo world site, zero command
offset, and absence of position clipping. It used LIBERO-10 task 0 with no
policy loaded and zero episode steps. This verifies command-to-goal mapping,
not achieved displacement, correction safety, or task performance.

The generated capture stays in artifact storage, outside this implementation
commit. Obtain its exact bytes from the evidence custodian, or make a new
permitted controller capture and retain its identity. Do not silently replace
the recorded file. A missing artifact leaves conversion disabled; synthetic
test evidence is not a runtime calibration substitute.

## Host setup and use

```python
from pathlib import Path
from translation_conversion import LiberoTranslationConverter

converter = LiberoTranslationConverter(
    Path(evidence_path).read_bytes(),
    expected_sha256=retained_evidence_sha256,
    runtime_controller=current_controller_contract,
    target='panda_arm',
    maximum_metres=0.03,  # illustrative host bound, not a safety calibration
)
converted = converter.convert(request, current_proposal)
```

The host supplies the evidence and expected hash independently of supervisor
responses. It must verify that the installed package/source identities match
the retained capture, and read current controller settings from its adapter.
The runtime contract uses the capture's controller fields: `name`, `robot`,
`reference_frame`, `translation_unit`, input/output minima and maxima,
`metres_per_native_translation_unit`, `native_translation_indices`,
`native_action_dimension`, `use_delta`, and `impedance_mode`. It additionally
requires an explicit `position_limits: None`; unknown or clipped geometry
needs separate verification. Do not fill the runtime contract by copying a
historical report. Reconstruct the converter after installation/configuration
changes. The converter checks declared values and evidence arithmetic; it
cannot itself attest what controller a caller has installed.

Construction verifies the artifact hash, supported mapping, and all six
signed probes, including their inverse conversion. There is no default
enabled converter. Runtime and report mappings must both match the supported
contract. Requests must name exactly `translation_x/y/z`, each in `m`, the
configured target and `world` frame. Camera/tool frames, normalized or other
units, missing components, rotation/gripper/coupled-arm requests, and stale
proposal identities raise `ValueError`. Host residual bounds are inclusive
and cannot exceed 0.05 metres per axis. Booleans and nonfinite values fail.
The policy proposal must be a finite seven-component list or tuple.

This module does not dispatch or compose corrections. Final composed-action
bounds, scene eligibility, expiry, readiness and operational budgets remain
execution-time gates; a baseline command plus a valid residual can still
exceed the controller range. No clipping or execution authority is implied.
No harness behavior changes, so these tests exercise the public conversion
boundary without an execution replay.

Run `python -m unittest discover -s tests -p test_translation_conversion.py -v`.
The seven tests use explicitly synthetic evidence and establish validation
and arithmetic behavior only.
