# Native action capabilities

Adapters expose `action_capabilities: ActionCapabilities` from
`action_capabilities.py`. A policy declares its required native interface after
preprocessing/postprocessing; the environment declares what it accepts.
`run_episode` validates the pair before reset, recording or inference.

The ordered `ActionComponent` tuple identifies every coordinate's name, arm,
coordination group, representation, unit, frame, native minimum/maximum and
positive scale into the declared unit. The contract also declares transport
layout, control frequency in Hz and supported operations. Policy ranges must
fit inside environment ranges, and required operations must be supported.
Other semantics, ordering, dimensionality, frequency and layout must match.
No implicit unit/frame conversion, reshaping or clipping is performed.

For example, two arms can each expose shoulder/elbow joint positions in radians,
in a shared `paired` group, as a four-value flat vector at 50 Hz. The harness
contains no seven-value or single-arm assumption. Groups declare command
coordination; they do not establish physical synchronization or safety.

The frozen LIBERO policy uses a single-vector batch with seven coordinates:
world translation deltas scaled by 0.05 m, world axis-angle deltas scaled by
0.5 rad, and a normalized gripper command. Each native range is [-1, 1], at
20 Hz, for the Panda arm. These are command semantics, not measured achieved
motion. Video FPS is unrelated. `read_native_capabilities` checks the installed
relative OSC_POSE controller, robot, ranges, scales, actual control frequency
and action space before the runner starts its episode. The baseline's inspected
environment/source identity remains part of its provenance. Unknown or changed
configurations fail; the declaration is not a universal robot default.

The runner stores the declaration in `result.json` and in the pre-action
`attempt.json` settings. `ResetOnResumePolicyAdapter` preserves the contract
across reset/resume. `policy_action` is the only declared baseline operation;
correction eligibility, reviewed bounds, conversion evidence and hold/stop
remain separate contracts. Native action capabilities grant no correction
permission and perform no physical calibration.

Historical opaque replay adapters may both omit declarations. They retain
their existing behavior and make no native-interface compatibility claim.
One declared adapter paired with an undeclared adapter is always rejected.
New policy/environment integrations must declare both contracts. Sealed replay
reproduces recorded execution; it does not recertify the original hardware.

Run the public contract checks with:

```powershell
python -m unittest discover -s tests -p test_native_capabilities.py -v
```

The checks include complete single-arm and synthetic two-arm episodes followed
by sealed replay, startup rejection for incompatible semantics, and synthetic
LIBERO readback failures. They establish software behavior only.
