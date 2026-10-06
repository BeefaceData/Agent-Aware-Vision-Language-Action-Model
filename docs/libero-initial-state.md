# Selecting and verifying a LIBERO initial state

The baseline runner accepts `--suite libero_10 --task-id 0 --initial-state-id 0`.
The state ID is an explicit zero-based index in that task's installed initial-state
catalog; the default is 0. `--seed` still controls randomness but is not state
identity. Task and state ranges are validated before policy asset loading or
simulator construction. Missing catalogs and malformed/nonfinite states fail closed.

`select_initial_state` resolves a selection, and
`LiberoEnvironmentAdapter(env, initial_state=selection)` applies and verifies it
on every reset. This integration targets the inspected LeRobot 0.4.3
`LiberoEnv` inside a single `SyncVectorEnv`. Its version-specific access is
isolated in `libero_initial_state.py`. Other adapters can retain the existing
reset interface without claiming verified LIBERO state identity.

The wrapper's normal reset calls LIBERO `set_init_state`, then performs its
configured settling actions. A temporary forwarding proxy compares the applied
vector and immediate `get_sim_state()` readback with the selected catalog state
**before settling**. Exact SHA-256 equality is required using finite,
one-dimensional little-endian float64 bytes. Inconsistent or absent application
fails reset and leaves policy stepping disabled. A separate digest captures the
state after settling; it need not equal the selected state. Normal settling and
policy preprocessing remain unchanged. The proxy is removed even if reset fails.

`initial-state.json` is written before the first policy proposal and records
suite, task/state IDs, selected/applied/settled digests, encoding, settling count
and verification status. `result.json` also retains this evidence, including an
unverified readback on failure. Raw simulator state and these evaluator identities
are not added to policy/supervisor observations. This sidecar is not currently
bound into the sealed replay manifest; a replay verifies recorded execution, not
a new simulator reset. Full run-identity binding belongs to issue #10.

Run offline contracts with:

```powershell
python -m unittest discover -s tests -p test_libero_initial_state.py -v
python -m unittest discover -s tests
```

The tests distinguish two synthetic initial states with the same seed, reject
inconsistent readback before settling/control, and seal/replay complete episodes
through the public adapter. They establish software behavior only. No live
inference, benchmark campaign or physical robot verification is implied.
