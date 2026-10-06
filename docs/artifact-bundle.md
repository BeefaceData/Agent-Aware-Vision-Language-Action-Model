# Final episode artifact integrity

The baseline runner publishes `bundle.json` only after the returned episode,
replay seal, video writers, frame index, result summary and environment cleanup
have completed. A failed seal marks the result artifacts incomplete. An attempt
without this final manifest is not a verified final bundle.

Verify before consuming a result or replaying an episode:

```powershell
python artifact_bundle.py outputs/my-attempt
python artifact_bundle.py outputs/my-attempt --replay
```

Python callers use `verify_artifact_bundle(directory)`. It returns a validated
recorded replay with `report()` and `run()` interfaces only after checking the
entire inventory. `seal_artifact_bundle(directory)` is the finalization API;
it refuses to overwrite a previous seal.

Version 1 lists SHA-256 digests for the pre-action attempt identity, inspected
environment, pinned policy assets, final result, step log, frame index, replay
manifest, observations, decisions and execution journal. Every camera with an
available indexed frame requires its video artifact. Missing observations remain
explicit gaps; they do not require invented frames. The manifest retains the
episode configuration, model identity and effective settings, cross-checked
against pre-action identity and the replay. Frame entries must match the recorded
observation sequence, camera availability and contiguous per-camera indices.

Inventory paths are relative so bundles can be moved. Historical absolute paths
inside recorder evidence are not followed. Unknown manifest versions, changed or
missing artifacts, omitted required files and inconsistent identities fail
verification. The existing `recorded_replay.py` interface still supports historical
trace-only fixtures; it does not certify the enclosing baseline artifact bundle.

Checksums detect changes relative to a retained manifest, not authenticity against
an author who can rewrite both evidence and hashes. Video bytes are hashed without
codec decoding; verification does not independently establish encoder correctness
or physical task success. Keep a trusted copy of the final manifest when sharing
permitted artifacts.

For a separate report that excludes raw evidence by default and records explicit
artifact permissions and omissions, use the [shareable export](shareable-report.md).

`python -m unittest discover -s tests -p test_artifact_bundle.py -v` exercises a
complete synthetic episode, camera gaps, relocation, CLI replay and damaged
evidence. Synthetic video sinks establish byte-integrity behavior only; these
checks perform no live inference, simulator campaign or physical actuation.
