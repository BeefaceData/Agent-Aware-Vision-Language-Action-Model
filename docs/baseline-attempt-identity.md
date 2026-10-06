# Baseline attempt identity

The baseline runner claims a new output directory before dependency or model
loading. Existing directories, including empty or failed attempts, are rejected.
Use a new `--output-dir` for each invocation; do not delete failed evidence to retry.

After reset verifies the selected initial state and before the first policy
proposal, `attempt.json` records the episode ID, task/instruction, applied and
settled state digests, global/reset seeds, verified policy and processor assets,
inspected environment, effective horizon, environment factory settings and
processor overrides. Simulator reset/settling precedes this manifest. It is
evaluator evidence and is never added to policy or supervisor observations.

The returned episode outcome's artifacts and the runner's `result.json` retain
`attempt_manifest` (relative to the output directory) and `attempt_sha256` (SHA-256
of the exact file bytes). A policy or recording failure retains the written
manifest and its reference. Failure before verified reset may have only preflight
or partial evidence; it must not be presented as an initialized attempt.

The sealed `replay/manifest.json` remains the action/observation replay contract;
it does not validate the separate attempt identity. Keep `attempt.json` and
`result.json` together with that replay bundle to inspect baseline provenance.
Hashes detect changed bytes, not authenticity against a malicious author.

Run the synthetic provenance checks with
`python -m unittest discover -s tests -p test_attempt_identity.py -v`.
They verify pre-proposal persistence, outcome linkage, state/revision distinctions,
collision protection, failure retention and complete episode replay. They do not
establish live model inference, simulator performance or scientific improvement.
