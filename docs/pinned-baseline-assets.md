# Frozen baseline asset resolution

Issue #3 fixes the baseline to `HuggingFaceVLA/smolvla_libero` revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. The runner's `--policy` option
accepts only that repository. There is no branch/revision override or fallback
to newly initialized processors.

`policy-assets.lock.json` records file sizes and upstream Git blob IDs or LFS
SHA-256 digests. `pinned_policy.resolve_policy_assets()` downloads only those
files at the declared commit, checks the returned snapshot revision and verifies
every byte digest, including normalization state. Missing files, mismatched
digests and unexpected root files fail before model construction or environment
creation. Files are verified again before loading to detect changes after
resolution. Treat the lock, Python code and local cache during loading as trusted
host configuration; this is not a sandbox against concurrent filesystem writers.

The policy configuration also references `HuggingFaceTB/SmolVLM2-500M-Instruct`.
That Hub name currently redirects to `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`;
the lock retains both names. Its separate, immutable revision is
`7b375e1b73b11138ff12fe22c8f2822d8fe03467` (upstream date 2025-04-08,
preceding the policy's 2025-09-17 revision). This pins the backbone, tokenizer
and image processor dependencies that are not stored in the policy repository.
The policy weights and processor state remain from the PRD's specified revision.

LeRobot 0.4.3 loads the configuration, strict policy weights and both processor
pipelines from the verified local snapshot. Only the device and the dependency's
local path are substituted; the tokenizer override points to that same verified
dependency. The model is put into evaluation mode with gradients disabled.
Strict loading rejects missing/unexpected weights instead of retaining silently
initialized parameters. LeRobot's processor factory does not forward revision
arguments, which is why passing a revision only to the policy loader is insufficient.

Before loading the model, the runner writes `policy-assets.json` with the verified
identities and file digests. The same record is included in `result.json` under
`policy_assets`, alongside `policy_revision`. A preflight error leaves an error
result with zero actions; a requested revision alone is not proof of resolved
assets. Existing caches can be reused after verification. An uncached invocation
downloads approximately 3.25 GB of weights plus tokenizer/configuration files;
startup hashes the files, so allow time for disk reads.

Run the offline contract checks with:

```powershell
python -m unittest discover -s tests -p test_pinned_policy.py -v
```

These use explicitly synthetic bytes and trusted loader doubles. They cover
unavailable, unpinned, missing, altered and incompatible assets; matching
processor sources; and a complete successful episode sealed and replayed through
public adapters. They do not establish live LeRobot/model compatibility, policy
quality, baseline parity or task performance. No live rollout was run for this
change; those checks retain their own readiness/resource gates.

Upstream provenance (read 2026-10-06):

- [Pinned policy files and metadata](https://huggingface.co/api/models/HuggingFaceVLA/smolvla_libero/revision/6721902bc4d61e50a3bfdb11dfb4cb626f05d102?blobs=true)
- [Pinned backbone files and metadata](https://huggingface.co/api/models/HuggingFaceTB/SmolVLM2-500M-Instruct/revision/7b375e1b73b11138ff12fe22c8f2822d8fe03467?blobs=true)
- [LeRobot processor factory](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/factory.py)
- [LeRobot strict policy loader](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/pretrained.py)
- [SmolVLM dependency loading](https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
