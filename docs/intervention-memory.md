# Immutable intervention experience records

`intervention_memory.InterventionMemory` implements issues #95 through #98 as an
offline append-only store with outcome-neutral candidate lookup. It does not
enable supervisor retrieval, select development data, update fixed-memory
evaluation snapshots, or authorize an intervention.

The trusted host retains four pieces of evidence:

- `attempt.json`, written before policy control by `AttemptIdentityRecorder`,
  including `task.instruction`, `policy_assets`, `settings.action_capabilities`
  (the serialized `ActionCapabilities`), and the frozen supervisor reference.
- The supervisor identity file referenced by `supervisor_manifest` and
  `supervisor_sha256` in that attempt.
- A decision-context JSON file retained at selection time. It requires
  `episode_id`, `proposal_id`, `observation_sequence`, `diagnosis`, and `request`.
  The diagnosis uses the temporal-diagnosis schema (`category`, `summary`,
  `evidence`, optional `conflicts`). The request is the original adjustment wire
  request or the serialized recovery request retained in the recovery sequence.
  It may also carry `progress_context`, a nonempty mapping of nonempty string
  identifiers (for example `{"stage": "grasp", "object": "target"}`). The host
  declares these labels from deployable evidence at decision time, not evaluator
  outcomes or privileged simulator state. They are not inferred from diagnosis
  or later task success. Omission records unavailable progress evidence.
- The completed episode's `trace/manifest.json`, published by `TraceRecorder.seal`.
  Its observation and decision hashes are checked through `load_recorded_replay`.

Retain each expected SHA-256 when its producer publishes the evidence. Do not
reconstruct a missing diagnosis or request from subsequent success, or compute a
new expected digest to bless changed evidence. Legacy episodes without complete
provenance are rejected. The host is responsible for the truth and timing of
retained context; hashing does not establish authorship or model immutability.

```python
from intervention_memory import InterventionMemory

memory = InterventionMemory("evidence/memory")
reference = memory.append(
    trace=("evidence/episode/trace/manifest.json", trace_sha256),
    attempt=("evidence/episode/attempt.json", attempt_sha256),
    supervisor=("evidence/episode/supervisor.json", supervisor_sha256),
    context=("evidence/episode/context.json", context_sha256),
)
# Keep this reference in the consuming manifest, independently of the store.
record = memory.read(reference["record_id"], expected_sha256=reference["sha256"])
```

The result includes task, robot capabilities, policy/supervisor identities,
effective configuration, diagnosis, request, acknowledged execution rows, and
the sealed episode outcome. Multiple recovery actions stay in one record, keyed
by the initiating episode/proposal. Recovery completion or abort checks remain
distinct from task success. An acknowledged adjustment with an unknown diagnosis
is preserved as unknown; the store provides no diagnostic or execution approval.
Rejected or unconfirmed actions are not completed intervention records.

New records use schema version 4 and expose `local_outcome` separately from
`episode_outcome`. The local status comes only from retained execution evidence:

| Local status | Meaning |
| --- | --- |
| `completed` | The recovery monitor confirmed the local completion conditions. |
| `aborted` | The recovery monitor stopped or cancelled the sequence; `reason` retains the exact cause. |
| `unknown` | No conclusive local check exists. A last `continuing` check yields `no_terminal_local_check`; adjustments without a local assessor yield `no_local_assessment`. |

An acknowledged adjustment is not automatically beneficial. Nor does completed
recovery establish task success or causal benefit: it can precede task failure.
Conversely, a task can succeed with an aborted recovery. Current execution cancels
recovery on terminal task results (`episode_terminated`); historical traces can
retain a failed local assessment alongside terminal success. These cases preserve
their separate outcomes and reasons. An abort due to unknown evidence or terminal
cancellation is not a claim that the correction caused harm.

Version 3 introduced `task_outcome` with a status and the exact episode stop
reason. It follows the harness's task-status contract: observed success is
`success`; unsuccessful termination or action-horizon exhaustion is `failure`;
other stops (including recovery abort or truncation) are `unknown`. A false
`episode_outcome.success` alone is not a verified task failure.

`evidence_limitations` retains machine-readable qualifications derived from the
same pinned evidence: unknown or conflicting diagnosis, unverified local/task
outcome, and missing or stale local assessment when applicable. Every record
includes `causal_benefit_unverified`: even two successful outcomes cannot show
that an intervention caused the success. Exact abort causes, assessment fields,
and observation freshness remain available in the original execution rows.
Neither an abort nor missing evidence establishes that a correction caused harm.

Version 4 adds `progress_context` from the pinned decision context, or `null`
when absent. Versions 1, 2 and 3 remain verifiable in their original schemas,
without adding fields or rewriting immutable bytes. Consumers must treat absent
outcome qualifications as unavailable, never infer local success from terminal
success. Unacknowledged execution and unsealed or corrupt episodes still fail
admission; uncertainty does not waive the provenance contract.

## Outcome-neutral candidate lookup

An offline evaluator can load a permitted list of pinned records for an exact
task and robot/control context:

```python
candidates = memory.candidates(
    [reference],
    task=record["task"],
    robot_capabilities=record["robot_capabilities"],
)
```

For inspectable task/progress and control compatibility filtering, use:

```python
result = memory.filter_candidates(
    permitted_references,
    task=active_task,  # complete declared task mapping, including instruction
    robot_capabilities=active_capabilities,  # serialized ActionCapabilities
    progress_context={"stage": "grasp", "object": "target"},
)
candidates = result["candidates"]
exclusions = result["excluded"]  # record_id, sha256, ordered reasons
```

The filter verifies each pinned record and all source evidence before checking
compatibility. Corruption raises `TraceError`, even for incompatible records;
it is never silently reported as an ordinary exclusion. Invalid active context
also raises `TraceError`, including on an empty reference list. Both result
lists preserve input order and contain detached data.

Task mappings and progress labels must match exactly. Missing legacy progress
produces `progress_context_missing`; unequal labels produce
`progress_context_mismatch`; a foreign task produces `task_mismatch`.
Control declarations are validated with `ActionCapabilities`. Compatibility is
deliberately conservative: layout, frequency, component count/order, name, arm,
coordination group, frame, representation, unit, scale and both bounds must
match. Supported operations must match as a set (their order has no meaning).
Each differing field yields `<field>_mismatch`, with `component_count_mismatch`
for dimension differences. When counts differ, individual coordinates are not
compared because no mapping is established. All applicable reasons are retained
in the documented comparison order: task, progress, layout, frequency,
operations, then component count or the ordered component fields above (frame
before representation). No automatic frame conversion, arm remapping, range
widening or task synonym matching is performed.

Compatibility does not authorize execution or establish positive benefit.
Failed/unknown outcomes remain eligible. Model compatibility, permitted data
splits, ranking and supervisor-safe redaction still belong to subsequent
retrieval policy. The original `candidates` API retains its exact task/control
lookup behavior and does not require progress evidence.

This returns detached, fully verified records in the supplied order. Failed
tasks, aborted recoveries, uncertain outcomes and unknown diagnoses receive the
same eligibility as successes. No success ranking or implicit directory scan is
performed. The task and complete capability declaration must match exactly;
empty context is rejected. Every supplied reference is verified before context
filtering; missing or changed evidence raises `TraceError` without returning a
partial candidate set. Reopening the store does not change these rules.

This is a storage-level candidate interface, not a compatibility or retrieval
policy. The caller must supply permitted references. General compatibility
filters (#98), model/configuration exclusions (#110), development/holdout splits,
ranking, budgets and supervisor-safe context preparation remain separate work.
Do not send these evaluator records directly to the supervisor.

Writes use exclusive file creation, flush, and fsync; a repeated episode/proposal
cannot replace an existing record. A failed write can leave an unreadable partial
file, which fails verification rather than becoming a usable experience. Reads
check both the externally retained record digest and all referenced evidence,
then recompute the record. Returned dictionaries are detached. Moving the common
parent of memory and source evidence preserves relative references. Retain source
files for the lifetime of the record; missing or changed evidence fails closed.
This is API-level immutability, not filesystem access control or signed storage.

Records contain private evaluator evidence and outcomes. They must not be passed
directly to a supervisor or exported without the applicable data-use permission.
Retrieval policy and evaluation splits remain separate work; these storage
contracts supply no claims about memory improving task performance.

Run the public offline contracts:

```powershell
python -m unittest discover -s tests -p test_intervention_memory.py -v
python -m unittest discover -s tests -p test_memory_compatibility.py -v
```

These tests cover completed recovery followed by task failure, failed local
recovery with terminal success, terminal cancellation, unknown local outcomes,
legacy records, successful adjustment and unsuccessful recovery episodes,
complete sealed replay, append-only behavior, portability, and missing, foreign
or changed provenance. They also verify negative and uncertain candidate lookup
after reopening, exact context selection, and rejection of relabeled outcomes or
removed evidence limitations. Synthetic evidence establishes storage behavior only.
Compatibility checks additionally cover mixed one-arm/two-arm records, complete
sealed replay, task and progress mismatches, ordered control semantics, missing
legacy progress, immutable progress provenance, and corrupt excluded evidence.
