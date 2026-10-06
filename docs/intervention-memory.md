# Immutable intervention experience records

`intervention_memory.InterventionMemory` implements issues #95, #96 and #97 as an
offline append-only store with outcome-neutral candidate lookup. It does not
enable supervisor retrieval, select development data, update fixed-memory
evaluation snapshots, or authorize an intervention.

The trusted host retains four pieces of evidence:

- `attempt.json`, written before policy control by `AttemptIdentityRecorder`,
  including `task.instruction`, `policy_assets`, `settings.action_capabilities`
  (the serialized `ActionCapabilities`), and the frozen supervisor reference.
- The supervisor identity file referenced by `supervisor_manifest` and
  `supervisor_sha256` in that attempt.
- A decision-context JSON file retained at selection time. It has exactly
  `episode_id`, `proposal_id`, `observation_sequence`, `diagnosis`, and `request`.
  The diagnosis uses the temporal-diagnosis schema (`category`, `summary`,
  `evidence`, optional `conflicts`). The request is the original adjustment wire
  request or the serialized recovery request retained in the recovery sequence.
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

New records use schema version 3 and expose `local_outcome` separately from
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

Version 3 also stores `task_outcome` with a status and the exact episode stop
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

Versions 1 and 2 remain verifiable and readable in their original schemas,
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
```

These tests cover completed recovery followed by task failure, failed local
recovery with terminal success, terminal cancellation, unknown local outcomes,
legacy records, successful adjustment and unsuccessful recovery episodes,
complete sealed replay, append-only behavior, portability, and missing, foreign
or changed provenance. They also verify negative and uncertain candidate lookup
after reopening, exact context selection, and rejection of relabeled outcomes or
removed evidence limitations. Synthetic evidence establishes storage behavior only.
