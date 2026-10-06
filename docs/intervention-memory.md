# Immutable intervention experience records

`intervention_memory.InterventionMemory` implements issues #95 through #100 as an
offline append-only store with candidate ranking and bounded context preparation. It does not
select development data, update fixed-memory
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
splits and supervisor-safe redaction still belong to subsequent
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
filters and ranking are available through the APIs described here;
model/configuration exclusions (#110), development/holdout splits,
budgets and supervisor-safe context preparation remain separate work.
Do not send these evaluator records directly to the supervisor.

## Deterministic relevance ordering

```python
ranked = memory.rank_candidates(
    permitted_references,
    task=active_task,
    robot_capabilities=active_capabilities,
    progress_context={"stage": "grasp", "object": "target"},
    failure_category="suspected_missed_grasp",
)
ordered_records = ranked["candidates"]
ranking_settings = ranked["ranking"]
exclusions = ranked["excluded"]
```

The `exact-context-failure-v1` policy first applies the same verified exact
task, progress and control compatibility gates as `filter_candidates`. Progress
is a hard applicability constraint, not a guessed distance between stages:
an experience from a different stage cannot outrank an applicable experience.
Among eligible records, exact matches to the declared diagnosis category rank
first; all other categories follow. Ties use ascending immutable `record_id`,
independent of input order, insertion time, outcome and store reopening.
The category must be one of the temporal-diagnosis schema's categories, including
`progress` and `unknown`. Unknown matches unknown only; it is not a wildcard.
There is no inferred similarity between other failure families or free-text
summaries, and no vector database or model call is involved.

Failed, successful and uncertain experiences have identical ranking rules.
Their original local/task outcomes and evidence limitations remain attached;
relevance is neither causal benefit nor correction authority. Repeated candidate
IDs occupy one place after **every** supplied reference has passed verification.
Invalid or corrupt duplicate references still fail the whole query. Exclusions
retain their pinned reasons and sort by record ID; repeated excluded references
remain visible. Invalid query context is rejected even for an empty input.

`ranking` records the policy version, queried failure category and progress
context, compatibility rule, sort fields, and absence of outcome preference.
Retain these settings along with the task/control query and permitted references
to reproduce an ordering. Results are detached evaluator data. This API neither
truncates nor prepares supervisor context. Use the separate bounded API below;
decision logging is available through the explicit adapter integration below.

## Bounded historical context

```python
retrieved = memory.retrieve_context(
    permitted_development_references,
    task=active_task, robot_capabilities=active_capabilities,
    progress_context={"stage": "grasp", "object": "target"},
    failure_category="suspected_missed_grasp",
    max_entries=4, max_summary_bytes=1024, max_context_bytes=4096,
)
memory_context = retrieved["context_json"]
```

Only `context_json` is the bounded model-facing payload. It is a compact JSON
array with sorted object keys and ASCII-escaped strings. All size limits count
its UTF-8 bytes, not characters or model tokens. The total includes brackets
and commas; even empty memory costs two bytes (`[]`). Entry and per-summary
limits may be zero; total allowance must be at least two. Booleans, fractional
limits and negative values are rejected, including for an empty query.

After verifying **all** references and ranking, selection walks rank order,
skipping any whole summary that cannot fit its individual or remaining total
allowance. Later smaller summaries may still fit. Once the entry cap is reached,
remaining candidates are omitted. Summaries are never cut mid-field: record ID,
pinned record digest, historical-evidence label, failure category, intervention
kind, local outcome/reason and evidence limitations remain together. Record pins
resolve to full verified provenance in the store. Unknown local outcomes and
unverified causal benefit remain explicit. Historical task outcomes are marked
`withheld` / `evaluator_only`; ground-truth success, raw traces, camera/state
payloads, model configuration, file paths, diagnosis prose and requests are not
copied into supervisor context. Full outcomes remain in the evaluator record.

`selected` is a detached structured copy of that same array. `context_bytes`
measures its serialized size. `settings`, `ranking`, `excluded` and `omitted`
are host audit metadata, outside the model-context allowance; do not send the
whole result to a model. Omission reasons are `entry_limit`, `summary_size_limit`
or `context_size_limit`, in that precedence. Verification is never short-circuited
by budgets, even zero entries. Corrupt duplicates and incompatible evidence
still fail the query. Retain settings and permitted references to reproduce it.

The caller remains responsible for development-data permission, split selection,
model compatibility and fixed-memory snapshot policy. This API does not scan
the store, attach memory to a provider request, change the frozen prompt, or
enable live retrieval. It establishes bounded preparation, not scientific
evidence that memory improves performance.

## Decision provenance (#101)

`ChronologicalVlmAdapter` accepts an optional `decision_memory=DecisionMemory(...)`.
This opt-in supplies only the bounded `context_json` as a separate user text
block; the frozen system prompt and VLA instruction remain unchanged. The host
must provide permitted development references and a query callback using current
deployable evidence:

```python
from decision_memory import DecisionMemory

selection = DecisionMemory(memory, permitted_development_references,
    query=lambda proposal: {
        "task": active_task, "robot_capabilities": active_capabilities,
        "progress_context": current_progress(proposal),
        "failure_category": current_failure_category(proposal),
        "max_entries": 4, "max_summary_bytes": 1024, "max_context_bytes": 4096,
    })
# Pass decision_memory=selection when constructing ChronologicalVlmAdapter.
```

Each preparation revalidates all pinned source evidence, including omitted
records. The adapter retains a detached host-owned `memory_context` on decoded
pass/abstention assessments. It contains selected IDs/digests and whole summaries,
the exact context string and its digest, query and budget/ranking settings,
permitted references, exclusions and omissions. The model cannot supply this
metadata through response JSON. Audit settings and raw evaluator records are
never appended to model input. Memory-informed active correction requests remain
rejected pending the current-scene integration in #108.

The optional `snapshot={"snapshot_id": ..., "sha256": ...}` records an externally
declared pin; `None` explicitly means no snapshot was declared. This field does
not create or verify snapshot membership, development splits or read-only
evaluation. Use the snapshot API below for creation/integrity; runtime enforcement
and development split certification remain #103/#109. It is not authorization for a live run.

Without this option, assessments explicitly retain `retrieval="disabled"` and
empty context/settings; an enabled query with no selected records instead has
`retrieval="enabled"`, `context_json="[]"` and its actual settings. Legacy traces
without the field remain readable; absence is historical unavailable evidence.

After sealing an episode, inspect the same records through the public replay
query, without invoking retrieval or the provider again:

```python
replay = load_recorded_replay(trace_directory)
for row in replay.evidence()["decisions"]:
    action = row["action_record"]
    assessment = action.get("supervisor_pass") or action.get("supervisor_abstention")
    if assessment:
        provenance = assessment.get("memory_context")
```

Loading validates the retained context digest, selected pins and summaries,
budget consistency, and disabled-mode invariants alongside existing decision
identity and trace checks. This verifies retained provenance, not remote model
attention or causal influence. The synthetic transport contracts compare the
exact sent text block with these sealed records for successful pass and
unsuccessful abstention episodes. They also cover corrupt source evidence,
tampered logs, empty retrieval, and model-authored provenance rejection.

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

## Immutable snapshots (#102)

Create a manifest from explicitly permitted development record pins. Supply
nonempty JSON metadata recording the selection provenance, plus the declared
retrieval policies and budgets. Model/task/control/configuration metadata and
outcomes are already bound by each record digest and its verified source evidence.

```python
from memory_snapshot import MemorySnapshot

snapshot = MemorySnapshot.freeze(snapshot_path, memory,
    permitted_development_references,
    metadata={"population": "development", "selection_revision": "review-1"},
    retrieval={"ranking_policy": "exact-context-failure-v1",
               "summary_policy": "whole-historical-summary-v1",
               "max_entries": 4, "max_summary_bytes": 1024,
               "max_context_bytes": 4096})
pin = snapshot.reference  # Retain separately in trusted experiment provenance.
loaded = MemorySnapshot(snapshot_path, expected_reference=pin)
manifest = loaded.read()
```

The content-derived `snapshot_id` covers sorted unique record pins, metadata,
retrieval configuration, schema version and relative store location. `sha256`
also pins the exact manifest bytes. Input order does not change identity;
duplicates are rejected. Empty snapshots are supported. Unsupported policies,
incomplete settings and invalid budgets fail before writing. Creation is
exclusive, flushed and fsynced; existing snapshots cannot be overwritten by the
API. Returned references and documents are detached copies.

Construction and every `read()` verify the manifest against the external pin,
recompute its identity, and verify **every** member through `InterventionMemory.read`,
including sealed replay payloads. Missing/changed records or their source evidence
raise `TraceError` identifying the affected record. Missing/changed manifests
also fail closed. Added store records do not enter snapshot membership.

This is a manifest over retained evidence, not a self-contained archive. Preserve
the snapshot, store, and all record-referenced evidence in their relative layout;
moving their common parent preserves validity. Only evidence bound by the memory
record and sealed trace contract is covered, not arbitrary auxiliary files in
the directories. Hashes detect drift against a trusted pin; they do not establish
authenticity or protect against filesystem owners replacing both artifacts and
pins. An interrupted write may leave an invalid file, which loading rejects.

To reproduce a query, use `manifest["records"]` with the store relative to the
snapshot directory and its three `max_*` settings. Supply the current task,
control, progress and failure-category query separately. This API records and
validates configuration; it does not automatically wire a runtime retriever,
prevent writes to the underlying store (#103), or certify development-only
selection (#109). `DecisionMemory` still treats a supplied snapshot pin as a
declaration. No execution policy, frozen prompt or model weights are changed.

Run the public offline contracts:

```powershell
python -m unittest discover -s tests -p test_intervention_memory.py -v
python -m unittest discover -s tests -p test_memory_compatibility.py -v
python -m unittest discover -s tests -p test_memory_ranking.py -v
python -m unittest discover -s tests -p test_memory_context.py -v
python -m unittest discover -s tests -p test_decision_memory.py -v
python -m unittest discover -s tests -p test_memory_snapshot.py -v
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
Ranking fixtures additionally verify failure-category precedence, outcome-neutral
ties, progress gates, permutations, reopening, duplicates, unknown queries and
fail-closed evidence checks over complete sealed synthetic episode replays.
