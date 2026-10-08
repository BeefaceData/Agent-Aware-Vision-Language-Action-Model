# Frozen evaluation protocol

`EvaluationProtocol.freeze` records a versioned declaration before an evaluation.
The caller must supply nonempty maps for conditions, splits, seeds, allocation,
horizons, analysis, and outcomes. The maps state the concrete scientific choices:
condition identities and memory modes; development and held-out membership;
pairing and analysis seeds; per-task trial allocation and order; action and
operational limits; estimators and interval settings; success, exclusion,
replacement, and system-failure rules. The API does not choose these values or
grant permission to run trials. Do not use placeholders for a real campaign.

`splits` has exactly `development` and `held_out` lists. Each member is the
`InitialStateSelection.identity()` value: `suite`, `task_id`, `state_id`,
`selected_sha256`, and `digest_encoding`. For this LIBERO-10 protocol the suite
is `libero_10`, task IDs are 0 through 9, and the digest encoding is
`little-endian-float64-vector-v1`. Every held-out task needs at least one
explicit state. A state cannot appear twice in either list or across the lists:
membership is checked by both catalog state ID and selected state digest within
the task. Pairing seeds do not establish state identity. A missing task or
overlap rejects the protocol before it is written and on every read. The
catalog and digest must still be verified against the installed simulator when
selecting each actual state.

```python
from evaluation_protocol import EvaluationProtocol

protocol = EvaluationProtocol.freeze(
    'evidence/protocol.json', conditions=conditions, splits=splits, seeds=seeds,
    allocation=allocation, horizons=horizons, analysis=analysis,
    outcomes=outcomes)
reference = protocol.reference  # Retain independently of protocol.json.
```

Freezing writes a canonical JSON manifest exclusively: an existing file cannot
be overwritten. `protocol_id` is SHA-256 over all declaration content and its
version; `sha256` pins the exact manifest bytes. Preserve both fields in an
independent run/report record. Before running or reporting, construct
`EvaluationProtocol(path, expected_reference=reference)` and call `read()`.
Every read verifies the external byte digest, required fields, and content
identity. A missing or changed file raises `ProtocolError`. A new protocol
choice requires a new file and identity; it cannot revise prior results.

This seal verifies declared bytes and the split structure, but it cannot prove
the declared state digests match the installed simulator. Trial results,
resource approval, and execution compliance remain separate gates.

## Paired trial schedule

`build_evaluation_schedule(protocol)` derives ordered attempt rows from a
verified seal. For this schedule, declare exactly `baseline`, `no_memory`, and
`fixed_memory`; give `seeds.pairing` a nonempty list of distinct nonnegative
integer environment seeds and `seeds.scheduling` a nonnegative integer. Declare
`allocation.task_0` through `allocation.task_9`, each with `pairs` equal to the
length of `seeds.pairing`. Each task receives the same number of paired
clusters. Its held-out states are selected by state ID in round-robin order;
the same selected-state identity and environment seed are used for A, B, and C
within a cluster. SHA-256 keys from the scheduling seed order task clusters
and choose the starting condition order. Cyclic rotations balance each
condition's position within a task and across the full schedule to a
difference of at most one when pair counts are not divisible by three.

```python
from evaluation_schedule import build_evaluation_schedule, audit_evaluation_schedule

rows = build_evaluation_schedule(protocol)
report = audit_evaluation_schedule(protocol, rows)
for row in rows:
    # Verify the selected-state digest against the installed simulator before
    # executing row['condition'] with row['environment_seed'].
    print(row['sequence'], row['attempt_id'], row['task_id'], row['condition'])
```

The audit checks the seal, exact seeded order, unique attempt IDs, all ten task
allocations, contiguous A/B/C clusters, paired state/seed identity, and
condition position balance by task and globally. Retain the protocol reference, rows, and audit
report with run evidence. The schedule assigns attempts only; it does not run
an evaluation, verify simulator state bytes, or grant resource approval.

## Resume a scheduled campaign

`run_evaluation_schedule(protocol, rows, directory, run, max_new=None)` audits
the sealed schedule and every existing attempt before it calls `run`. The
callback receives one row and a newly reserved directory named by its
`attempt_id`; it returns the complete `EpisodeOutcome`. It must seal a recorded
replay under `replay/` or a complete artifact bundle in that directory. The
runner checks that evidence and its replayed outcome before writing
`schedule-result.json`. Later calls verify it again and skip the row. The
returned outcome list follows schedule order; `max_new` can stop after a
declared number of newly completed rows and resume later.

```python
from evaluation_resume import run_evaluation_schedule

outcomes = run_evaluation_schedule(protocol, rows, 'evidence/attempts', run_one)
```

Keep the directory and protocol reference together. A reserved directory
without a verified completion record is an incomplete attempt, even when the
callback raised before the first action. Resume blocks before starting any
other row. Inspect its evidence and apply the protocol's pre-start exclusion
or post-start failure rule explicitly; do not erase it or silently run the
same attempt ID again. This interface does not implement replacement policy
or authorize a live campaign.

## Primary attempt accounting

`score_evaluation_outcomes(protocol, rows, attempts)` audits the sealed schedule
and returns one disposition per row. Pass a mapping from `attempt_id` to a
completed `EpisodeOutcome`, its dictionary from `run_evaluation_schedule`, or
an `EpisodeInterruption` captured from a raised harness exception. Verify
retained replay or bundle evidence before scoring persisted dictionaries; this
function accounts for outcomes but does not authenticate artifacts.

The valid-start boundary is the accepted initial observation. A completed
outcome has crossed it. An interruption with `pre_start_failure` and no last
observation is listed as a pre-start exclusion. Once an initial observation
exists, controller errors, timeouts, exhausted intervention budgets, and other
post-start interruptions each add one failure to the primary denominator,
even when zero actions were executed. Completed unsuccessful episodes do the
same. Only evaluator success adds to the numerator. Every scheduled identity
appears in `dispositions`, including missing evidence. Incomplete artifacts,
conflicting fields, and unknown start boundaries are marked unscoreable.

`success_rate` is `None` while any row is missing or unscoreable, or while
there are no valid starts. This prevents a partial campaign from being
presented as its final end-to-end success rate. Pre-start replacement is a
separate declared policy; this scorer does not replace attempts or launch
trials.

## Pre-start replacement

Declare `outcomes.pre_start_replacement` before freezing the protocol, with an
`eligible_stages` list drawn from `policy_reset`, `environment_reset`, and
`initial_observation`, plus a nonnegative integer
`max_replacements_per_attempt`. The separate
`next_pre_start_replacement(protocol, rows, row, history)` interface verifies
that rule and the complete ordered history. Its first `(attempt_id, evidence)`
pair is the scheduled attempt; subsequent pairs use the replacement identities
returned by earlier calls. Evidence must be an `EpisodeInterruption` with a
matching environment seed, a declared pre-start stage, no accepted initial
observation, and no executed or failed action. A completed outcome or a
post-start interruption cannot be replaced. The allowance applies to each
original scheduled attempt, including repeated initialization failures.

```python
from evaluation_replacement import next_pre_start_replacement

plan = next_pre_start_replacement(
    protocol, rows, row, [(row['attempt_id'], original_interruption)])
# Retain plan and original evidence; start the rerun under plan['attempt_id'].
```

The plan contains the original schedule row, new attempt ID, immediate parent
ID, replacement number, and the provenance of every failed initialization.
Store the plan and all attempt evidence with the campaign. Never delete or
reuse the original directory reserved by `run_evaluation_schedule`. Run the
replacement as a new attempt with the same declared state, condition, and
environment seed. A successful replacement supplies the outcome for the
scheduled row in primary scoring, while the original and every replacement
remain in the infrastructure report. This planner does not verify persisted
artifacts or launch a live attempt; the caller must validate each replay or
bundle before scoring it.
