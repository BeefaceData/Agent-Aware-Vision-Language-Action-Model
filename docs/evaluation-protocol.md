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
