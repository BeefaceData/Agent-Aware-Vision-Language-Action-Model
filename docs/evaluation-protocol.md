# Frozen evaluation protocol

`EvaluationProtocol.freeze` records a versioned declaration before an evaluation.
The caller must supply nonempty maps for conditions, splits, seeds, allocation,
horizons, analysis, and outcomes. The maps state the concrete scientific choices:
condition identities and memory modes; development and held-out membership;
pairing and analysis seeds; per-task trial allocation and order; action and
operational limits; estimators and interval settings; success, exclusion,
replacement, and system-failure rules. The API does not choose these values or
grant permission to run trials. Do not use placeholders for a real campaign.

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

This seal verifies declared bytes, not the truth of split membership, trial
results, resource approval, or execution compliance. Those are separate
readiness and analysis gates.
