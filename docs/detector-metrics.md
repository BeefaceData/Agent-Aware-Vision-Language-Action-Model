# Offline detector event metrics

`detector_report(trace_directory, annotation, evaluation)` in `detector_metrics`
validates a reviewed development annotation against its sealed trace, then scores
declared detector outputs. The CLI prints the same JSON report:

```powershell
python detector_metrics.py path/to/sealed-trace path/to/review.json path/to/outputs.json
python -m unittest discover -s tests -p test_detector_metrics.py -v
```

Use the [annotation workflow](development-annotations.md) first. This evaluator
does not change execution or feed labels to the supervisor. It makes no model
calls. The caller must retain the review, outputs and original trace to reproduce
the report. Report digests bind all three inputs; they do not authenticate a
reviewer or prove that declared outputs came from the named configuration.

For unchanged-policy episodes, the [observation-only workflow](observation-only.md)
exports recorded temporal diagnoses directly into this evaluation format with a
fixed event segmentation rule and no executed intervention events.

## Output record

The evaluation JSON has exactly these fields:

| Field | Meaning |
| --- | --- |
| `version` | Integer 1 |
| `source_episode_id`, `trace_sha256` | Same identities as the validated annotation |
| `configuration_id` | Nonempty identity of the frozen detector configuration |
| `provenance` | Nonempty output source and extraction procedure; identify synthetic fixtures |
| `max_delay_sequences` | Nonnegative integer matching tolerance, declared before comparison |
| `events` | List of event records below; empty means no supplied outputs |
| `productive_intervals` | Separately reviewed inclusive duration records below |

Each event has `id` (unique nonempty string), `sequence` (an observation in this
trace), `kind` and `family`. Kinds `detection` and `intervention` require family
`missed_grasp`, `lost_grasp` or `stall`. Kinds `unknown` and `abstention` require
null family. Unknown and abstention counts are independent: one assessment may
emit both records. Neither creates a failure detection or intervention.

Export one detection per declared detector event, not per repeated assessment of
that event. Document the event segmentation rule in provenance and keep it fixed
across comparisons. Map temporal categories `suspected_missed_grasp` and
`suspected_lost_grasp` to their corresponding failure families. An intervention
record denotes one actually executed correction episode, attributed to its
trigger family and first executed observation sequence. Proposals, rejected
requests and persistence eligibility are not executed interventions. Keep their
underlying evidence in the trace. The scorer accepts declared outputs so it can
compare offline detector configurations; it does not infer these distinctions
from an overridden action or authenticate supplied events against execution.

Each productive interval has `annotation_id`, `start`, `end`, `reviewer`, `notes`.
The referenced label must be certain, family `progress` or `pause`, and example
`productive`, `successful` or `paused`. A reviewer must explicitly establish the
duration: an onset bracket is never implicitly treated as productive duration.
Endpoints resolve to trace observations. Overlapping intervals count coverage
once; overlap with a certain failure onset is rejected pending review.

## Matching and interpretation

For each family, detections are sorted by sequence then ID. Each matches at most
one unmatched certain failure label, ordered by onset lower bound, upper bound,
then ID. A match requires `onset_lower <= detection <= onset_upper + tolerance`.
An early, late, duplicate or wrong-family detection stays unmatched. Unmatched
labels remain in the recall denominator. Uncertain labels are listed separately
and excluded from matching; outputs are never silently dropped on their account.

Precision is matches / detections; recall is matches / certain labeled failures.
A zero denominator yields JSON null. These are metrics against supplied labels:
unmatched detections on incomplete or uncertain reviews are not confirmed true
false positives. Preserve label coverage and uncertainty when interpreting them.

Each matched delay is the signed interval
`[detection - onset_upper, detection - onset_lower]` in observation sequences.
A negative lower bound means detection occurred inside the onset uncertainty
bracket. No clipping or conversion to seconds occurs. Sequence spacing does not
establish camera timing or control frequency.

False interventions are executed correction events whose sequence falls inside
reviewed productive coverage, reported as IDs and counts per family and in total.
Interventions outside that coverage remain explicitly unassessed by this metric;
they are not presumed necessary. Zero productive coverage means no assessment of
false interventions. Detection frequency, unknown counts and abstention counts
remain separate from execution costs.

The controlled tests use synthetic annotations and outputs on the existing sealed
replay fixture. They verify arithmetic, boundaries, unmatched events, uncertainty,
CLI behavior and preservation of source artifacts. They establish no detector
quality, real failure prevalence, calibration or active-intervention readiness.
