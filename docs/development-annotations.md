# Review development traces offline

Use `development_annotations.py` with a sealed `TraceRecorder` bundle. This
workflow needs only Python's standard library; it makes no inference calls and
does not modify execution, model weights, policy instructions or source traces.
Annotations are evaluator records, never supervisor input or memory entries.

1. Select a permitted **development** trace. Record its split/access provenance;
   do not use held-out episodes for detector development. The tool validates a
   declaration, not actual permission or membership in an external split registry.
2. Inspect the observations chronologically using
   `load_recorded_replay(trace).evidence()` from `recorded_replay`. This public
   API validates the sealed bundle and returns observation sequences, capture
   times, camera availability, state, actions and terminal outcome. Render the
   retained pixels in an approved local viewer when reviewing real imagery.
   Missing views are unknown evidence. Privileged outcomes may describe the
   example but must not substitute for visible evidence of failure onset.
3. Create a separate draft (the destination must not already exist):

   ```powershell
   python development_annotations.py template tests/fixtures/recorded_episode .agent/review-draft.json --reviewer "Your name" --provenance "Synthetic repository fixture; workflow practice only"
   ```

4. Edit the draft JSON, adding one or more events using the schema below. Keep
   uncertain examples and negative examples. Add review notes explaining what
   is visible and what remains ambiguous. Set `status` to `reviewed` and
   `reviewed_at` to your actual ISO 8601 review time including timezone only
   after inspection. These fields are an attribution declaration, not a digital
   signature or an independent second review.
5. Validate against the same trace:

   ```powershell
   python development_annotations.py validate tests/fixtures/recorded_episode .agent/review-draft.json
   ```

For a verified programmatic round-trip, call `load_annotation(trace, path)`,
`save_annotation(trace, new_path, value)`, then `load_annotation(trace, new_path)`.
The saved value must equal the first loaded value. Saving refuses overwrites;
retain prior review versions when revising labels or resolving disagreement.
Keep raw traces and private review notes in permitted artifact storage.

## Version 1 schema

The top-level fields are `version` (1), `split` (`development`), `provenance`
(nonempty text), `source_episode_id`, `trace_sha256`, `reviewer`, `reviewed_at`,
`status` (`draft` or `reviewed`) and `events` (a nonempty list for validation).
The template sets identity fields automatically. The SHA-256 covers the public
replay evidence serialized as sorted compact UTF-8 JSON, including configuration,
observations, decisions and outcome. It detects changed content; it does not
prove authenticity. Drafts cannot pass validation.

Each event has exactly these fields:

| Field | Meaning |
| --- | --- |
| `id` | Nonempty identifier unique within this review |
| `family` | `missed_grasp`, `lost_grasp`, `stall`, `progress`, `pause`, `occlusion`, or `unknown` |
| `example` | `failed`, `successful`, `paused`, `occluded`, or `productive` |
| `onset` | Inclusive `[earliest_sequence, latest_sequence]` bracketing onset of the labeled phenomenon, or `null` if unavailable |
| `uncertain` | Explicit boolean; must be true for unknown family or null onset |
| `evidence` | Nonempty list of `{ "sequence": integer, "description": "visible evidence and limitations" }` |
| `notes` | Nonempty explanation of the label and ambiguity |

Onset endpoints and evidence references must resolve to observations in this
trace. Exact onset uses equal endpoints. An event already present in the first
frame may use `[0, 0]` with uncertainty and a left-censoring explanation. Use
null onset when even a bracket is unsupported. These are observation sequence
coordinates, not action counts or sensor capture timestamps; consult the trace
to recover timing and preserve its time-basis limitations. Evidence may occur
before or after onset. Multiple events may overlap.

`example` describes the reviewed segment, not a new terminal outcome verdict.
Successful attempts may contain stalls; failed attempts may contain productive
segments. Label intentional waiting as `pause`, productive work as `progress`,
and obscured evidence as `occlusion` or `unknown` as appropriate. Do not infer
missed grasp solely from a closed gripper or stall solely from still pixels.

## Supplied round-trip

`tests/fixtures/development_annotation.json` is an explicitly synthetic schema
example tied to the existing unsuccessful replay fixture. Its uncertain onset
is illustrative, not a human-reviewed detector label. Verify without models,
client demonstrations, downloads or robot access:

```powershell
python development_annotations.py validate tests/fixtures/recorded_episode tests/fixtures/development_annotation.json
python -m unittest discover -s tests -p test_development_annotations.py -v
```

The tests cover all five example types, unknown onset, invalid intervals and
references, duplicate IDs/JSON fields, incomplete review, changed trace identity,
CLI behavior and lossless save/load without source modification. Real reviewed
trace curation, annotation agreement and detector calibration remain subsequent
work; this sample establishes none of those results.
