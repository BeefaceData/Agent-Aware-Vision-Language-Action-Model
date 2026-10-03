# Chronological VLM requests

`supervisor_vlm.py` implements an Anthropic Messages API adapter. Its wire
format follows the provider's [Messages reference](https://platform.claude.com/docs/en/api/messages/create)
and [vision guide](https://platform.claude.com/docs/en/build-with-claude/vision).
The model identifier, output token limit, history bound, image encoder and
transport are caller configured. No model is selected or called by construction.

Use `ChronologicalVlmAdapter` inside `BoundedSupervisorProvider` as the harness's
`supervisor_decider`. Each adapter belongs to one serial stream of episodes.
The task instruction must be present on the first observation; main and wrist
frame references must describe present or missing images. The existing
observation contract strips evaluator data before building the request.

Each request contains the current episode/observation/proposal identity and
unchanged task instruction, followed by observations in ascending sequence.
For each observation, state and timestamps precede main then wrist metadata and
their available images. Missing views get metadata only. Capture times,
synchronization and time basis remain explicit; observation-return timestamps
are not relabeled as measured camera exposure times. Truncation and sequence
gaps are reported. At most `2 * max_observations` images are included; this is a
sample bound, not an image-byte or input-token budget.

History resets on a new episode ID. Duplicate/reversed observations and changed
task instructions fail before sending. This adapter supplies observation history
and the current proposed action only; executed-action history is explicitly
unavailable. It does not infer that earlier proposals executed.

## Integration

The image converter is caller owned, as in the camera recorder. It receives
sanitized numeric pixels and must return PNG bytes, preserving the declared
camera orientation and channel semantics. For an adapter declaring integer RGB
HWC (or single-batch BHWC) pixels, a converter can use NumPy and Pillow:

```python
from io import BytesIO
import numpy as np
from PIL import Image

from supervisor_provider import BoundedSupervisorProvider
from supervisor_vlm import ChronologicalVlmAdapter, VlmSettings

def encode_png(pixels):
    array = np.asarray(pixels)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if (array.ndim != 3 or array.shape[2] != 3 or array.size == 0
            or array.dtype.kind not in 'ui'
            or array.min() < 0 or array.max() > 255):
        raise ValueError('expected integer RGB HWC pixels in [0, 255]')
    output = BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(output, format='PNG')
    return output.getvalue()

# Select an explicit provider model under the applicable resource/data gate.
def configure_supervisor(model_id, transport):
    adapter = ChronologicalVlmAdapter(
        VlmSettings(model_id, max_tokens=512, max_observations=4),
        encode_png, transport=transport)
    return BoundedSupervisorProvider(adapter, timeout_seconds=10)

# run_episode(config, policy, environment, recorder,
#             supervisor_decider=configure_supervisor(model_id, transport))
```

Inject `transport(payload, monotonic_deadline, cancellation_event)` for offline
fixtures. It returns the decoded provider response envelope. The default
`AnthropicMessagesTransport` performs one HTTPS POST to the fixed provider host,
reading `ANTHROPIC_API_KEY` only into its authentication header. The key is not
part of settings, prompts, manifests or traces. No redirects or retries occur.
Do not put credentials in model IDs, task text or image-conversion output.

The outer provider bounds waiting and discards late responses. Socket timeout
and cancellation do not guarantee remote cancellation or avoid provider charges.
Live use requires the PRD's resource and data-use authorization; the checks below
are mocked and make no paid requests or uploads.

## Responses and verification

### Per-episode resource accounting

Create a new `ModelCallJournal` beside the trace and pass it to the adapter:

```python
from model_usage import ModelCallJournal

# trace is a TraceRecorder with a newly created destination directory.
journal = ModelCallJournal(trace.directory / 'model_calls.jsonl')
adapter = ChronologicalVlmAdapter(
    VlmSettings(model_id), encode_png, transport=transport,
    call_journal=journal)
provider = BoundedSupervisorProvider(adapter, timeout_seconds=10)
# Run the episode with supervisor_decider=provider, then inspect:
# journal.summary()
```

Each actual transport invocation records a flushed start and a finish with its
episode/observation/proposal identity, provider, configured model, elapsed
monotonic wall time and reported usage. Calls rejected before transport (including
busy and pre-cancelled requests) are not counted as model calls. `returned` means
the transport returned; it does not establish a valid supervisor decision or task
success. Usage is retained even when the subsequent decision parsing fails.
Transport exceptions produce an `error` finish with unknown usage/cost, without
recording exception text, images, prompts, credentials or response bodies.

The Anthropic adapter allowlists reported input/output and cache token counters.
It does not infer prices: cost remains `null`. For another instrumented provider,
use `journal.call(proposal, provider_id, model_id, operation)`, where `operation()`
returns `ModelReply(response, usage={...}, cost=..., currency=...)` only for known,
reported amounts. Identity and metric names must be approved nonsecret metadata.
Ordinary responses have unknown usage and cost. Every explicit retry must pass
through `call` separately; optional `retry_of` links to a prior summary call ID.
Neither this journal nor the existing provider initiates retries.

The summary distinguishes reported subtotals from complete totals. Any unknown
cost makes the total unknown; currencies are kept separate. Missing unit counters
produce unknown totals for those units, and `unknown_usage_calls` also exposes
calls with no usage at all. Call wall time includes transport waiting, not the
whole episode, and is not a measure of provider billing time.

A start without a finish remains pending after a timeout, process interruption,
or failed journal write; it is never treated as free. A late worker may append
usage without changing the already rejected decision. The journal therefore
remains separate from the immutable replay manifest and its integrity guarantee.
Retain the JSONL alongside unsuccessful attempts too. Do not present pending
accounting as final; the existing replay seal does not seal this journal. Use a
new journal and adapter for each episode; existing paths and foreign episode
identities are rejected. Accounting is opt-in and does not yet instrument local
VLA inference or infer its resource use.

Verify offline with
`python -m unittest discover -s tests -p test_model_usage.py -v`.

The fixed prompt requests a temporal diagnosis with pass or explicit uncertain
abstention as one JSON object. Responses must be one text block with `end_turn`; truncated output,
provider errors, invalid JSON and unsupported blocks fail safely. The shared
decoder validates the decision's fields and current request identity. Correction
execution is not enabled by this adapter. Transport exceptions are reduced to a
safe generic reason by the bounded provider; raw exception messages are not
recorded. The harness records rejected attempts and finalizes their evidence.

Run:

```text
python -m unittest discover -s tests -p test_supervisor_vlm.py -v
python -m unittest discover -s tests -v
```

Tests cover request ordering and history bounds, missing cameras and gaps,
episode reset, private-field exclusion, credential isolation at the HTTP header,
pass/abstention parsing, stale/malformed/error responses, pre-send cancellation,
and complete sealed episode replay plus a retained failed attempt. Mocked image
encoding uses a tiny PNG fixture; these tests establish software contracts, not
live model compatibility, diagnosis quality or manipulation performance.

## Required observation eligibility

Wrap a synchronous supervisor (including `BoundedSupervisorProvider`) before
passing it to `run_episode`:

```python
from supervisor_eligibility import ObservationEligibleSupervisor

eligible_supervisor = ObservationEligibleSupervisor(
    provider, {"main": 0.5, "wrist": 0.5, "robot_state": 0.5})
# run_episode(..., supervisor_decider=eligible_supervisor)
```

These age limits are illustrative, caller-owned configuration, not calibrated
readiness thresholds. Every listed input must have a present deployable payload
and verified sensor capture age within its limit at assessment time. Inputs
omitted from the mapping are optional: for example, a main-camera-only assessment
can proceed without a wrist view. Choose required inputs for the intended
diagnosis/correction capabilities before operation, never from a model reply.

Missing or stale required evidence yields a proposal-bound `SupervisorAbstention`
with named evidence statuses and reasons. Unknown sensor capture times (including
observation-return timestamps) and invalid capture metadata also abstain. No
supervisor/provider call occurs on ineligible observations. Each new proposal is
checked independently, so restored evidence resumes assessment without carrying
an old abstention or correction forward. Skipped observations are not appended to
the chronological adapter; its next request exposes the resulting sequence gaps.

The gate uses the same monotonic time basis as observation capture and copies the
configured limits. Its abstention records uncertainty, not task failure or a
hold/stop command. Existing harness behavior may continue the unchanged baseline
proposal; execution health checks and correction-time validation remain separate
requirements. This adapter is opt-in, and return-time-only adapters must supply
actual sensor timestamps before satisfying these strict required-input rules.

Offline verification:
`python -m unittest discover -s tests -p test_supervisor_eligibility.py -v`.
The sealed replay covers required camera loss, stale robot state, restored
assessment, and unchanged baseline actions. These fixtures establish interface
behavior, not empirical detection quality.

## Temporal progress diagnosis

The prompt asks for `temporal_diagnosis` with `category`, a nonempty `summary`,
and an `evidence` list. Categories are `progress`, `suspected_missed_grasp`,
`suspected_lost_grasp`, `stall`, and `unknown`. Each evidence entry names an
`observation_sequence`, a deployable `source` (`main`, `wrist`, or `robot_state`),
and a description of the observed cue. Non-unknown assessments require at least
one reference; an unknown assessment can leave the list empty when no useful
observation is available. Summaries and descriptions are capped at 2,000
characters each and evidence lists at 32 entries.

Non-unknown categories accompany `kind="pass"`: even a suspected failure only
reports an assessment and continues the unchanged policy proposal. Unknown
accompanies `kind="abstain"` and retains its reason and evidence availability.
Neither case authorizes correction or changes the frozen VLA task instruction.
There is no task-success, reward, hidden object-pose or confidence field in this
contract. Free-text descriptions remain model claims, not evaluator truth.

The shared decoder and harness validate the structure and category/decision
consistency. The trace retains the diagnosis and sealed replay validates it.
Legacy pass/abstention responses without a temporal diagnosis remain supported;
absence is not synthesized into a progress assessment.

The chronological adapter returns a `WindowedSupervisorResponse` containing the
model JSON and the detached observation window used for that request. This is
adapter-owned context, never a field accepted from model JSON. The bounded
provider's shared decoder resolves every temporal reference against that exact
window, including references attached to an unknown assessment. References are
episode-scoped by the response identity; the window and all its packets must
belong to the current episode and end at the current observation. Evicted,
missing, future, and foreign-episode evidence cannot support a diagnosis.
Camera references require matching metadata and supplied pixels; state references
require a supplied deployable measurement and matching capture identity when
capture metadata is present. This checks existence, not the truth of a model's
description or sensor freshness (which has its separate eligibility gate).

Direct `SupervisorResponseDecoder.decode(response, proposal)` calls have only
the current observation available. Callers that actually supply history can pass
`window=builder.snapshot()` explicitly; a recorded model response cannot supply
its own history. Unresolved references raise `SupervisorResponseError`. The
bounded provider reports `rejected` with no decision, and the harness retains the
reference index and reason in a finalized failed-attempt trace before terminating
without executing that proposal. It does not fabricate an observation or silently
replace a diagnosis. Failed attempts are retained but not sealed as successful
replays. Existing correction request decoders still return request data only;
resolving a diagnosis does not grant intervention authority. Contradiction and
persistence gates remain in #52/#53; no calibrated detector quality is claimed.

`tests/fixtures/temporal_diagnoses.json` contains synthetic responses for every
category. Run `python -m unittest discover -s tests -p test_temporal_diagnosis.py -v`
for public decoding, malformed-field rejection, missing-evidence uncertainty and
complete sealed episode checks through the mocked chronological VLM adapter.
These offline examples are contract fixtures, not human-reviewed development
labels or measured model performance.

Run `python -m unittest discover -s tests -p test_diagnosis_evidence.py -v`
for bounded-window resolution, absent sources, episode isolation and retained
rejected-attempt checks.
