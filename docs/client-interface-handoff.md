# Client interface handoff packet

Prepared 2026-10-03 for Boniface under [issue #152](https://github.com/BeefaceData/Agent-Aware-Vision-Language-Action-Model/issues/152), implementing the interface-discovery portion of [PRD #1](https://github.com/BeefaceData/Agent-Aware-Vision-Language-Action-Model/issues/1) (stories 33, 34, 37).

**Status: prepared, unsent, awaiting client answers.** This packet records no client commitments, supplied contract, verified inference result, or physical-operation permission. Boniface coordinates follow-up; Thabo reviews the technical interface; Almond reviews integration and control assumptions. These are project coordination roles, not evidence that the client has agreed to a date or capability.

## Request draft for Boniface to review

Please provide the smallest permitted interface package that lets us understand your frozen policy and bimanual towel-folding controller while simulation development continues: a runnable inference example, its environment and model identity, observation/action schemas, and documented timing and interruption behavior. An example that consumes an approved synthetic or sanitized input and returns actions without actuating hardware is sufficient for initial contract review. No demonstrations or training dataset are requested or assumed.

Please also identify your interface contact and gate reviewer, the expected handoff date, gate-review turnaround, and arrangements for access and resets after approval. For each item below, supply an answer and an authorized evidence reference, or explicitly mark it unavailable/unsupported with an owner and expected follow-up date. Do not send credentials or private payloads through GitHub Issues.

This is a draft for coordination, not an automatically sent message. Receipt of the package permits contract review; physical integration still depends on the simulation gate, client acknowledgement, and documented access/resource permission.

## Technical response checklist

All answers below are **MISSING / unverified** as of preparation. The requested fields describe evidence to obtain, not implemented client capabilities. Thabo owns technical follow-up through Boniface; Almond reviews control and interruption evidence.

| ID | Requested answer and minimum evidence | Current answer |
| --- | --- | --- |
| T1 | Runnable inference entry point and exact command; dependency lock or resolved versions, OS/runtime/hardware requirements, transport/service version, and expected output for a permitted input. Explain how to run without robot actuation, including any side effects. | MISSING |
| T2 | Frozen policy/checkpoint identifier and immutable revision/hash; matching processors, normalization assets, preprocessing, expected input/output shapes and dtypes, action chunk length/cadence, and deterministic settings where supported. Give authorized access instructions separately from credentials. | MISSING |
| T3 | Exact task instruction and how it reaches the policy; reset of policy state/action queues, proposal errors, and how fresh observations replace queued actions after an override. V1 preserves model weights and does not rewrite the policy instruction. | MISSING |
| T4 | Camera field names, camera/arm identities, resolution, dtype, color order, calibration/frame references, timestamps and clock domain, capture cadence, synchronization, maximum age, and missing/stale-view representation. Include a permitted schema/example. | MISSING |
| T5 | Robot-state fields and shapes, arm/joint ordering, poses, velocities and gripper measurements as available; units, frames, timestamp source and availability flags. Identify deployable fields separately from private or evaluator-only truth; task-success labels must not enter supervisor inputs. | MISSING |
| T6 | Native action vector/message schema, arm/channel ordering, units and reference frames; absolute versus delta commands, rotation convention, gripper sign/range, scaling/clipping/deadbands, allowed values, and controller configuration identity. Supply documented conversions and calibration references, not assumed LIBERO scaling. | MISSING |
| T7 | Supported translation, rotation and gripper adjustments per arm; coupled-arm constraints, collision/workspace limits, numerical bounds and unsupported operations. Identify who approves limits and which evidence establishes them. Unsupported capabilities remain disabled. | MISSING |
| T8 | Whether two-arm commands are atomic or sequential, synchronization/skew bounds, partial acceptance/failure behavior, ordering and duplicate-command handling. Explain command ownership and whether policy and correction execution can be made exclusive. | MISSING |
| T9 | Controller frequency, command lifetime, buffering, transport latency/jitter, observation-to-action timing, and whether stepping/pausing is supported. Distinguish receipt, acceptance and execution acknowledgements; identify executed action IDs and how unknown completion is reported. | MISSING |
| T10 | Documented hold and stop commands and their acknowledgement semantics, stopping latency, failure/timeout behavior, transport-loss/watchdog response and operator emergency-stop arrangements. Cover interruption during normal policy actions and corrections. A zero vector is not assumed to hold safely. | MISSING |
| T11 | Resume prerequisites after hold/stop or correction: fresh observations, controller health, queue invalidation, ownership reconciliation and required human authorization. State when automatic resumption is prohibited or unsupported. | MISSING |
| T12 | Episode start/end identity, terminal observations, automatic reset behavior, independent folding-success rubric/contact and assistance recording. Explain how an interrupted or unscoreable attempt is retained instead of silently discarded. Detailed rubric approval continues in #160. | MISSING |

## Coordination and permission response checklist

Boniface owns every coordination follow-up below and records client responses. Thabo supports interface/access/reset planning; Almond reviews gate and control evidence. Client-side names and all dates remain unconfirmed.

| ID | Requested answer / follow-up | Current answer |
| --- | --- | --- |
| C1 | Name and contact route for the client technical interface owner, controller/safety owner, access coordinator and simulation-gate reviewer; who can approve each item? | MISSING; Boniface to identify |
| C2 | Date/time/timezone for initial interface package delivery, version/update process, and turnaround for incomplete-field questions or integration faults. | MISSING; Boniface to agree |
| C3 | Required simulation evidence for gate review, submission route, reviewer availability, expected review duration and written acknowledgement format. Record submission and decision dates when they exist. | MISSING; Boniface to agree |
| C4 | Conditional integration slot after acceptance, robot/site or remote access arrangements, authorized operators, supervision, safety briefing and permitted commands. | MISSING; Boniface with Thabo |
| C5 | Who resets the robot/towel, how starting configurations are established and logged, reset duration, assistance rules, and what happens when reset fails. A reset begins a new attempt. | MISSING; Boniface with Thabo |
| C6 | Available hardware/provider and compute/API allowance, run/time/call caps, access windows and stop authority. No paid calls or hardware execution are authorized by this packet. | MISSING; Boniface to obtain resource approval |
| C7 | Permission to access, retain and process each code/model/schema/image/state artifact; approved storage, viewers, external model/provider use, redaction/export rules and retention/deletion requirements. | MISSING; Boniface to obtain data-use approval |
| C8 | Client-approved fallback schedule if the simulation gate or contract delivery slips; escalation contact and decision date. Preserve the original physical milestone and report unmet evidence. | MISSING; Boniface to coordinate |

The PRD's 2026-10-31 milestone and Oct 11-17 gate-review / Oct 18-24 integration windows are provisional project planning dates, not client commitments. Boniface should obtain C2-C4 timing while simulation work proceeds and record any revised plan through the issue tracker. Contract discovery need not wait for simulation acceptance.

## Evidence and follow-up record

Keep raw client materials only in the designated, authorized artifact location. That location is currently **MISSING**; Boniface must establish it before receipt/storage of private payloads. Public tracker updates should contain permitted summaries and non-sensitive evidence identifiers only. Do not commit credentials, private imagery, raw client documents or restricted artifact URLs.

For each T/C item, complete this record in the permitted location:

| Field | Value to fill |
| --- | --- |
| Item ID and status | MISSING / received-unverified / verified / unsupported / blocked |
| Answer and applicability | Exact supported behavior and relevant policy/controller version |
| Source | Named client respondent, response date and permitted reference |
| Evidence identity | Artifact version/hash and approved storage reference, where permitted |
| Verification | Reviewer, date, command/check performed and actual result; mark unrun checks explicitly |
| Open gap | Consequence for integration and the next question/action |
| Follow-up | Project owner, client counterpart and agreed due date/timezone, or MISSING |
| Permission | Who authorized access/use/export, scope and evidence reference, or MISSING |

Boniface maintains the unresolved-item list in [#153](https://github.com/BeefaceData/Agent-Aware-Vision-Language-Action-Model/issues/153) using permitted summaries; Thabo verifies the received example and contract there. Missing critical fields keep #153 open. Preparing this packet completes #152 only; it does not complete contract validation.

## Review and gate sequence

1. Boniface reviews this draft and coordinates any later communication through the agreed client route. No communication or demonstration request occurs as part of packet preparation.
2. Boniface obtains the approved delivery/storage channel and tracks missing answers. Thabo reviews supplied schemas and any authorized non-actuating example; record exactly what ran and its actual output, without implying a live integration test.
3. Under #153, Thabo verifies identities, observations, actions, timing and interruption semantics, with Almond reviewing control assumptions. Replay/stub evidence supports interface development only; it cannot establish physical readiness or towel-folding competence.
4. Physical operation remains blocked until demonstrated simulation acceptance, client gate acknowledgement, compatible interfaces, verified interruption behavior and physical resource/access permission are recorded under [#163](https://github.com/BeefaceData/Agent-Aware-Vision-Language-Action-Model/issues/163). Unavailable or ambiguous hold/stop behavior blocks active physical use.
5. The initial transfer probe freezes harness logic and supervisor prompt templates, permits necessary adapters/task instruction/robot configuration, and starts with empty cross-episode memory before client-specific tuning. Physical improvement requires its own frozen-policy baseline and evaluation; simulation results do not establish it.

These boundaries follow [ADR-0003](adr/0003-develop-in-libero-before-client-integration.md) and [ADR-0005](adr/0005-bounded-recovery-tools-and-action-adjustments.md). This packet changes no model weights, policy instructions, control bounds or execution behavior.
