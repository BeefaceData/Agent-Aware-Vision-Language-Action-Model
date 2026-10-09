# Resource and data-use record for issue #55

Status checked 2026-10-09 against PRD #1, issue #55, and the prepared (unsent)
[client interface handoff](client-interface-handoff.md). The user established the
simulation supervisor allowance below. It does not authorize a live trial while
the separate run gates remain unresolved.

| Gate | Evidence currently recorded | Missing confirmation |
| --- | --- | --- |
| Simulation compute | The user states all tests run on this machine. Local inspection on 2026-10-09 found an NVIDIA GeForce RTX 2050 with 4,096 MiB VRAM and driver 595.95. This establishes a local test host, not VLM capacity or permission for a particular live job. | Approved local job scope and time caps; confirm the selected models fit this hardware before any live run. |
| Model/API calls | `ralph/supervisor-api-policy.json` names Anthropic `claude-sonnet-5-5`; an ignored local `.env` contains `ANTHROPIC_API_KEY` (presence checked, value not read). On 2026-10-09, the user identified themselves as the approver and replaced the prior USD $50 hard limit with a soft goal to stay below USD $50. The policy's pricing validity ended 2026-10-08. | Billing owner, current model access and pricing, durable usage reporting, and the allowed run window. |
| Simulation data use | On 2026-10-09, the user authorized sending the data needed for Anthropic simulation supervision, including synthetic fixtures and public LIBERO observations with camera frames. This does not cover private client or physical-robot data. | Confirm provider storage and retention rules before live calls; obtain separate permission for any private client or physical-robot data. |
| Client hardware and data | The handoff's C4, C6 and C7 answers are missing; the packet is unsent. | Client-approved access, operator and stop authority, run limits, permitted artifacts/providers, storage and retention rules, and approving contacts. |

On 2026-10-09, the user stated that all tests run on this machine. Cloud compute
is not part of the stated test plan. The earlier hard limit is superseded by the
user's no-cap instruction; USD $50 is a soft target. Credential presence does
not establish current model access. The user personally approved the Anthropic
supervisor plan and the simulation data scope above. Live calls still need
current pricing, durable usage reporting, and a permitted run window.
`supervisor_vlm.py` reads `ANTHROPIC_API_KEY` from the process environment; it
does not load the ignored `.env` file itself.

Boniface coordinates the resource and client permission follow-up under PRD #1;
Jean Gabriel supports resource readiness. These coordination roles do not imply
approval. Record each answer with its scope, cap, approving person, date, and a
permitted evidence reference. Keep credentials and private client artifacts out of
the repository and public issue. Until those answers are confirmed, dependent
paid inference, benchmark campaigns, data transfers, and robot operation remain
blocked. Offline replay and other separately authorized implementation can continue.
