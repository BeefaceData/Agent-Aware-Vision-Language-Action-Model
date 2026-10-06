# ENPIRE and ASPIRE: targeted source verification

Checked 2026-09-27 for the capstone PRD interview. This verifies identification and reported experimental conditions; it does not reproduce results or establish a frontier ranking.

## Identification and availability

| Work | Verified identity and version | Authors and affiliations |
| --- | --- | --- |
| ENPIRE | *ENPIRE: Agentic Robot Policy Self-Improvement in the Real World*, arXiv:2606.19980; v1 June 18, 2026; latest listed v2 September 20, 2026. The arXiv comment names CoRL 2026; acceptance was not independently checked. | Wenli Xiao, Jia Xie, Tonghe Zhang, and Haotian Lin are the four equal-contribution authors; 17 authors total. NVIDIA, CMU, UC Berkeley. |
| ASPIRE | *ASPIRE: Agentic /Skills Discovery for Robotics*, arXiv:2607.00272v1, June 30, 2026; one version listed. Expansion: Agentic Skill Programming through Iterative Robot Exploration. | Runyu Lu, Yubo Wu, and Ethan Kou are equal-contribution authors; 14 authors total. NVIDIA, University of Michigan, UIUC, UC Berkeley, CMU. |

Metadata and complete author lists: [ENPIRE arXiv](https://arxiv.org/abs/2606.19980), [ENPIRE official page](https://research.nvidia.com/labs/gear/enpire/), [ASPIRE arXiv](https://arxiv.org/abs/2607.00272), [ASPIRE official page](https://research.nvidia.com/labs/gear/aspire/). The robotics paper is distinct from NVIDIA's [ASPIRE research group](https://research.nvidia.com/labs/aspire/).

Both have public NVlabs repositories with source and reproduction guidance: [ENPIRE](https://github.com/NVlabs/ENPIRE) and [ASPIRE](https://github.com/NVlabs/ASPIRE). Their NVIDIA-owned material uses Apache-2.0; dependencies retain separate terms. Availability is verified; installation, completeness, and reproducibility are untested.

## What changes, and what the results measure

**ENPIRE — verified author report.** Its environment, policy-improvement, rollout, and evolution modules support agents revising policy/training code from physical feedback. Supported regimes include heuristics, tool composition, behavior cloning, and reinforcement learning; this is not a framework restricted to frozen policy weights. Its showcased tasks include Push-T, pin/GPU insertion, and zip-tie manipulation. The official page qualifies its 99% headline as pass@8: up to eight failure-conditioned retries within a rollout, rather than independent best-of-eight samples or first-attempt success. [Official method and metric explanation](https://research.nvidia.com/labs/gear/enpire/)

The paper evaluates Codex/GPT-5.5 xhigh, Claude Code/Opus 4.7 High, and Kimi Code/Kimi K2.6 thinking on bimanual YAM hardware. Crucially, §3.5 also combines GR00T VLA calls with generated perception/planning procedures in RoboCasa365, then transfers a hover/grasp strategy to real scissors use. Thus ENPIRE is relevant to VLA orchestration; describing it solely as policy retraining would be inaccurate. Its simulation comparisons use matching 40 seed/layout/style triplets and native success predicates; this is a different protocol from its real-world retry metric. [ENPIRE v2, §§3, 3.5, D.2](https://arxiv.org/html/2606.19980v2)

**ASPIRE — verified author report.** Adaptation changes robot programs and a library of validated repair guidance through trace-based debugging and evolutionary search. Skills become in-context guidance; the claimed learning mechanism is not gradient updates to a VLA or supervisor. [Official method](https://research.nvidia.com/labs/gear/aspire/), [repository explanation](https://github.com/NVlabs/ASPIRE)

Simulation uses Claude Code/Opus 4.6 with a 1M-token context; real transfer uses Codex/GPT-5.5 xhigh on YAM. Reported gains are percentage-point differences: up to 77 on LIBERO-Pro, Robosuite handover 20%→92%, and BEHAVIOR-1K radio pickup 56%→88%. LIBERO-Pro uses 10 tasks × 50 held-out seeds per suite/perturbation, after separate debugging seeds; Robosuite uses 100 held-out trials per task. Long-task transfer reports approximately 31% versus 4%, without further debugging/retries/library updates for ASPIRE. Real soda-can results are 13/20→19/20 with transferred skills, following real-world program debugging, not direct deployment of a simulation policy. [ASPIRE v1, §§3.1–3.6](https://arxiv.org/html/2607.00272v1)

## Verification verdict and PRD consequence

Identities, source availability, and the above reported conditions are confirmed by author-controlled primary sources. Grade: approximately Level III (controlled computational/robotic comparisons), not independently reproduced evidence. Institutional/intellectual interests apply: the authors evaluate their own systems. No predatory-journal claim is supported; these are arXiv records, and journal screening is inapplicable. DOI resolution and Semantic Scholar API requests failed through the browsing tool (`S2-API-UNAVAILABLE`); arXiv records, full texts, official sites, and repositories were checked directly. Those corroborating surfaces share authorship and are not independent replications.

**Inference for scope:** a +10-percentage-point gain against the client's own frozen-VLA baseline would not establish superiority over either paper. A frontier comparison needs a named method, matched task/embodiment, allowed observations/actions, adaptation and retry budgets, held-out conditions, and cost/latency accounting. No verified result above supplies a directly interchangeable towel-folding benchmark.
