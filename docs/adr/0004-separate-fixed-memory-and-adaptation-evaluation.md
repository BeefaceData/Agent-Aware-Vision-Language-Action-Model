---
status: accepted
---

# Separate fixed-memory and adaptation evaluation

The primary controlled comparison evaluates the frozen VLA, the same policy with a supervisor but no cross-episode memory, and the supervised system with a fixed development-memory snapshot. A separate adaptation experiment starts from a declared snapshot, permits memory updates, and reports performance against experience consumed. The team accepted this separation in Q19 on 2026-09-27 so that improvement from accumulating test experience is distinguished from performance of a fixed system.

## Comparison budgets

The primary baseline and supervised evaluations share an action horizon; recovery actions consume it, and resets begin new attempts. An optional ablation may remove the standard action horizon, with its stopping rule and elapsed-time/cost reporting declared separately. These Q24 decisions were accepted on 2026-09-29 so that additional trial budget does not silently change the primary acceptance claim.
