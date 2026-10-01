---
status: accepted
---

# Improve through memory with frozen models

Both the VLA policy and the supervisor VLM retain fixed learned parameters in the initial system. The supervisor adapts its decisions using outcome-linked cross-episode memory; automatic prompt, procedure, and code modification remain outside the initial core. This choice, accepted in the answer to Q7 on 2026-09-27, isolates memory's contribution for comparison with the identical supervisor without memory and avoids adding a separate model-training or harness-modification system.
