"""Legacy pass-through supervisor used by the baseline runner's --harness-config hook.

Contains the PASS/ABORT decision schemas, traces, safety admission, the verified
recovery-skill registry and its development-only candidates. The planner-driven
runtime lives in ``harness.runtime`` and ``harness.skills`` and does not use this package.
"""
