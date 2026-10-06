# Correction processing at episode termination

`run_episode` accepts evaluator terminal flags only with an accepted observation.
Success takes precedence over termination, then truncation. It captures the
terminal observation identity and reason before recovery assessment, recording
or progress callbacks. A terminal recovery records an `episode_terminated`
abort, cancels its remaining commands and skips local assessment and policy
resumption. This does not imply that the recovery locally succeeded or failed.

Simulation decisions are synchronous: a pending decision blocks environment
advancement. After a terminal result the harness never asks for another
decision, drains a provider queue or dispatches another command. A provider
thread may finish computing; its result has no execution authority. A deliberate
new episode has a new identity, and executor resolutions tied to the old
proposal are rejected. Direct adapter calls outside the harness remain outside
this contract, as do physical controllers advancing independently of `step()`.

Completed outcomes retain `terminal_observation` and `stop_reason`. When recording,
progress reporting or finalization propagates an exception, `episode_interruption`
retains `terminal_observation`, `terminal_reason` and the detached
`last_observation` alongside the infrastructure/interruption `stop_reason`.
The original exception and incomplete-artifact status remain visible.

Sealed replay validates terminal cancellation against evaluator flags and
rejects subsequent decisions. Historical traces with a local recovery check
at termination still load without rewriting their retained evidence.

Run the public synthetic checks with:

```powershell
python -m unittest discover -s tests -p test_terminal_corrections.py -v
```

The fixtures force late adjustment/recovery delivery after success, failure
and truncation; verify rejection on restart; replay terminal recovery at either
command; and retain terminal evidence through callback and cleanup faults.
These checks establish software behavior, not physical stopping performance.
