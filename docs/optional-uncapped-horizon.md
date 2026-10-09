# Optional uncapped action horizon

The primary baseline and supervised comparison uses the same finite
`EpisodeConfig.max_steps`. An optional diagnostic episode may set `max_steps=None`
to allow more actions than that standard horizon. Declare the matched
`standard_action_horizon` and explicit `max_episode_seconds`,
`max_supervisor_calls`, and `max_interventions` before resetting an environment:

```python
config = EpisodeConfig(
    seed=17, max_steps=None, standard_action_horizon=500,
    max_episode_seconds=120, max_supervisor_calls=8, max_interventions=3,
)
```

These numbers are illustrative, not approved run allowances. The wall-clock
limit bounds the entire active episode. Supervisor calls and interventions are
charged by the existing episode counters; reaching a configured call or
intervention limit follows their existing stop or guarded fallback behavior.
The environment may also terminate or truncate. A live run still requires the
applicable resource and data-use permissions.
An environment with its own finite episode limit must be configured to permit
the intended extension; this harness setting cannot override simulator or
controller limits.

Seal the complete episode with `TraceRecorder.seal(outcome)` and load it with
`load_recorded_replay(path)`. The manifest labels this mode
`evaluation_scope: optional_uncapped_action_horizon` and retains all four
limits, including the standard reference, in `config`. `replay.report()` labels
it an `optional_ablation` with `primary_acceptance_evidence: false`. Its result
must be reported separately with executed actions, elapsed time, supervisor
calls, interventions, and cost evidence where available. It cannot replace the
equal-horizon primary comparison or establish the +10-point claim.
