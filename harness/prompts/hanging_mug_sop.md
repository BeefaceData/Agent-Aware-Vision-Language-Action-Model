# Hanging Mug Task SOP

This is the task-level operating procedure for supervising the frozen Neotix
policy. Follow it together with the active scope prompt, public observations,
predicate results, and fixed skill schemas. The active scope may intentionally
stop before later phases. Never use simulator-oracle evidence as planner input.

## Objective

Move the mug from its initial support through left-arm acquisition and bimanual
handoff, then have the right arm carry and hang it on the rack. A phase change
requires public evidence for its milestone. Mug movement, proximity to the rack,
gripper commands, or apparent alignment alone do not prove grasp, handoff, or
successful hanging. Accept completion only when the runtime reports
`task_success`.

## Operating procedure

1. **Acquire with the VLA.** Start from the current observation and let
   `VLA_ACT` perform the normal approach and contact-rich grasp. Reassess at
   each configured feedback boundary. Continue with a fresh bounded VLA call
   while the approach remains plausible and safe. A false lift predicate means
   lift is unconfirmed; it is not by itself proof of a failed grasp.

2. **Recover a demonstrated grasp miss.** Intervene only when current public
   evidence supports a miss or stalled attempt and the action result is known,
   not uncertain. For every grasp retry, whichever arm failed, issue exactly
   one MOVE_TO to `mug_recovery_clearance`, the fixed pose grounded 10 cm above
   the current mug, while preserving the current wrist orientation. The runtime
   plans vertical clearance, overhead travel and descent inside that single
   MOVE_TO. Do not choose a new grasp orientation or add a separate wrist or
   re-staging action. After the clearance move succeeds, require current
   `release_ready=true` and `mug_toppled=false`, then open the cleared gripper.
   Request fresh VLA inference only after opening completes. Do not replay stale actions from a
   previous VLA chunk. If the move fails, execution is uncertain, or public
   evidence confirms the mug has toppled, abort instead of handing control back
   from an unclear pose.

3. **Let the VLA coordinate handoff.** The frozen policy controls both arms
   while the left arm places the mug on the table and withdraws and the right
   arm approaches. Do not lock the right arm or wait for a separate left-arm
   clearance predicate. Use public evidence to distinguish left-arm lift,
   table staging, and right-arm pickup. A raised mug advances to the right-arm
   phase only when the evidence attributes that lift to the right arm. Preserve
   an already confirmed milestone through later visual occlusion.

4. **Carry and hang.** Let the VLA perform the ordinary right-arm pickup and
   rack approach. Inspect fresh views and public evidence at feedback
   boundaries. If a visible miss, drop, stalled approach, or pose error calls
   for intervention, use only a registered bounded skill and a currently
   grounded target. A cached rack frame or `rack_seat_aligned=true` can guide
   alignment; neither proves physical contact or a successful hang. Do not
   release a contacting hand without the required public authorization.

## Evidence and decision rules

- Treat predicate values as three-valued: `true` is confirmed, `false` is
  evidence against the predicate at that observation, and `null` is unknown.
  Never turn unknown into success or failure by assumption.
- Use images, public predicates, measured public state, and prior public
  results. Do not expose or rely on privileged simulator poses, contacts, or
  task-checker internals when choosing a planner action.
- After each bounded action, inspect the new observation and result before
  choosing the next action. Chunk exhaustion is a feedback boundary, not a
  task failure or success. Any intervention ends the old action sequence; the
  next VLA call must use fresh observation and policy history as managed by the
  runtime.
- Choose exactly one available structured skill per planner turn. Do not
  invent a skill, pose, joint action, or unsupported postcondition. The runtime
  safety checks and action limits remain binding.
- Continue through unknown perception only with a safe bounded action. Report
  the episode incomplete when limits are exhausted without confirmed success.
  Abort for hard safety conditions or uncertain execution; do not retry an
  action whose execution may have partially occurred.
