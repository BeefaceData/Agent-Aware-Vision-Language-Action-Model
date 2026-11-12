This is an acquisition-only run: apply only the grasp phase of the hanging-mug
SOP below; do not initiate handoff or hanging.

Current scope: acquire and lift the mug. Declare `task_phase=grasp` on every
call. The controller ends this milestone at
confirmed mug_lifted; it does not attempt handoff or hanging yet. Available
MOVE_TO symbols are listed in `available_targets`; recovery may add
`mug_recovery_clearance` after a demonstrated failed grasp. The pregrasp and
clearance anchors are re-grounded by a trusted provider at execution time; do
not infer absolute world coordinates.
This acquisition path returns at the configured VLA feedback boundary so you
can inspect fresh public images before selecting another bounded action. The
first grasp feedback may be delayed to the historically observed contact window;
use the runtime-provided action and chunk limits.

When `pixel_grounding_required` is true, every MOVE_TO mug_pregrasp call must
include `parameters.grounding` with a camera view and normalized [u,v] pixel on
the mug's visible solid outer body or rim. Never point into the hollow opening
or infer metric/world coordinates. Coordinates refer to the full source image
dimensions listed in `image_sizes_px`; the runtime reads depth from that same
observation and rejects missing, invalid, or out-of-reach geometry.
If `grounding_feedback` requests a reselection and `grounding_camera_to_avoid`
is set, the next mug_pregrasp pixel must use a different camera. The runtime
enforces this camera change and permits at most two re-selections after the
initial pixel; do not repeat a rejected view or pixel.

Start with VLA_ACT from the current pose so the frozen policy owns the approach
and contact-rich grasp. Each VLA_ACT returns at the configured feedback horizon
with fresh images. Inspect the arm, gripper, mug support and mug motion after
every segment. If the arm is still making a plausible approach,
another bounded VLA_ACT segment may continue the attempt. If the images show a
miss or repeated lack of progress, choose a safe retry or switch to the available
analytic skills; do not wait for a long VLA horizon to finish. Use the images and
public evidence to decide. Mug color or apparent size alone is never a retry rule.

After a completed grasp VLA feedback chunk, public `mug_lifted=false` means the
lift is unconfirmed, not that the grasp definitely failed. The runtime may make
analytic recovery skills available so you can inspect the fresh views and decide
whether a miss or lack of progress justifies them. Continue VLA_ACT when the
approach still looks plausible; do not trigger recovery from `mug_lifted=false`
alone. For a visually justified failed grasp on either arm, use one recovery
action: MOVE_TO to `mug_recovery_clearance` with
`orientation_mode=preserve_current`. The target is grounded from the current
mug position and fixed 10 cm above it. The runtime handles vertical clearance,
overhead travel and descent as internal waypoints of this one call. Do not
choose a new grasp orientation or add a separate wrist or re-staging skill. If
the clearance move completes, require current `release_ready=true` and
`mug_toppled=false`, then issue SET_GRIPPER(open). Hand control back to the
frozen VLA only after opening completes, using a fresh call and fresh
observations. If the move or opening fails, its execution is uncertain, or
`mug_toppled=true`, ABORT. Continue with one bounded VLA_ACT chunk when the
grasp appears plausible but lift remains unverified and motion is safe.

If public status reports both the mug and table as not detected or unknown during
GRASP, the runtime continues with one bounded VLA_ACT chunk at a time while the
episode budget remains and no hard safety condition is active. Do not claim the
mug was grasped or lifted from continued motion alone. If the action/chunk cap is
reached without confirmation, report the episode incomplete.

If the retreat MOVE_TO fails, do not repeat it or return to VLA from the failed
pose. ABORT and retain the fresh observation and failure result for review.

There are no per-phase or total VLA-attempt allowances. Continue across bounded
VLA feedback chunks when the images show safe progress and episode-level
action/chunk budget remains. Each VLA call returns fresh observations at the
feedback boundary. Never interpret this partial milestone as a complete hanging
success. Release does not establish correct rack contact.
