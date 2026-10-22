Current scope: acquire and lift the mug. Declare `task_phase=grasp` on every
call. The controller ends this milestone at
confirmed mug_lifted; it does not attempt handoff or hanging yet. Available MOVE_TO
symbols are current_ee and mug_pregrasp. The pregrasp anchor is re-grounded by a
trusted provider at execution time; do not infer absolute world coordinates.
This acquisition path returns at the configured VLA feedback boundary so you
can inspect fresh public images before selecting another bounded action.

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
and contact-rich grasp. Each VLA_ACT returns after at most the configured
feedback chunks with fresh images. Inspect the arm, gripper, mug support and mug
motion after every segment. If the arm is still making a plausible approach,
another bounded VLA_ACT segment may continue the attempt. If the images show a
miss or repeated lack of progress, choose a safe retry or switch to the available
analytic skills; do not wait for a long VLA horizon to finish. Use the images and
public evidence to decide. Mug color or apparent size alone is never a retry rule.

Analytic re-approach is available only after public failure evidence or a
recoverable skill error. For a visually justified recovery, MOVE_TO
mug_pregrasp may use a
planner-selected normalized pixel on the solid mug body or rim. The runtime
unprojects that pixel through same-observation RGB-D and starts from measured TCP
pose; it does not accept planner-provided world coordinates or joint actions.
Use bounded MOVE_TO, ROTATE_WRIST and SET_GRIPPER calls only when the resulting
motion is justified by the current view and state. Preserve support before
opening, and do not open a potentially held unsupported mug. If grasp appears
plausible but lift remains unverified, continue with one bounded VLA_ACT chunk
when motion is safe and budget remains, then inspect the fresh evidence. ABORT
if no safe bounded continuation remains.

If the retreat MOVE_TO returns a pre-motion `planning_failure` (zero executed
physics ticks and unchanged command state), do not repeat that rejected move.
You may spend the remaining bounded VLA attempt from the current measured pose
only when the images show the mug still supported and the hand open/empty; VLA_ACT
owns the contact-rich re-approach. Otherwise ABORT. This exception does not apply
to uncertain execution or any skill that may have moved the robot.

Use at most the configured total VLA attempt count and current phase allowance
stated in runtime_limits. Grasp, handoff, and hang have separate allowances;
exhausting one phase does not spend a later phase's allowance. Each VLA call
returns fresh observations at the feedback boundary. You may re-stage using the
same fixed tools. Never interpret this partial milestone as a complete hanging
success. Release does not establish correct rack contact.
