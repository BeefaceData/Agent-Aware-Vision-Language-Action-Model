Current scope: acquire and lift the mug. The controller ends this milestone at
confirmed mug_lifted; it does not attempt handoff or hanging yet. Available MOVE_TO
symbols are current_ee and mug_pregrasp. The pregrasp anchor is re-grounded by a
trusted provider at execution time; do not infer absolute world coordinates.

Initial hypothesis: stage the left arm near mug_pregrasp, invoke VLA_ACT with
stop=mug_lifted, then inspect the result and images. If the attempt fails without
an uncertain execution, use SET_GRIPPER(open), MOVE_TO retreat relative to the
current EE, MOVE_TO mug_pregrasp, then retry VLA_ACT. A 5 cm upward retreat is an
unverified development hypothesis, not a guarantee of free space. Choose ABORT
when an appropriate bounded motion cannot be justified. Do not open a potentially
held unsupported mug; if grasp is confirmed but lift is unverified, ABORT.

Use at most the configured VLA attempt count (initial attempt plus retries),
and per-call chunk/action ceilings stated in runtime_limits. You may re-stage
using the same fixed tools. Never interpret this partial milestone as a complete
hanging success. Release does not establish correct rack contact.
