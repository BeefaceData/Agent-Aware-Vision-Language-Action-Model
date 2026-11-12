You are locating the visible hanger on a stationary mug rack in one RGB image.
The rack is a freestanding stand on the tabletop, separate from both robot arms.
Do not select a robot arm, gripper, camera hardware, or their shadows as rack
material, even when a dark robot part resembles a hook or post. If you cannot
distinguish the rack itself from the robots, report it as not visible.

Report three ordered pixels, all on clearly visible rack material:
1. The usable contact point on the selected peg where a mug handle should seat.
2. A point along that peg toward its free end (the insertion direction).
3. A point above the first point on the rack, to define the rack's vertical plane.

Return each point as an object with ``x_px`` (horizontal column) and ``y_px``
(vertical row), measured in the supplied image's native pixel dimensions. The
origin is the top-left pixel. Do not return normalized coordinates, swap the
axes, or use a resized-image coordinate system. Do not infer hidden geometry,
depth, world coordinates, robot motion, or simulator state. If any landmark is
hidden, ambiguous, or too small to locate, report the target as not visible or
provide confidence 0 and explain the ambiguity.
