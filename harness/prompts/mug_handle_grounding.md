You are locating the handle of a mug held by a robot in one RGB image.

Report three ordered pixels on visible solid handle material:
1. A point on the inside of the handle loop nearest its center.
2. A second point along the loop toward the mug body.
3. A point higher on the loop, to define its plane.

Use full-resolution pixel coordinates with origin at the image's top-left. Do
not put a point in the empty hole, infer hidden geometry, depth, world
coordinates, robot motion, or simulator state. If all three landmarks cannot
be located unambiguously, report the target as not visible and set keypoints to
null.
