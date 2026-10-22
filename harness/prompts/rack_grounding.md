You are locating one fixed object in the first image of a hanging-mug episode.

The rack is a freestanding mug rack/tree: a vertical post on a square base, with
one or more short pegs/hooks extending sideways or diagonally from the post. It
may be low-contrast, near an image edge, or partly cropped. Use only the current
RGB image. Return whether rack pixels are visible, a tight box around the visible
rack, and a point on a clearly visible projecting peg where a mug could hang.
Coordinates use the original image size and top-left origin. Do not infer hidden
parts or guess a peg point; if uncertain, report not visible and null coordinates.

This call only grounds a visual reference. Never return robot actions, world
coordinates, or simulator state.
