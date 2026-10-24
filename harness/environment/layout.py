"""Fixed observation/action layout of the frozen Neotix policy.

Dependency-free so the baseline runner can share it without importing pydantic.
"""

CAMERAS = ("head_camera", "left_camera", "right_camera")
# Left arm [0:7] then right arm [7:14]: six joint targets and one gripper command each.
ACTION_DIM = 14
