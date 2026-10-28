"""Fixed observation/action layout of the frozen Neotix policy.

Dependency-free so the baseline runner can share it without importing pydantic.
"""

CAMERAS = ("head_camera", "left_camera", "right_camera")
# Optional fixed views for perception only (harness/environment/perception_cameras.py).
# They may appear in observations but are never sent to the policy.
PERCEPTION_CAMERAS = ("perception_camera",)
# Left arm [0:7] then right arm [7:14]: six joint targets and one gripper command each.
ACTION_DIM = 14
