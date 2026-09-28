"""Stable wire and ROI landmark contracts, independent of model output order."""

LEGACY_BRIDGE_POINT_NAMES = (
    "hl_ankle_l", "hl_ankle_r", "hl_hip_l", "hl_hip_r", "hl_toes_l", "hl_toes_r",
)
KNEE_BRIDGE_POINT_NAMES = LEGACY_BRIDGE_POINT_NAMES + ("hl_knee_l", "hl_knee_r")

# These are exactly the hl_* landmarks of the deployed 15-point model. New
# landmarks must not silently become ROI anchors when a model is replaced.
LEGACY_ROI_ANCHOR_NAMES = (
    "hl_toes_l", "hl_ankle_l", "hl_hip_l", "hl_iliac_l",
    "hl_toes_r", "hl_ankle_r", "hl_hip_r", "hl_iliac_r",
)
