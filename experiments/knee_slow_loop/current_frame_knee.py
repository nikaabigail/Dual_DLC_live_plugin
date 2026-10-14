"""Current-frame 2D circle correction; no camera, model, history or controller.

The caller supplies the already selected side's observations and a frozen,
explicit calibration in the same image coordinates. Projected segment lengths
are not anatomical ground truth. No default animal/calibration is embedded.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Sequence


XY = tuple[float, float]
Observation = Sequence[float]
_NAN_XY: XY = (math.nan, math.nan)


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class CircleCalibration:
    """Frozen effective projected lengths; any anatomy adjustment is already applied.

    ``canonical_branch_sign`` is the old calibrated circle-branch sign in a
    forward-right body orientation; multiply it by current heading (+1/-1).
    This profile must match the subject, camera, scale and coordinate system.
    """
    width: int
    height: int
    femur_px: float
    tibia_px: float
    canonical_branch_sign: int
    calibration_id: str = ''

    def __post_init__(self):
        for name in ('width', 'height'):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('femur_px', 'tibia_px'):
            value = getattr(self, name)
            if not _finite_number(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        if (not isinstance(self.canonical_branch_sign, Integral)
                or isinstance(self.canonical_branch_sign, bool)
                or self.canonical_branch_sign not in (-1, 1)):
            raise ValueError('canonical_branch_sign must be the calibrated -1 or +1.')
        if not isinstance(self.calibration_id, str):
            raise ValueError('calibration_id must be a string.')


@dataclass(frozen=True)
class FusionConfig:
    pcutoff: float = 0.6
    fusion_geometry_weight: float = 0.25
    geometry_shift_cap_fraction: float = 0.15
    maximum_disagreement_fraction: float = 0.5
    heading_minimum_dx_px: float = 10.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if not _finite_number(getattr(self, name)):
                raise ValueError(f'{name} must be finite and numeric.')
        if not 0 <= self.pcutoff <= 1:
            raise ValueError('pcutoff must be in [0, 1].')
        if self.fusion_geometry_weight < 0:
            raise ValueError('fusion_geometry_weight must be nonnegative.')
        if not 0 <= self.geometry_shift_cap_fraction <= 1:
            raise ValueError('geometry_shift_cap_fraction must be in [0, 1].')
        if self.maximum_disagreement_fraction <= 0:
            raise ValueError('maximum_disagreement_fraction must be positive.')
        if self.heading_minimum_dx_px <= 0:
            raise ValueError('heading_minimum_dx_px must be positive.')


@dataclass(frozen=True)
class KneeFusionResult:
    raw_xy: XY
    fused_xy: XY
    geometry_xy: XY
    hip_xy: XY
    ankle_xy: XY
    radii: XY
    status: str
    applied: bool
    likelihood: float
    correction_px: float


def _row(value: Observation | None) -> tuple[XY, float, bool]:
    """Retain finite raw xy even when confidence is missing/invalid."""
    if value is None:
        return _NAN_XY, math.nan, False
    try:
        entries = tuple(value)
        if len(entries) != 3:
            return _NAN_XY, math.nan, False
        x, y, p = (float(v) for v in entries)
    except (TypeError, ValueError, OverflowError):
        return _NAN_XY, math.nan, False
    xy = (x, y) if math.isfinite(x) and math.isfinite(y) else _NAN_XY
    valid = math.isfinite(x) and math.isfinite(y) and math.isfinite(p) and 0 <= p <= 1
    return xy, p, valid


def _in_bounds(xy: XY, calibration: CircleCalibration) -> bool:
    return 0 <= xy[0] < calibration.width and 0 <= xy[1] < calibration.height


def _intersections(hip: XY, ankle: XY, femur: float, tibia: float):
    """The existing planar intersection convention, including tangent tolerance."""
    dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-8:
        return (), 'coincident_anchors'
    if distance > femur + tibia + 1e-7 or distance < abs(femur - tibia) - 1e-7:
        return (), 'incompatible_lengths'
    along = (femur * femur - tibia * tibia + distance * distance) / (2 * distance)
    square = femur * femur - along * along
    if not math.isfinite(along) or not math.isfinite(square):
        return (), 'numerical_failure'
    if square < -1e-6:
        return (), 'incompatible_lengths'
    height = math.sqrt(max(0.0, square))
    ex, ey = dx / distance, dy / distance
    signs = (1,) if height <= 1e-7 else (1, -1)
    return tuple((sign, (hip[0] + along * ex - sign * height * ey,
                         hip[1] + along * ey + sign * height * ex))
                 for sign in signs), None


def fuse_current_frame(
    hip: Observation | None,
    knee: Observation | None,
    ankle: Observation | None,
    *,
    calibration: CircleCalibration | None,
    nose: Observation | None = None,
    eye: Observation | None = None,
    config: FusionConfig = FusionConfig(),
) -> KneeFusionResult:
    """Blend this frame's raw knee toward its calibrated circle hypothesis.

    Observations are (x, y, original DLC likelihood), in full-image coordinates.
    ``eye`` must belong to the selected side. Nose is preferred; the eye is a
    fallback when nose is invalid, weak or too close in x to establish heading.

    Every refusal preserves finite raw knee coordinates exactly. That does not
    make a weak/out-of-bounds raw point valid: consumers must use status and the
    original confidence. Missing knees are never inserted. ``likelihood`` stays
    original DLC metadata and is NOT the uncertainty of the fused estimate.

    Inputs are never modified. There is no history, future-frame lookahead,
    interpolation, calibration fitting or inference. Caller-owned temporal
    smoothing remains an explicit separate operation.
    """
    if not isinstance(config, FusionConfig):
        raise TypeError('config must be FusionConfig.')
    if calibration is not None and not isinstance(calibration, CircleCalibration):
        raise TypeError('calibration must be CircleCalibration or None.')
    h, ph, hv = _row(hip)
    k, pk, kv = _row(knee)
    a, pa, av = _row(ankle)
    result = KneeFusionResult(k, k, _NAN_XY, h, a, _NAN_XY,
                              'raw_missing_calibration', False, pk, 0.0)
    if calibration is None:
        return result
    if not kv or not _in_bounds(k, calibration):
        return replace(result, status='raw_invalid_knee')
    if pk < config.pcutoff:
        return replace(result, status='raw_low_knee_confidence')
    if not hv or not av or not _in_bounds(h, calibration) or not _in_bounds(a, calibration):
        return replace(result, status='raw_invalid_anchor')
    if ph < config.pcutoff or pa < config.pcutoff:
        return replace(result, status='raw_low_anchor_confidence')
    heading = None
    for source in (nose, eye):
        head, phead, valid = _row(source)
        if valid and phead >= config.pcutoff and _in_bounds(head, calibration):
            dx = head[0] - h[0]
            if abs(dx) >= config.heading_minimum_dx_px:
                heading = 1 if dx > 0 else -1
                break
    if heading is None:
        return replace(result, status='raw_missing_current_heading')
    radii = (float(calibration.femur_px), float(calibration.tibia_px))
    result = replace(result, radii=radii)
    candidates, reason = _intersections(h, a, *radii)
    if not candidates:
        return replace(result, status='raw_geometry_' + reason)
    if not any(_in_bounds(xy, calibration) for _, xy in candidates):
        return replace(result, status='raw_geometry_all_candidates_out_of_bounds')
    branch = int(calibration.canonical_branch_sign) * heading
    geo = candidates[0][1] if len(candidates) == 1 else next(xy for sign, xy in candidates if sign == branch)
    if not _in_bounds(geo, calibration):
        return replace(result, status='raw_geometry_recommended_out_of_bounds')
    result = replace(result, geometry_xy=geo)
    dx, dy = geo[0] - k[0], geo[1] - k[1]
    difference = math.hypot(dx, dy)
    total = sum(radii)
    if difference > config.maximum_disagreement_fraction * total:
        return replace(result, status='raw_large_disagreement')
    wg = config.fusion_geometry_weight
    if wg == 0:
        return replace(result, status='raw_geometry_disabled')
    wd = max(0.2, min(pk, 1.0))
    fraction = wg / (wd + wg)
    dx, dy = dx * fraction, dy * fraction
    shift = math.hypot(dx, dy)
    cap = config.geometry_shift_cap_fraction * total
    if shift > cap and shift > 0:
        dx, dy = dx * (cap / shift), dy * (cap / shift)
    return replace(result, fused_xy=(k[0] + dx, k[1] + dy),
                   status='fused_current_frame', applied=True,
                   correction_px=math.hypot(dx, dy))
