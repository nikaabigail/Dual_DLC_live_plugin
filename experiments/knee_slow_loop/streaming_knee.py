"""Bounded, CPU-only knee smoothing prototype; never controls stimulation.

Input is an already estimated pose and optional already computed geometry.
No camera, DLC, GPU, heading estimation, circle intersection or controller is
included. A seven-sample ring emits only its centre, with three future samples.
Output is returned to the caller, not retained in another queue by this class.

Continuous fully observed windows use exactly the weighted second-difference
objective used in knee_fusion_slow_20260928/run_experiment.py. Startup, end of
stream and discontinuities are deliberately conservative: no partial flush and
no lookahead beyond the currently available ring. NaN/low-confidence knees can
be filled only when BOTH observed boundaries are already in that ring.
"""
from collections import Counter, deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class StreamConfig:
    sample_period_s: float = 0.01
    expected_frame_stride: int = 1
    cadence_tolerance_fraction: float = 0.25
    max_window_span_s: float = 0.075
    max_capture_age_s: float = 0.05
    minimum_likelihood: float = 0.2
    radius: int = 3
    max_missing_knee_frames: int = 3
    temporal_lambda: float = 1.0
    geometry_weight: float = 0.25
    geometry_shift_cap_fraction: float = 0.15

    def __post_init__(self):
        if self.radius != 3:
            raise ValueError('This prototype is explicitly restricted to 7 samples.')
        if self.sample_period_s <= 0 or self.expected_frame_stride < 1:
            raise ValueError('Positive cadence and frame stride are required.')
        if not 0 <= self.cadence_tolerance_fraction < 1:
            raise ValueError('Cadence tolerance must be in [0, 1).')
        if self.max_window_span_s < 6 * self.sample_period_s - 1e-12:
            raise ValueError('Maximum span must allow six nominal intervals.')
        if self.max_capture_age_s < 3 * self.sample_period_s - 1e-12:
            raise ValueError('Deadline must allow three future nominal samples.')
        if not 1 <= self.max_missing_knee_frames <= self.radius:
            raise ValueError('Missing-knee limit must be between 1 and radius.')
        if self.temporal_lambda <= 0:
            raise ValueError('Positive temporal penalty is required.')


@dataclass(frozen=True)
class PoseSample:
    frame_id: int
    capture_time_s: float
    hip_xy: object
    knee_xy: object
    ankle_xy: object
    knee_likelihood: float = 0.8
    side: str = 'right'
    route: str = 'camera_0'
    stim_version: int = 0
    geometry_xy: object = None
    radii: object = None


@dataclass(frozen=True)
class KneeResult:
    frame_id: int
    capture_time_s: float
    processing_started_at_s: float
    knee_xy: object
    hip_xy: object
    ankle_xy: object
    likelihood: float
    status: str
    side: str
    route: str
    stim_version: int
    evidence_frame_ids: tuple
    future_wait_s: float
    capture_age_at_start_s: float
    geometry_used: bool


@dataclass(frozen=True)
class PushOutcome:
    reason: str
    result: object = None


def _xy(value):
    if value is None:
        return np.full(2, np.nan)
    a = np.asarray(value, dtype=float)
    if a.shape != (2,):
        raise ValueError('Coordinate must have shape (2,).')
    return a.copy()


_PENALTIES = {}
for _n in range(2, 8):
    _d = np.diff(np.eye(_n), n=2, axis=0)
    _PENALTIES[_n] = _d.T @ _d


def solve_centre(values, weights, centre, lam=1.0):
    """Exact small linear solve, equivalent to the prior offline smoother."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values).all(axis=1) & (weights > 0)
    if finite.sum() < 2:
        return values[centre].copy() if finite[centre] else np.full(2, np.nan)
    w = np.where(finite, weights, 0.0)
    matrix = np.diag(w) + lam * _PENALTIES[len(values)]
    rhs = w[:, None] * np.where(finite[:, None], values, 0.0)
    return np.linalg.solve(matrix, rhs)[centre]


class StreamingKnee:
    """Constant-space ring. Serial calls from one pose producer are assumed.

    The caller must separately bound its ingress/result queues. Chronologically
    ordered capture timestamps and a shared monotonic clock for `now_s` are
    required; wall-clock time is not used. Intentional decimation must configure
    expected_frame_stride and sample_period_s before feeding a new instance.
    Deadline guards here use call-start time. After this function returns, the
    owning worker MUST recheck completion time, current generation and freshness
    before consuming the result. Numerical compute time is not included in the
    returned capture_age_at_start_s or processing_started_at_s metadata.
    """
    def __init__(self, config=None):
        self.config = config or StreamConfig()
        self.buffer = deque(maxlen=7)
        self.counters = Counter()
        self.max_buffer_seen = 0
        self._last_seen_id = None
        self._last_seen_time = None
        self._last_now = None
        self._generation = None

    @property
    def buffered_samples(self):
        return len(self.buffer)

    def _reset(self, reason):
        self.buffer.clear()
        self.counters['reset_' + reason] += 1

    def push(self, sample, now_s=None):
        c = self.config
        now = sample.capture_time_s if now_s is None else float(now_s)
        t = float(sample.capture_time_s)
        if not math.isfinite(t) or not math.isfinite(now) or now < t - 1e-12:
            self.counters['rejected_invalid_time'] += 1
            return PushOutcome('invalid_time')
        if self._last_now is not None and now < self._last_now - 1e-12:
            self.counters['rejected_nonmonotonic_arrival'] += 1
            return PushOutcome('nonmonotonic_arrival')
        self._last_now = now
        if self._last_seen_id is not None and (
                sample.frame_id <= self._last_seen_id or t <= self._last_seen_time):
            self.counters['rejected_duplicate_or_reordered'] += 1
            return PushOutcome('duplicate_or_reordered')
        self._last_seen_id = sample.frame_id
        self._last_seen_time = t
        if now - t > c.max_capture_age_s + 1e-12:
            self._reset('stale_input')
            self.counters['rejected_stale_input'] += 1
            return PushOutcome('stale_input')

        hip, knee, ankle = _xy(sample.hip_xy), _xy(sample.knee_xy), _xy(sample.ankle_xy)
        if not np.isfinite([hip, ankle]).all():
            self._reset('invalid_anchor')
            self.counters['rejected_invalid_anchor'] += 1
            return PushOutcome('invalid_anchor')
        generation = (sample.side, sample.route, sample.stim_version)
        if self.buffer:
            prev = self.buffer[-1]
            dt = t - prev['time']
            if generation != self._generation:
                self._reset('generation_boundary')
            elif sample.frame_id - prev['frame_id'] != c.expected_frame_stride:
                self._reset('frame_gap')
            elif abs(dt - c.sample_period_s) > c.sample_period_s * c.cadence_tolerance_fraction + 1e-12:
                self._reset('time_gap')
        self._generation = generation

        p = float(sample.knee_likelihood)
        observed = bool(np.isfinite(knee).all() and math.isfinite(p)
                        and c.minimum_likelihood <= p <= 1.0)
        wd = float(np.clip(p, c.minimum_likelihood, 1.0)) if observed else 0.0
        mixed = knee.copy() if observed else np.full(2, np.nan)
        geo_used = False
        wg = 0.0
        geo, radii = _xy(sample.geometry_xy), _xy(sample.radii)
        if (observed and np.isfinite(geo).all() and np.isfinite(radii).all()
                and (radii > 0).all() and c.geometry_weight > 0):
            wg = c.geometry_weight
            delta = (geo - knee) * (wg / (wd + wg))
            norm = float(np.linalg.norm(delta))
            cap = c.geometry_shift_cap_fraction * float(radii.sum())
            if norm > cap and norm > 1e-8:
                delta *= cap / norm
            mixed = knee + delta
            geo_used = True
        self.buffer.append({'frame_id': int(sample.frame_id), 'time': t,
                            'hip': hip, 'knee': knee, 'ankle': ankle,
                            'mixed': mixed, 'weight': wd + wg, 'observed': observed,
                            'likelihood': p, 'generation': generation, 'geometry_used': geo_used})
        self.counters['accepted_samples'] += 1
        self.max_buffer_seen = max(self.max_buffer_seen, len(self.buffer))
        if len(self.buffer) < 7:
            return PushOutcome('warming_up')
        window = list(self.buffer)
        if window[-1]['time'] - window[0]['time'] > c.max_window_span_s + 1e-12:
            # Keep only the newest source sample, so no partial old span survives.
            newest = self.buffer[-1]
            self._reset('window_span')
            self.buffer.append(newest)
            return PushOutcome('window_span')
        centre = window[3]
        if now - centre['time'] > c.max_capture_age_s + 1e-12:
            self.counters['rejected_stale_output'] += 1
            return PushOutcome('stale_output')

        observed_mask = np.array([r['observed'] for r in window], dtype=bool)
        process = observed_mask.copy()
        # Only gaps with both real observations already present are eligible.
        a = 0
        while a < 7:
            if observed_mask[a]:
                a += 1
                continue
            b = a + 1
            while b < 7 and not observed_mask[b]:
                b += 1
            if a > 0 and b < 7 and b - a <= c.max_missing_knee_frames:
                process[a:b] = True
            a = b
        if not process[3]:
            xy = np.full(2, np.nan)
            status = 'missing_unbounded_or_long_gap'
            used_window = (centre,)
        else:
            lo, hi = 3, 4
            while lo > 0 and process[lo - 1]:
                lo -= 1
            while hi < 7 and process[hi]:
                hi += 1
            used_window = window[lo:hi]
            values = np.array([r['mixed'] - r['hip'] for r in used_window])
            weights = np.array([r['weight'] for r in used_window])
            xy = solve_centre(values, weights, 3 - lo, c.temporal_lambda) + centre['hip']
            if not centre['observed']:
                status = 'interpolated' if np.isfinite(xy).all() else 'missing'
            elif np.linalg.norm(xy - centre['knee']) > 1e-8:
                status = 'observed_adjusted'
            else:
                status = 'observed_unadjusted'
        result = KneeResult(
            frame_id=centre['frame_id'], capture_time_s=centre['time'], processing_started_at_s=now,
            knee_xy=xy, hip_xy=centre['hip'].copy(), ankle_xy=centre['ankle'].copy(),
            likelihood=centre['likelihood'] if status == 'observed_unadjusted' else float('nan'),
            status=status, side=generation[0], route=generation[1], stim_version=generation[2],
            evidence_frame_ids=tuple(r['frame_id'] for r in used_window),
            future_wait_s=window[-1]['time'] - centre['time'],
            capture_age_at_start_s=now - centre['time'], geometry_used=any(r['geometry_used'] for r in used_window))
        self.counters['emitted_results'] += 1
        self.counters['status_' + status] += 1
        return PushOutcome('emitted', result)


def synthetic_sample(i, source_period_s=0.01, frame_stride=1, **overrides):
    """Repeatable moving-hip trajectory with smooth motion plus deterministic noise."""
    t = i * source_period_s
    hip = np.array([200.0 + 10.0 * np.sin(t), 90.0 + 2.0 * np.cos(2*t)])
    clean = hip + np.array([28.0 + 6.0*np.sin(12*t), 37.0 + 10.0*np.cos(12*t)])
    knee = clean + np.array([1.3*np.sin(91*t), 1.7*np.cos(77*t)])
    payload = dict(frame_id=i*frame_stride, capture_time_s=t, hip_xy=hip,
                   knee_xy=knee, ankle_xy=hip+np.array([57.0, 80.0]),
                   knee_likelihood=0.7+0.2*np.cos(t), geometry_xy=clean,
                   radii=np.array([45.0, 50.0]))
    payload.update(overrides)
    return PoseSample(**payload)
