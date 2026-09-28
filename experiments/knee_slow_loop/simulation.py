"""Deterministic offline scheduler model; no model/GPU/device or production imports.

Time is integer microseconds. Events at equal times run completion, configuration /
wake-up, arrival, then observation. Ordinary FIFO computes old queued jobs; both
policies apply the identical completion freshness/context gate. The latest policy
also drops an ineligible pending job before starting it. No work is drained after
the 60-second observation horizon. All thresholds are synthetic engineering inputs.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

US = 1_000_000
FRESHNESS_US = 150_000


@dataclass(frozen=True)
class Job:
    seq: int
    release_us: int
    capture_us: int
    version: int
    segment: int
    side: str
    service_us: int
    mixed_context: bool = False


@dataclass
class Scenario:
    name: str
    duration_us: int = 60 * US
    interval_us: int = 100_000
    service_us: int = 20_000
    lookahead_us: int = 30_000
    freshness_us: int = FRESHNESS_US
    special_service: dict[int, int] = field(default_factory=dict)
    stalls: tuple[tuple[int, int], ...] = ()
    blackouts: tuple[tuple[int, int], ...] = ()
    context_changes: tuple[tuple[int, str, object], ...] = ()
    description: str = ""


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def stats_ms(values):
    return {
        "n": len(values),
        "mean": sum(values) / len(values) / 1000 if values else None,
        "median": percentile(values, 50) / 1000 if values else None,
        "p95": percentile(values, 95) / 1000 if values else None,
        "max": max(values) / 1000 if values else None,
    }


def blocked_until(now, stalls):
    for start, end in stalls:
        if start <= now < end:
            return end
    return now


def finish_after_service(start, work, stalls):
    """Pause useful worker time during a stall; stalls must not overlap."""
    current, remaining = start, work
    for lo, hi in stalls:
        if hi <= current:
            continue
        available = max(0, lo - current)
        if remaining <= available:
            return current + remaining
        remaining -= available
        current = max(current, hi)
    return current + remaining


def timing_demonstration():
    series = {
        "100_hz": [i * 10_000 for i in range(7)],
        "50_hz": [i * 20_000 for i in range(7)],
        "irregular_with_missing_frames": [0, 10_000, 20_000, 30_000, 40_000, 100_000, 140_000],
    }
    return {
        name: {
            "capture_times_us": times,
            "window_frames": 7,
            "future_frames": 3,
            "actual_lookahead_ms": (times[6] - times[3]) / 1000,
            "largest_interframe_gap_ms": max(b - a for a, b in zip(times, times[1:])) / 1000,
            "continuous_at_max_gap_30ms": max(b - a for a, b in zip(times, times[1:])) <= 30_000,
        }
        for name, times in series.items()
    }


def simulate(scenario: Scenario, policy: str, keep_trace=True):
    if policy not in ("fifo", "latest"):
        raise ValueError(policy)
    if scenario.lookahead_us < 0 or scenario.interval_us <= 0 or scenario.service_us <= 0:
        raise ValueError("Nonnegative lookahead, positive arrival interval/service required")
    if any(a >= b for a, b in scenario.stalls):
        raise ValueError("Invalid stall interval")
    if any(b > c for (_, b), (c, _) in zip(scenario.stalls, scenario.stalls[1:])):
        raise ValueError("Stalls must be ordered and nonoverlapping")

    events, serial = [], 0

    def event(time, priority, kind, payload=None):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (time, priority, serial, kind, payload))

    for change in scenario.context_changes:
        event(change[0], 1, "config", change[1:])
    for _, end in scenario.stalls:
        event(end, 1, "wake")
    seq = 0
    for time in range(scenario.lookahead_us, scenario.duration_us, scenario.interval_us):
        if not any(a <= time < b for a, b in scenario.blackouts):
            event(time, 2, "arrival", seq)
        seq += 1
    for time in range(0, scenario.duration_us + 1, 100_000):
        event(time, 3, "sample")
    if scenario.duration_us % 100_000:
        event(scenario.duration_us, 3, "sample")

    context = {"version": 0, "segment": 0, "side": "r"}
    pending = deque()
    active = None
    counts = Counter()
    accepted_ages, completed_ages, started_ages = [], [], []
    accepted_times, accepted_captures = [], []
    trace = []
    max_queue = max_outstanding = max_last_age = 0
    last_accepted_capture = last_accepted_at = newest_capture = None
    first_recovery_after_stall = {}
    seen_ids = set()
    dispositions = {}

    def reasons(job, now):
        result = []
        if job.version != context["version"]:
            result.append("version")
        if job.segment != context["segment"] or job.side != context["side"]:
            result.append("segment_side")
        if job.mixed_context:
            result.append("mixed_context")
        if now - job.capture_us > scenario.freshness_us:
            result.append("stale")
        return result

    def record(now, kind, job=None, reason=""):
        nonlocal max_queue, max_outstanding, max_last_age
        max_queue = max(max_queue, len(pending))
        max_outstanding = max(max_outstanding, len(pending) + int(active is not None))
        if policy == "latest":
            assert len(pending) <= 1
            assert len(pending) + int(active is not None) <= 2
        last_age = None if last_accepted_capture is None else now - last_accepted_capture
        if last_age is not None:
            max_last_age = max(max_last_age, last_age)
        if keep_trace:
            trace.append({
                "scenario": scenario.name, "policy": policy,
                "time_us": now, "event": kind, "job_id": job.seq if job else "",
                "source_capture_us": job.capture_us if job else "",
                "job_release_us": job.release_us if job else "",
                "job_age_us": now - job.capture_us if job else "",
                "version": context["version"],
                "job_version": job.version if job else "",
                "segment": context["segment"], "side": context["side"],
                "job_segment": job.segment if job else "", "job_side": job.side if job else "",
                "job_mixed_context": job.mixed_context if job else "",
                "pending": len(pending), "active": int(active is not None),
                "accepted_so_far": counts["accepted"],
                "newest_usable_source_age_us": "" if newest_capture is None else now - newest_capture,
                "last_accepted_source_age_us": "" if last_age is None else last_age,
                "time_since_last_accept_us": "" if last_accepted_at is None else now - last_accepted_at,
                "reason": reason,
            })

    def start_next(now):
        nonlocal active
        if active is not None or blocked_until(now, scenario.stalls) > now:
            return
        while pending:
            job = pending.popleft()
            invalid = reasons(job, now)
            if policy == "latest" and invalid:
                counts["prestart_rejected"] += 1
                counts["prestart_rejected_" + invalid[0]] += 1
                dispositions[job.seq] = "prestart_rejected"
                record(now, "prestart_rejected", job, ";".join(invalid))
                continue
            active = job
            counts["started"] += 1
            started_ages.append(now - job.capture_us)
            event(finish_after_service(now, job.service_us, scenario.stalls), 0, "complete", job)
            record(now, "started", job)
            return

    while events:
        now, _, _, kind, payload = heapq.heappop(events)
        if now > scenario.duration_us:
            break
        # Peak age occurs just BEFORE a fresh accepted result replaces the previous
        # source timestamp. Measuring only after replacement underestimates AoI.
        if last_accepted_capture is not None:
            max_last_age = max(max_last_age, now - last_accepted_capture)
        if kind == "complete":
            job = payload
            assert active == job
            active = None
            counts["completed"] += 1
            age = now - job.capture_us
            completed_ages.append(age)
            invalid = reasons(job, now)
            if invalid:
                counts["completion_rejected"] += 1
                counts["completion_rejected_" + invalid[0]] += 1
                for reason in invalid:
                    counts["completion_violation_" + reason] += 1
                dispositions[job.seq] = "completion_rejected"
                record(now, "completion_rejected", job, ";".join(invalid))
            else:
                counts["accepted"] += 1
                counts["invalid_accepted"] += int(age > scenario.freshness_us or job.version != context["version"] or job.segment != context["segment"] or job.side != context["side"] or job.mixed_context)
                accepted_ages.append(age)
                accepted_times.append(now)
                accepted_captures.append(job.capture_us)
                last_accepted_capture, last_accepted_at = job.capture_us, now
                dispositions[job.seq] = "accepted"
                for _, stall_end in scenario.stalls:
                    if now >= stall_end and stall_end not in first_recovery_after_stall:
                        first_recovery_after_stall[stall_end] = now - stall_end
                record(now, "accepted", job)
            start_next(now)
        elif kind == "config":
            key, value = payload
            if key not in context:
                raise ValueError(key)
            context[key] = value
            record(now, "config", reason=f"{key}={value}")
        elif kind == "wake":
            record(now, "wake")
            start_next(now)
        elif kind == "arrival":
            capture = now - scenario.lookahead_us
            source_context = {"version": 0, "segment": 0, "side": "r"}
            for change_time, key, value in sorted(scenario.context_changes, key=lambda row: row[0]):
                if change_time <= capture:
                    source_context[key] = value
            # A symmetric 7-frame snapshot spans three past/future intervals. Do
            # not present a window crossing a config/side boundary as homogeneous.
            mixed = any(capture - scenario.lookahead_us < change_time <= now
                        for change_time, _, _ in scenario.context_changes)
            job = Job(payload, now, capture,
                      source_context["version"], source_context["segment"], source_context["side"],
                      scenario.special_service.get(payload, scenario.service_us), mixed)
            assert job.seq not in seen_ids
            seen_ids.add(job.seq)
            counts["arrived"] += 1
            newest_capture = job.capture_us
            if policy == "latest" and pending:
                removed = pending.popleft()
                counts["replaced_pending"] += 1
                dispositions[removed.seq] = "replaced"
                record(now, "replaced_pending", removed)
            pending.append(job)
            record(now, "arrival", job)
            start_next(now)
        elif kind == "sample":
            record(now, "sample")

    counts["end_pending"] = len(pending)
    counts["end_active"] = int(active is not None)
    if active:
        dispositions[active.seq] = "end_active"
    for job in pending:
        dispositions[job.seq] = "end_pending"
    counts["end_unfinished"] = counts["end_pending"] + counts["end_active"]
    conservation = {
        "arrival_partition": counts["arrived"] == counts["completed"] + counts["replaced_pending"] + counts["prestart_rejected"] + counts["end_unfinished"],
        "started_partition": counts["started"] == counts["completed"] + counts["end_active"],
        "completion_partition": counts["completed"] == counts["accepted"] + counts["completion_rejected"],
        "unique_final_disposition": len(dispositions) == counts["arrived"] and set(dispositions) == seen_ids,
        "invalid_accepted_zero": counts["invalid_accepted"] == 0,
    }
    assert all(conservation.values()), conservation
    count_keys = ["arrived", "started", "completed", "accepted", "invalid_accepted",
                  "replaced_pending", "prestart_rejected", "prestart_rejected_stale",
                  "prestart_rejected_version", "prestart_rejected_segment_side", "prestart_rejected_mixed_context",
                  "completion_rejected", "completion_rejected_stale", "completion_rejected_version",
                  "completion_rejected_segment_side", "completion_rejected_mixed_context", "completion_violation_stale",
                  "completion_violation_version", "completion_violation_segment_side", "completion_violation_mixed_context",
                  "end_pending", "end_active", "end_unfinished"]
    result = {
        "scenario": scenario.name, "policy": policy, "description": scenario.description,
        "horizon_s": scenario.duration_us / US,
        "arrival_interval_ms": scenario.interval_us / 1000,
        "service_ms": scenario.service_us / 1000,
        "lookahead_ms": scenario.lookahead_us / 1000,
        "freshness_threshold_ms": scenario.freshness_us / 1000,
        "counts": {key: counts[key] for key in count_keys},
        "max_pending": max_queue, "max_active_plus_pending": max_outstanding,
        "accepted_age_ms": stats_ms(accepted_ages),
        "completed_age_ms": stats_ms(completed_ages),
        "started_age_ms": stats_ms(started_ages),
        "last_accepted_source_age_at_end_ms": (scenario.duration_us - last_accepted_capture) / 1000 if last_accepted_capture is not None else None,
        "max_age_of_last_accepted_source_ms": max_last_age / 1000 if accepted_ages else None,
        "time_without_new_accepted_at_end_ms": (scenario.duration_us - last_accepted_at) / 1000 if last_accepted_at is not None else scenario.duration_us / 1000,
        "max_interval_between_accepted_ms": max((b - a for a, b in zip(accepted_times, accepted_times[1:])), default=0) / 1000 if accepted_times else None,
        "accepted_rate_hz": counts["accepted"] / (scenario.duration_us / US),
        "first_fresh_recovery_after_stall_ms": {str(k / US): v / 1000 for k, v in first_recovery_after_stall.items()},
        "conservation": conservation,
        "accepted_times_us": accepted_times,
        "accepted_source_capture_us": accepted_captures,
    }
    return result, trace


def scenarios():
    return [
        Scenario("nominal", description="10 Hz jobs, 20 ms service; no overload"),
        Scenario("sustained_overload", service_us=250_000, description="10 Hz jobs, 250 ms service; each completed job already exceeds the 150 ms freshness budget"),
        Scenario("slow_burst", special_service={200: 2 * US}, description="One job released at 20.03 s needs 2 s; other jobs need 20 ms"),
        Scenario("worker_stall", stalls=((20 * US, 22 * US),), description="Worker unavailable in [20,22) s; input continues"),
        Scenario("always_600ms", service_us=600_000, description="Every job needs 600 ms; bounded queue does not guarantee any usable updates"),
        Scenario("version_change_during_job", special_service={200: 600_000}, context_changes=((20_100_000, "version", 1),), description="Stimulus version changes while an old-version job runs; pending version is checked before start"),
        Scenario("version_changes_during_pending", service_us=80_000, stalls=((20_000_000, 20_500_000),), context_changes=((20_490_000, "version", 1),), description="Pending snapshot has old version when worker resumes, so latest policy rejects it before computation"),
        Scenario("source_blackout", blackouts=((20 * US, 25 * US),), context_changes=((25 * US, "segment", 1),), description="No source snapshots for 5 s; accepted-update age can rise despite every applied update being fresh"),
        Scenario("side_segment_change", special_service={200: 80_000}, context_changes=((20_060_000, "side", "l"), (20_060_000, "segment", 1)), description="A job from the old side/segment must not apply to the new segment"),
        Scenario("frame_stress_100hz", interval_us=10_000, service_us=16_667, description="Every 100 Hz frame creates a 16.667 ms job; naive FIFO accumulates"),
        Scenario("window_jobs_10hz", service_us=16_667, description="Same service but one snapshot job per 100 ms; scheduling frequency is separate from pose frequency"),
        Scenario("poses_50hz", lookahead_us=60_000, description="50 Hz poses require 60 ms for 3 future frames; analysis job release remains 10 Hz"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('out') / 'knee_slow_loop',
                        help='Output directory for synthetic JSON/CSV (default: out/knee_slow_loop).')
    folder = parser.parse_args(argv).output_dir
    folder.mkdir(parents=True, exist_ok=True)
    results, traces = [], []
    for scenario in scenarios():
        for policy in ("fifo", "latest"):
            result, trace = simulate(scenario, policy)
            results.append(result)
            traces.extend(trace)
    payload = {
        "schema_version": 1,
        "experiment": "deterministic integer-time offline backlog simulation",
        "units": "microseconds in traces, explicitly named milliseconds in aggregate metrics",
        "engineer_selected_freshness_threshold_ms": FRESHNESS_US / 1000,
        "threshold_is_biologically_validated": False,
        "event_priority": ["completion", "configuration or wake-up", "arrival", "observation"],
        "completion_exactly_at_config_time": "completion uses the context valid immediately before the configuration event",
        "horizon_policy": "arrivals at t < horizon, completions at t <= horizon; no end drain",
        "source_timestamp_definition": "capture_us is the newest usable kinematic output anchor; release_us is the capture time of its third future input frame; fixed pose lookahead is included in age; source context is attributed at capture, not release",
        "mixed_context_policy": "a symmetric window crossing a version/side/segment configuration boundary is ineligible; same completion gate for FIFO/latest, plus latest prestart gate",
        "fifo_policy": "ordinary unbounded FIFO computes queued jobs, then uses the same freshness/context completion gate as latest",
        "latest_policy": "one active plus at most one replaceable pending job; pending age/version/segment checked before start; active job is not cancelled",
        "service_policy": "cost assigned to each job at release; identical arrivals and per-job work requirement for both schedulers; stalls pause worker time",
        "important_limits": [
            "No measured DLC inference speed, CPU/GPU contention, camera latency or biological phase cycles are modelled.",
            "150 ms is a test budget, not an experimentally supported safe stimulus-update delay.",
            "A finite queue and fresh-only acceptance do not establish liveness or input continuity.",
            "No automatic invalidation of already applied settings or stimulation safety controller is modelled.",
            "A real implementation needs a separate stale-data watchdog; no accepted updates can persist indefinitely.",
        ],
        "timing_demonstration": timing_demonstration(),
        "results": results,
    }
    (folder / "simulation_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (folder / "simulation_trace.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    for result in results:
        counts = result["counts"]
        print(f"{result['scenario']:30} {result['policy']:6} accepted={counts['accepted']:4} pending_max={result['max_pending']:4} completed_age_max_ms={result['completed_age_ms']['max']} accepted_age_max_ms={result['accepted_age_ms']['max']} unfinished={counts['end_unfinished']}")
    print(f"Wrote {len(results)} runs, {len(traces)} trace events")


if __name__ == "__main__":
    main()
