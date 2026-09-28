import unittest

from simulation import Scenario, US, finish_after_service, scenarios, simulate, timing_demonstration


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = {scenario.name: scenario for scenario in scenarios()}
        cls.cache = {}

    def run_case(self, name, policy):
        key = (name, policy)
        if key not in self.cache:
            self.cache[key] = simulate(self.scenarios[name], policy)
        return self.cache[key]

    def test_nominal_same_outputs(self):
        fifo, _ = self.run_case("nominal", "fifo")
        latest, _ = self.run_case("nominal", "latest")
        self.assertEqual(fifo["accepted_times_us"], latest["accepted_times_us"])
        self.assertEqual(fifo["counts"]["accepted"], 600)
        self.assertEqual(fifo["accepted_age_ms"]["max"], 50)
        self.assertEqual(fifo["max_age_of_last_accepted_source_ms"], 150)

    def test_sustained_fifo_growth_latest_bound(self):
        fifo, _ = self.run_case("sustained_overload", "fifo")
        latest, _ = self.run_case("sustained_overload", "latest")
        self.assertGreaterEqual(fifo["max_pending"], 350)
        self.assertLessEqual(latest["max_pending"], 1)
        self.assertEqual(latest["max_active_plus_pending"], 2)
        self.assertGreater(latest["counts"]["replaced_pending"], 0)

    def test_all_conservation_and_acceptance_gates(self):
        for name in self.scenarios:
            for policy in ("fifo", "latest"):
                result, trace = self.run_case(name, policy)
                with self.subTest(name=name, policy=policy):
                    self.assertTrue(all(result["conservation"].values()))
                    self.assertEqual(result["counts"]["invalid_accepted"], 0)
                    for row in trace:
                        if row["event"] == "accepted":
                            self.assertLessEqual(row["job_age_us"], 150_000)
                            self.assertEqual(row["job_version"], row["version"])

    def test_prechange_job_is_never_relabelled(self):
        for policy in ("fifo", "latest"):
            result, trace = self.run_case("version_change_during_job", policy)
            old_completions = [row for row in trace if row["time_us"] > 20_100_000 and row["job_version"] == 0 and row["event"] in ("accepted", "completion_rejected")]
            self.assertTrue(old_completions)
            self.assertTrue(all(row["event"] == "completion_rejected" for row in old_completions))
            self.assertGreater(result["counts"]["completion_rejected_version"], 0)

    def test_pending_version_checked_before_computation(self):
        latest, _ = self.run_case("version_changes_during_pending", "latest")
        self.assertGreater(latest["counts"]["prestart_rejected_version"], 0)

    def test_pending_age_checked_before_computation(self):
        case = Scenario("old_pending", duration_us=US, stalls=((0, 700_000),), blackouts=((150_000, 900_000),))
        latest, trace = simulate(case, "latest")
        self.assertGreater(latest["counts"]["prestart_rejected_stale"], 0)
        rejected_ids = {row["job_id"] for row in trace if row["event"] == "prestart_rejected"}
        started_ids = {row["job_id"] for row in trace if row["event"] == "started"}
        self.assertTrue(rejected_ids.isdisjoint(started_ids))

    def test_finite_recovery_after_stall(self):
        latest, _ = self.run_case("worker_stall", "latest")
        self.assertLessEqual(latest["first_fresh_recovery_after_stall_ms"]["22.0"], 50)

    def test_permanent_overload_has_no_updates(self):
        for name in ("always_600ms", "sustained_overload"):
            for policy in ("fifo", "latest"):
                result, _ = self.run_case(name, policy)
                self.assertEqual(result["counts"]["accepted"], 0)
                self.assertIsNone(result["accepted_age_ms"]["max"])
                self.assertGreater(result["counts"]["completed"], 0)

    def test_blackout_liveness_is_distinct_from_accepted_freshness(self):
        result, _ = self.run_case("source_blackout", "latest")
        self.assertEqual(result["accepted_age_ms"]["max"], 50)
        # The first post-blackout window crosses the new segment boundary and is
        # ineligible; a complete new-segment window arrives one release later.
        self.assertEqual(result["max_age_of_last_accepted_source_ms"], 5250)
        self.assertEqual(result["max_interval_between_accepted_ms"], 5200)

    def test_side_segment_gate(self):
        result, _ = self.run_case("side_segment_change", "latest")
        self.assertGreater(result["counts"]["completion_rejected_segment_side"], 0)

    def test_fifty_hz_and_missing_frame_times(self):
        timing = timing_demonstration()
        self.assertEqual(timing["100_hz"]["actual_lookahead_ms"], 30)
        self.assertEqual(timing["50_hz"]["actual_lookahead_ms"], 60)
        self.assertEqual(timing["irregular_with_missing_frames"]["actual_lookahead_ms"], 110)
        self.assertFalse(timing["irregular_with_missing_frames"]["continuous_at_max_gap_30ms"])
        result, _ = self.run_case("poses_50hz", "latest")
        self.assertEqual(result["accepted_age_ms"]["max"], 80)

    def test_frame_jobs_build_fifo_backlog_window_jobs_do_not(self):
        frame, _ = self.run_case("frame_stress_100hz", "fifo")
        window, _ = self.run_case("window_jobs_10hz", "fifo")
        latest, _ = self.run_case("frame_stress_100hz", "latest")
        self.assertGreater(frame["max_pending"], 2000)
        self.assertEqual(window["counts"]["arrived"], window["counts"]["accepted"])
        self.assertLessEqual(latest["accepted_age_ms"]["max"], 60)

    def test_no_end_drain(self):
        result, _ = simulate(Scenario("end", duration_us=40_000), "fifo")
        self.assertEqual(result["counts"]["arrived"], 1)
        self.assertEqual(result["counts"]["completed"], 0)
        self.assertEqual(result["counts"]["end_active"], 1)

    def test_event_priority_completion_before_configuration_before_arrival(self):
        case = Scenario("tie", duration_us=250_000, service_us=100_000, context_changes=((130_000, "version", 1),))
        result, trace = simulate(case, "latest")
        at_tie = [row for row in trace if row["time_us"] == 130_000]
        kinds = [row["event"] for row in at_tie]
        self.assertLess(kinds.index("accepted"), kinds.index("config"))
        self.assertLess(kinds.index("config"), kinds.index("arrival"))
        arrivals = [row for row in at_tie if row["event"] == "arrival"]
        self.assertEqual(arrivals[0]["job_version"], 0)
        self.assertIn("prestart_rejected", kinds)

    def test_context_is_at_capture_not_release(self):
        case = Scenario("capture_version", duration_us=250_000, context_changes=((20_000, "version", 1),))
        _, trace = simulate(case, "latest")
        first = next(row for row in trace if row["event"] == "arrival")
        self.assertEqual(first["version"], 1)
        self.assertEqual(first["job_version"], 0)
        self.assertFalse(any(row["event"] == "started" and row["job_id"] == first["job_id"] for row in trace))

    def test_mixed_context_window_is_ineligible(self):
        case = Scenario("mixed", duration_us=250_000, context_changes=((95_000, "version", 1),))
        for policy in ("fifo", "latest"):
            result, trace = simulate(case, policy)
            # Anchor at100ms is after the change, but its past window starts70ms.
            self.assertFalse(any(row["event"] == "accepted" and row["job_id"] == 1 for row in trace))
            rejected = result["counts"]["prestart_rejected_mixed_context"] + result["counts"]["completion_rejected_mixed_context"]
            self.assertGreater(rejected, 0)

    def test_stall_service_accounting(self):
        self.assertEqual(finish_after_service(90, 20, ((100, 200),)), 210)
        self.assertEqual(finish_after_service(90, 10, ((100, 200),)), 100)
        self.assertEqual(finish_after_service(100, 10, ((100, 200),)), 210)


if __name__ == "__main__":
    unittest.main(verbosity=2)
