"""Deterministic correctness tests for the independent phase-one replay."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PhaseOneReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.binary = cls.directory / "replay"
        subprocess.run([
            "g++", "-O2", "-std=c++17", "-pthread",
            "-I" + str(ROOT / "cpp/include"),
            str(ROOT / "cpp/src/line_admission.cpp"),
            str(ROOT / "experiments/frequency_momentum_ablation/replay.cpp"),
            "-o", str(cls.binary),
        ], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def replay(self, rows, policy="dual_max", capacity=2,
               frequency_threshold=3, momentum_threshold=1,
               hit_ns=11915, fill_ns=72000, bypass_ns=65977):
        trace = self.directory / f"trace_{self.id().rsplit('.', 1)[-1]}.txt"
        trace.write_text("".join(f"{' '.join(map(str, row))}\n" for row in rows))
        completed = subprocess.run([
            str(self.binary), str(trace), policy, str(capacity),
            "8192", "32768", str(frequency_threshold),
            "1024", "128", str(momentum_threshold),
            str(64 * 1024), str(hit_ns), str(fill_ns), str(bypass_ns),
        ], check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)

    def test_signed_net_saved_formula(self):
        result = self.replay(
            [(1, 0, 0, 3), (1, 0, 0, 3)], policy="cache_all", capacity=2)
        expected = (
            result["hits"] * (65977 - 11915)
            - result["admissions"] * (72000 - 65977)
        )
        self.assertEqual(result["net_saved_ns"], expected)
        self.assertEqual(result["hit_saving_ns"] - result["admission_cost_ns"], expected)

    def test_momentum_one_captures_second_burst_request(self):
        result = self.replay([(41, 0, 0, 3), (41, 0, 0, 3)])
        self.assertEqual(result["two_ref_second_requests"], 1)
        self.assertEqual(result["two_ref_second_hits"], 1)
        self.assertEqual(result["two_ref_second_hit_ratio"], 1.0)

    def test_dual_score_rejects_one_touch_scan_against_hot_residents(self):
        rows = []
        for _ in range(6):
            rows.extend([(1, 0, 1, 1), (2, 0, 1, 1)])
        rows.extend((100 + index, 0, 0, 2) for index in range(10))
        result = self.replay(rows, policy="dual_max", capacity=2)
        self.assertGreater(result["scan_replacement_attempts"], 0)
        self.assertGreater(result["scan_replacement_rejections"], 0)
        self.assertEqual(result["scan_hot_evictions"], 0)

    def test_shadow_pollution_is_distinct_from_victim_reaccess(self):
        pollution = self.replay(
            [(1, 0, 1, 1), (2, 0, 0, 0), (1, 0, 1, 1)],
            policy="cache_all", capacity=1)
        self.assertEqual(pollution["victim_reaccess_misses"], 1)
        self.assertEqual(pollution["counterfactual_pollution_misses"], 1)

        reaccess_only = self.replay(
            [(1, 0, 1, 1), (2, 0, 0, 0), (3, 0, 0, 0), (2, 0, 0, 0)],
            policy="cache_all", capacity=1)
        self.assertGreaterEqual(reaccess_only["victim_reaccess_misses"], 1)
        self.assertEqual(reaccess_only["counterfactual_pollution_misses"], 0)

    def test_all_score_modes_execute(self):
        rows = [(1, 0, 1, 1), (2, 0, 1, 1), (3, 0, 0, 2), (1, 0, 1, 1)]
        for policy in (
            "dual_max", "dual_weighted", "dual_multiplicative", "dual_lexicographic"
        ):
            with self.subTest(policy=policy):
                result = self.replay(rows, policy=policy)
                self.assertEqual(result["policy"], policy)
                self.assertEqual(result["hits"] + result["misses"], 4)


if __name__ == "__main__":
    unittest.main()
