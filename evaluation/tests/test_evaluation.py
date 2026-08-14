"""Offline unit and result-integrity tests for the evaluation workflow."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.run_python_evaluation import (
    ATOL,
    BLOCK_SIZE,
    RTOL,
    VOCAB_SIZE,
    prepare_context,
    production_mask,
    select_stratified,
    stable_softmax,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"


class PreprocessingTests(unittest.TestCase):
    def test_prepare_context_inserts_no_event_and_reserves_target(self) -> None:
        rows = np.asarray(
            [
                [101, 0, 1],
                [101, 1, 10],
                [101, 2, 11],
            ],
            dtype=np.float64,
        )

        tokens, ages = prepare_context(rows)

        np.testing.assert_array_equal(tokens, np.asarray([2, 11, 1], dtype=np.int64))
        np.testing.assert_array_equal(ages, np.asarray([0, 1, 1], dtype=np.float32))
        self.assertEqual(tokens.dtype, np.dtype("int64"))
        self.assertEqual(ages.dtype, np.dtype("float32"))

    def test_prepare_context_matches_upstream_tied_age_order(self) -> None:
        # This fixture distinguishes upstream torch.argsort's tie order from a
        # stable NumPy sort: the second raw tied event precedes the first.
        ages = [0, 1000, 1000, *range(3000, 15000, 1000)]
        rows = np.column_stack(
            (np.full(len(ages), 303), ages, np.arange(20, 20 + len(ages)))
        )

        tokens, prepared_ages = prepare_context(rows)

        np.testing.assert_array_equal(tokens[:6], [21, 1, 23, 22, 1, 24])
        np.testing.assert_array_equal(prepared_ages[:6], [0, 1, 1000, 1000, 1827.25, 3000])

    def test_prepare_context_uses_right_window_before_target_reservation(self) -> None:
        rows = np.column_stack(
            (
                np.full(60, 202),
                np.arange(60),
                np.arange(10, 70),
            )
        )

        tokens, ages = prepare_context(rows)

        self.assertEqual(len(tokens), BLOCK_SIZE)
        np.testing.assert_array_equal(tokens, np.arange(22, 70, dtype=np.int64))
        np.testing.assert_array_equal(ages, np.arange(11, 59, dtype=np.float32))

    def test_production_mask_combines_checkpoint_ignores_and_seen_events(self) -> None:
        tokens = np.asarray([2, 13, 13, 1, 14], dtype=np.int64)
        mask = production_mask(tokens, [0, *range(2, 13)])

        self.assertEqual(mask.shape, (VOCAB_SIZE,))
        self.assertTrue(mask[[0, 2, 12, 13, 14]].all())
        self.assertFalse(mask[[1, 15]].any())

    def test_stable_softmax_excludes_masked_events_and_reports_log_rate(self) -> None:
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        logits[11] = 1.0
        mask = np.ones(VOCAB_SIZE, dtype=bool)
        mask[[10, 11]] = False

        probabilities, log_rate = stable_softmax(logits, mask)

        expected = np.exp([0.0, 1.0])
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities[[10, 11]], expected, rtol=0, atol=1e-15)
        self.assertEqual(float(probabilities[mask].sum()), 0.0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=15)
        self.assertAlmostEqual(log_rate, float(np.log1p(np.e)), places=15)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _synthetic_groups():
        groups = []
        patient_id = 1
        for sex_token in (1, 2):
            for length in (4, 8, 12, 16):
                for _ in range(40):
                    rows = np.column_stack(
                        (
                            np.full(length, patient_id),
                            np.arange(length),
                            np.r_[sex_token, np.full(length - 1, 20)],
                        )
                    )
                    groups.append((patient_id, rows))
                    patient_id += 1
        return groups

    def test_stratified_selection_is_seeded_unique_and_balanced(self) -> None:
        groups = self._synthetic_groups()
        selected, boundaries = select_stratified(groups)
        repeated, repeated_boundaries = select_stratified(groups)

        self.assertEqual(selected, repeated)
        self.assertEqual(boundaries, repeated_boundaries)
        self.assertEqual(len(selected), 256)
        self.assertEqual(len(set(selected)), 256)

        lookup = {patient_id: rows for patient_id, rows in groups}
        strata = Counter()
        for patient_id in selected:
            rows = lookup[patient_id]
            sex = "female" if int(rows[0, 2]) == 1 else "male"
            quartile = int(np.searchsorted(boundaries, len(rows), side="left"))
            strata[(sex, quartile)] += 1
        self.assertEqual(
            strata,
            Counter({(sex, quartile): 32 for sex in ("female", "male") for quartile in range(4)}),
        )


class SavedResultIntegrityTests(unittest.TestCase):
    @staticmethod
    def _load(name: str):
        path = RESULTS / name
        if not path.exists():
            raise unittest.SkipTest(f"generated result not present: {path}")
        return json.loads(path.read_text())

    def test_python_conversion_diagnostic_covers_fixed_cohort(self) -> None:
        result = self._load("python_fidelity.json")
        self.assertEqual(result["cohort"]["complete_held_out_trajectories"], 7143)
        comparison = result["pytorch_vs_python_onnxruntime"]
        for scope in ("all_cohort_final", "stratified_all_position"):
            logits = comparison[f"{scope}_logits"]
            distributions = comparison[f"{scope}_distributions"]
            self.assertEqual(logits["atol"], ATOL)
            self.assertEqual(logits["rtol"], RTOL)
            self.assertEqual(logits["within_tolerance_fraction"], 1.0)
            self.assertLessEqual(distributions["maximum_absolute_probability_error"], 1e-4)
            self.assertLessEqual(distributions["total_variation_distance"]["maximum"], 1e-4)
            self.assertLessEqual(
                distributions["absolute_log_total_rate_error"]["maximum"], 1e-4
            )
            self.assertEqual(distributions["top1_agreement"], 1.0)

    def test_browser_results_meet_prespecified_fidelity_criteria(self) -> None:
        for backend in ("wasm", "webgpu"):
            with self.subTest(backend=backend):
                result = self._load(f"browser_{backend}.json")
                self.assertEqual(result["schema_version"], 1)
                self.assertEqual(result["backend"], backend)
                self.assertTrue(result["preprocessing"]["exact"])
                self.assertEqual(result["preprocessing"]["cases"], 257)
                self.assertEqual(
                    result["determinism"]["bitwise_equal_runs"],
                    result["determinism"]["repeated_runs"],
                )
                for scope in ("all_cohort_final", "stratified_all_position"):
                    logits = result["fidelity"][f"{scope}_logits"]
                    distributions = result["fidelity"][f"{scope}_distributions"]
                    self.assertGreater(logits["count"], 0)
                    self.assertEqual(logits["atol"], ATOL)
                    self.assertEqual(logits["rtol"], RTOL)
                    self.assertEqual(logits["within_tolerance_fraction"], 1)
                    self.assertEqual(distributions["mask_cell_mismatches"], 0)
                    self.assertEqual(distributions["top1_agreement"], 1)
                    self.assertTrue(
                        all(value == 0 for value in distributions["acceptance_failures"].values())
                    )

    def test_latency_and_memory_measurements_have_expected_repetitions(self) -> None:
        for backend in ("wasm", "webgpu"):
            with self.subTest(backend=backend):
                result = self._load(f"browser_{backend}.json")
                self.assertEqual(result["benchmark_configuration"]["cold_trials"], 10)
                for length in ("12", "24", "48"):
                    samples = result["latency"]["model_only"][length]
                    self.assertEqual(samples["n"], 200)
                    self.assertEqual(len(samples["samples_ms"]), 200)
                self.assertEqual(result["latency"]["full_trajectory"]["n"], 30)

                memory = self._load(f"browser_memory_{backend}.json")["process_memory"]
                self.assertGreater(memory["peak_rss_bytes"], memory["idle_rss_bytes"])
                self.assertEqual(
                    memory["incremental_peak_rss_bytes"],
                    memory["peak_rss_bytes"] - memory["idle_rss_bytes"],
                )


if __name__ == "__main__":
    unittest.main()
