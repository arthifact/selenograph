"""Fast, synthetic benchmark tests; extraction is mocked and reports are temporary.

Run: .venv/bin/python -B -m unittest poster.tests.test_benchmark_features
"""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from poster import benchmark_data
from poster import benchmark_features as benchmark

GROUPS = ("site-a", "site-b", "site-c")
SEED = 7
CAP = 3
TREES = 2


def arm_named(name):
    return next(arm for arm in benchmark.ARMS if arm.name == name)


def digest(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def synthetic_data():
    # Each site has a natural class-3 majority, but a capped draw is balanced.
    labels = np.tile(np.repeat([1, 2, 3], [4, 8, 12]), len(GROUPS)).astype(np.uint8)
    groups = np.repeat(GROUPS, 24)
    names = list(benchmark_data.FEATURE_NAMES)
    X = np.random.default_rng(91).normal(size=(len(labels), len(names))).astype(np.float32)
    support = np.tile([0.0, 0.89, benchmark_data.MIN_COVERAGE, 1.0], len(labels) // 4)
    X[support < benchmark_data.MIN_COVERAGE, names.index("best_resolution")] = np.nan
    X[::24, :] = np.nan  # Even completely missing feature rows remain test targets.
    return {
        "X": X,
        "feature_names": names,
        "y_saved": labels,
        "y_draft": (labels % 3 + 1).astype(np.uint8),
        "groups": groups,
        "sample_ids": np.array([f"{group}:cell-{i}" for i, group in enumerate(groups)]),
        "support_fraction": support,
        "metadata": {"contract_version": 1, "limitations": ["Synthetic fixtures only."]},
    }


def summary_runs(arms, snapshots=("saved",), seeds=(0, 17)):
    """Unequal site sizes expose accidental cell-weighted summary averaging."""
    runs = []
    for arm_index, arm in enumerate(arms):
        for snapshot in snapshots:
            for group, cells, score in zip(GROUPS, (5, 50, 500), (0.2, 0.5, 0.8)):
                for seed, offset in zip(seeds, (-0.1, 0.1)):
                    f1 = score + offset + arm_index * 0.05
                    runs.append({
                        "arm": arm.name, "snapshot": snapshot, "held_out": group,
                        "seed": seed, "test_cells": cells,
                        "fit_seconds": 0.01, "predict_seconds": 0.002,
                        "metrics": {"macro_f1": f1, "mean_iou": f1 / 2,
                                    "accuracy": f1, "balanced_accuracy": f1,
                                    "brier_multiclass": 1 - f1},
                    })
    return runs


class SyntheticTestCase(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.extract = self.enterContext(patch.object(
            benchmark_data, "load_data",
            side_effect=AssertionError("Tests must never extract real reference data")))

    def run_small(self, data=None, **options):
        config = {"seeds": (SEED,), "snapshots": ("saved",), "cap": CAP, "trees": TREES}
        config.update(options)
        return benchmark.run_benchmark(self.data if data is None else data, **config)


class SamplingAndFoldTests(SyntheticTestCase):
    def test_sampling_is_deterministic_balanced_capped_and_without_replacement(self):
        y, groups = self.data["y_saved"], self.data["groups"]
        eligible = groups != GROUPS[0]
        before = eligible.copy()
        selected = benchmark.training_rows(y, groups, eligible, SEED, CAP)
        repeated = benchmark.training_rows(y, groups, eligible, SEED, CAP)
        np.testing.assert_array_equal(selected, repeated)
        np.testing.assert_array_equal(eligible, before)
        np.testing.assert_array_equal(selected, np.unique(selected))
        self.assertTrue(eligible[selected].all())
        self.assertEqual(len(selected), 2 * len(benchmark.CLASSES) * CAP)
        for group in GROUPS[1:]:
            for code in benchmark.CLASSES:
                self.assertEqual(np.sum((groups[selected] == group) & (y[selected] == code)), CAP)
        different = benchmark.training_rows(y, groups, eligible, SEED + 1, CAP)
        self.assertFalse(np.array_equal(selected, different))
        uncapped = benchmark.training_rows(y, groups, eligible, SEED, 100)
        np.testing.assert_array_equal(uncapped, np.flatnonzero(eligible))
        self.assertEqual(benchmark.training_rows(y, groups, np.zeros_like(eligible), SEED, CAP).size, 0)

    def test_all_arms_share_test_rows_and_only_screen_training_support(self):
        original = deepcopy(self.data)
        classifier = benchmark.RandomForestClassifier
        models = []

        def tracked_classifier(**kwargs):
            model = classifier(**kwargs)
            model.fit = Mock(wraps=model.fit)
            model.predict = Mock(wraps=model.predict)
            models.append(model)
            return model

        with patch.object(benchmark, "RandomForestClassifier", side_effect=tracked_classifier):
            for group in GROUPS:
                test = np.flatnonzero(self.data["groups"] == group)
                for arm in benchmark.ARMS:
                    with self.subTest(group=group, arm=arm.name):
                        fold = benchmark.fit_fold(self.data, arm, "saved", group, SEED, CAP, TREES)
                        self.assertEqual(fold["test_rows_sha256"], digest(self.data["sample_ids"][test]))
                        self.assertEqual(fold["test_cells"], len(test))
                        self.assertEqual(fold["metrics"]["cells"], len(test))
                        self.assertEqual(fold["test_supported_cells"], 12)
                        self.assertEqual(fold["supported_metrics"]["cells"], 12)
                        self.assertEqual(fold["low_support_metrics"]["cells"], 12)
                        np.testing.assert_array_equal(
                            fold["metrics"]["confusion_matrix"],
                            np.array(fold["supported_metrics"]["confusion_matrix"])
                            + np.array(fold["low_support_metrics"]["confusion_matrix"]))
                        self.assertNotIn(group, fold["train_groups"])
                        self.assertEqual(set(fold["train_groups"]), set(GROUPS) - {group})
                        self.assertEqual(fold["training_cells_available"], 48)
                        eligible = self.data["groups"] != group
                        if arm.screen_training_support:
                            eligible &= self.data["support_fraction"] >= benchmark_data.MIN_COVERAGE
                        self.assertEqual(fold["training_cells_after_screen"], int(eligible.sum()))
                        if arm.name == "majority":
                            continue  # Natural-prevalence provenance is checked separately.
                        train = benchmark.training_rows(self.data["y_saved"], self.data["groups"],
                                                        eligible, SEED, CAP)
                        self.assertFalse(np.intersect1d(train, test).size)
                        self.assertEqual(fold["train_rows_sha256"], digest(self.data["sample_ids"][train]))
                        self.assertEqual(fold["training_cells_sampled"], len(train))
                        columns = [self.data["feature_names"].index(f) for f in arm.features]
                        model = models[-1]
                        model.fit.assert_called_once()
                        np.testing.assert_array_equal(model.fit.call_args.args[0], self.data["X"][np.ix_(train, columns)])
                        np.testing.assert_array_equal(model.fit.call_args.args[1], self.data["y_saved"][train])
                        np.testing.assert_array_equal(model.predict.call_args.args[0], self.data["X"][np.ix_(test, columns)])
                        self.assertTrue(np.isnan(model.predict.call_args.args[0][0]).all())
        for key in ("X", "y_saved", "y_draft", "groups", "sample_ids", "support_fraction"):
            np.testing.assert_array_equal(self.data[key], original[key])
        self.assertEqual(self.data["metadata"], original["metadata"])
        self.extract.assert_not_called()

    def test_repeated_fit_has_identical_sampling_and_scores(self):
        args = (self.data, arm_named("terrain_full_nac_quality"), "saved", GROUPS[0], SEED, CAP, TREES)
        first, second = benchmark.fit_fold(*args), benchmark.fit_fold(*args)
        for key in ("train_rows_sha256", "test_rows_sha256", "training_class_counts",
                    "metrics", "supported_metrics", "low_support_metrics"):
            self.assertEqual(first[key], second[key], key)

    def test_majority_uses_natural_prevalence_and_all_eligible_row_provenance(self):
        for snapshot, majority in (("saved", 3), ("draft", 1)):
            with self.subTest(snapshot=snapshot):
                train = np.flatnonzero(self.data["groups"] != GROUPS[0])
                test = np.flatnonzero(self.data["groups"] == GROUPS[0])
                y = self.data[f"y_{snapshot}"]
                with patch.object(benchmark, "RandomForestClassifier") as classifier:
                    fold = benchmark.fit_fold(self.data, arm_named("majority"), snapshot,
                                              GROUPS[0], SEED, 1, TREES)
                classifier.assert_not_called()
                expected = np.zeros((3, 3), dtype=int)
                expected[:, majority - 1] = [np.sum(y[test] == c) for c in benchmark.CLASSES]
                self.assertEqual(fold["metrics"]["confusion_matrix"], expected.tolist())
                self.assertAlmostEqual(fold["metrics"]["accuracy"], 0.5)
                self.assertAlmostEqual(fold["metrics"]["brier_multiclass"], 1.0)
                self.assertEqual(fold["training_cells_sampled"], len(train))
                self.assertEqual(fold["train_rows_sha256"], digest(self.data["sample_ids"][train]))
                self.assertEqual(fold["training_class_counts"],
                                 {str(c): int(np.sum(y[train] == c)) for c in benchmark.CLASSES})

    def test_probability_columns_keep_class_order_when_training_class_is_absent(self):
        train = self.data["groups"] != GROUPS[0]
        self.data["y_saved"][train & (self.data["y_saved"] == 2)] = 1
        with patch.object(benchmark, "metrics", wraps=benchmark.metrics) as score:
            fold = benchmark.fit_fold(self.data, arm_named("terrain_full"), "saved",
                                      GROUPS[0], SEED, CAP, TREES)
        probability = score.call_args_list[0].args[2]
        self.assertEqual(probability.shape, (24, 3))
        np.testing.assert_array_equal(probability[:, 1], np.zeros(24))
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(fold["metrics"]["per_class"]["2"]["support"], 8)
        truth = np.eye(3)[self.data["y_saved"][:24] - 1]
        self.assertAlmostEqual(fold["metrics"]["brier_multiclass"],
                               float(np.mean(np.sum((truth - probability) ** 2, axis=1))))

    def test_screening_with_insufficient_training_classes_fails_explicitly(self):
        self.data["support_fraction"][:] = 0
        with self.assertRaisesRegex(ValueError, "Insufficient training classes"):
            benchmark.fit_fold(self.data, arm_named("terrain_full_nac_screened"),
                               "saved", GROUPS[0], SEED, CAP, TREES)


class MetricTests(unittest.TestCase):
    def test_known_confusion_f1_iou_and_multiclass_brier(self):
        y = np.array([1, 1, 2, 2, 3, 3])
        pred = np.array([1, 2, 2, 3, 3, 3])
        probability = np.array([[0.75, 0.25, 0], [0, 1, 0], [0, 0.5, 0.5],
                                [0, 0, 1], [0, 0, 1], [0.25, 0.25, 0.5]])
        result = benchmark.metrics(y, pred, probability)
        assert result is not None
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [0, 0, 2]])
        self.assertEqual(result["cells"], 6)
        self.assertEqual(result["absent_target_classes"], [])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 59 / 90)
        self.assertAlmostEqual(result["mean_iou"], 0.5)
        self.assertAlmostEqual(result["brier_multiclass"], 5 / 6)
        for code, precision, recall, f1, iou in (
                (1, 1, 0.5, 2 / 3, 0.5), (2, 0.5, 0.5, 0.5, 1 / 3),
                (3, 2 / 3, 1, 0.8, 2 / 3)):
            row = result["per_class"][str(code)]
            self.assertEqual(row["support"], 2)
            for name, expected in (("precision", precision), ("recall", recall), ("f1", f1), ("iou", iou)):
                self.assertAlmostEqual(row[name], expected)

    def test_absent_classes_still_count_in_macro_scores_and_empty_stratum_is_none(self):
        result = benchmark.metrics(np.array([1, 1]), np.array([1, 1]))
        assert result is not None
        self.assertEqual(result["absent_target_classes"], [2, 3])
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertAlmostEqual(result["mean_iou"], 1 / 3)
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertNotIn("brier_multiclass", result)
        for code in ("2", "3"):
            self.assertTrue(all(value == 0 for value in result["per_class"][code].values()))
        self.assertIsNone(benchmark.metrics(np.array([], dtype=int), np.array([], dtype=int)))

    def test_invalid_target_or_prediction_class_is_rejected(self):
        for y, pred in (([0], [1]), ([4], [1]), ([1], [0]), ([1], [4])):
            with self.subTest(y=y, pred=pred), self.assertRaises(ValueError):
                benchmark.metrics(np.array(y), np.array(pred))


class BenchmarkRunTests(SyntheticTestCase):
    def test_saved_and_draft_evaluate_the_same_cells_with_their_own_targets(self):
        arms = (arm_named("terrain_full"), arm_named("terrain_full_nac_screened"))
        report = self.run_small(arms=arms, snapshots=("saved", "draft"))
        self.assertTrue(report["complete"])
        self.assertEqual(report["sample_count"], len(self.data["sample_ids"]))
        self.assertEqual(report["sample_ids_sha256"], digest(self.data["sample_ids"]))
        self.assertEqual(len(report["runs"]), len(arms) * len(GROUPS) * 2)
        index = {(r["arm"], r["snapshot"], r["held_out"]): r for r in report["runs"]}
        for arm in arms:
            for group in GROUPS:
                saved = index[arm.name, "saved", group]
                draft = index[arm.name, "draft", group]
                self.assertEqual(saved["test_rows_sha256"], draft["test_rows_sha256"])
                test = self.data["groups"] == group
                for snapshot, fold in (("saved", saved), ("draft", draft)):
                    counts = [int(np.sum(self.data[f"y_{snapshot}"][test] == c)) for c in benchmark.CLASSES]
                    np.testing.assert_array_equal(np.sum(fold["metrics"]["confusion_matrix"], axis=1), counts)
                self.assertNotEqual(saved["metrics"]["per_class"]["1"]["support"],
                                    draft["metrics"]["per_class"]["1"]["support"])
        for snapshot in ("saved", "draft"):
            self.assertEqual(set(report["summary"][snapshot]), {arm.name for arm in arms})
        self.assertFalse(report["recommendation"]["app_recipe_changed"])

    def test_absent_or_all_nan_optional_features_are_explicit_skips(self):
        optional = set(benchmark_data.RADAR + benchmark_data.LOLA)
        arms = tuple(arm for arm in benchmark.ARMS
                     if arm.name == "terrain_full" or optional.intersection(arm.features))
        skipped = {arm.name for arm in arms if optional.intersection(arm.features)}
        for mode in ("absent", "all_nan"):
            with self.subTest(mode=mode):
                data = synthetic_data()
                if mode == "absent":
                    columns = [i for i, name in enumerate(data["feature_names"]) if name not in optional]
                    data["X"] = data["X"][:, columns]
                    data["feature_names"] = [data["feature_names"][i] for i in columns]
                else:
                    for name in optional:
                        data["X"][:, data["feature_names"].index(name)] = np.nan
                report = self.run_small(data, arms=arms)
                self.assertTrue(report["complete"])
                self.assertEqual(set(report["skipped_arms"]), skipped)
                self.assertEqual({r["arm"] for r in report["runs"]}, {"terrain_full"})
                self.assertEqual(set(report["summary"]["saved"]), {"terrain_full"})
                for arm in arms:
                    if arm.name in skipped:
                        reason = report["skipped_arms"][arm.name]
                        self.assertIn("Unavailable features", reason)
                        for name in optional.intersection(arm.features):
                            self.assertIn(name, reason)

    def test_custom_arm_is_included_in_run_summary(self):
        custom = benchmark.Arm("synthetic_slope", ("slope",))
        report = self.run_small(arms=(custom,))
        self.assertTrue(report["complete"])
        self.assertEqual(set(report["summary"]["saved"]), {custom.name})
        self.assertEqual(report["summary"]["saved"][custom.name]["features"], ["slope"])
        self.assertEqual(report["recommendation"]["status"], "incomplete_no_recommendation")

    def test_expired_deadline_never_fits_or_recommends(self):
        with patch.object(benchmark.time, "monotonic", return_value=10), \
                patch.object(benchmark, "fit_fold") as fit:
            report = self.run_small(arms=(arm_named("terrain_full"),), deadline=9)
        fit.assert_not_called()
        self.assertFalse(report["complete"])
        self.assertEqual(report["runs"], [])
        self.assertTrue(report["stopped_reason"])
        self.assertEqual(report["summary"]["saved"], {})
        self.assertEqual(report["recommendation"]["status"], "incomplete_no_recommendation")
        self.assertFalse(report["recommendation"]["app_recipe_changed"])
        self.assertNotIn("highest_mean_macro_f1", report["recommendation"])

    def test_partial_runs_never_recommend_even_with_a_complete_baseline(self):
        arms = (arm_named("terrain_full"), arm_named("terrain_minimal"))
        fit_fold = benchmark.fit_fold
        for exception in (TimeoutError, KeyboardInterrupt):
            with self.subTest(exception=exception.__name__):
                calls = 0

                def interrupted_fit(*args, exception=exception, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == len(GROUPS) + 2:
                        raise exception("Synthetic interruption")
                    return fit_fold(*args, **kwargs)

                with patch.object(benchmark, "fit_fold", side_effect=interrupted_fit):
                    report = self.run_small(arms=arms)
                self.assertFalse(report["complete"])
                self.assertEqual(len(report["runs"]), len(GROUPS) + 1)
                self.assertEqual(set(report["summary"]["saved"]), {"terrain_full"})
                self.assertEqual(report["recommendation"]["status"], "incomplete_no_recommendation")
                self.assertFalse(report["recommendation"]["app_recipe_changed"])
                self.assertNotIn("consistent_candidates", report["recommendation"])
                self.assertIn("Synthetic interruption", report["stopped_reason"])

    def test_only_two_report_files_are_written_and_both_are_readable(self):
        with tempfile.TemporaryDirectory(prefix="selenograph-benchmark-report-") as temporary:
            root = Path(temporary)
            output = root / "output" / "experiments" / "synthetic"
            report = self.run_small(arms=(arm_named("majority"), arm_named("terrain_full")),
                                    snapshots=("saved", "draft"), output=output)
            benchmark.save_report(output, report)  # Repeated checkpoint writes leave no temporary files.
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["results.json", "summary.md"])
            self.assertEqual({p for p in root.rglob("*") if p.is_file()},
                             {output / "results.json", output / "summary.md"})
            decoded = json.loads((output / "results.json").read_text(encoding="utf-8"),
                                 parse_constant=lambda token: self.fail(f"Nonfinite JSON value: {token}"))
            self.assertEqual(decoded, json.loads(json.dumps(report, allow_nan=False)))
            text = (output / "summary.md").read_text(encoding="utf-8")
            for expected in ("Private development benchmark", "Saved-preferred", "Draft-preferred",
                             "terrain_full", "majority", report["recommendation"]["status"]):
                self.assertIn(expected, text)
            self.assertNotIn("sample_ids", decoded)
            self.assertNotIn("X", decoded)
            for fold in decoded["runs"]:
                self.assertNotIn("predictions", fold)
                self.assertNotIn("model", fold)
        self.assertFalse(root.exists())
        self.extract.assert_not_called()


class SummaryTests(unittest.TestCase):
    def test_custom_arms_site_balancing_seed_ranges_and_paired_deltas(self):
        arms = (arm_named("terrain_full"), benchmark.Arm("synthetic_candidate", ("slope",)))
        summary = benchmark.summarize(summary_runs(arms), ("saved",), GROUPS, (0, 17), arms=arms)["saved"]
        self.assertEqual(set(summary), {arm.name for arm in arms})
        baseline, candidate = summary["terrain_full"], summary["synthetic_candidate"]
        self.assertAlmostEqual(baseline["site_balanced"]["macro_f1"], 0.5)
        self.assertAlmostEqual(candidate["site_balanced"]["macro_f1"], 0.55)
        self.assertAlmostEqual(candidate["site_balanced"]["mean_iou"], 0.275)
        self.assertAlmostEqual(candidate["site_balanced"]["brier_multiclass"], 0.45)
        np.testing.assert_allclose(baseline["seed_mean_macro_f1_range"], [0.4, 0.6])
        self.assertEqual(candidate["feature_count"], 1)
        self.assertAlmostEqual(candidate["delta_macro_f1_vs_full"], 0.05)
        self.assertEqual(candidate["groups_improved"], 3)
        for group, cells, f1 in zip(GROUPS, (5, 50, 500), (0.2, 0.5, 0.8)):
            self.assertEqual(baseline["by_group"][group]["cells"], cells)
            self.assertAlmostEqual(baseline["by_group"][group]["macro_f1"], f1)
            np.testing.assert_allclose(baseline["by_group"][group]["seed_macro_f1_range"], [f1 - 0.1, f1 + 0.1])
            self.assertAlmostEqual(candidate["delta_by_group"][group], 0.05)

    def test_incremental_comparisons_use_matching_parents_in_each_snapshot(self):
        comparisons = (
            ("terrain_full_nac_radar", "terrain_full_nac", (0.12, -0.03, 0.0), (0.03, -0.01, 0.01)),
            ("terrain_full_nac_quality", "terrain_full_nac", (-0.06, 0.0, -0.03), (-0.02, 0.0, -0.01)),
            ("terrain_full_nac_screened", "terrain_full_nac", (-0.03, 0.0, 0.03), (0.0, 0.0, 0.0)),
            ("terrain_local_global_sky", "terrain_local", (0.03, 0.06, 0.0), (0.01, 0.02, 0.0)),
            ("terrain_local_local_sky", "terrain_local", (-0.03, 0.0, -0.06), (-0.01, 0.0, -0.02)),
        )
        parents = {"terrain_local": 0.3, "terrain_full_nac": 0.6, "terrain_full": 0.8}
        # Put children before parents; neither arm order nor the full-terrain baseline
        # should determine the incremental comparison's reference recipe.
        arms = tuple(arm_named(name) for name in [*(row[0] for row in comparisons), *parents])
        runs = summary_runs(arms, ("saved", "draft"))
        expected = {name: (parent, f1, iou) for name, parent, f1, iou in comparisons}
        for run in runs:
            group_index = GROUPS.index(run["held_out"])
            offset = 0.02 * group_index + (-0.01 if run["seed"] == 0 else 0.01)
            sign = 1 if run["snapshot"] == "saved" else -1
            if run["arm"] in expected:
                parent, f1_delta, iou_delta = expected[run["arm"]]
                f1 = parents[parent] + offset + sign * f1_delta[group_index]
                iou = parents[parent] / 2 + offset + sign * iou_delta[group_index]
            else:
                f1 = parents[run["arm"]] + offset
                iou = parents[run["arm"]] / 2 + offset
            run["metrics"] = {"macro_f1": f1, "mean_iou": iou, "accuracy": f1,
                              "balanced_accuracy": f1, "brier_multiclass": 1 - f1}
        summary = benchmark.summarize(runs, ("saved", "draft"), GROUPS, (0, 17), arms=arms)
        for snapshot, sign in (("saved", 1), ("draft", -1)):
            for name, parent, f1_delta, iou_delta in comparisons:
                with self.subTest(snapshot=snapshot, arm=name):
                    row = summary[snapshot][name]
                    comparison = row["incremental_comparison"]
                    self.assertEqual(comparison["parent"], parent)
                    self.assertAlmostEqual(comparison["delta_macro_f1"], sign * sum(f1_delta) / 3)
                    self.assertAlmostEqual(comparison["delta_mean_iou"], sign * sum(iou_delta) / 3)
                    self.assertEqual(set(comparison["delta_by_group"]), set(GROUPS))
                    for group, delta in zip(GROUPS, f1_delta):
                        self.assertAlmostEqual(comparison["delta_by_group"][group], sign * delta)
                    self.assertNotAlmostEqual(comparison["delta_macro_f1"], row["delta_macro_f1_vs_full"])

    def test_incremental_comparison_requires_a_complete_parent_in_the_same_snapshot(self):
        pairs = (
            ("terrain_full_nac_radar", "terrain_full_nac"),
            ("terrain_full_nac_quality", "terrain_full_nac"),
            ("terrain_full_nac_screened", "terrain_full_nac"),
            ("terrain_local_global_sky", "terrain_local"),
            ("terrain_local_local_sky", "terrain_local"),
        )
        for child, parent in pairs:
            for missing in ("all_folds", "one_seed_fold"):
                with self.subTest(arm=child, missing=missing):
                    arms = tuple(arm_named(name) for name in ("terrain_full", parent, child))
                    runs = summary_runs(arms, ("saved", "draft"))
                    runs = [run for run in runs if not (
                        run["arm"] == parent and run["snapshot"] == "saved"
                        and (missing == "all_folds" or
                             (run["held_out"] == GROUPS[-1] and run["seed"] == 17)))]
                    summary = benchmark.summarize(runs, ("saved", "draft"), GROUPS, (0, 17), arms=arms)
                    self.assertNotIn(parent, summary["saved"])
                    self.assertIn("terrain_full", summary["saved"])
                    self.assertNotIn("incremental_comparison", summary["saved"][child])
                    self.assertEqual(summary["draft"][child]["incremental_comparison"]["parent"], parent)
                    self.assertAlmostEqual(summary["draft"][child]["incremental_comparison"]["delta_macro_f1"], 0.05)

    def test_incremental_comparison_does_not_require_full_terrain_baseline(self):
        for parent, child in (("terrain_full_nac", "terrain_full_nac_radar"),
                              ("terrain_local", "terrain_local_local_sky")):
            with self.subTest(parent=parent, arm=child):
                arms = (arm_named(parent), arm_named(child))
                summary = benchmark.summarize(summary_runs(arms), ("saved",), GROUPS, (0, 17), arms=arms)["saved"]
                self.assertNotIn("terrain_full", summary)
                self.assertNotIn("delta_macro_f1_vs_full", summary[child])
                comparison = summary[child]["incremental_comparison"]
                self.assertEqual(comparison["parent"], parent)
                self.assertAlmostEqual(comparison["delta_macro_f1"], 0.05)
                self.assertAlmostEqual(comparison["delta_mean_iou"], 0.025)
                for group in GROUPS:
                    self.assertAlmostEqual(comparison["delta_by_group"][group], 0.05)

    def test_default_summary_excludes_an_arm_missing_a_seed_fold(self):
        arms = (arm_named("terrain_full"), arm_named("terrain_minimal"))
        runs = summary_runs(arms)
        summary = benchmark.summarize(runs[:-1], ("saved",), GROUPS, (0, 17))
        self.assertEqual(set(summary["saved"]), {"terrain_full"})
        decision = benchmark.recommendation(summary, ("saved",), complete=False)
        self.assertEqual(decision["status"], "incomplete_no_recommendation")
        self.assertFalse(decision["app_recipe_changed"])

    def test_recommendation_requires_consistency_in_both_snapshots_and_every_group(self):
        arms = (arm_named("terrain_full"), arm_named("terrain_minimal"))
        summary = benchmark.summarize(summary_runs(arms, ("saved", "draft")),
                                      ("saved", "draft"), GROUPS, (0, 17))
        decision = benchmark.recommendation(summary, ("saved", "draft"), complete=True)
        self.assertEqual(decision["consistent_candidates"], ["terrain_minimal"])
        self.assertEqual(decision["status"], "development_only_not_promoted")
        self.assertFalse(decision["app_recipe_changed"])
        for failure in ("snapshot_gain", "group_loss", "iou_loss"):
            with self.subTest(failure=failure):
                changed = deepcopy(summary)
                draft = changed["draft"]["terrain_minimal"]
                if failure == "snapshot_gain":
                    draft["delta_macro_f1_vs_full"] = 0.005
                elif failure == "group_loss":
                    draft["delta_by_group"][GROUPS[0]] = -0.01
                else:
                    draft["site_balanced"]["mean_iou"] = changed["draft"]["terrain_full"]["site_balanced"]["mean_iou"] - 0.01
                decision = benchmark.recommendation(changed, ("saved", "draft"), complete=True)
                self.assertEqual(decision["consistent_candidates"], [])


class ValidationTests(SyntheticTestCase):
    def test_nan_and_low_support_targets_are_valid_and_unique_cells_are_required(self):
        benchmark.validate_data(self.data)
        self.data["sample_ids"][1] = self.data["sample_ids"][0]
        with self.assertRaisesRegex(ValueError, "Repeated original cells"):
            benchmark.validate_data(self.data)

    def test_invalid_matrix_features_groups_and_support(self):
        invalid = [
            ("X", self.data["X"].ravel()),
            ("X", self.data["X"][:-1]),
            ("X", np.full_like(self.data["X"], np.inf)),
            ("feature_names", [self.data["feature_names"][0]] * len(self.data["feature_names"])),
            ("groups", np.where(self.data["groups"] == GROUPS[2], GROUPS[0], self.data["groups"])),
        ]
        for key in ("groups", "y_saved", "y_draft", "support_fraction"):
            invalid.append((key, self.data[key][:-1]))
        for value in (-0.01, 1.01, np.nan, np.inf):
            support = self.data["support_fraction"].copy()
            support[0] = value
            invalid.append(("support_fraction", support))
        for key, value in invalid:
            with self.subTest(key=key, shape=np.shape(value)), self.assertRaises(ValueError):
                benchmark.validate_data({**self.data, key: value})
        empty = deepcopy(self.data)
        for key in ("X", "sample_ids", "groups", "y_saved", "y_draft", "support_fraction"):
            empty[key] = empty[key][:0]
        with self.assertRaises(ValueError):
            benchmark.validate_data(empty)

    def test_each_snapshot_rejects_unknown_or_nodata_targets(self):
        for key in ("y_saved", "y_draft"):
            for value in (0, 4, 255, np.nan):
                with self.subTest(snapshot=key, value=value):
                    targets = self.data[key].astype(float)
                    targets[0] = value
                    with self.assertRaises(ValueError):
                        benchmark.validate_data({**self.data, key: targets})

    def test_invalid_run_options_fail_before_any_fitting_or_report_writes(self):
        cases = ({"seeds": ()}, {"seeds": (1, 1)}, {"seeds": (-1,)},
                 {"snapshots": ()}, {"snapshots": ("saved", "saved")},
                 {"snapshots": ("unknown",)}, {"cap": 0}, {"cap": -1},
                 {"trees": 0}, {"trees": -1})
        for options in cases:
            with self.subTest(options=options), patch.object(benchmark, "fit_fold") as fit, \
                    patch.object(benchmark, "save_report") as save:
                with self.assertRaises(ValueError):
                    self.run_small(arms=(arm_named("terrain_full"),), **options)
                fit.assert_not_called()
                save.assert_not_called()


class CliTests(SyntheticTestCase):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="selenograph-benchmark-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "poster" / "experiments" / "synthetic"
        locations = {
            "PROJECT_ROOT": self.root, "DATA_DIR": self.root / "data",
            "DEM_DIR": self.root / "data" / "processed_data",
            "MODEL_DIR": self.root / "models", "OUTPUT_DIR": self.root / "output",
            "REFERENCE_DIR": self.root / "data" / "reference_data",
            "PROFESSOR_MAPS_DIR": self.root / "data" / "reference_data" / "professor_geologic_maps",
        }
        for name, value in locations.items():
            self.enterContext(patch.object(benchmark.paths, name, value))
        self.enterContext(patch.object(benchmark.storage, "ROOT", self.root / "poster"))
        self.extract.side_effect = None
        self.extract.return_value = self.data

    def invoke(self, options):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = benchmark.main(options)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue() + stderr.getvalue()

    def assert_rejected_before_extraction(self, options):
        self.extract.reset_mock()
        with patch.object(benchmark, "run_benchmark", return_value={"complete": True, "runs": []}) as run:
            code, text = self.invoke(options)
        self.extract.assert_not_called()
        run.assert_not_called()
        self.assertNotEqual(code, 0, text)
        self.assertTrue(text.strip())

    def test_invalid_options_are_rejected_before_extraction(self):
        cases = (["--trees", "0"], ["--trees", "-1"],
                 ["--cap-per-group-class", "0"], ["--cap-per-group-class", "-1"],
                 ["--seeds", "-1"], ["--seeds", "7", "7"], ["--seeds"],
                 ["--snapshots", "saved", "saved"], ["--snapshots", "unknown"],
                 ["--max-seconds", "0"], ["--max-seconds", "-1"],
                 ["--max-seconds", "nan"], ["--max-seconds", "inf"])
        for options in cases:
            with self.subTest(options=options):
                self.assert_rejected_before_extraction(["--output", str(self.output), *options])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_output_must_resolve_under_experiments(self):
        targets = (self.root, self.root / "data", self.root / "models", self.root / "app",
                   self.root / "tests", self.root / "outside", self.root / "output" / "maps",
                   self.root / "output" / "drafts", self.root / "output" / "experiments-other" / "run",
                   self.root / "poster" / "experiments" / ".." / "maps" / "run")
        for target in targets:
            with self.subTest(target=target):
                self.assert_rejected_before_extraction(["--output", str(target)])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_symlink_cannot_escape_experiments(self):
        outside = self.root / "outside"
        outside.mkdir()
        experiments = self.output.parent
        experiments.mkdir(parents=True)
        (experiments / "escape").symlink_to(outside, target_is_directory=True)
        self.assert_rejected_before_extraction(["--output", str(experiments / "escape" / "run")])
        self.assertEqual(list(outside.iterdir()), [])

    def test_nonempty_directory_is_rejected_without_overwriting(self):
        self.output.mkdir(parents=True)
        existing = self.output / "results.json"
        existing.write_text('{"existing": true}\n', encoding="utf-8")
        self.assert_rejected_before_extraction(["--output", str(self.output)])
        self.assertEqual(existing.read_text(encoding="utf-8"), '{"existing": true}\n')
        self.assertEqual(list(self.output.iterdir()), [existing])

    def test_valid_new_or_empty_directory_passes_small_options_to_runner(self):
        for exists in (False, True):
            with self.subTest(existing_empty_directory=exists):
                if exists:
                    self.output.mkdir(parents=True)
                self.extract.reset_mock()
                with patch.object(benchmark, "run_benchmark", return_value={"complete": True, "runs": []}) as run:
                    code, text = self.invoke([
                        "--output", str(self.output), "--trees", str(TREES),
                        "--cap-per-group-class", str(CAP), "--seeds", str(SEED),
                        "--snapshots", "saved", "--max-seconds", "5"])
                self.assertEqual(code, 0, text)
                self.extract.assert_called_once()
                self.assertEqual(self.extract.call_args.args,
                                 (benchmark.paths.DEM_DIR, benchmark.paths.PROFESSOR_MAPS_DIR,
                                  benchmark.paths.DATA_DIR / "raw_data"))
                run.assert_called_once()
                self.assertIs(run.call_args.args[0], self.data)
                options = run.call_args.kwargs
                self.assertEqual(options["output"], self.output)
                self.assertEqual(options["trees"], TREES)
                self.assertEqual(options["cap"], CAP)
                self.assertEqual(options["seeds"], (SEED,))
                self.assertEqual(options["snapshots"], ("saved",))
                self.assertTrue(np.isfinite(options["deadline"]))
                self.assertTrue(callable(options["progress"]))
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
