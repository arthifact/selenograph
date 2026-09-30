"""Processed-catalog evaluation checks; all estimator fitting is mocked."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from app import assistant_model as assistant
from app import core, evaluation, paths
from tests.raster_fixtures import TRANSFORM, raster, sha, write_json


class PixelDouble:
    def predict(self, rows):
        return np.clip(np.rint(rows[:, 0]), 1, 3).astype(np.uint8)

    def predict_proba(self, rows):
        result = np.full((len(rows), 3), 0.05, np.float32)
        result[np.arange(len(rows)), self.predict(rows) - 1] = 0.9
        return result


def sample(group, labels=(1, 1, 2), **extra) -> dict[str, Any]:
    return dict(X=np.ones((len(labels), 6), np.float32), y=np.array(labels, np.uint8),
                w=np.ones(len(labels), np.float32), site_id=group, **extra)


def meta(name, group, path, *, available=True, verified=True):
    return {"name": name, "tile_id": name, "map_id": name, "site_id": group, "group": group,
            "geography_aliases": [], "source_kind": "processed_map", "section": "survey",
            "path": str(path), "snapshot": "saved", "verified": verified, "restricted": False,
            "painted": available, "available": available,
            "reason": None if available else "No usable cell labels",
            "label_source": "synthetic cell painting" if available else None,
            "label_status": "training-painting agreement; not independent truth"}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="selenograph-evaluation-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.processed = self.root / "processed_data"
        self.sources = self.processed / "survey"
        self.models, self.output, self.poster = [self.root / p for p in ("models", "output", "poster")]
        self.registry, self.paintings = {}, {}
        for index, (name, group) in enumerate((("a", "alpha"), ("b", "beta"), ("c", "gamma"), ("unlabelled", "alpha"))):
            path = self.sources / f"{name}.tif"
            values = np.full((8, 10), 3, np.float32)
            values[:, :4] = 1
            raster(path, values, transform=TRANSFORM @ rasterio.Affine.translation(index * 10000, 0))
            self.registry[name] = meta(name, group, path, available=name != "unlabelled")
            cells = np.ones((120, 120), np.uint8)
            cells[:, 48:] = 3
            if name == "unlabelled":
                cells[:] = 0
            self.paintings[name] = cells
        self.samples = {name: sample(group) for name, group in (("a", "alpha"), ("b", "beta"), ("c", "gamma"))}
        self.patch(core, "MODEL", "rf")
        self.patch(core, "DEM_DIR", self.processed)
        self.patch(assistant, "MODEL_DIR", self.models)
        self.patch(core, "map_site", side_effect=lambda n: self.registry.get(n, {}).get("group"))
        self.patch(core, "map_record", side_effect=lambda n: {"site": self.registry[n]["site_id"], "group": self.registry[n]["group"]} if n in self.registry else None)
        self.patch(core, "dem_path", side_effect=lambda n: self.registry.get(n, {}).get("path") or str(self.root / "absent.tif"))
        self.patch(core, "meta_get", side_effect=lambda n: {"verified": self.registry.get(n, {"verified": True})["verified"]})
        self.patch(core, "held_out_maps", side_effect=lambda names=None: set(names or ()))
        self.patch(evaluation, "catalog", side_effect=lambda: copy.deepcopy(self.registry))
        self.patch(core, "build", side_effect=self.build)
        self.patch(core, "_painting_source", side_effect=lambda n, final=None: (self.paintings[n].copy(), True, False))
        self.patch(core, "load_confidence", side_effect=lambda n: np.where(self.paintings[n] > 0, core.SURE, 0))
        self.patch(core, "load_accepted", side_effect=lambda n: np.zeros((120, 120), bool))
        self.patch(assistant, "retained_samples", side_effect=AssertionError("No source-specific retention path"))
        self.patch(core.backend(), "fit_rows", side_effect=lambda *args: PixelDouble())
        self.bundle = {"schema": assistant.schema(), "samples": self.samples,
                       "model": PixelDouble(), "revision": "active-revision"}

    def patch(self, module, key, *args, **kwargs):
        context = patch.object(module, key, *args, **kwargs)
        value = context.start()
        self.addCleanup(context.stop)
        return value

    @staticmethod
    def build(path):
        with rasterio.open(path) as src:
            values = src.read(1)
            profile = {"height": src.height, "width": src.width, "transform": src.transform, "crs": src.crs}
        return np.stack([values + i for i in range(6)]), list(core.BASE_FEATS), np.ones(values.shape, np.float32), profile, 5.0

    def run_selected(self, names):
        with patch.object(joblib, "dump", side_effect=AssertionError("No fold model artifacts")), \
                patch.object(np, "savez_compressed", side_effect=AssertionError("No OOF arrays")), \
                patch.object(np, "savez", side_effect=AssertionError("No OOF arrays")):
            return evaluation.run(self.bundle, names, output_root=self.output,
                                  model_root=self.models, poster_root=self.poster)

    def test_floor_bincount_handles_edge_rasters_and_zero_pixel_cells(self):
        for shape in ((1, 1), (8, 10), (119, 127), (240, 241)):
            with self.subTest(shape=shape):
                cells = (np.arange(14400).reshape(120, 120) % 3 + 1).astype(np.uint8)
                pixels = core.per_pixel(cells, shape)
                area = evaluation.cell_sum(np.ones(shape, bool))
                result = evaluation.cell_majority(pixels)
                np.testing.assert_array_equal(result[area > 0], cells[area > 0])
                self.assertTrue((result[area == 0] == 255).all())
                self.assertEqual(area.sum(), np.prod(shape))
        self.assertEqual(evaluation.cell_majority(np.full((2, 2), 255, np.uint8))[0, 0], 255)

    def test_one_selected_location_is_valid_if_other_geography_trains(self):
        with patch.object(assistant, "fit", side_effect=AssertionError("planning must not fit")), \
                patch.object(core, "build", side_effect=AssertionError("planning must not build terrain")), \
                patch.object(core, "read", side_effect=AssertionError("planning must not read full rasters")):
            planned = evaluation.plan(self.bundle, ["a"])
        self.assertEqual(planned[0]["train_keys"], ["b", "c"])
        self.assertEqual(planned[0]["excluded_keys"], ["a"])
        self.assertIsNone(planned[0]["reason"])
        _, report = self.run_selected(["a"])
        self.assertEqual(len(report["validation"]["folds"]), 1)
        self.assertEqual(report["validation"]["method"], "independent selected-map evaluation")
        self.assertEqual(report["validation"]["aggregate"]["n"], 80)

    def test_sole_training_map_refused_but_one_remaining_class_can_score(self):
        self.bundle["samples"] = {"a": self.samples["a"]}
        self.assertIn("No verified training samples remain", evaluation.plan(self.bundle, ["a"])[0]["reason"])
        with self.assertRaisesRegex(ValueError, "No evaluable folds"):
            self.run_selected(["a"])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.models.exists())
        self.assertFalse(self.poster.exists())
        self.bundle["samples"] = {"a": self.samples["a"], "b": sample("beta", (2, 2))}
        self.assertIsNone(evaluation.plan(self.bundle, ["a"])[0]["reason"])
        with patch.object(assistant, "fit", side_effect=AssertionError("Single class needs no classifier fit")):
            _, report = self.run_selected(["a"])
        fold = report["validation"]["folds"][0]
        self.assertEqual(fold["model_kind"], "single-class")
        self.assertEqual(fold["majority_class"], 2)
        self.assertEqual(fold["train_tiles"], ["b"])
        self.assertGreater(fold["metrics"]["n"], 0)

    def test_arbitrary_rotation_model_and_sources_unchanged(self):
        self.models.mkdir()
        active = self.models / "rf.joblib"
        active.write_bytes(b"active model must not be loaded or rewritten")
        before = active.read_bytes(), active.stat().st_mtime_ns
        sources = {p: (sha(p), p.stat().st_mtime_ns) for p in self.sources.glob("*.tif")}
        samples = {k: s["X"].copy() for k, s in self.samples.items()}
        bundle_hash = joblib.hash(self.bundle)
        active_model = self.bundle["model"]
        with patch.object(assistant, "fit", wraps=assistant.fit) as fit, \
                patch.object(assistant, "save", side_effect=AssertionError("No active model writes")), \
                patch.object(assistant, "load", side_effect=AssertionError("No active model reads")):
            report_path, report = self.run_selected(["a", "b"])
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(len(report["validation"]["folds"]), 2)
        self.assertEqual(report["model_revision"], "active-revision")
        for fold, called in zip(report["validation"]["folds"], fit.call_args_list):
            self.assertNotIn(fold["group"], fold["train_groups"])
            self.assertFalse(set(fold["test_tiles"]) & set(called.args[0]))
            self.assertEqual(fold["majority_class"], 1)
            self.assertNotIn("model_path", fold)
            self.assertTrue(fold["model_revision"])
        self.assertEqual((active.read_bytes(), active.stat().st_mtime_ns), before)
        self.assertEqual(joblib.hash(self.bundle), bundle_hash)
        self.assertIs(self.bundle["model"], active_model)
        self.assertEqual({p: (sha(p), p.stat().st_mtime_ns) for p in self.sources.glob("*.tif")}, sources)
        for key, expected in samples.items():
            np.testing.assert_array_equal(self.samples[key]["X"], expected)
        self.assertEqual(report_path, self.output / "evaluations" / f"evaluation-{report['run_id']}.json")
        self.assertEqual(list((self.output / "evaluations").iterdir()), [report_path])
        self.assertEqual(json.loads(report_path.read_text()), report)
        self.assertEqual(report["provenance"], {"maps": {k: self.registry[k] for k in ("a", "b")}})
        self.assertEqual(evaluation.latest_report(self.output), (report_path, report))
        self.assertNotIn("exports", report)
        self.assertNotIn("poster_dir", report)
        self.assertFalse(self.poster.exists())
        self.assertEqual(list(self.models.iterdir()), [active])
        self.assertFalse(list(self.output.rglob("*.npz")))
        json.dumps(report, allow_nan=False)

    def test_rotation_accepts_arbitrary_group_counts_and_deduplicates_selections(self):
        for count in (2, 5, 8):
            with self.subTest(groups=count):
                self.registry = {k: v for k, v in self.registry.items() if k in ("a", "b")}
                self.samples = {"a": sample("alpha"), "b": sample("beta")}
                for index in range(2, count):
                    name, group = f"tile-{index}", f"location-{index}"
                    path = self.sources / f"{name}.tif"
                    raster(path, np.ones((8, 10), np.float32),
                           transform=TRANSFORM @ rasterio.Affine.translation(index * 10000, 0))
                    self.registry[name] = meta(name, group, path)
                    self.paintings[name] = np.ones((120, 120), np.uint8)
                    self.samples[name] = sample(group)
                self.bundle["samples"] = self.samples
                names = list(self.registry)
                with patch.object(assistant, "fit", wraps=assistant.fit) as fit:
                    _, report = self.run_selected(names + names)
                self.assertEqual(fit.call_count, count)
                self.assertEqual(report["selected_maps"], names)
                self.assertEqual(len(report["tiles"]), count)
                self.assertEqual(report["validation"]["aggregate"]["n"], count * 80)
                for fold in report["validation"]["folds"]:
                    self.assertEqual(len(fold["train_groups"]), count - 1)
                    self.assertNotIn(fold["group"], fold["train_groups"])

    def test_metrics_count_original_cells_not_pixels_or_repeated_selections(self):
        raster(Path(self.registry["a"]["path"]), core.per_pixel(self.paintings["a"], (240, 241)).astype(np.float32))
        _, report = self.run_selected(["a", "a"])
        self.assertEqual(report["validation"]["aggregate"]["n"], 14400)
        self.assertEqual(report["validation"]["aggregate"]["accuracy"], 1.0)
        self.assertEqual(report["tiles"][0]["eligible_cells"], 14400)
        self.assertEqual(len(report["tiles"]), 1)
        self.assertNotIn("exports", report)
        self.assertFalse(self.models.exists())

    def test_unverified_painted_maps_score_but_unpainted_maps_do_not(self):
        self.registry["b"]["verified"] = False
        _, report = self.run_selected(["a", "unlabelled", "b", "not-in-catalog"])
        self.assertEqual({row["name"] for row in report["skipped"]}, {"unlabelled", "not-in-catalog"})
        self.assertEqual([t["tile_id"] for t in report["tiles"]], ["a", "b"])
        plans = evaluation.plan(self.bundle, ["a", "b"])
        self.assertEqual(plans[0]["train_keys"], ["c"])
        self.assertEqual(plans[1]["train_keys"], ["a", "c"])
        self.assertEqual(plans[1]["mode"], "painted-holdout")
        self.assertFalse(report["tiles"][1]["verified"])

    def test_revoked_verification_excludes_remembered_samples_from_every_fold(self):
        self.registry["b"]["verified"] = False
        before = joblib.hash(self.bundle)
        plan = evaluation.plan(self.bundle, ["a"])[0]
        self.assertEqual(plan["train_keys"], ["c"])
        self.assertEqual(set(plan["excluded_keys"]), {"a", "b"})
        with patch.object(assistant, "fit", wraps=assistant.fit) as fit:
            self.run_selected(["a"])
        self.assertEqual(set(fit.call_args.args[0]), {"c"})
        self.assertEqual(joblib.hash(self.bundle), before)
        self.registry["c"]["verified"] = False
        self.assertIn("No verified training samples", evaluation.plan(self.bundle, ["a"])[0]["reason"])

    def test_unverified_evaluation_does_not_require_verification_before_loading(self):
        with patch.object(core, "meta_get", side_effect=lambda name: {"verified": name != "a"}):
            _, report = self.run_selected(["a"])
        self.assertEqual(report["validation"]["folds"][0]["train_tiles"], ["b", "c"])

    def test_same_location_maps_rotate_individually_and_restore_for_next_turn(self):
        for name in ("a", "b", "c"):
            self.registry[name]["group"] = "same-location"
            self.samples[name]["site_id"] = "same-location"
        with patch.object(assistant, "fit", wraps=assistant.fit) as fit:
            _, report = self.run_selected(["a", "b", "c"])
        self.assertEqual([set(c.args[0]) for c in fit.call_args_list],
                         [{"b", "c"}, {"a", "c"}, {"a", "b"}])
        self.assertEqual([f["test_tiles"] for f in report["validation"]["folds"]], [["a"], ["b"], ["c"]])
        self.assertEqual(report["validation"]["aggregate"]["n"], 240)
        self.assertIn("map_mean_macro_f1", report["validation"])

    def test_neighbours_and_shared_feature_sources_are_not_removed_with_selected_map(self):
        self.registry["b"]["geography_aliases"] = ["alpha"]
        self.samples["b"].update(public_tiles=["a"], geographic_group="alpha")
        with patch.object(core, "held_out_maps", side_effect=AssertionError("No persistent/geographic holdouts")), \
                patch.object(rasterio, "open", side_effect=AssertionError("No raster reads while planning")):
            plans = evaluation.plan(self.bundle, ["a", "b"])
        self.assertEqual([p["train_keys"] for p in plans], [["b", "c"], ["a", "c"]])
        self.assertEqual([p["maps"] for p in plans], [["a"], ["b"]])

    def test_multiple_unverified_maps_each_use_the_complete_verified_training_set(self):
        for name in ("b", "c"):
            self.registry[name]["verified"] = False
        with patch.object(assistant, "fit", wraps=assistant.fit) as fit:
            _, report = self.run_selected(["b", "c"])
        self.assertEqual([set(c.args[0]) for c in fit.call_args_list], [{"a"}, {"a"}])
        self.assertEqual([f["mode"] for f in report["validation"]["folds"]],
                         ["painted-holdout", "painted-holdout"])
        self.assertEqual(report["validation"]["aggregate"]["n"], 160)

    def test_plan_uses_no_terrain_or_fit_and_keeps_selection_order(self):
        before = joblib.hash(self.bundle)
        with patch.object(core, "build", side_effect=AssertionError("No terrain extraction")), \
                patch.object(assistant, "fit", side_effect=AssertionError("No fit while planning")):
            plans = evaluation.plan(self.bundle, ["c", "a", "c"])
        self.assertEqual([p["maps"] for p in plans], [["c"], ["a"]])
        self.assertEqual(joblib.hash(self.bundle), before)

    def test_all_source_types_use_core_build_never_dataset_extraction(self):
        self.registry["b"].update(source_kind="arbitrary-bundled-labels", section="archived-campaign", label_status="unreviewed import")
        with patch.object(core, "build", side_effect=self.build) as build:
            _, report = self.run_selected(["a", "b"])
        self.assertEqual(build.call_count, 2)
        self.assertEqual({Path(c.args[0]) for c in build.call_args_list},
                         {Path(self.registry[n]["path"]) for n in ("a", "b")})
        imported = next(t for t in report["tiles"] if t["tile_id"] == "b")
        self.assertEqual(imported["label_status"], "unreviewed import")
        self.assertTrue(imported["verified"])

    def test_target_terrain_and_labels_survive_feature_source_coverage_holes(self):
        def partial(path):
            X, names, shade, profile, px = self.build(path)
            X[:, :, 5:] = np.nan
            shade[:, 5:] = np.nan
            return X, names, shade, profile, px

        self.registry["a"]["feature_sources"] = ["survey/source.tif"]
        with patch.object(core, "build", side_effect=partial), patch.object(core, "read", wraps=core.read) as read:
            tile = evaluation._load_map(self.registry["a"])
        read.assert_called_once_with(self.registry["a"]["path"], core.MAX_SIDE)
        self.assertEqual(tile["reference_terrain_pixels"], 80)
        self.assertEqual(tile["public_coverage_fraction"], 0.5)
        self.assertEqual(int(tile["eligible"].sum()), 40)
        self.assertTrue((tile["reference"][:, 5:] == 3).all())
        self.assertTrue(np.isnan(tile["hillshade"][:, 5:]).all())

    def test_target_dem_nodata_is_not_counted_as_missing_source_coverage(self):
        values = np.ones((8, 10), np.float32)
        values[:, 5:] = np.nan
        raster(Path(self.registry["a"]["path"]), values)
        tile = evaluation._load_map(self.registry["a"])
        self.assertEqual(tile["reference_terrain_pixels"], 40)
        self.assertEqual(tile["public_coverage_fraction"], 1.0)
        self.assertEqual(int(tile["eligible"].sum()), 40)
        self.assertTrue((tile["reference"][:, 5:] == 255).all())

    def test_feature_grid_must_match_decimated_target_dem(self):
        def decimated(path):
            z, transform, crs, px = core.read(path, core.MAX_SIDE)
            return np.stack([z + i for i in range(6)]), list(core.BASE_FEATS), np.ones(z.shape), {
                "height": z.shape[0], "width": z.shape[1], "transform": transform, "crs": crs}, px

        with patch.object(core, "MAX_SIDE", 5), patch.object(core, "build", side_effect=decimated):
            tile = evaluation._load_map(self.registry["a"])
        self.assertEqual(tile["valid"].shape, (4, 5))
        self.assertEqual(tile["reference_terrain_pixels"], 20)
        with patch.object(core, "MAX_SIDE", 5), self.assertRaisesRegex(ValueError, "Feature grid differs"):
            evaluation._load_map(self.registry["a"])

    def test_accepted_unsure_and_no_pixel_cells_are_not_scored(self):
        accepted = np.zeros((120, 120), bool)
        confidence = np.ones((120, 120), np.uint8)
        ids = evaluation._cell_ids((8, 10))
        accepted.ravel()[ids[0, 0]] = True
        confidence.ravel()[ids[0, 1]] = core.UNSURE
        with patch.object(core, "load_accepted", return_value=accepted), \
                patch.object(core, "load_confidence", return_value=confidence):
            tile = evaluation._load_map(self.registry["a"])
        self.assertEqual(int(tile["eligible"].sum()), 78)
        self.assertEqual(tile["exclusions"]["no_pixel_cells"], 14400 - 80)

    def test_partial_impossible_fold_disclosed_without_in_sample_fallback(self):
        self.bundle["samples"] = {"a": sample("alpha", (1, 2))}
        self.registry["b"]["verified"] = False
        _, report = self.run_selected(["a", "b"])
        self.assertEqual([f["test_tiles"] for f in report["validation"]["folds"]], [["b"]])
        self.assertEqual([s["name"] for s in report["skipped"]], ["a"])
        self.assertIn("No verified training samples remain", report["skipped"][0]["reason"])

    def test_backend_refusal_discloses_skipped_fold_without_active_model_fallback(self):
        original_fit = assistant.fit
        with patch.object(assistant, "fit", side_effect=lambda samples: {"model": None} if "b" in samples else original_fit(samples)), \
                patch.object(self.bundle["model"], "predict", side_effect=AssertionError("No active predictions")), \
                patch.object(self.bundle["model"], "predict_proba", side_effect=AssertionError("No active confidence")):
            _, report = self.run_selected(["a", "b"])
        self.assertEqual([f["group"] for f in report["validation"]["folds"]], ["beta"])
        self.assertEqual([row["name"] for row in report["skipped"]], ["a"])
        self.assertIn("no in-sample fallback", report["skipped"][0]["reason"])

    def test_default_output_has_one_json_per_run_and_no_external_artifacts(self):
        with patch.object(paths, "OUTPUT_DIR", self.output):
            first, _ = evaluation.run(self.bundle, ["a"])
            second, report = evaluation.run(self.bundle, ["b"])
        self.assertNotEqual(first, second)
        self.assertEqual(set(self.output.rglob("*")), {self.output / "evaluations", first, second})
        self.assertEqual(json.loads(second.read_text()), report)
        self.assertEqual(evaluation.latest_report(self.output), (second, report))
        self.assertFalse(self.models.exists())
        self.assertFalse(self.poster.exists())

    def test_historical_snapshot_metadata_is_preserved_in_json(self):
        self.registry["a"].update(label_source="Historical cell snapshot", restricted=True,
                                  label_status="user verified historical snapshot; not independent certification")
        with patch.object(core, "_painting_source", return_value=(self.paintings["a"].copy(), True, True)), \
                patch.object(core, "map_record", return_value={"annotations": {"provenance": {"snapshot": "draft"}}}):
            _, report = self.run_selected(["a"])
        tile = report["tiles"][0]
        self.assertTrue(report["restricted"])
        self.assertTrue(tile["verified"])
        self.assertEqual(tile["snapshot"], "draft")
        self.assertEqual(tile["source_label"], "Historical cell snapshot")
        self.assertEqual(tile["review_status"], self.registry["a"]["label_status"])

    def test_failed_publication_removes_temporary_file_and_preserves_latest(self):
        report_path, report = self.run_selected(["a"])
        before = report_path.read_bytes()
        with patch.object(evaluation.report_helpers.os, "replace", side_effect=OSError("publish failure")), \
                self.assertRaisesRegex(OSError, "publish failure"):
            self.run_selected(["b"])
        self.assertEqual(list((self.output / "evaluations").iterdir()), [report_path])
        self.assertEqual(report_path.read_bytes(), before)
        self.assertEqual(evaluation.latest_report(self.output), (report_path, report))
        self.assertFalse(self.models.exists())

    def test_old_model_and_poster_roots_are_ignored(self):
        ignored = self.processed / "unused-root"
        report_path, _ = evaluation.run(self.bundle, ["a"], output_root=self.output,
                                        model_root=ignored, poster_root=ignored)
        self.assertFalse(ignored.exists())
        self.assertFalse(self.models.exists())
        self.assertEqual(list((self.output / "evaluations").iterdir()), [report_path])

    def test_gbm_uses_current_backend_schema_without_fold_artifacts(self):
        with patch.object(core, "MODEL", "gbm"), \
                patch.object(core.backend("gbm"), "fit_rows", side_effect=lambda *args: PixelDouble()):
            self.bundle["schema"] = assistant.schema()
            _, report = self.run_selected(["a"])
        self.assertEqual(report["schema"]["backend"], "gbm")
        self.assertNotIn("exports", report)
        self.assertFalse(self.models.exists())


class ReportDiscoveryTests(unittest.TestCase):
    OLD = "20260929T120000.000000Z-aaaaaaaa"
    NEW = "20260929T130000.000000Z-bbbbbbbb"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="selenograph-report-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "output"

    def report(self, run_id):
        return {"run_id": run_id, "created_at": "2026-09-29T12:00:00+00:00",
                "selected_maps": [], "skipped": [], "tiles": [],
                "validation": {"aggregate": {"n": 12, "accuracy": 0.75, "macro_f1": 0.7}},
                "provenance": {"maps": {}}}

    def publish(self, run_id):
        report = self.report(run_id)
        path = self.output / "evaluations" / f"evaluation-{run_id}.json"
        write_json(path, report)
        return path, report

    def test_missing_output_is_read_only(self):
        self.assertIsNone(evaluation.latest_report(self.output))
        self.assertFalse(self.output.exists())

    def test_newest_valid_date_wins_not_mtime(self):
        old, _ = self.publish(self.OLD)
        latest = self.publish(self.NEW)
        os.utime(old, ns=(old.stat().st_atime_ns, latest[0].stat().st_mtime_ns + 10_000_000))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.output / "evaluations").iterdir()}
        self.assertEqual(evaluation.latest_report(self.output), latest)
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.output / "evaluations").iterdir()}, before)

    def test_corrupt_invalid_and_unsafe_newer_reports_do_not_hide_valid_older_report(self):
        previous = self.publish(self.OLD)
        candidate = self.output / "evaluations" / f"evaluation-{self.NEW}.json"
        invalid = [None, [], {}, dict(self.report(self.NEW), run_id="../escape"),
                   dict(self.report(self.NEW), validation=[]),
                   dict(self.report(self.NEW), provenance=None),
                   dict(self.report(self.NEW), selected_maps="not a list"),
                   dict(self.report(self.NEW), validation={"aggregate": {"n": 12, "accuracy": "wrong"}}),
                   dict(self.report(self.NEW), validation={"aggregate": {"n": 12, "accuracy": float("nan")}})]
        for content in ["{truncated", *[json.dumps(value) for value in invalid]]:
            with self.subTest(content=content):
                candidate.write_text(content)
                self.assertEqual(evaluation.latest_report(self.output), previous)
        candidate.unlink()
        outside = self.root / "outside.json"
        write_json(outside, self.report(self.NEW))
        candidate.symlink_to(outside)
        self.assertEqual(evaluation.latest_report(self.output), previous)

    def test_invalid_dates_directories_and_unpublished_files_are_ignored(self):
        previous = self.publish(self.OLD)
        for run_id in ("20261329T140000.000000Z-cccccccc", "not-a-date"):
            self.publish(run_id)
        (self.output / "evaluations" / f"evaluation-{self.NEW}.json").mkdir()
        write_json(self.output / "evaluations" / f".evaluation-{self.NEW}.json-staging.tmp", self.report(self.NEW))
        write_json(self.output / "evaluations" / f"evaluation-{self.NEW}.json.tmp", self.report(self.NEW))
        self.assertEqual(evaluation.latest_report(self.output), previous)

    def test_reports_outside_evaluations_are_ignored(self):
        report = self.report(self.NEW)
        write_json(self.output / f"evaluation-{self.NEW}.json", report)
        write_json(self.output / "evaluation" / self.NEW / "results.json", report)
        write_json(self.output / "evaluation/latest.json",
                   {"run_id": self.NEW, "results": f"{self.NEW}/results.json"})
        self.assertIsNone(evaluation.latest_report(self.output))
        self.assertFalse((self.output / "evaluations").exists())
        self.assertEqual(evaluation.latest_report(self.output), None)
        latest = self.publish(self.OLD)
        self.assertEqual(evaluation.latest_report(self.output), latest)

    def test_progress_cache_refreshes_on_new_deleted_and_repaired_reports(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        old, _ = self.publish(self.OLD)
        page = Path(__file__).resolve().parents[1] / "app/app_pages/progress.py"
        entry = "import streamlit as st\nst.navigation([st.Page(" + repr(str(page)) + ", title='Progress')]).run()"
        with patch.object(paths, "OUTPUT_DIR", self.output), \
                patch.object(assistant, "load", return_value={"samples": {}, "model": None}), \
                patch.object(assistant, "retained_samples", return_value={}), \
                patch.object(evaluation, "catalog", return_value={}), \
                patch.object(core, "test_maps", return_value=[]), \
                patch.object(core, "held_out_maps", return_value=set()), \
                patch.object(core, "painted_maps", return_value=(set(), set())), \
                patch.object(evaluation, "latest_report", wraps=evaluation.latest_report) as latest:
            app = AppTest.from_string(entry, default_timeout=30).run()

            def displayed(run_id):
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertTrue(any(f"Run: {run_id}" in c.value for c in app.caption))

            displayed(self.OLD)
            app.run()
            displayed(self.OLD)
            self.assertEqual(latest.call_count, 1)
            newer, _ = self.publish(self.NEW)
            app.run()
            displayed(self.NEW)
            self.assertEqual(latest.call_count, 2)
            newer.write_text("{broken")
            app.run()
            displayed(self.OLD)
            self.assertEqual(latest.call_count, 3)
            self.publish(self.NEW)
            app.run()
            displayed(self.NEW)
            self.assertEqual(latest.call_count, 4)
            newer.unlink()
            app.run()
            displayed(self.OLD)
            # The restored listing can reuse its still-valid cached older report.
            self.assertEqual(latest.call_count, 4)
            self.assertEqual(list((self.output / "evaluations").iterdir()), [old])


class CatalogTests(unittest.TestCase):
    def test_processed_only_arbitrary_sections_verification_and_label_status(self):
        with tempfile.TemporaryDirectory(prefix="selenograph-catalog-test-") as temporary:
            root = Path(temporary) / "processed_data"
            files = [root / p for p in ("survey-x/a.tif", "archive-y/b.tif", "reference/c.tif", "flat.tif", "survey-x/empty.tif")]
            records = {
                "a": {"group": "location-a", "collection": "private", "section": "declared-section", "geography_aliases": ["old-location"],
                      "annotations": {"verified": True}},
                "b": {"group": "old-location", "annotations": {"painting": "cells.npy", "verified": True,
                                                                "label_status": "unreviewed imported cells", "provenance": "origin.json"}},
                "c": {"group": "elsewhere", "collection": "reference"},
                "flat": {"group": "elsewhere"}, "empty": {"group": "empty-place"},
            }
            verified = {"a": False, "b": True, "c": True, "flat": False, "empty": True}

            def painting(name):
                cells = np.full((120, 120), 0 if name == "empty" else 1, np.uint8)
                return cells, cells > 0, "bundled" if name == "b" else "saved"

            with patch.object(core, "DEM_DIR", root), \
                    patch.object(core, "dem_files", return_value=[str(p) for p in files]), \
                    patch.object(core, "map_record", side_effect=records.get), \
                    patch.object(core, "section", wraps=core.section) as section, \
                    patch.object(core, "map_site", side_effect=lambda n: records[n]["group"]), \
                    patch.object(core, "meta_get", side_effect=lambda n: {"verified": verified[n]}), \
                    patch.object(evaluation, "_painting", side_effect=painting), \
                    patch.object(core, "build", side_effect=AssertionError("catalog must not build terrain")), \
                    patch.dict("sys.modules", {"poster": None}):
                registry = evaluation.catalog()
            self.assertEqual(set(registry), {p.stem for p in files} - {"empty"})
            self.assertEqual(section.call_count, len(files))
            self.assertEqual(registry["a"]["section"], "declared-section")
            self.assertEqual({registry[n]["section"] for n in registry}, {"declared-section", "archive-y", "reference", "processed_data"})
            self.assertIsNone(registry["a"]["reason"])
            self.assertTrue(registry["a"]["available"])
            self.assertFalse(registry["a"]["verified"])
            self.assertTrue(registry["b"]["available"])
            self.assertTrue(registry["c"]["available"])  # collection name has no special filtering
            self.assertEqual(registry["b"]["label_status"], "unreviewed imported cells")
            self.assertEqual(registry["b"]["provenance"], "origin.json")
            self.assertEqual(registry["b"]["group"], "location-a")
            self.assertNotIn("empty", registry)
            self.assertFalse(hasattr(evaluation, "ALIASES"))
            self.assertFalse(hasattr(evaluation, "references"))

    def test_real_processed_catalog_and_bundled_snapshot_precedence(self):
        with tempfile.TemporaryDirectory(prefix="selenograph-catalog-test-") as temporary:
            root = Path(temporary)
            processed = root / "processed"
            section = processed / "arbitrary-archive"
            output = root / "output"
            out, drafts = output / "paintings", output / "painting_drafts"
            cells = np.ones((120, 120), np.uint8)
            raster(section / "bundled.tif", np.ones((8, 10), np.float32))
            raster(section / "unregistered.tif", np.ones((8, 10), np.float32))
            raster(root / "reference_data" / "not-processed.tif", np.ones((8, 10), np.float32))
            np.save(section / "cells.npy", cells)
            provenance = {"snapshot": "draft", "original_provenance": {"id": "historical-labels"}}
            record = {"working_dem": "bundled.tif", "group": "a-location", "source_kind": "any-import",
                      "annotations": {"painting": "cells.npy", "verified": True,
                                      "label_status": "user verified historical snapshot; not independent truth",
                                      "provenance": provenance}}
            write_json(section / "working_dems.json", [record])
            original_files = {p: (sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            locations = {"DEM_DIR": processed, "OUT_DIR": out, "DRAFT_DIR": drafts,
                         "MAP_DIR": output / "maps", "LEGACY_OUT_DIR": output / "maps", "LEGACY_DRAFT_DIR": output / "drafts",
                         "LEGACY_MIRRORED_DIR": output}
            with patch.multiple(core, **locations), \
                    patch.multiple(paths, OUTPUT_DIR=output, **locations), \
                    patch.object(core, "build", side_effect=AssertionError("No catalog terrain extraction")), \
                    patch.dict("sys.modules", {"poster": None}):
                registry = evaluation.catalog()
                self.assertEqual(set(registry), {"bundled"})
                self.assertEqual(registry["bundled"]["section"], "arbitrary-archive")
                self.assertTrue(registry["bundled"]["available"])
                self.assertEqual(registry["bundled"]["snapshot"], "draft")
                self.assertEqual(registry["bundled"]["provenance"], provenance)
                self.assertIn("user verified historical", registry["bundled"]["label_status"])
                self.assertFalse(output.exists())
                self.assertEqual({p: (sha(p), p.stat().st_mtime_ns) for p in original_files}, original_files)
                # User saves/drafts must win without inheriting the bundle's snapshot.
                core.save_painting("bundled", cells * 2, final=True)
                self.assertEqual(evaluation.catalog()["bundled"]["snapshot"], "saved")
                core.save_painting("bundled", cells * 3, final=False)
                self.assertEqual(evaluation.catalog()["bundled"]["snapshot"], "draft")
                np.testing.assert_array_equal(evaluation._painting("bundled")[0], cells * 3)
                core.save_painting("bundled", np.zeros_like(cells), final=False)
                self.assertNotIn("bundled", evaluation.catalog())
                core.save_painting("bundled", cells, final=False)
                core.meta_set("bundled", verified=False)
                self.assertTrue(evaluation.catalog()["bundled"]["available"])
                self.assertFalse(evaluation.catalog()["bundled"]["verified"])
            self.assertEqual({p: (sha(p), p.stat().st_mtime_ns) for p in original_files}, original_files)

    def test_catalog_excludes_all_accepted_or_unsure_paintings_from_availability(self):
        cells = np.ones((120, 120), np.uint8)
        with tempfile.TemporaryDirectory(prefix="selenograph-catalog-test-") as temporary:
            root = Path(temporary)
            with patch.object(core, "DEM_DIR", root), patch.object(core, "dem_files", return_value=[str(root / "map.tif")]), \
                    patch.object(core, "map_record", return_value={}), patch.object(core, "map_site", return_value="somewhere"), \
                    patch.object(core, "meta_get", return_value={"verified": True}), \
                    patch.object(core, "_painting_source", return_value=(cells, True, False)):
                for accepted, confidence in ((np.ones(cells.shape, bool), np.full(cells.shape, core.SURE)),
                                             (np.zeros(cells.shape, bool), np.full(cells.shape, core.UNSURE))):
                    with patch.object(core, "load_accepted", return_value=accepted), \
                            patch.object(core, "load_confidence", return_value=confidence):
                        record = evaluation.catalog()["map"]
                    self.assertFalse(record["available"])
                    self.assertIn("No sure manual cells", record["reason"])

    def test_bundled_painting_uses_same_flag_apis(self):
        cells = np.ones((120, 120), np.uint8)
        accepted = np.zeros(cells.shape, bool)
        accepted[0, 0] = True
        confidence = np.full(cells.shape, core.SURE, np.uint8)
        confidence[0, 1] = core.UNSURE
        with patch.object(core, "_painting_source", return_value=(cells, True, True)), \
                patch.object(core, "map_record", return_value={}), \
                patch.object(core, "load_accepted", return_value=accepted), \
                patch.object(core, "load_confidence", return_value=confidence):
            loaded, eligible, snapshot = evaluation._painting("arbitrary-bundled-map")
        self.assertEqual(snapshot, "bundled")
        self.assertEqual(int(eligible.sum()), 14400 - 2)
        np.testing.assert_array_equal(loaded, cells)


if __name__ == "__main__":
    unittest.main()
