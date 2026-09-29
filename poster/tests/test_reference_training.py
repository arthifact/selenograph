"""Offline synthetic checks only: no real dataset extraction or estimator fitting."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import rasterio

from app import assistant_model as assistant
from poster import benchmark_data as data
from app import core
from poster import reference_training as training, storage
from poster.tests import test_benchmark_data as benchmark_fixtures
from poster.tests.test_benchmark_data import (
    MOON,
    TRANSFORM,
    fake_build,
    raster,
    sha,
    write_json,
)


class FixturePixelModel:
    """Pickleable backend double; exercises real app save/load/prediction, not RF fit."""

    n_features_in_ = 6

    def predict(self, rows):
        return np.clip(np.rint(rows[:, 0]), 1, 3).astype(np.uint8)

    def predict_proba(self, rows):
        result = np.full((len(rows), 3), 0.05, np.float32)
        result[np.arange(len(rows)), self.predict(rows) - 1] = 0.9
        return result


def fixture_tiles():
    tiles = []
    for index, (site, map_id) in enumerate((
        ("mons-mouton", "MM026"), ("nobile1", "N1014"),
        ("nobile1-ms1", "Nobile1-MS1"), ("nobile2", "N2005"),
    )):
        cells = np.full((120, 120), 3, np.uint8)
        cells[:, :25], cells[:, 25:50] = 1, 2
        grid = data._Grid(cells.shape, TRANSFORM @ rasterio.Affine.translation(index * 200, 0), MOON)
        X = np.stack([cells.astype(np.float32) + i for i in range(6)])
        group, key = data.GROUPS[site], f"reference-{index}"
        provenance = {
            "tile_id": key, "map_id": map_id, "site_id": site, "snapshot": "saved",
            "reference_group": group, "geographic_group": group,
            "public_region": data.REGIONS[site], "source_kind": "reference_snapshot",
            "public_tiles": [f"public_dem_{index}"], "public_tile_ids": [f"public-{index}"],
            "painting_hash": assistant.painting_hash(cells), "source_revision": {"fixture": "v1"},
            "footprint": grid.metadata(), "grid": grid.metadata(),
            "confidence_observed": False, "accepted_observed": {"saved": False},
        }
        tiles.append({
            "tile_id": key, "map_id": map_id, "site_id": site, "group": group,
            "snapshot": "saved", "grid": grid.metadata(), "px_m": 5.0,
            "X": X, "hillshade": np.full(cells.shape, 0.5, np.float32),
            "reference": cells.copy(), "cells": cells, "eligible": np.ones(cells.shape, bool),
            "valid": np.ones(cells.shape, bool), "support": np.zeros(cells.shape, bool),
            "reference_terrain_pixels": cells.size, "public_coverage_fraction": 1.0,
            "exclusions": {}, "provenance": provenance,
            "prediction": np.full(cells.shape, 255, np.uint8),
            "confidence": np.full(cells.shape, np.nan, np.float32),
            "cell_prediction": np.full(cells.shape, 255, np.uint8),
        })
    return tiles


class GeometryAndScoreTests(unittest.TestCase):
    def test_original_floor_cell_geometry_unknown_and_ties(self):
        cells = (np.arange(14400).reshape(120, 120) % 3 + 1).astype(np.uint8)
        pixels = data._expand(cells, (1024, 1023)).copy()
        np.testing.assert_array_equal(training.cell_majority(pixels), cells)
        rows, cols = data._edges(pixels.shape)
        self.assertEqual(set(np.diff(rows)), {8, 9})
        pixels[:rows[1], :cols[1]] = 255
        pixels[:rows[1], cols[1]:cols[2]] = 0
        pixels[:rows[1] // 2, cols[1]:cols[2]] = 2
        pixels[rows[1] // 2:rows[1], cols[1]:cols[2]] = 1
        prediction = training.cell_majority(pixels)
        self.assertEqual(prediction[0, 0], 255)
        self.assertEqual(prediction[0, 1], 1)
        valid = np.ones(pixels.shape, bool)
        valid[:rows[1], cols[1]:cols[2]] = False
        self.assertEqual(training.cell_majority(pixels, valid)[0, 1], 255)
        with self.assertRaisesRegex(ValueError, "120 pixels"):
            training.cell_majority(np.ones((100, 120), np.uint8))

    def test_metrics_contract_and_nodata(self):
        metrics = training.score(np.array([1, 1, 2, 2, 3, 0, 255]),
                                 np.array([1, 2, 2, 3, 3, 255, 0]))
        self.assertEqual(metrics["n"], 5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 3 / 5)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.5 + 2 / 3) / 3)
        self.assertEqual(metrics["per_class"]["2"],
                         {"precision": 0.5, "recall": 0.5, "f1": 0.5, "iou": 1 / 3, "support": 2})
        empty = training.score(np.array([0, 255]), np.array([0, 255]))
        self.assertEqual((empty["n"], empty["macro_f1"]), (0, 0.0))
        self.assertAlmostEqual(training.score(np.array([1]), np.array([1]))["macro_f1"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "Missing/invalid"):
            training.score(np.array([1]), np.array([255]))
        json.dumps(metrics, allow_nan=False)

    def test_majority_uses_unsampled_cells_not_balanced_rows(self):
        tiles = fixture_tiles()
        samples = training.sample_tiles(tiles)
        counts = np.bincount(np.concatenate([s["y"] for s in samples.values()]), minlength=4)[1:]
        np.testing.assert_array_equal(counts, [8000, 8000, 8000])
        self.assertEqual(int(counts.argmax() + 1), 1)
        self.assertEqual(training._majority_class(tiles), 3)
        tiles[0]["cells"][:] = 1
        tiles[1]["cells"][:] = 2
        self.assertEqual(training._majority_class(tiles[:2]), 1)


class SourceTests(unittest.TestCase):
    def setUp(self):
        # Reuse the audited small TIFF/manifest fixture, without running its tests.
        self.fixture = benchmark_fixtures.ExtractionTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture
        path = self.f.reference / "sites" / self.f.site / "site.json"
        write_json(path, {"site_id": self.f.site, "map_id": "Nobile1-MS1"})
        self.publish_reference()

    def publish_reference(self):
        self.f._publish_reference()
        path = self.f.reference / "dataset.json"
        manifest = data._json(path)
        relative = f"sites/{self.f.site}/site.json"
        manifest["sites"] = [{"site_id": self.f.site, "path": relative}]
        manifest["artifacts"].append({"path": relative, "sha256": sha(self.f.reference / relative)})
        write_json(path, manifest)

    def load(self, build=fake_build):
        with patch.object(core, "build", side_effect=build) as called, \
                patch.object(data, "_public_features", side_effect=AssertionError("NAC/cell means forbidden")), \
                patch.object(data, "_optional", side_effect=AssertionError("No LOLA/radar extraction")):
            result = training.load_data(self.f.public, self.f.reference, self.f.raw)
        self.assertEqual(called.call_count, 1)
        self.assertEqual(Path(called.call_args.args[0]), self.f.folder / "tile/sfs.tif")
        return result

    def test_six_pixel_features_saved_preference_accepted_union_and_immutable_sources(self):
        before = {p: (p.stat().st_mtime_ns, sha(p)) for p in self.f.root.rglob("*") if p.is_file()}
        result = self.load()
        after = {p: (p.stat().st_mtime_ns, sha(p)) for p in self.f.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        tile = result["tiles"][0]
        self.assertEqual(tile["X"].shape, (6, 240, 240))
        self.assertEqual(tile["X"].dtype, np.float32)
        np.testing.assert_array_equal(tile["X"][:, 10, 10], [10, 11, 14, 15, 16, 17])
        self.assertEqual(tile["map_id"], "Nobile1-MS1")
        self.assertEqual(tile["group"], "nobile-1")
        self.assertEqual(tile["snapshot"], "saved")
        self.assertFalse(tile["eligible"][0, 0])  # insufficient reference terrain
        self.assertFalse(tile["eligible"][0, 1])  # saved unknown
        self.assertTrue(tile["eligible"][0, 2])   # draft unknown does not exclude saved
        self.assertTrue(tile["eligible"][0, 3])   # conflicting draft must not duplicate rows
        self.assertFalse(tile["eligible"][0, 4])  # accepted flag exists only in draft
        self.assertTrue(tile["eligible"][0, 5])   # NAC hole/valid_data=0 still usable terrain
        self.assertEqual(tile["reference"][0, 0], 255)
        self.assertEqual(tile["reference"][0, 2], 255)
        self.assertTrue(np.isnan(tile["X"][:, 0, 0]).all())
        self.assertFalse(tile["support"].any())
        self.assertEqual(tile["provenance"]["public_tiles"], ["sfs"])
        self.assertEqual(tile["provenance"]["public_tile_ids"], ["public-1"])
        self.assertFalse(tile["provenance"]["confidence_observed"])
        self.assertIsNone(tile["provenance"]["confidence_counts"])
        self.assertIn("NOT certified sure", tile["provenance"]["weight_policy"])
        with patch.object(assistant, "sample", wraps=assistant.sample) as sample:
            samples = training.sample_tiles(result["tiles"])
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(len(samples), 1)
        item = samples[self.f.tile_id]
        self.assertEqual(len(item["y"]), 2000)
        self.assertTrue((item["y"] == 1).all())  # draft's class 2 never gets sampled
        self.assertTrue((item["w"] == 1.0).all())
        labels = sample.call_args.args[1]
        self.assertEqual(labels[0, 8], 255)
        self.assertEqual(labels[0, 4], 1)
        again = training.sample_tiles(result["tiles"])[self.f.tile_id]
        np.testing.assert_array_equal(item["X"], again["X"])
        self.assertGreaterEqual(result["metadata"]["timings_seconds"]["extraction"], 0)
        self.assertEqual(result["metadata"]["software_versions"]["numpy"], np.__version__)
        json.dumps(result["metadata"], allow_nan=False)

    def test_draft_fallback_and_missing_flags_stay_unknown(self):
        self.f._make_reference(variants=("draft",), accepted=False)
        self.publish_reference()
        tile = self.load()["tiles"][0]
        self.assertEqual(tile["snapshot"], "draft")
        np.testing.assert_array_equal(tile["cells"], self.f.draft)
        self.assertTrue(tile["eligible"][0, 4])
        self.assertEqual(tile["provenance"]["accepted_observed"], {"draft": False})
        self.assertFalse(tile["provenance"]["confidence_observed"])
        self.assertFalse(tile["eligible"][0, 2])

    def test_no_nac_layer_required_and_all_six_features_must_be_finite(self):
        self.f.public_record["layers"].pop("nac")
        (self.f.folder / "tile/nac.tif").unlink()
        write_json(self.f.folder / "working_dems.json", [self.f.public_record])
        self.f.public_meta["outputs"]["working_dems.json"]["checksum"]["sha256"] = sha(
            self.f.folder / "working_dems.json")
        self.f._publish_public_meta()

        def partially_missing(path):
            X, names, shade, profile, px = fake_build(path)
            X[names.index("curv"), 10, 10] = np.nan
            return X, names, shade, profile, px

        tile = self.load(build=partially_missing)["tiles"][0]
        self.assertFalse(tile["eligible"][5, 5])  # 3/4 pixels <90%, although channel 0 finite
        self.assertTrue(tile["eligible"][0, 5])
        self.assertFalse(tile["valid"][10, 10])
        self.assertTrue(np.isnan(tile["X"][:, 10, 10]).all())

    def test_nearest_pixels_across_whole_public_tiles_are_not_cell_means(self):
        self.f._split_public_tiles()
        catalog = self.f.folder / "working_dems.json"
        records = data._json(catalog)
        for record in records:
            side = record["tile_id"]
            relative = f"{side}/sfs-{side}.tif"
            old_path = self.f.folder / record["layers"]["sfs"]
            with rasterio.open(old_path) as src:
                values, transform = src.read(1), src.transform
            if side == "right":
                values[:] = 30
            raster(self.f.folder / relative, values, transform=transform)
            record["layers"]["sfs"] = record["working_dem"] = relative
            self.f.public_meta["outputs"][relative] = {"checksum": {"sha256": sha(self.f.folder / relative)}}
        write_json(catalog, records)
        self.f.public_meta["outputs"]["working_dems.json"]["checksum"]["sha256"] = sha(catalog)
        self.f._publish_public_meta()
        # Isolate nearest transport from real neighbourhood calculations; the
        # separate feature-equivalence test covers the joined terrain recipe.
        with (patch.object(core, "build", side_effect=fake_build) as build,
              patch.object(core, "_source_contexts", return_value={})):
            result = training.load_data(self.f.public, self.f.reference, self.f.raw)
        self.assertEqual(build.call_count, 2)
        tile = result["tiles"][0]
        self.assertTrue(tile["eligible"][10, 60])
        np.testing.assert_array_equal(tile["X"][0, 20, 120:122], [10, 30])
        self.assertEqual(tile["provenance"]["public_tiles"], ["sfs-left", "sfs-right"])
        self.assertEqual(result["metadata"]["public_tiles_used"], 2)
        # Remove the right public tile: a half-covered original cell is ineligible.
        write_json(catalog, records[:1])
        self.f.public_meta["outputs"]["working_dems.json"]["checksum"]["sha256"] = sha(catalog)
        self.f._publish_public_meta()
        with patch.object(core, "build", side_effect=fake_build):
            tile = training.load_data(self.f.public, self.f.reference, self.f.raw)["tiles"][0]
        self.assertTrue(tile["eligible"][10, 59])
        self.assertFalse(tile["eligible"][10, 60])
        self.assertFalse(tile["eligible"][10, 61])
        self.assertFalse(tile["valid"][:, 121:].any())
        self.assertTrue(np.isnan(tile["hillshade"][:, 121:]).all())
        self.assertTrue((tile["reference"][:, 121:] == 1).all())

    def test_public_overlap_rejected_even_when_support_is_zero(self):
        duplicate = dict(self.f.public_record, tile_id="public-duplicate")
        write_json(self.f.folder / "working_dems.json", [self.f.public_record, duplicate])
        self.f.public_meta["outputs"]["working_dems.json"]["checksum"]["sha256"] = sha(
            self.f.folder / "working_dems.json")
        self.f._publish_public_meta()
        with self.assertRaisesRegex(ValueError, "Duplicate public DEM|overlapping public"):
            self.load()

    def test_public_mask_hash_and_reference_hash_fail_closed(self):
        path = self.f.folder / "tile/sfs_support.tif"
        raster(path, np.ones(self.f.shape, np.float32))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.load()
        self.f._make_public()
        np.save(self.f.reference / "saved.npy", np.zeros((120, 120), np.uint8), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.load()

    def test_map_id_is_not_guessed_from_tile_id_or_snapshot_provenance(self):
        path = self.f.reference / "sites" / self.f.site / "site.json"
        write_json(path, {"site_id": self.f.site})
        self.publish_reference()
        with self.assertRaisesRegex(ValueError, "No map_id"):
            self.load()


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="selenograph-report-finalization-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.directory = self.root / "fixture-run"
        self.directory.mkdir()
        (self.directory / "figures").mkdir()
        self.figure_report = self.directory / "figures/report.json"
        self.figure_report.write_bytes(b"old figure report")
        self.summary = self.directory / "summary.md"
        self.summary.write_bytes(b"old summary")
        self.bundle = self.directory / "poster-assets.zip"
        metrics = training.score(np.array([1, 1, 2, 3]), np.array([1, 2, 2, 3]))
        baseline = training.score(np.array([1, 1, 2, 3]), np.ones(4, np.uint8))
        self.report = {
            "run_id": self.directory.name, "created_at": "2026-09-29T00:00:00+00:00",
            "model_revision": "fixture-revision", "schema": {"classes": {"1": "smooth", "2": "rough", "3": "floor"}},
            "feature_names": list(core.BASE_FEATS), "timings_seconds": {"final_fit": 0.125},
            "training": {"tiles": 2, "map_groups": 2, "geography_groups": 2, "sampled_pixels": 321,
                         "samples_by_tile": {"train-a": 123, "train-b": 198},
                         "installation": "not installed: existing user model preserved",
                         "excluded_reference_groups": ["nobile-1"]},
            "validation": {"method": "synthetic held-out fold", "aggregate": metrics,
                           "baseline_aggregate": baseline, "group_mean_macro_f1": metrics["macro_f1"],
                           "folds": [{"group": "nobile-1", "metrics": metrics,
                                      "baseline": baseline, "majority_class": 1}]},
            "tiles": [{"tile_id": "held-a", "map_id": "N1014", "snapshot": "saved",
                       "eligible_cells": 4, "reference_terrain_pixels": 100,
                       "public_coverage_fraction": 0.75}],
            "limitations": ["These labels were already used for feature selection.",
                            "Class 3 is not an ice detector."],
            "exports": {"files": [{"path": "figures/asset.png", "label": "Test image"}],
                        "bundle": "poster-assets.zip"},
        }
        info = zipfile.ZipInfo("figures/asset.png", date_time=(2026, 9, 29, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o600 << 16
        info.comment = b"entry metadata"
        with zipfile.ZipFile(self.bundle, "w") as archive:
            archive.comment = b"local restricted fixture"
            archive.writestr(info, bytes(range(256)) * 100)
            archive.writestr("captions.md", b"unchanged captions\n", compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr("report.json", b"old bundled report")
            archive.writestr("summary.md", b"old bundled summary")

    def test_final_report_summary_and_zip_contents_are_consistent_and_repeatable(self):
        before_report = copy.deepcopy(self.report)
        with zipfile.ZipFile(self.bundle) as archive:
            untouched = {i.filename: (archive.read(i), i.date_time, i.compress_type, i.external_attr, i.comment)
                         for i in archive.infolist() if i.filename not in ("report.json", "summary.md")}
        with patch.object(assistant, "fit", side_effect=AssertionError("must not fit")), \
                patch.object(assistant, "save", side_effect=AssertionError("must not modify a model")):
            for _ in range(2):
                training.finalize_report_exports(self.directory, self.report)
                self.assertEqual(data._json(self.figure_report), self.report)
                with zipfile.ZipFile(self.bundle) as archive:
                    self.assertEqual(archive.comment, b"local restricted fixture")
                    self.assertEqual(archive.namelist().count("report.json"), 1)
                    self.assertEqual(archive.namelist().count("summary.md"), 1)
                    self.assertEqual(archive.read("report.json"), self.figure_report.read_bytes())
                    self.assertEqual(archive.read("summary.md"), self.summary.read_bytes())
                    for name, original in untouched.items():
                        info = archive.getinfo(name)
                        self.assertEqual((archive.read(info), info.date_time, info.compress_type,
                                          info.external_attr, info.comment), original)
        self.assertEqual(self.report, before_report)
        self.assertFalse((self.directory / "results.json").exists())
        self.assertFalse((self.root / "latest.json").exists())
        summary = self.summary.read_text(encoding="utf-8")
        for expected in ("75.000%", "50.000%", "| nobile-1 | 4 |",
                         "already used for feature selection", "poster.reference_training", "app model is never changed"):
            self.assertIn(expected, summary)
        self.assertFalse(list(self.directory.rglob("*.tmp")))

    def test_renderer_outputs_can_be_absent(self):
        self.figure_report.unlink()
        self.figure_report.parent.rmdir()
        self.bundle.unlink()
        self.report["exports"] = {}
        training.finalize_report_exports(self.directory, self.report)
        self.assertTrue(self.summary.is_file())
        self.assertFalse(self.figure_report.parent.exists())
        self.assertFalse(self.bundle.exists())

    def test_corrupt_or_missing_declared_zip_does_not_publish_reports(self):
        self.bundle.write_bytes(b"not a zip")
        with self.assertRaises(zipfile.BadZipFile):
            training.finalize_report_exports(self.directory, self.report)
        self.assertEqual(self.bundle.read_bytes(), b"not a zip")
        self.assertEqual(self.figure_report.read_bytes(), b"old figure report")
        self.assertEqual(self.summary.read_bytes(), b"old summary")
        self.bundle.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Declared report bundle"):
            training.finalize_report_exports(self.directory, self.report)
        self.assertFalse(list(self.directory.rglob("*.tmp")))

    def test_zip_replace_failure_preserves_previous_archive_and_cleans_staging(self):
        before = self.bundle.read_bytes()
        with patch.object(training.os, "replace", side_effect=OSError("injected publish failure")), \
                self.assertRaisesRegex(OSError, "injected publish failure"):
            training.finalize_report_exports(self.directory, self.report)
        self.assertEqual(self.bundle.read_bytes(), before)
        self.assertEqual(self.figure_report.read_bytes(), b"old figure report")
        self.assertEqual(self.summary.read_bytes(), b"old summary")
        self.assertFalse(list(self.directory.rglob("*.tmp")))

    def test_bundle_paths_cannot_escape_run(self):
        outside = self.root / "outside.zip"
        outside.write_bytes(b"do not touch")
        self.report["exports"]["bundle"] = "../outside.zip"
        with self.assertRaisesRegex(ValueError, "escapes root"):
            training.finalize_report_exports(self.directory, self.report)
        self.assertEqual(outside.read_bytes(), b"do not touch")
        self.assertEqual(self.summary.read_bytes(), b"old summary")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="selenograph-reference-training-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.models, self.output = self.root / "models", self.root / "output"
        self.tiles = fixture_tiles()
        for module, name, value in ((core, "MODEL", "rf"), (assistant, "MODEL_DIR", self.models)):
            context = patch.object(module, name, value)
            context.start()
            self.addCleanup(context.stop)
        # Real assistant.fit/save/load with a pickleable estimator double. NO RF is trained.
        self.fit_rows = patch.object(core.backend(), "fit_rows", side_effect=lambda *args: FixturePixelModel())
        self.fit_rows_mock = self.fit_rows.start()
        self.addCleanup(self.fit_rows.stop)

    def test_three_fresh_models_no_geographic_leakage_or_existing_user_model(self):
        samples = training.sample_tiles(self.tiles)
        before = {k: s["X"].copy() for k, s in samples.items()}
        with patch.object(assistant, "fit", wraps=assistant.fit) as fit, \
                patch.object(assistant, "load", side_effect=AssertionError("Never use an existing user model")):
            result, exports = training.validate(self.tiles, samples)
        self.assertEqual(fit.call_count, 3)
        self.assertEqual(len({f["model_revision"] for f in result["folds"]}), 3)
        for call, fold in zip(fit.call_args_list, result["folds"]):
            groups = {s["geographic_group"] for s in call.args[0].values()}
            self.assertNotIn(fold["group"], groups)
            self.assertEqual(groups, set(fold["train_groups"]))
            self.assertFalse(set(fold["train_tiles"]) & set(fold["test_tiles"]))
            self.assertEqual(fold["majority_class"], 3)
            self.assertEqual(fold["sampled_pixels"], len(fold["train_tiles"]) * 6000)
            self.assertEqual(fold["baseline"]["accuracy"], 70 / 120)
            self.assertNotIn("model_path", fold)
            if fold["group"] == "nobile-1":
                self.assertEqual(fold["test_tiles"], ["reference-1", "reference-2"])
        self.assertEqual(result["aggregate"]["n"], 4 * 14400)
        self.assertEqual(result["aggregate"]["macro_f1"], 1.0)
        self.assertEqual(result["group_mean_macro_f1"], 1.0)
        self.assertEqual(exports, {})
        self.assertFalse(self.models.exists())
        self.assertFalse(self.output.exists())
        for tile in self.tiles:
            np.testing.assert_array_equal(tile["cell_prediction"], tile["cells"])
        for key in samples:
            np.testing.assert_array_equal(samples[key]["X"], before[key])
        json.dumps(result, allow_nan=False)

    def test_fold_prediction_nodata_and_all_groups_required(self):
        tile = self.tiles[0]
        tile["valid"][0, 0] = False
        tile["X"][:, 0, 0] = np.nan
        tile["eligible"][0, 0] = False
        tile["reference"][0, 0] = 255
        samples = training.sample_tiles(self.tiles)
        training.validate(self.tiles, samples)
        self.assertEqual(tile["prediction"][0, 0], 255)
        self.assertTrue(np.isnan(tile["confidence"][0, 0]))
        self.assertEqual(tile["cell_prediction"][0, 0], 255)
        self.assertEqual(tile["metrics"]["n"], 14399)
        with self.assertRaisesRegex(ValueError, "all three"):
            training.validate(self.tiles[:1], samples)
        samples["reference-0"]["geographic_group"] = "nobile-1"
        with self.assertRaisesRegex(ValueError, "geography disagrees"):
            training.validate(self.tiles, samples)

    def renderer(self, tiles, report, directory):
        self.assertEqual(len(tiles), 4)
        self.assertEqual(set(tiles[0]), set(training.RENDER_FIELDS))
        self.assertTrue(all(t["prediction"].dtype == np.uint8 for t in tiles))
        self.assertEqual(report["validation"]["aggregate"]["n"], 57600)
        self.assertEqual(directory.name, "figures")
        directory.mkdir()
        (directory / "fixture.txt").write_text("synthetic figure", encoding="utf-8")
        write_json(directory / "report.json", report)
        with zipfile.ZipFile(directory.parent / "poster-assets.zip", "w") as archive:
            archive.write(directory / "fixture.txt", "figures/fixture.txt")
            archive.writestr("report.json", json.dumps(report))
        return {"files": [{"path": "figures/fixture.txt", "label": "Fixture"}], "bundle": "poster-assets.zip"}

    def run_fixture(self, renderer=None):
        metadata = {"source_hashes": {}, "public_tiles_used": 4, "public_manifest_tiles": 12,
                    "public_catalog_tiles_by_region": {"fixture": 12}}
        with patch.object(storage, "ROOT", self.root / "poster"), \
                patch.object(training, "load_data", return_value={"tiles": copy.deepcopy(self.tiles), "metadata": metadata}), \
                patch.object(assistant, "save", side_effect=AssertionError("No model writes")), \
                patch.object(assistant, "load", side_effect=AssertionError("No active model reads")), \
                patch.object(np, "savez_compressed", side_effect=AssertionError("No saved arrays")), \
                patch.object(assistant, "fit", wraps=assistant.fit) as fit:
            result = training.run(public_root=self.root / "public", reference_root=self.root / "reference",
                                  raw_root=self.root / "raw", renderer=renderer or self.renderer, progress=None)
        self.assertEqual(fit.call_count, 3)
        return result

    def test_results_figures_and_provenance_stay_in_poster(self):
        directory, report = self.run_fixture()
        self.assertTrue(directory.is_relative_to((self.root / "poster").resolve()))
        self.assertEqual(data._json(directory / "results.json"), report)
        self.assertTrue(report["provenance"]["samples"])
        self.assertEqual(report["training"]["tiles"], 4)
        self.assertEqual(report["training"]["geography_groups"], 3)
        self.assertTrue((directory / "summary.md").is_file())
        self.assertTrue((directory / "poster-assets.zip").is_file())
        self.assertFalse(self.models.exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(list(directory.rglob("*.npz")))
        self.assertFalse(list(directory.rglob("*.joblib")))
        self.assertFalse(list(directory.rglob("*.tmp")))

    def test_real_renderer_preserves_full_labels_and_consistent_json(self):
        from poster import reference_figures
        tile = self.tiles[0]
        tile["valid"][:, 100:] = False
        tile["eligible"][:, 100:] = False
        tile["X"][:, :, 100:] = np.nan
        tile["hillshade"][:, 100:] = np.nan
        image = np.asarray(reference_figures._map_images(tile)[0])
        expected = (np.asarray(reference_figures.CLASS_COLOURS[3]) * (0.78 + 0.22 * 0.65)).astype(np.uint8)
        np.testing.assert_array_equal(image[20, 110], expected)
        directory, report = self.run_fixture(renderer=reference_figures.export_figures)
        self.assertTrue(all((directory / item["path"]).is_file() for item in report["exports"]["files"]))
        self.assertEqual(data._json(directory / "figures/report.json"), report)
        with zipfile.ZipFile(directory / report["exports"]["bundle"]) as archive:
            self.assertEqual(json.loads(archive.read("report.json")), report)
        self.assertIn("python", report["software_versions"])
        self.assertGreaterEqual(report["timings_seconds"]["figure_export"], 0)

    def test_existing_user_model_unchanged_and_runs_unique(self):
        self.models.mkdir()
        target = self.models / "rf.joblib"
        target.write_bytes(b"existing user model")
        before = target.read_bytes(), target.stat().st_mtime_ns
        first, _ = self.run_fixture()
        second, _ = self.run_fixture()
        self.assertNotEqual(first, second)
        self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)
        self.assertTrue((first / "results.json").is_file())
        self.assertTrue((second / "results.json").is_file())

    def test_failed_export_does_not_publish_results_or_change_completed_run(self):
        directory, _ = self.run_fixture()
        before = (directory / "results.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "render failed"):
            self.run_fixture(renderer=Mock(side_effect=RuntimeError("render failed")))
        self.assertEqual((directory / "results.json").read_bytes(), before)
        self.assertEqual(list(directory.parent.glob("*/results.json")), [directory / "results.json"])
        self.assertFalse(self.models.exists())

    def test_empty_renderer_still_publishes_results_and_summary(self):
        directory, report = self.run_fixture(renderer=lambda *args: {})
        self.assertTrue((directory / "summary.md").is_file())
        self.assertTrue((directory / "results.json").is_file())
        self.assertFalse((directory / "figures").exists())
        self.assertNotIn("bundle", report["exports"])

    def test_output_cannot_escape_poster_or_write_into_source_data(self):
        root = self.root / "poster"
        root.mkdir()
        (root / "escape").symlink_to(self.root, target_is_directory=True)
        with patch.object(storage, "ROOT", root), \
                patch.object(training, "load_data", side_effect=AssertionError("must fail before source access")):
            for destination in (self.output, self.models, root, root / "escape/results"):
                with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "inside poster"):
                    training.run(output_root=destination, renderer=self.renderer)
            with self.assertRaisesRegex(ValueError, "immutable source"):
                training.run(public_root=root / "data", output_root=root / "data/results")

    def test_cli_has_no_model_install_option(self):
        with patch.object(training, "run") as run:
            training.main([])
            run.assert_called_once_with(output_root=None)
            run.reset_mock()
            training.main(["--output", "poster/results/example"])
            run.assert_called_once_with(output_root=Path("poster/results/example"))


if __name__ == "__main__":
    unittest.main()
