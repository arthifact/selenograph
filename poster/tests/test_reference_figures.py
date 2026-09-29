"""Synthetic offline poster tests; no reference data, pipeline or persistent artifacts."""
import copy
import csv
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from rasterio.crs import CRS

from poster import reference_figures as figures
from poster.benchmark_data import GRID, _expand
from app.core import per_pixel

MOON = CRS.from_string("+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m")
FEATURES = ["slope", "rough", "svf", "rel_local", "curv", "svf_local"]


def metrics(truth, prediction):
    cm = np.array([[np.count_nonzero((truth == a) & (prediction == b))
                    for b in (1, 2, 3)] for a in (1, 2, 3)])
    result = {"n": int(cm.sum()), "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else None,
              "confusion_matrix": cm.tolist(), "per_class": {}}
    for index, code in enumerate((1, 2, 3)):
        tp, actual, predicted = int(cm[index, index]), int(cm[index].sum()), int(cm[:, index].sum())
        result["per_class"][str(code)] = {
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / actual if actual else 0.0,
            "f1": 2 * tp / (actual + predicted) if actual + predicted else 0.0,
            "iou": tp / (actual + predicted - tp) if actual + predicted - tp else 0.0,
            "support": actual,
        }
    result["macro_f1"] = float(np.mean([p["f1"] for p in result["per_class"].values()]))
    return result


def make_tile(index=0, shape=(137, 181)):
    groups = ("mons-mouton", "mons-mouton", "nobile-1", "nobile-1", "nobile-2", "nobile-2")
    sites = ("mons-mouton", "mons-mouton", "nobile1", "nobile1-ms1", "nobile2", "nobile2")
    maps = ("MM", "MM", "N1014", "MS1", "N2", "N2")
    rows, cols = np.indices((GRID, GRID))
    cells = ((rows // 35 + cols // 40 + index) % 3 + 1).astype(np.uint8)
    row_widths, col_widths = (np.diff(np.arange(GRID + 1) * n // GRID) for n in shape)
    eligible = (row_widths[:, None] > 0) & (col_widths[None, :] > 0)
    eligible[0:8, 0:10] = False
    cell_prediction = cells.copy()
    changed = ((rows + cols + index) % (5 + index)) == 0
    cell_prediction[changed] = cell_prediction[changed] % 3 + 1
    reference = per_pixel(cells, shape).copy()
    reference[-3:, -3:] = 255
    prediction = per_pixel(cell_prediction, shape).copy()
    # A genuinely finer pattern ensures maps aren't just expanded reference cells.
    prediction[::7, ::5] = prediction[::7, ::5] % 3 + 1
    rr, cc = np.indices(shape)
    valid = np.ones(shape, bool)
    valid[:, :index + 2] = False
    valid[-2:, :] = False
    prediction[~valid] = 255
    confidence = (0.34 + 0.66 * cc / max(1, shape[1] - 1)).astype(np.float32)
    confidence[~valid] = np.nan
    support = valid.copy()
    support[30:60, 35:90] = False
    X = np.stack([(np.sin(rr / (7 + i)) + cc / (25 + i)) * (i + 1)
                  for i in range(6)]).astype(np.float32)
    X[:, ~valid] = np.nan
    # Synthetic lunar-grid metadata, not external data or a claimed real location.
    x, y = -80_000 + index * 1700, 60_000 - index * 3100
    transform = [5.0, 0.0, x, 0.0, -5.0, y]
    return {
        "tile_id": f"synthetic-{index + 1}", "map_id": maps[index], "site_id": sites[index],
        "group": groups[index], "snapshot": "saved" if index % 2 else "draft",
        "grid": {"shape": list(shape), "transform": transform, "crs_wkt": MOON.to_wkt(),
                 "bounds": [x, y - shape[0] * 5, x + shape[1] * 5, y]},
        "px_m": 5.0, "X": X, "hillshade": (0.6 + 0.35 * np.sin(rr / 11)).astype(np.float32),
        "reference": reference, "cells": cells, "eligible": eligible, "valid": valid,
        "support": support, "prediction": prediction, "confidence": confidence,
        "cell_prediction": cell_prediction,
    }


def make_report(tiles):
    groups = sorted({t["group"] for t in tiles})
    records = []
    for tile in tiles:
        mask = tile["eligible"]
        records.append({k: copy.deepcopy(tile[k]) for k in ("tile_id", "map_id", "site_id", "group", "snapshot", "px_m", "grid")})
        records[-1].update(eligible_cells=int(mask.sum()), reference_terrain_pixels=int(np.count_nonzero(tile["reference"] != 255)),
                           public_coverage_fraction=float(tile["valid"].mean()), exclusions={"synthetic": int((~mask).sum())},
                           metrics=metrics(tile["cells"][mask], tile["cell_prediction"][mask]))
    folds, all_truth, all_prediction, all_baseline = [], [], [], []
    for group in groups:
        train = [t for t in tiles if t["group"] != group]
        test = [t for t in tiles if t["group"] == group]
        truth = np.concatenate([t["cells"][t["eligible"]] for t in test])
        prediction = np.concatenate([t["cell_prediction"][t["eligible"]] for t in test])
        if train:
            train_truth = np.concatenate([t["cells"][t["eligible"]] for t in train])
            majority = int(np.bincount(train_truth, minlength=4)[1:].argmax() + 1)
        else:
            majority = 1  # Synthetic empty-training edge case, not a measured result.
        baseline = np.full(truth.shape, majority, np.uint8)
        folds.append({"group": group, "train_groups": [g for g in groups if g != group],
                      "train_tiles": [t["tile_id"] for t in train], "test_tiles": [t["tile_id"] for t in test],
                      "sampled_pixels": int(sum(t["valid"].sum() for t in train)),
                      "metrics": metrics(truth, prediction), "baseline": metrics(truth, baseline), "majority_class": majority})
        all_truth.append(truth)
        all_prediction.append(prediction)
        all_baseline.append(baseline)
    truth = np.concatenate(all_truth)
    return {
        "run_id": "synthetic-only", "created_at": "synthetic timestamp", "schema": {"classes": [1, 2, 3]},
        "feature_names": FEATURES, "model_revision": "synthetic-revision",
        "training": {"tiles": len(tiles), "map_groups": len({t["map_id"] for t in tiles}), "geography_groups": len(groups),
                     "sampled_pixels": int(sum(t["valid"].sum() for t in tiles)),
                     "samples_by_tile": {t["tile_id"]: int(t["valid"].sum()) for t in tiles}},
        "validation": {"method": "Synthetic group holdout <&> OOF", "aggregate": metrics(truth, np.concatenate(all_prediction)),
                       "group_mean_macro_f1": float(np.mean([f["metrics"]["macro_f1"] for f in folds])),
                       "baseline_aggregate": metrics(truth, np.concatenate(all_baseline)), "folds": folds},
        "tiles": records, "policies": {"review": "all unreviewed", "support": "not a scoring exclusion"},
        "limitations": ["Synthetic fixture only; no scientific performance claims."],
    }


class RenderingTests(unittest.TestCase):
    def test_reference_extent_missing_outputs_and_unscored_hatching(self):
        tile = make_tile(shape=(240, 360))
        tile["hillshade"][:] = 1
        tile["eligible"][:] = True
        tile["eligible"][15:20, 15:20] = False
        tile["prediction"][80, 80] = 255
        tile["confidence"][100, 100] = np.nan
        reference, prediction, difference, confidence = map(np.asarray, figures._map_images(tile))
        # Labels outside the public terrain are not lost in the reference panel.
        self.assertFalse(np.array_equal(reference[40, 0], figures.WHITE))
        for values in (prediction, difference, confidence):
            np.testing.assert_array_equal(values[40, 0], figures.WHITE)
            np.testing.assert_array_equal(values[80, 80], figures.WHITE)
        np.testing.assert_array_equal(confidence[100, 100], figures.WHITE)
        self.assertFalse(np.array_equal(prediction[100, 100], figures.WHITE))
        np.testing.assert_array_equal(reference[-1, -1], figures.UNKNOWN)
        scored = _expand(tile["eligible"], tile["reference"].shape)
        present = tile["valid"] & (tile["prediction"] != 255)
        unscored = difference[present & ~scored]
        self.assertTrue(len(np.unique(unscored, axis=0)) > 1)
        self.assertFalse(np.any(np.all(unscored == figures.AGREE, axis=1)))
        # Low support must not be silently screened out.
        low_support = present & ~tile["support"] & scored
        self.assertTrue(low_support.any())
        self.assertFalse(np.any(np.all(prediction[low_support] == figures.WHITE, axis=1)))

    def test_difference_is_exact_cell_expansion_not_pixel_disagreement(self):
        tile = make_tile(shape=(241, 367))
        tile["valid"][:] = True
        tile["eligible"][:] = True
        tile["cells"][:] = 1
        tile["reference"][:] = 1
        tile["prediction"][:] = 3  # All pixels differ, but this is not the cell score.
        tile["cell_prediction"][:] = 1
        tile["cell_prediction"][4, 9] = 2
        scored, equal, n, agreement = figures._cell_comparison(tile)
        self.assertEqual(n, GRID * GRID)
        self.assertEqual(agreement, (GRID * GRID - 1) / (GRID * GRID))
        with patch.object(figures, "per_pixel", wraps=per_pixel) as expand:
            difference = np.asarray(figures._map_images(tile)[2])
        self.assertGreaterEqual(expand.call_count, 2)
        expected = _expand(~equal, tile["reference"].shape)
        np.testing.assert_array_equal(np.all(difference == figures.DISAGREE, axis=2), expected)
        np.testing.assert_array_equal(np.all(difference == figures.AGREE, axis=2), ~expected)
        self.assertTrue(scored.all())

    def test_cell_prediction_vector_and_no_scored_cells(self):
        tile = make_tile()
        expected = tile["cell_prediction"][tile["eligible"]].copy()
        tile["cell_prediction"] = expected
        np.testing.assert_array_equal(figures._cell_predictions(tile)[tile["eligible"]], expected)
        tile["eligible"][:] = False
        tile["cell_prediction"] = np.empty(0, np.uint8)
        self.assertEqual(figures._cell_comparison(tile)[2:], (0, None))
        images = figures._map_images(tile)
        self.assertEqual(len(images), 4)
        self.assertNotIn(figures.AGREE, map(tuple, np.asarray(images[2]).reshape(-1, 3)))
        tile["cell_prediction"] = np.ones(12, np.uint8)
        with self.assertRaisesRegex(ValueError, "eligible-cell vector"):
            figures._cell_predictions(tile)

    def test_scale_bar_accounts_for_affine_spacing_and_display_resize(self):
        for source_width, display_width, px in ((1024, 1225, 5), (181, 721, 7.5), (120, 510, 12)):
            distance, length = figures._scale_bar(source_width, display_width, px)
            self.assertAlmostEqual(length / display_width * source_width * px, distance)
            self.assertLessEqual(length, display_width * 0.27 + 1e-9)
        tile = make_tile(shape=(1024, 1024))
        self.assertEqual(figures._cell_size_text(tile), "40/45 m across x 40/45 m along rows")
        tile["grid"]["transform"] = [3, 4, 100, 4, -3, 200]
        tile["px_m"] = 999  # The affine grid, not a stale nominal scalar, controls scale.
        self.assertEqual(figures._spacing(tile), (5, 5))

    def test_tiny_floor_geometry_preserves_original_cells_and_omits_zero_sizes(self):
        cells = np.arange(GRID * GRID).reshape(GRID, GRID)
        for shape in ((1, 1), (2, 7), (17, 29), (63, 181), (241, 367)):
            with self.subTest(shape=shape):
                rows, cols = figures._edges(shape)
                expected = np.repeat(np.repeat(cells, np.diff(rows), axis=0), np.diff(cols), axis=1)
                np.testing.assert_array_equal(per_pixel(cells, shape), expected)
                self.assertEqual(expected.shape, shape)
                if min(shape) >= GRID:
                    np.testing.assert_array_equal(expected, _expand(cells, shape))
                tile = make_tile(shape=shape)
                before = tile["cells"].copy()
                tile["valid"][:] = True
                tile["prediction"][:] = 1
                scored, equal, _, _ = figures._cell_comparison(tile)
                difference = np.asarray(figures._map_images(tile)[2])
                expected_difference = per_pixel(scored & ~equal, shape)
                np.testing.assert_array_equal(np.all(difference == figures.DISAGREE, axis=2), expected_difference)
                np.testing.assert_array_equal(tile["cells"], before)
                self.assertEqual(tile["cells"].shape, (GRID, GRID))
                self.assertNotRegex(figures._cell_size_text(tile), r"\b0(?:/| m)")
        self.assertEqual(int(per_pixel(cells, (1, 1))[0, 0]), int(cells[-1, -1]))
        self.assertEqual(figures._cell_size_text(make_tile(shape=(17, 29))), "5 m across x 5 m along rows")
        self.assertEqual(figures._cell_size_text(make_tile(shape=(63, 181))), "5/10 m across x 5 m along rows")

    def test_restriction_and_review_labels_are_report_scoped(self):
        tile = make_tile()
        tile["snapshot"] = "manual"
        public = make_report([tile])
        public["restricted"] = False
        public["policies"] = {}
        private = copy.deepcopy(public)
        private["restricted"] = True
        private["limitations"] = ["Restricted model seeds were used with user map labels."]
        for report, restriction in ((public, figures.PUBLIC_NOTICE), (private, figures.RESTRICTION),
                                    (make_report([tile]), figures.RESTRICTION), (public, figures.PUBLIC_NOTICE)):
            with self.subTest(restricted=report.get("restricted", "legacy")):
                _, svg = figures._score_figures(report)
                text = " ".join(ET.fromstring(svg).itertext())
                self.assertIn(restriction, text)
                self.assertEqual("Unreviewed reference snapshots" in text, "restricted" not in report)
                self.assertEqual("Earlier feature selection" in text, "restricted" not in report)
                if report.get("restricted") is False:
                    self.assertNotIn("CUI", text)
        self.assertEqual(figures._source_label(tile, private), ("Map labels / manual", "review status not supplied"))
        private["tiles"][0]["source_label"] = "User annotations"
        private["tiles"][0]["review_status"] = "author-reviewed"
        self.assertEqual(figures._source_label(tile, private), ("User annotations / manual", "author-reviewed"))
        self.assertEqual(figures.RESTRICTION, "CUI / LOCAL-ONLY  |  Publication permission required")

    def test_arbitrary_spatial_group_counts_drive_chart_and_footprint_layout(self):
        tiles = []
        for index in range(7):
            tile = make_tile(index % 6, shape=(23, 31))
            tile.update(tile_id=f"selected-{index}", map_id=f"map-{index}", group=f"spatial-{index}")
            tiles.append(tile)
        report = make_report(tiles)
        report["restricted"] = False
        figures._validate(tiles, report)
        score_image, svg = figures._score_figures(report)
        text = " ".join(ET.fromstring(svg).itertext())
        self.assertIn("7 held-out spatial groups", text)
        self.assertNotIn("3 held-out", text)
        self.assertGreater(score_image.height, 2400)
        with patch.object(figures, "_page", wraps=figures._page) as page:
            footprint_image = figures._location_figure(tiles, figures._footprints(tiles), report)
        self.assertIn("7 raster windows  /  7 map groups  /  7 spatial groups", page.call_args.args[3])
        self.assertGreater(footprint_image.height, 2600)

    def test_feature_channels_constant_missing_and_score_independent_selection(self):
        values = np.full((120, 170), 7.0, np.float32)
        valid = np.ones(values.shape, bool)
        valid[:, 0] = False
        picture, extent = figures._feature_picture(values, valid)
        self.assertEqual(extent, (7, 7))
        np.testing.assert_array_equal(np.asarray(picture)[50, 50], figures.RAMP[1])
        np.testing.assert_array_equal(np.asarray(picture)[50, 0], figures.WHITE)
        picture, extent = figures._feature_picture(np.full_like(values, np.nan), valid)
        self.assertIsNone(extent)
        self.assertTrue(np.all(np.asarray(picture) == 255))
        tiles = [make_tile(i) for i in range(6)]
        chosen = figures._representative(tiles)["tile_id"]
        for tile in tiles:
            tile["prediction"][:] = 255
            tile["confidence"][:] = np.nan
            tile["cell_prediction"][:] = 3
        self.assertEqual(figures._representative(list(reversed(tiles)))["tile_id"], chosen)
        tiles[0]["X"][:] = np.nan
        self.assertNotEqual(figures._representative(tiles)["tile_id"], tiles[0]["tile_id"])

    def test_footprints_use_actual_rotated_lunar_affine_and_reject_earth(self):
        tile = make_tile()
        tile["grid"]["transform"] = [3, 4, 12345, 4, -3, -67890]
        crs, outlines = figures._footprints([tile])
        self.assertEqual(crs, CRS.from_wkt(tile["grid"]["crs_wkt"]))
        h, w = tile["reference"].shape
        for col, row in ((0, 0), (w, 0), (w, h), (0, h)):
            expected = (3 * col + 4 * row + 12345, 4 * col - 3 * row - 67890)
            self.assertTrue(np.any(np.all(np.isclose(outlines[0], expected), axis=1)))
        tile["grid"]["crs_wkt"] = CRS.from_epsg(3031).to_wkt()
        with self.assertRaisesRegex(ValueError, "lunar polar"):
            figures._footprints([tile])

    def test_font_fallback_and_strict_json_metadata(self):
        figures._font.cache_clear()
        with patch.object(figures, "_font_path", return_value=None):
            self.assertIsNotNone(figures._font(23))
        figures._font.cache_clear()
        self.assertEqual(figures._json_value({"a": np.int64(7), "b": np.nan}), {"a": 7, "b": None})
        with self.assertRaisesRegex(ValueError, "raster arrays"):
            figures._json_value(np.zeros((120, 120)))

    def test_invalid_inputs_fail_before_creating_outputs(self):
        tiles = [make_tile(i) for i in range(6)]
        report = make_report(tiles)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp).resolve() / "figures"
            with self.assertRaisesRegex(ValueError, "At least one"):
                figures.export_figures([], report, out)
            invalid = copy.deepcopy(report)
            invalid["feature_names"] = ["slope"]
            with self.assertRaisesRegex(ValueError, "six"):
                figures.export_figures(tiles, invalid, out)
            invalid = copy.deepcopy(report)
            invalid["validation"]["folds"][0]["train_groups"].append(invalid["validation"]["folds"][0]["group"])
            with self.assertRaisesRegex(ValueError, "held-out"):
                figures.export_figures(tiles, invalid, out)
            tile = copy.deepcopy(tiles[0])
            tile["X"] = tile["X"][:, :-1]
            with self.assertRaisesRegex(ValueError, "X must"):
                figures.export_figures([tile], report, out)
            self.assertFalse(out.exists())

    def test_group_validation_uses_supplied_field_not_site_aliases(self):
        tiles = [make_tile(2), make_tile(3)]
        tiles[1]["group"] = "caller-defined-spatial-group"
        report = make_report(tiles)
        report["restricted"] = False
        figures._validate(tiles, report)
        # Site/map names do not override group metadata, but fold/group mismatches fail.
        tiles[1]["group"] = tiles[0]["group"]
        with self.assertRaisesRegex(ValueError, "held-out"):
            figures._validate(tiles, report)

    def test_repeated_group_folds_and_same_geography_training_are_rejected(self):
        tiles = [make_tile(i) for i in range(6)]
        report = make_report(tiles)
        report["validation"]["folds"].append(copy.deepcopy(report["validation"]["folds"][0]))
        with self.assertRaisesRegex(ValueError, "one held-out fold"):
            figures._validate(tiles, report)
        report = make_report(tiles)
        fold = report["validation"]["folds"][0]
        fold["train_tiles"].append(fold["test_tiles"][1])
        with self.assertRaisesRegex(ValueError, "held-out geography appears in training"):
            figures._validate(tiles, report)

    def test_score_svg_preserves_report_values_and_escapes_text(self):
        report = make_report([make_tile(i) for i in range(6)])
        report["validation"]["aggregate"]["per_class"]["3"]["support"] = 0
        report["validation"]["aggregate"]["accuracy"] = 0.123456
        report["validation"]["aggregate"]["macro_f1"] = 0.456789
        report["validation"]["group_mean_macro_f1"] = 0.654321
        _, svg = figures._score_figures(report)
        root = ET.fromstring(svg)
        text = " ".join(root.itertext())
        self.assertIn("12.3%", text)
        self.assertIn("0.457", text)
        self.assertIn("0.654", text)
        self.assertIn("--", text)
        self.assertIn(report["validation"]["method"], text)
        self.assertIn("Crater interiors / floor", text)
        self.assertNotIn("<image", svg)
        self.assertNotIn("http://", svg.replace("http://www.w3.org/2000/svg", ""))
        self.assertEqual(len(root.findall("{http://www.w3.org/2000/svg}image")), 0)


class ExportTests(unittest.TestCase):
    def test_six_tile_export_api_dpi_bundle_captions_metrics_and_read_only_inputs(self):
        tiles = [make_tile(i, shape=(127 + i * 2, 173 + i * 3)) for i in range(6)]
        report = make_report(tiles)
        original_report = copy.deepcopy(report)
        originals = [{key: value.copy() for key, value in tile.items() if isinstance(value, np.ndarray)} for tile in tiles]
        for tile in tiles:
            for value in tile.values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
        with tempfile.TemporaryDirectory(prefix="reference-figures-test-") as temp:
            run = Path(temp).resolve()
            out = run / "figures"
            out.mkdir()
            sentinel = out / "existing-source.tif"
            sentinel.write_bytes(b"must not modify or archive this raster")
            with patch("socket.create_connection", side_effect=AssertionError("No network allowed")):
                result = figures.export_figures(reversed(tiles), report, out)
            self.assertEqual(set(result), {"files", "bundle"})
            self.assertEqual(result["bundle"], "poster-assets.zip")
            pngs = [item for item in result["files"] if item["path"].endswith(".png")]
            self.assertEqual(len(pngs), len(tiles) + 3)
            self.assertEqual(len(result["files"]), len(tiles) + 4)
            for item in result["files"]:
                self.assertEqual(set(item), {"path", "label"})
                self.assertTrue(item["path"].startswith("figures/"))
                self.assertTrue(item["label"])
                self.assertTrue((run / item["path"]).is_file())
            for item in pngs:
                with Image.open(run / item["path"]) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertGreaterEqual(min(image.size), 2400)
                    for dpi in image.info["dpi"]:
                        self.assertAlmostEqual(dpi, 300, delta=0.02)
                    # Not a blank placeholder: ink/terrain and paper are both present.
                    extrema = np.asarray(image.getextrema())
                    self.assertTrue(np.all(extrema[:, 0] < 100))
                    self.assertTrue(np.all(extrema[:, 1] == 255))
            names = [Path(item["path"]).name for item in pngs]
            self.assertEqual(names[:6], [f"tile-{i:02d}-synthetic-{i}.png" for i in range(1, 7)])
            self.assertIn("feature-montage.png", names)
            self.assertIn("scores-by-group-class.png", names)
            self.assertIn("reference-footprints.png", names)
            caption = (out / "captions.md").read_text(encoding="utf-8")
            for required in ("CUI", "LOCAL-ONLY", "Publication permission required", "unreviewed",
                             "Earlier feature selection", "not an untouched final test", "3 held-out spatial groups",
                             "no full-resolution geological truth", "Missing reference vectors are not reconstructed",
                             "MS1 + N1014 held out together", "shadowed_floor", "NOT PSR / ice", "uncalibrated", "not geological error",
                             "benchmark_data._expand", "core.per_pixel", "not silently removed", "6 tiles, 4 map groups, 3 spatial groups"):
                self.assertIn(required, caption)
            for tile in tiles:
                self.assertIn(tile["tile_id"], caption)
            self.assertEqual(json.loads((out / "report.json").read_text()), report)
            csv_rows = list(csv.DictReader(io.StringIO((out / "scores.csv").read_text())))
            pooled = [r for r in csv_rows if r["scope"] == "aggregate" and r["estimate"] == "OOF"]
            self.assertEqual(len(pooled), 3)
            self.assertEqual(float(pooled[0]["accuracy"]), report["validation"]["aggregate"]["accuracy"])
            class_three = pooled[2]
            self.assertEqual(class_three["class_label"], "Crater interiors / floor")
            self.assertEqual(float(class_three["recall"]), report["validation"]["aggregate"]["per_class"]["3"]["recall"])
            with zipfile.ZipFile(run / result["bundle"]) as archive:
                members = archive.namelist()
                self.assertEqual(len(set(members)), len(members))
                expected = {item["path"] for item in result["files"]} | {"captions.md", "report.json", "scores.csv"}
                self.assertEqual(set(members), expected)
                self.assertFalse(any(name.endswith((".tif", ".npy", ".npz")) for name in members))
                self.assertEqual(archive.read("captions.md").decode(), caption)
                for item in result["files"]:
                    self.assertEqual(archive.read(item["path"]), (run / item["path"]).read_bytes())
            self.assertEqual(sentinel.read_bytes(), b"must not modify or archive this raster")
            original_bundle = (run / result["bundle"]).read_bytes()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                figures.export_figures(tiles, report, out)
            self.assertEqual((run / result["bundle"]).read_bytes(), original_bundle)
        self.assertEqual(report, original_report)
        for tile, before in zip(tiles, originals):
            for key, expected in before.items():
                np.testing.assert_array_equal(tile[key], expected)

    def test_two_group_public_tiny_export_under_poster_has_generic_provenance(self):
        tiles = [make_tile(i, shape=shape) for i, shape in zip((0, 1, 4), ((17, 29), (63, 95), (37, 141)))]
        for index, (tile, snapshot) in enumerate(zip(tiles, ("manual", "saved", "draft"))):
            tile.update(tile_id=f"selected-{index}", map_id=f"user-map-{index}", site_id=f"location-{index}",
                        group="spatial-west" if index < 2 else "spatial-east", snapshot=snapshot)
        report = make_report(tiles)
        report["restricted"] = False
        report["sources"] = [{"label": "Synthetic public terrain", "restricted": False}]
        report["policies"] = {"review": "User map status is supplied per tile"}
        report["limitations"] = ["Manual labels have not been independently audited.",
                                 "Only two selected spatial groups are evaluated; generalisation elsewhere is unknown."]
        report["tiles"][1].update(source_label="User annotations", review_status="author-reviewed")
        before = copy.deepcopy(report)
        arrays = [{key: value.copy() for key, value in tile.items() if isinstance(value, np.ndarray)} for tile in tiles]
        with tempfile.TemporaryDirectory(prefix="selected-location-posters-test-") as temp:
            run = Path(temp).resolve() / "poster" / "selected-run"
            out = run / "figures"
            with patch.object(figures, "_text", wraps=figures._text) as text, \
                    patch("socket.create_connection", side_effect=AssertionError("No network allowed")):
                result = figures.export_figures(tiles, report, out)
            rendered = "\n".join(str(call.args[2]) for call in text.call_args_list)
            self.assertIn(figures.PUBLIC_NOTICE, rendered)
            self.assertNotIn("CUI", rendered)
            self.assertNotIn("unreviewed", rendered.lower())
            self.assertNotIn("Earlier feature selection", rendered)
            self.assertIn("A  Map labels / manual", rendered)
            self.assertIn("A  User annotations / saved", rendered)
            self.assertIn("A  Map labels / draft", rendered)
            self.assertIn("CELL-LEVEL / SPATIAL HELD-OUT EVALUATION", rendered)
            self.assertIn("2 held-out spatial groups", rendered)
            self.assertIn("3 raster windows / 3 map groups / 2 spatial groups", rendered)
            self.assertEqual(result["bundle"], "poster-assets.zip")
            self.assertEqual(len(result["files"]), len(tiles) + 4)
            for item in result["files"]:
                self.assertTrue(item["path"].startswith("figures/"))
                path = run / item["path"]
                self.assertTrue(path.is_file())
                if path.suffix == ".png":
                    with Image.open(path) as image:
                        self.assertGreaterEqual(min(image.size), 2400)
                        self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.02)
            caption = (out / "captions.md").read_text()
            for required in (figures.PUBLIC_NOTICE, "3 tiles, 3 map groups, 2 spatial groups", "2 held-out spatial groups",
                             "snapshot `manual`", "snapshot `saved`", "snapshot `draft`", "User annotations / saved",
                             "author-reviewed", "positive-width rasterized cells are 5 m across x 5 m along rows",
                             "core.per_pixel", "zero width or height", *report["limitations"]):
                self.assertIn(required, caption)
            for absent in ("CUI", "unreviewed", "N1014", "MS1", "3 held-out", "output/reference_training"):
                self.assertNotIn(absent, caption)
            self.assertEqual(json.loads((out / "report.json").read_text()), report)
            with zipfile.ZipFile(run / result["bundle"]) as archive:
                expected = {item["path"] for item in result["files"]} | {"captions.md", "report.json", "scores.csv"}
                self.assertEqual(set(archive.namelist()), expected)
                self.assertEqual(len(archive.namelist()), len(expected))
                self.assertEqual(archive.read("captions.md").decode(), caption)
            self.assertEqual([p.name for p in Path(temp).resolve().iterdir()], ["poster"])
        self.assertEqual(report, before)
        for tile, saved in zip(tiles, arrays):
            for key, value in saved.items():
                np.testing.assert_array_equal(tile[key], value)

    def test_safe_output_names_for_untrusted_tile_identifiers(self):
        tile = make_tile()
        tile["tile_id"] = "../../escape/<tile>"
        report = make_report([tile])
        # Exercise the public naming path without rendering another full poster set.
        image = Image.new("RGB", (8, 8), "white")
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(figures, "_tile_figure", return_value=image), \
                patch.object(figures, "_feature_figure", return_value=(image, [None] * 6)), \
                patch.object(figures, "_score_figures", return_value=(image, "<svg/>")), \
                patch.object(figures, "_location_figure", return_value=image):
            root = Path(temp).resolve()
            result = figures.export_figures([tile], report, root / "figures")
            for item in result["files"]:
                path = root / item["path"]
                self.assertTrue(path.resolve().is_relative_to(root / "figures"))
                self.assertNotIn("..", item["path"])
            self.assertIn("tile-01-escape-tile.png", result["files"][0]["path"])


if __name__ == "__main__":
    unittest.main()
