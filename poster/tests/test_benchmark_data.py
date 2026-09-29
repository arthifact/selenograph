"""Synthetic, offline extractor tests; fixture writes are confined to temporary roots."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.crs import CRS

from poster import benchmark_data as data

MOON = CRS.from_string("+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m")
TRANSFORM = rasterio.Affine(5, 0, 1000, 0, -5, 2000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def raster(path, values, transform=TRANSFORM, nodata=np.nan, crs=MOON):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", count=1, height=values.shape[0],
                       width=values.shape[1], dtype=values.dtype, transform=transform,
                       crs=crs, nodata=nodata) as dst:
        dst.write(values, 1)


def fake_build(path):
    with rasterio.open(path) as src:
        values = src.read(1)
        profile = {"height": src.height, "width": src.width, "transform": src.transform, "crs": src.crs}
    stack = np.stack([values + i for i in range(len(data.core.LAYERS))]).astype(np.float32)
    return stack, list(data.core.LAYERS), np.zeros(values.shape), profile, 5.0


class CellTests(unittest.TestCase):
    def test_floor_boundaries_and_40_45_m_means(self):
        rows, cols = data._edges((1024, 1024))
        self.assertEqual(set(np.diff(rows) * 5), {40, 45})
        self.assertEqual(int(np.diff(rows).sum()), 1024)
        cells = np.arange(14400, dtype=np.float32).reshape(120, 120)
        expanded = data._expand(cells, (1024, 1024))
        terrain = np.ones(expanded.shape, bool)
        np.testing.assert_array_equal(data._cell_mean(expanded, terrain), cells)
        counts = data._cell_sum(terrain)
        np.testing.assert_array_equal(counts, np.outer(np.diff(rows), np.diff(cols)))
        self.assertEqual(expanded[rows[1] - 1, 0], cells[0, 0])
        self.assertEqual(expanded[rows[1], 0], cells[1, 0])

    def test_channel_coverage_denominator_is_valid_reference_terrain(self):
        a = np.ones((240, 240), np.float32)
        terrain = np.ones(a.shape, bool)
        a[0, 0] = np.nan
        self.assertTrue(np.isnan(data._cell_mean(a, terrain)[0, 0]))
        terrain[0, 0] = False
        self.assertEqual(data._cell_mean(a, terrain)[0, 0], 1)

    def test_exclusion_order_and_unknown_not_certified(self):
        saved = np.ones((2, 4), np.uint8)
        draft = saved.copy()
        saved[0, 0], draft[0, 1] = 0, 0
        accepted = np.zeros(saved.shape, bool)
        accepted[0, 2] = True
        ground, area = np.full(saved.shape, 100), np.full(saved.shape, 100)
        ground[0, 3] = 89
        joint = ground.copy()
        joint[1, 0], joint[1, 1] = 89, 90
        keep, excluded = data._eligible(saved, draft, accepted, ground, area, joint)
        self.assertEqual(int(keep.sum()), 3)
        self.assertEqual(list(excluded.values()), [2, 1, 1, 1])
        self.assertTrue(keep[1, 1])

    def test_lunar_reprojection_unmapped_coverage_and_ownership(self):
        source = data._Grid((4, 4), TRANSFORM, MOON)
        target = data._Grid((4, 6), TRANSFORM, MOON)
        warped = data._warp(np.ones(source.shape, np.float32), source, target)
        self.assertTrue(np.all(warped[:, :4] == 1))
        self.assertTrue(np.isnan(warped[:, 4:]).all())
        ref = {"record": {"tile_id": "ref"}, "occupied": np.zeros(target.shape, bool)}
        footprint = warped == 1
        data._claim(ref, footprint, "a")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            data._claim(ref, footprint, "b")
        shifted = data._Grid((4, 4), TRANSFORM @ rasterio.Affine.translation(100, 0), MOON)
        self.assertFalse(data._intersects(shifted, target))

    def test_nac_statistics_nan_aware_and_constant_field(self):
        a = np.full((120, 120), 100, np.float32)
        a[50:55, 50:55] = np.nan
        features = data._image_features(a, 5)
        good = np.isfinite(a)
        for name in data.IMAGE[1:]:
            finite = np.isfinite(features[name])
            self.assertTrue(finite.any())
            np.testing.assert_allclose(features[name][finite], 0, atol=1e-5)
            self.assertTrue(np.isnan(features[name][~good]).all())
        self.assertTrue(np.isnan(features["nac_gradient"][49, 51]))

    def test_hash_validation_cached_once_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = root / "a"
            path.write_bytes(b"reference")
            hashes = data._Hashes()
            with patch.object(hashes, "stream", wraps=hashes.stream) as stream:
                hashes.file(path, "a", sha(path))
                hashes.file(path, "a", sha(path))
                self.assertEqual(stream.call_count, 1)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                hashes.file(path, "a", "0" * 64)
            with self.assertRaisesRegex(ValueError, "escapes"):
                data._path(root, "../elsewhere")

    def test_group_contract(self):
        self.assertEqual(data.GROUPS, {"mons-mouton": "mons-mouton", "nobile1": "nobile-1",
                                      "nobile1-ms1": "nobile-1", "nobile2": "nobile-2"})
        self.assertEqual(data.REGIONS["nobile1-ms1"], "nobile_rim_1")


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="selenograph-benchmark-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.public, self.reference, self.raw = [self.root / p for p in ("public", "reference", "raw")]
        self.site = "nobile1-ms1"
        self.region = "nobile_rim_1"
        self.tile_id = "reference-1"
        self.shape = (240, 240)
        self.terrain = np.ones(self.shape, bool)
        self.terrain[0, 0] = False
        self.folder = self.public / self.region
        self.reference.mkdir()
        self.raw.mkdir()
        self.saved = np.ones((120, 120), np.uint8)
        self.saved[0, 1] = 0
        self.draft = self.saved.copy()
        self.draft[0, 2], self.draft[0, 3] = 0, 2
        self._make_public()
        self._make_reference()

    def _make_public(self):
        layers, outputs = {}, {}
        for name in ("sfs", "nac", "valid_data", "sfs_support", "image_count", "solar_bins", "best_resolution"):
            relative = f"tile/{name}.tif"
            path = self.folder / relative
            value = np.full(self.shape, 1 if name in ("valid_data", "image_count") else 10, np.float32)
            if name == "sfs_support":
                value[:] = 0
            if name in ("nac", "valid_data"):
                value[0:2, 10:12] = np.nan if name == "nac" else 0
            if name == "solar_bins":
                value[2, 0] = np.nan
            raster(path, value)
            layers[name] = relative
            outputs[relative] = {"checksum": {"sha256": sha(path)}}
        self.public_record = {"tile_id": "public-1", "collection": "public", "site": self.region,
                              "layers": layers, "working_dem": layers["sfs"]}
        write_json(self.folder / "working_dems.json", [self.public_record])
        outputs["working_dems.json"] = {"checksum": {"sha256": sha(self.folder / "working_dems.json")}}
        self.public_meta = {"grid": data._Grid(self.shape, TRANSFORM, MOON).metadata(),
                            "outputs": outputs, "sources": {"nac": {"native": {"grid": {"spacing": [10, 10]}}}}}
        self._publish_public_meta()

    def _publish_public_meta(self):
        path = self.folder / "metadata" / f"{self.region}_metadata.json"
        write_json(path, self.public_meta)
        write_json(self.public / "manifest.json", {"regions": {self.region: {"metadata_sha256": sha(path)}}})

    def _make_reference(self, variants=("saved", "draft"), accepted=True, radar=False):
        dem_path = self.reference / "dem.tif"
        raster(dem_path, np.where(self.terrain, 100, -9999).astype(np.float32), nodata=-9999)
        annotations = []
        for variant in variants:
            cells = self.saved if variant == "saved" else self.draft
            painting = self.reference / f"{variant}.npy"
            np.save(painting, cells, allow_pickle=False)
            files = [{"role": "painting", "path": painting.name, "sha256": sha(painting)}]
            if accepted and variant == "draft":
                flags = np.zeros(cells.shape, bool)
                flags[0, 4] = True
                path = self.reference / "accepted.npy"
                np.save(path, flags, allow_pickle=False)
                files.append({"role": "accepted", "path": path.name, "sha256": sha(path)})
            labels = data._expand(cells, self.shape).astype(np.uint16)
            labels[~self.terrain] = 65535
            label_path = self.reference / f"{variant}.tif"
            raster(label_path, labels, nodata=65535)
            provenance = {"schema_id": "selenograph-geologic-units-v1", "files": files,
                          "private_map_id": "Nobile1-MS1", "review_status": "unreviewed_snapshot"}
            write_json(self.reference / f"{variant}.json", provenance)
            annotations.append({"snapshot_variant": variant, "path": label_path.name,
                                "sha256": sha(label_path), "provenance": f"{variant}.json"})
        layers: dict[str, dict[str, Any]] = {"dem": {"path": "dem.tif", "sha256": sha(dem_path)}}
        if radar:
            for name in ("radar-s1", "radar-cpr"):
                path = self.reference / f"{name}.tif"
                values = np.full(self.shape, 3, np.float32)
                values[2:4, :2] = -2 if name == "radar-s1" else 1e35
                raster(path, values)
                layers[name] = {"path": path.name, "sha256": sha(path), "native_pixel_size_m": [118.45, 118.45]}
        self.reference_record = {"tile_id": self.tile_id, "site_id": self.site, "annotations": annotations,
                                 "layers": layers, "legacy": {"preferred_seed": "draft"}}
        write_json(self.reference / "tile.json", self.reference_record)
        self._publish_reference()

    def _publish_reference(self):
        files = sorted(p for p in self.reference.iterdir() if p.is_file() and p.name != "dataset.json")
        manifest = {"tiles": [{"tile_id": self.tile_id, "site_id": self.site, "path": "tile.json"}],
                    "artifacts": [{"path": p.name, "sha256": sha(p)} for p in files]}
        write_json(self.reference / "dataset.json", manifest)

    def load(self, **kwargs):
        with patch.object(data.core, "build", side_effect=fake_build) as build:
            result = data.load_data(self.public, self.reference, self.raw, **kwargs)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(Path(build.call_args.args[0]), self.folder / "tile/sfs.tif")
        return result

    def test_contract_exclusions_support_retention_and_read_only(self):
        before = {p: (p.stat().st_mtime_ns, sha(p)) for p in self.root.rglob("*") if p.is_file()}
        messages = []
        result = self.load(progress=messages.append)
        after = {p: (p.stat().st_mtime_ns, sha(p)) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["X"].shape, (14395, 23))
        self.assertEqual(result["X"].dtype, np.float32)
        self.assertEqual(result["y_saved"].dtype, np.uint8)
        self.assertEqual(result["y_draft"].dtype, np.uint8)
        self.assertEqual(result["support_fraction"].dtype, np.float32)
        self.assertTrue(np.all(result["support_fraction"] == 0))
        self.assertEqual(set(result["groups"]), {"nobile-1"})
        self.assertEqual(result["groups"].dtype.kind, "U")
        self.assertEqual(len(set(result["sample_ids"])), len(result["sample_ids"]))
        for col in (0, 1, 2, 4, 5):
            self.assertNotIn(f"reference-1:0:{col}", result["sample_ids"])
        row = np.flatnonzero(result["sample_ids"] == "reference-1:0:3")[0]
        self.assertEqual((result["y_saved"][row], result["y_draft"][row]), (1, 2))
        self.assertEqual(result["feature_names"], data.FEATURE_NAMES)
        meta = result["metadata"]
        self.assertEqual(meta["snapshot_differences_kept"], 1)
        self.assertEqual(meta["references"][0]["snapshots"]["saved"]["accepted_observed"], False)
        self.assertIsNone(meta["references"][0]["snapshots"]["saved"]["confidence_counts"])
        self.assertEqual(meta["references"][0]["saved_variant_used"], "saved")
        self.assertEqual(meta["references"][0]["kept_support_below_90pct"], 14395)
        self.assertEqual(meta["public_regions"][self.region]["sources"]["nac"]["native"]["grid"]["spacing"], [10, 10])
        self.assertTrue(messages)
        json.dumps(meta, allow_nan=False)
        for name in ["uncertainty", *data.RADAR, *data.LOLA]:
            self.assertTrue(np.isnan(result["X"][:, data.FEATURE_NAMES.index(name)]).all())
            self.assertFalse(meta["feature_availability"][name])
        row = np.flatnonzero(result["sample_ids"] == "reference-1:1:0")[0]
        self.assertTrue(np.isnan(result["X"][row, data.FEATURE_NAMES.index("solar_bins")]))

    def test_snapshot_fallback_missing_flags_kept_as_unknown(self):
        self._make_reference(variants=("draft",), accepted=False)
        result = self.load()
        np.testing.assert_array_equal(result["y_saved"], result["y_draft"])
        self.assertIn("reference-1:0:4", result["sample_ids"])
        meta = result["metadata"]["references"][0]
        self.assertEqual(meta["saved_variant_used"], "draft")
        self.assertFalse(meta["snapshots"]["draft"]["accepted_observed"])
        self.assertFalse(meta["snapshots"]["draft"]["confidence_observed"])

    def test_features_do_not_use_label_values(self):
        first = self.load()
        self.saved[10:20, 10:20] = 3
        self.draft[10:20, 10:20] = 2
        self._make_reference()
        second = self.load()
        np.testing.assert_array_equal(first["sample_ids"], second["sample_ids"])
        np.testing.assert_array_equal(first["X"], second["X"])
        self.assertFalse(np.array_equal(first["y_saved"], second["y_saved"]))

    def test_optional_radar_and_bounded_lola(self):
        self._make_reference(radar=True)
        path = self.raw / "future-experiments/LDEM_83S_10MPP_ADJ.tiff"
        y, x = np.mgrid[:120, :120]
        raster(path, (x + y).astype(np.float32), transform=TRANSFORM @ rasterio.Affine.scale(2))
        result = self.load()
        row = np.flatnonzero(result["sample_ids"] == "reference-1:1:0")[0]
        self.assertTrue(np.isnan(result["X"][row, data.FEATURE_NAMES.index("radar_s1_log")]))
        self.assertTrue(np.isnan(result["X"][row, data.FEATURE_NAMES.index("radar_cpr")]))
        other = np.flatnonzero(result["sample_ids"] == "reference-1:2:2")[0]
        self.assertAlmostEqual(result["X"][other, data.FEATURE_NAMES.index("radar_s1_log")], np.log(4), places=6)
        for name in data.LOLA:
            self.assertTrue(np.isfinite(result["X"][:, data.FEATURE_NAMES.index(name)]).all())
        self.assertEqual(result["metadata"]["references"][0]["optional"]["lola"]["native_pixel_size_m"], [10, 10])

    def _split_public_tiles(self, include_right=True):
        records = []
        for tile_id, left, right in (("left", 0, 121), ("right", 121, 240)):
            if tile_id == "right" and not include_right:
                continue
            layers = {}
            for name, original in self.public_record["layers"].items():
                with rasterio.open(self.folder / original) as src:
                    values = src.read(1)[:, left:right]
                if name == "nac" and tile_id == "right":
                    values[:] = 30
                relative = f"{tile_id}/{name}.tif"
                path = self.folder / relative
                raster(path, values, transform=TRANSFORM @ rasterio.Affine.translation(left, 0))
                layers[name] = relative
                self.public_meta["outputs"][relative] = {"checksum": {"sha256": sha(path)}}
            records.append({"tile_id": tile_id, "collection": "public", "site": self.region,
                            "layers": layers, "working_dem": layers["sfs"]})
        path = self.folder / "working_dems.json"
        write_json(path, records)
        self.public_meta["outputs"]["working_dems.json"] = {"checksum": {"sha256": sha(path)}}
        self._publish_public_meta()

    def test_cell_spans_two_public_tiles_without_duplicate_rows(self):
        self._split_public_tiles()
        with patch.object(data.core, "build", side_effect=fake_build) as build:
            result = data.load_data(self.public, self.reference, self.raw)
            self.assertEqual(build.call_count, 2)
        self.assertEqual(result["X"].shape, (14395, 23))
        self.assertEqual(len(set(result["sample_ids"])), 14395)
        nac = data.FEATURE_NAMES.index("nac")
        for column, expected in ((59, 10), (60, 20), (61, 30)):
            row = np.flatnonzero(result["sample_ids"] == f"reference-1:10:{column}")[0]
            self.assertEqual(result["X"][row, nac], expected)

    def test_unmapped_and_half_covered_cells_are_excluded(self):
        self._split_public_tiles(include_right=False)
        with patch.object(data.core, "build", side_effect=fake_build):
            result = data.load_data(self.public, self.reference, self.raw)
        self.assertEqual(result["X"].shape, (7195, 23))
        self.assertIn("reference-1:10:59", result["sample_ids"])
        self.assertNotIn("reference-1:10:60", result["sample_ids"])
        self.assertNotIn("reference-1:10:61", result["sample_ids"])
        self.assertTrue(np.all(result["support_fraction"] == 0))

    def test_reference_and_public_checksum_failure(self):
        path = self.reference / "saved.npy"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            data.load_data(self.public, self.reference, self.raw)
        self._make_reference()
        path = self.folder / "tile/nac.tif"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            data.load_data(self.public, self.reference, self.raw)

    def test_label_tiff_legend_and_exact_expansion_enforced(self):
        annotation = self.reference_record["annotations"][0]
        with rasterio.open(self.reference / "dem.tif") as src:
            grid = data._Grid.read(src)
        path = self.reference / annotation["path"]
        raster(path, np.ones(self.shape, np.uint8), nodata=255)
        annotation["sha256"] = sha(path)
        with self.assertRaisesRegex(ValueError, "dtype/nodata/grid"):
            data._snapshot(self.reference, annotation, grid,
                           self.terrain, data._Hashes())
        wrong = data._expand(self.saved, self.shape).astype(np.uint16)
        wrong[~self.terrain] = 65535
        wrong[20, 20] = 2
        raster(path, wrong, nodata=65535)
        annotation["sha256"] = sha(path)
        with self.assertRaisesRegex(ValueError, "exactly expand"):
            data._snapshot(self.reference, annotation, grid,
                           self.terrain, data._Hashes())

    def test_raw_archive_source_member_hash_validation(self):
        path = self.raw / "essentials/source.zip"
        path.parent.mkdir()
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("source.bin", b"source")
        self.public_meta["archive"] = {"name": path.name, "sha256": sha(path)}
        self.public_meta["sources"] = {"nac": {"member": "source.bin", "sha256": hashlib.sha256(b"source").hexdigest()}}
        self._publish_public_meta()
        result = self.load()
        self.assertIn("members verified", result["metadata"]["public_regions"][self.region]["raw_source_verification"])
        self.public_meta["sources"]["nac"]["sha256"] = "0" * 64
        self._publish_public_meta()
        with self.assertRaisesRegex(ValueError, "raw source nac"):
            data.load_data(self.public, self.reference, self.raw)


if __name__ == "__main__":
    unittest.main()
