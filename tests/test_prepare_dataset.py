"""Offline synthetic tests; all filesystem fixtures live in TemporaryDirectory.

Run: python -m unittest discover -s tests -p test_prepare_dataset.py
The preparation program and its interprocess-lock tests require Unix.
"""
import hashlib
import importlib.util
import io
import json
import select
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.crs import CRS as RasterCRS
from rasterio.io import MemoryFile

PREP_PATH = Path(__file__).resolve().parents[1] / "data/prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", PREP_PATH)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)

CRS = RasterCRS.from_string("+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m")
GRID = rasterio.Affine(5, 0, 100, 0, -5, 200)
NAC_GRID = rasterio.Affine(10, 0, 95, 0, -10, 205)
COARSE_GRID = rasterio.Affine(15, 0, 100, 0, -15, 200)


def raster_bytes(values, transform=GRID, crs=CRS, nodata=None,
                 scale=1.0, offset=0.0, unit=None, mask=None):
    values = np.asarray(values)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), MemoryFile() as memory:
        with memory.open(driver="GTiff", height=values.shape[0], width=values.shape[1],
                         count=1, dtype=values.dtype, transform=transform, crs=crs, nodata=nodata) as dst:
            dst.write(values, 1)
            dst.scales, dst.offsets = (scale,), (offset,)
            if unit:
                dst.set_band_unit(1, unit)
            if mask is not None:
                dst.write_mask(np.asarray(mask, dtype="uint8") * 255)
        return memory.read()


def snapshot(root):
    """Include content, inode and mtime, but not atime (reads may change it)."""
    return {p.relative_to(root).as_posix():
            (p.lstat().st_ino, p.lstat().st_mtime_ns,
             p.read_bytes() if p.is_file() and not p.is_symlink() else None)
            for p in [root, *sorted(root.rglob("*"))]}


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="selenograph-offline-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.raw, self.output = self.root / "raw_data", self.root / "dataset"
        (self.raw / "essentials").mkdir(parents=True)
        self.enterContext(mock.patch.object(prep, "PROJECT", self.root))
        self.enterContext(rasterio.Env(GDAL_PAM_ENABLED="NO"))
        self.sfs = np.arange(36, dtype="float32").reshape(6, 6)
        self.sfs[1, 1], self.sfs[2, 2], self.sfs[4, 4] = -9999, np.nan, np.inf
        self.sfs[-1, :] = -9999
        self.nac = np.array([[0, 1, 2, 3], [4, -9999, np.nan, 7],
                             [8, 9, np.inf, 11], [12, 13, 14, 15]], dtype="float64")
        self.nac_mask = np.ones((4, 4), dtype="uint8")
        self.nac_mask[3, 3] = 0
        self.count = np.array([[0, 2], [1, 3]], dtype="uint8")
        self.products = {
            "GLDELEV": raster_bytes(self.sfs, nodata=-9999, scale=2, offset=10, unit="m"),
            "GLDOMOS": raster_bytes(self.nac, NAC_GRID, nodata=-9999, mask=self.nac_mask),
            "GLDMASK": raster_bytes(self.count, COARSE_GRID),
            "GLDSBCT": raster_bytes(np.array([[0, 1], [2, 3]], dtype="int16"), COARSE_GRID),
            "GLDBRES": raster_bytes(np.full((2, 2), 1.25, dtype="float32"), COARSE_GRID, unit="m"),
            "GLDSIGM": raster_bytes(np.ones((6, 6), dtype="float32"), nodata=-1, unit="m"),
        }

    def archive(self, ident=0, region="Test_Region", omit=(), changes=None, extras=()):
        path = self.raw / "essentials" / f"A3CLR22_{ident}_{region}.zip"
        products = {**self.products, **(changes or {})}
        members = [(f"{path.stem}/A3{ident:02d}_{key}_005.tif", content)
                   for key, content in products.items() if key not in omit]
        if "README" not in omit:
            members.append((f"{path.stem}/README.txt", b"Synthetic provenance; uncertainty filename caveat.\n"))
        members.extend(extras)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
            for name, content in members:
                z.writestr(name, content)
        return path

    def cli(self, *arguments, expected=0):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = prep.main(["--raw-data", str(self.raw), "--output", str(self.output), *arguments])
        self.assertEqual(code, expected, out.getvalue() + err.getvalue())
        return out.getvalue() + err.getvalue()

    def manifest(self):
        return prep.read_json(self.output / "manifest.json")

    def test_zips_dropped_directly_into_raw_data_build_and_verify(self):
        archive = self.archive()
        archive.rename(self.raw / archive.name)
        self.cli()
        self.assertEqual(self.manifest()["tile_count"], 1)
        self.cli("--verify")

    def test_duplicate_regions_across_raw_and_legacy_folder_are_rejected(self):
        archive = self.archive()
        (self.raw / archive.name).write_bytes(archive.read_bytes())
        self.cli(expected=1)

    def metadata(self, region="test_region"):
        return prep.read_json(self.output / region / f"metadata/{region}_metadata.json")

    def layer(self, layer, region="test_region", row=0, col=0):
        tile_id = f"zenodo-{region}-r{row:03d}-c{col:03d}"
        return self.output / region / prep.layer_path(tile_id, layer)

    def read_layer(self, layer):
        with rasterio.open(self.layer(layer)) as src:
            return src.read(1)

    def refresh_output_checksum(self, name):
        m = self.metadata()
        m["outputs"][name]["checksum"] = prep.digest(self.output / "test_region" / name)
        (self.output / "test_region/metadata/test_region_metadata.json").write_text(json.dumps(m))

    def refresh_checksum(self, layer):
        self.refresh_output_checksum(prep.layer_path("zenodo-test_region-r000-c000", layer))

    def catalog(self, region="test_region"):
        return prep.read_json(self.output / region / "working_dems.json")

    def registered_site(self, name="unrelated-site", map_id="other-dem"):
        root = self.output / name
        (root / "elevation").mkdir(parents=True)
        dem = f"elevation/{map_id}.tif"
        (root / dem).write_bytes(self.products["GLDELEV"])
        (root / "working_dems.json").write_text(json.dumps([
            {"working_dem": dem, "site": name, "layers": {"sfs": dem}}]))
        (root / "provenance.txt").write_text("Not owned by the public preparer.\n")
        return root

    def test_select_exact_root_products_not_all_mac_dot_or_nested(self):
        prefix = "A3CLR22_0_Test_Region"
        archive = self.archive(extras=[
            (f"{prefix}/A300_GLDSBCTALL_005.tif", b"not a raster"),
            (f"{prefix}/nested/A300_GLDOMOS_005.tif", b"not a raster"),
            (f"__MACOSX/{prefix}/._A300_GLDOMOS_005.tif", b"resource fork"),
            (f"{prefix}/.A300_GLDOMOS_005.tif", b"dotfile"),
            (f"{prefix}/evidence.csv", b"image,count\nexample,1\n")])
        selected = prep.inventory(archive)
        self.assertEqual(set(selected), prep.REQUIRED | prep.OPTIONAL | {"evidence_00"})
        self.assertTrue(selected["solar_bins"].filename.endswith("_GLDSBCT_005.tif"))
        self.cli()
        self.assertEqual(self.metadata()["sources"]["evidence_00"]["member"], f"{prefix}/evidence.csv")
        self.assertFalse(any("ALL" in p.name or "nested" in p.parts for p in self.output.rglob("*")))

    def test_missing_required_and_missing_optional(self):
        for key in ("GLDOMOS", "GLDELEV", "GLDMASK", "README"):
            with self.subTest(key=key):
                archive = self.archive(omit=(key,))
                with self.assertRaisesRegex(ValueError, "Missing required"):
                    prep.inventory(archive)
        self.archive(omit=("GLDSBCT", "GLDBRES", "GLDSIGM"))
        self.cli()
        self.assertEqual(self.metadata()["quality"]["missing_optional"], sorted(prep.OPTIONAL))
        self.cli("--verify")

    def test_ambiguous_duplicate_and_uncertainty_alias(self):
        prefix = "A3CLR22_0_Test_Region"
        for suffix in ("A300_GLDOMOS_010.tif", "A300_GLDISGM_005.tif", "A300_GLDOMOS_005.tif"):
            with self.subTest(suffix=suffix), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive = self.archive(extras=[(f"{prefix}/{suffix}", b"ambiguous")])
                with self.assertRaisesRegex(ValueError, "Ambiguous|Duplicate"):
                    prep.inventory(archive)
        archive = self.archive(omit=("GLDSIGM",), changes={"GLDISGM": self.products["GLDSIGM"]})
        self.assertIn("GLDISGM", prep.inventory(archive)["uncertainty"].filename)

    def test_zip_traversal_and_symlink_rejected_even_if_not_selected(self):
        for name in ("../outside", "/absolute", "C:/drive", "x/../outside", "x/./file", "x\\file"):
            with self.subTest(name=name):
                archive = self.archive(extras=[(name, b"bad")])
                with self.assertRaisesRegex(ValueError, "Unsafe ZIP path"):
                    prep.inventory(archive)
        link = zipfile.ZipInfo("A3CLR22_0_Test_Region/unused_link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive = self.archive(extras=[(link, b"../outside")])
        with self.assertRaisesRegex(ValueError, "Special ZIP member"):
            prep.inventory(archive)

    def test_selected_member_crc_is_read_to_eof(self):
        archive = self.archive()
        with zipfile.ZipFile(archive) as z:
            info = z.getinfo(f"{archive.stem}/README.txt")
        data = bytearray(archive.read_bytes())
        name_length, extra_length = struct.unpack_from("<HH", data, info.header_offset + 26)
        data[info.header_offset + 30 + name_length + extra_length] ^= 1
        archive.write_bytes(data)
        self.assertIn("CRC", self.cli(expected=1))
        self.assertEqual(self.manifest()["status"], "partial")
        self.assertFalse((self.output / "test_region").exists())
        self.assertFalse((self.output / ".staging").exists())

    def test_published_md5_and_local_sha256(self):
        archive = self.archive()
        md5 = hashlib.md5(archive.read_bytes()).hexdigest()
        record = self.raw / "zenodo-record-17954508.json"
        for checksum in ("md5:" + "0" * 32, "sha256:" + "0" * 64):
            record.write_text(json.dumps({"files": [{"key": archive.name, "checksum": checksum}]}))
            self.assertIn("MD5", self.cli(expected=1))
        record.write_text(json.dumps({"files": []}))
        self.assertIn("MD5", self.cli(expected=1))
        record.write_text(json.dumps({"files": [{"key": archive.name, "checksum": "md5:" + md5}]}))
        self.cli()
        source = self.metadata()["archive"]
        self.assertEqual(source["published_md5"], md5)
        self.assertEqual(source["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.cli("--verify")

    def test_holes_zero_half_pixel_alignment_coarse_counts_and_native_metadata(self):
        self.archive()
        self.cli()
        expected_sfs = self.sfs.copy()
        expected_sfs[(expected_sfs == -9999) | ~np.isfinite(expected_sfs)] = np.nan
        # Destination centers map to source indices [0,1,1,2,2,3], not [0,0,1,1,2,2].
        indices = [0, 1, 1, 2, 2, 3]
        native = self.nac.copy()
        native[(native == -9999) | ~np.isfinite(native) | (self.nac_mask == 0)] = np.nan
        expected_nac = native[np.ix_(indices, indices)].astype("float32")
        expected_count = np.repeat(np.repeat(self.count, 3, axis=0), 3, axis=1).astype("float32")
        np.testing.assert_equal(self.read_layer("sfs"), expected_sfs)
        np.testing.assert_equal(self.read_layer("nac"), expected_nac)
        np.testing.assert_equal(self.read_layer("image_count"), expected_count)
        np.testing.assert_equal(self.read_layer("valid_data"), np.isfinite(expected_sfs) & np.isfinite(expected_nac))
        np.testing.assert_equal(self.read_layer("sfs_support"), np.isfinite(expected_sfs) & (expected_count > 0))
        self.assertEqual(self.read_layer("valid_data")[0, 0], 1)  # Finite zero is valid.
        self.assertEqual(self.read_layer("sfs_support")[0, 0], 0)  # Count zero is unsupported.
        m = self.metadata()
        self.assertEqual(m["sources"]["nac"]["native"]["grid"]["transform"], list(NAC_GRID)[:6])
        self.assertEqual(m["sources"]["image_count"]["native"]["grid"]["spacing"], [15, 15])
        self.assertEqual(m["sources"]["sfs"]["native"]["scales"], [2])
        # Compare with the source GeoTIFF CRS, not the pre-serialization PROJ string.
        with MemoryFile(self.products["GLDELEV"]) as memory, memory.open() as original, \
                rasterio.open(self.layer("sfs")) as src:
            self.assertEqual((src.scales, src.offsets, src.units), ((2.0,), (10.0,), ("m",)))
            self.assertEqual((src.width, src.height, src.transform, src.crs), (6, 6, GRID, original.crs))
        for item in m["outputs"].values():
            if "pixels" in item:
                self.assertEqual(item["pixels"]["total"], 36)
        self.assertFalse(any("label" in p.name for p in self.output.rglob("*")))

    def test_full_nodata_region_is_not_cropped_or_skipped(self):
        self.archive(changes={"GLDELEV": raster_bytes(np.full((6, 6), -9999, dtype="float32"), nodata=-9999)})
        self.cli()
        self.assertEqual(self.read_layer("sfs").shape, (6, 6))
        self.assertTrue(np.isnan(self.read_layer("sfs")).all())
        self.assertFalse(self.read_layer("valid_data").any())
        self.assertFalse(self.read_layer("sfs_support").any())
        self.cli("--verify")

    def test_coarse_quality_mask_and_uncovered_border_stay_invalid_not_zero_filled(self):
        mask = np.array([[1, 0], [1, 1]], dtype="uint8")
        self.archive(changes={"GLDELEV": raster_bytes(np.zeros((7, 7), dtype="float32")),
                              "GLDMASK": raster_bytes(self.count, COARSE_GRID, mask=mask)})
        self.cli()
        expected = np.full((7, 7), np.nan, dtype="float32")
        coarse = self.count.astype("float32")
        coarse[mask == 0] = np.nan
        expected[:6, :6] = np.repeat(np.repeat(coarse, 3, axis=0), 3, axis=1)
        np.testing.assert_equal(self.read_layer("image_count"), expected)
        np.testing.assert_equal(self.read_layer("sfs_support"), np.isfinite(expected) & (expected > 0))
        self.assertEqual(self.read_layer("sfs").shape, (7, 7))
        self.cli("--verify")

    def test_count_float32_exactness_and_positive_integral_semantics(self):
        for layer in ("image_count", "solar_bins"):
            for value, dtype, accepted in ((2**24 + 2, "uint32", True), (2**24 + 1, "uint32", False),
                                           (-1, "int16", False), (0.5, "float32", False)):
                with self.subTest(layer=layer, value=value):
                    original, result = self.root / "count.tif", self.root / "aligned.tif"
                    original.write_bytes(raster_bytes(np.full((2, 2), value, dtype=dtype), COARSE_GRID))
                    with MemoryFile(self.products["GLDELEV"]) as mem, mem.open() as src:
                        target = prep.grid(src)
                    if accepted:
                        prep.align_raster(original, result, target, layer)
                        with rasterio.open(result) as src:
                            self.assertTrue((src.read(1).astype("float64") == value).all())
                    else:
                        with self.assertRaisesRegex(ValueError, "Inexact/invalid counts"):
                            prep.align_raster(original, result, target, layer)

    def test_native_sfs_grid_validation(self):
        cases = [(rasterio.Affine(5, 1, 100, 0, -5, 200), CRS),
                 (rasterio.Affine(0, 0, 100, 0, -5, 200), CRS),
                 (rasterio.Affine(-5, 0, 100, 0, -5, 200), CRS),
                 (rasterio.Affine(5, 0, 100, 0, 5, 200), CRS), (GRID, RasterCRS.from_epsg(4326)),
                 (GRID, None), (rasterio.Affine(5, 0, float("inf"), 0, -5, 200), CRS)]
        for transform, crs in cases:
            with self.subTest(transform=transform, crs=crs):
                src = mock.Mock(count=1, width=6, height=6, crs=crs, transform=transform, bounds=(100, 170, 130, 200))
                with self.assertRaisesRegex(ValueError, "Invalid native SfS grid"):
                    prep.validate_sfs_grid(src)
        self.archive(changes={"GLDELEV": raster_bytes(self.sfs, cases[0][0], nodata=-9999)})
        self.assertIn("Invalid native SfS grid", self.cli(expected=1))
        self.assertFalse((self.output / "test_region").exists())

    def test_reuse_is_immutable_and_verify_is_read_only(self):
        self.archive()
        self.cli()
        before = snapshot(self.output / "test_region")
        with mock.patch.object(prep, "build_region", side_effect=AssertionError("must reuse")):
            self.cli()
        self.assertEqual(before, snapshot(self.output / "test_region"))
        before = snapshot(self.output)
        with mock.patch.object(prep, "write_json", side_effect=AssertionError("verify cannot write")):
            self.cli("--verify")
        self.assertEqual(before, snapshot(self.output))

    def test_registered_sites_allow_initialization_resume_and_read_only_verification(self):
        self.archive()
        sites = [self.registered_site(name, f"other-{i}") for i, name in enumerate(
            ("Arbitrary Site", "second", "Third (reference)", "fourth"))]
        catalog = sites[0] / "working_dems.json"
        records = prep.read_json(catalog)
        records[0].update(
            collection="anything", layers={"sfs": {"path": "elevation/other-0.tif", "status": "available"},
                "quality": {"count": "../second/elevation/other-1.tif"},
                "optional": {"path": None, "status": "unavailable"}},
            annotations={"painting": "provenance.txt"},
            feature_sources=[str(sites[1] / "elevation/other-1.tif")])
        catalog.write_text(json.dumps(records))
        before = [snapshot(site) for site in sites]
        raw_before = snapshot(self.raw)
        self.assertIn("No dataset manifest", self.cli("--verify", expected=1))
        self.cli("--max-seconds", "0.000000000001", expected=2)
        self.assertEqual(self.manifest()["included_regions"], [])
        with mock.patch.object(prep, "verify_region", wraps=prep.verify_region) as verify:
            self.cli()
        self.assertEqual([call.args[1] for call in verify.call_args_list], ["test_region"])
        self.assertEqual(self.manifest()["requested_regions"], ["test_region"])
        self.assertEqual(self.manifest()["included_regions"], ["test_region"])
        self.assertEqual(self.manifest()["tile_count"], 1)
        with mock.patch.object(prep, "build_region", side_effect=AssertionError("must reuse")):
            self.cli()
        published = snapshot(self.output)
        with mock.patch.object(prep, "write_json", side_effect=AssertionError("verify cannot write")), \
                mock.patch.object(prep, "verify_region", wraps=prep.verify_region) as verify:
            self.cli("--verify")
        self.assertEqual([call.args[1] for call in verify.call_args_list], ["test_region"])
        self.assertEqual(published, snapshot(self.output))
        self.assertEqual(before, [snapshot(site) for site in sites])
        self.assertEqual(raw_before, snapshot(self.raw))

    def test_moved_public_regions_keep_hashes_and_reuse_without_rebuilding(self):
        processed = self.root / "processed_data"
        self.output = processed / "public"
        self.archive()
        self.cli()
        region_before = snapshot(self.output / "test_region")
        manifest_bytes = (self.output / "manifest.json").read_bytes()
        manifest_before = self.manifest()
        for path in self.output.iterdir():
            path.rename(processed / path.name)
        self.output.rmdir()
        self.output = processed
        site = self.registered_site()
        site_before = snapshot(site)
        with mock.patch.object(prep, "build_region", side_effect=AssertionError("must reuse moved data")):
            self.cli("--verify")
            self.assertEqual(manifest_bytes, (self.output / "manifest.json").read_bytes())
            self.cli()
        self.assertEqual(manifest_before, self.manifest())
        self.assertEqual(manifest_bytes, (self.output / "manifest.json").read_bytes())
        self.assertEqual(region_before, snapshot(self.output / "test_region"))
        self.assertEqual(site_before, snapshot(site))
        self.assertEqual(self.metadata()["config"]["collection"], "public")
        self.assertFalse((self.output / "public").exists())

    def test_registered_catalog_must_be_nonempty_and_structurally_valid(self):
        self.archive()
        site = self.registered_site()
        catalog = site / "working_dems.json"
        valid = prep.read_json(catalog)[0]
        invalid = [None, {}, [], [None], [{}], [valid, {}], [valid, valid],
                   [{**valid, "working_dem": "missing.tif"}],
                   [{**valid, "working_dem": "elevation"}],
                   [{**valid, "working_dem": 12}],
                   [{**valid, "layers": []}], [{**valid, "layers": {"nac": 12}}],
                   [{**valid, "annotations": []}], [{**valid, "annotations": {"painting": False}}],
                   [{**valid, "feature_sources": "elevation/other-dem.tif"}],
                   [{**valid, "feature_sources": [None]}]]
        for records in invalid:
            with self.subTest(records=records):
                catalog.write_text(json.dumps(records))
                before = snapshot(self.output)
                self.cli(expected=1)
                self.assertFalse((self.output / "manifest.json").exists())
                self.assertEqual(before, snapshot(self.output))
        catalog.write_text("{not json")
        before = snapshot(self.output)
        self.cli(expected=1)
        self.assertEqual(before, snapshot(self.output))

    def test_registered_paths_reject_escapes_hidden_components_and_invalid_paths(self):
        self.archive()
        site = self.registered_site()
        catalog = site / "working_dems.json"
        valid = prep.read_json(catalog)[0]
        outside = self.root / "outside.tif"
        outside.write_bytes(self.products["GLDELEV"])
        outside_before = snapshot(outside)
        for value in (str(outside), "../../outside.tif", "elevation/../../../outside.tif",
                      "../.hidden/dem.tif", ".hidden/../elevation/other-dem.tif",
                      "C:/outside.tif", "elevation\\other-dem.tif", "", "bad\x00path"):
            records = [{**valid, "working_dem": value},
                       {**valid, "layers": {"nac": value}},
                       {**valid, "layers": {"nac": {"path": value, "status": "available"}}},
                       {**valid, "layers": {"quality": {"count": value}}},
                       *({**valid, "annotations": {kind: value}}
                         for kind in ("painting", "confidence", "accepted", "labels")),
                       {**valid, "feature_sources": [value]}]
            for record in records:
                with self.subTest(value=value, record=record):
                    catalog.write_text(json.dumps([record]))
                    before = snapshot(self.output)
                    self.cli(expected=1)
                    self.assertEqual(before, snapshot(self.output))
        self.assertEqual(outside_before, snapshot(outside))

    def test_registered_datasets_do_not_authorize_strays_or_hidden_roots(self):
        self.archive()
        base = self.output
        for has_manifest in (False, True):
            self.output = base / str(has_manifest)
            site = self.registered_site()
            if has_manifest:
                self.cli()
            for name in ("junk.txt", "unknown-directory", ".hidden", ".staging"):
                with self.subTest(has_manifest=has_manifest, name=name):
                    stray = self.output / name
                    if name == "junk.txt":
                        stray.write_text("do not delete")
                    elif name.startswith("."):
                        self.registered_site(name, "hidden-dem")
                    else:
                        stray.mkdir()
                    before = snapshot(self.output)
                    self.cli(expected=1)
                    if has_manifest:
                        self.cli("--verify", expected=1)
                    self.assertEqual(before, snapshot(self.output))
                    if stray.is_dir():
                        prep.shutil.rmtree(stray)
                    else:
                        stray.unlink()
            self.assertTrue((site / "working_dems.json").is_file())
            if not has_manifest:
                self.assertFalse((self.output / "manifest.json").exists())

    def test_registered_hardlinked_raster_inputs_preserve_external_sources_and_metadata(self):
        self.archive()
        site = self.registered_site("Arbitrary Site")
        preserved = self.root / "preserved"
        preserved.mkdir()
        sources = {}
        for relative, product in (("elevation/other-dem.tif", "GLDELEV"), ("image.tif", "GLDOMOS"),
                                  ("feature-source.tif", "GLDELEV"), ("labels.tif", "GLDMASK")):
            source = preserved / Path(relative).name
            source.write_bytes(self.products[product])
            source.chmod(0o444)
            linked = site / relative
            linked.unlink(missing_ok=True)
            linked.hardlink_to(source)
            self.assertEqual(linked.stat().st_ino, source.stat().st_ino)
            self.assertEqual(source.stat().st_nlink, 2)
            sources[relative] = source
        catalog = site / "working_dems.json"
        records = prep.read_json(catalog)
        records[0]["layers"].update(nac={"path": "image.tif", "status": "available"})
        records[0].update(feature_sources=["feature-source.tif"], annotations={"labels": "labels.tif"})
        catalog.write_text(json.dumps(records))
        site_before, preserved_before = snapshot(site), snapshot(preserved)
        self.cli()
        with mock.patch.object(prep, "build_region", side_effect=AssertionError("must reuse")):
            self.cli()
        output_before = snapshot(self.output)
        with mock.patch.object(prep, "write_json", side_effect=AssertionError("verify cannot write")):
            self.cli("--verify")
        self.assertEqual(output_before, snapshot(self.output))
        self.assertEqual(site_before, snapshot(site))
        self.assertEqual(preserved_before, snapshot(preserved))
        for relative, source in sources.items():
            self.assertEqual((site / relative).stat().st_ino, source.stat().st_ino)
            self.assertEqual(source.stat().st_nlink, 2)
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o444)

    def test_registered_metadata_hardlinks_remain_rejected(self):
        self.archive()
        site = self.registered_site()
        alias = self.root / "preserved-metadata"
        for target in (site / "working_dems.json", site / "provenance.txt"):
            with self.subTest(target=target):
                alias.hardlink_to(target)
                before = snapshot(self.output)
                self.cli(expected=1)
                self.assertEqual(before, snapshot(self.output))
                self.assertFalse((self.output / "manifest.json").exists())
                alias.unlink()

    def test_owned_public_outputs_and_publication_metadata_still_reject_hardlinks(self):
        self.archive()
        self.cli()
        alias = self.root / "external-alias"
        for target in (self.output / "manifest.json", self.output / "test_region/working_dems.json",
                       self.output / "test_region/metadata/test_region_metadata.json", self.layer("sfs")):
            with self.subTest(target=target):
                alias.hardlink_to(target)
                before = snapshot(self.output)
                self.cli(expected=1)
                self.cli("--verify", expected=1)
                self.assertEqual(before, snapshot(self.output))
                alias.unlink()
        self.cli("--verify")

    def test_registered_symlinks_and_undeclared_hardlinks_are_rejected_without_following(self):
        self.archive()
        site = self.registered_site()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("never walk or modify this directory")
        outside_before = snapshot(outside)
        catalog = site / "working_dems.json"
        dem = site / "elevation/other-dem.tif"
        for target in (self.output / "linked-site", site / "nested-link", catalog, dem):
            with self.subTest(target=target):
                content = target.read_bytes() if target.is_file() else None
                if content is not None:
                    target.unlink()
                target.symlink_to(outside, target_is_directory=True)
                before = snapshot(self.output)
                with mock.patch.object(prep, "read_json", wraps=prep.read_json) as read:
                    self.cli(expected=1)
                self.assertTrue(all(not call.args[0].resolve().is_relative_to(outside)
                                    for call in read.call_args_list))
                self.assertEqual(before, snapshot(self.output))
                target.unlink()
                if content is not None:
                    target.write_bytes(content)
        alias = site / "hardlink.tif"
        alias.hardlink_to(dem)
        before = snapshot(self.output)
        self.assertIn("linked output", self.cli(expected=1))
        self.assertEqual(before, snapshot(self.output))
        self.assertEqual(outside_before, snapshot(outside))

    def test_registered_ids_cannot_collide_with_owned_or_new_public_ids(self):
        self.archive()
        base = self.output
        for has_manifest in (False, True):
            with self.subTest(has_manifest=has_manifest):
                self.output = base / str(has_manifest)
                if has_manifest:
                    self.cli()
                site = self.registered_site(map_id="zenodo-test_region-r000-c000")
                before = snapshot(site)
                published = snapshot(self.output)
                self.assertIn("Duplicate map ID", self.cli(expected=1))
                self.assertEqual(before, snapshot(site))
                if has_manifest:
                    self.assertIn("Duplicate map ID", self.cli("--verify", expected=1))
                    self.assertEqual(published, snapshot(self.output))
                else:
                    self.assertEqual(self.manifest()["included_regions"], [])
                    self.assertFalse((self.output / "test_region").exists())
                    self.assertFalse((self.output / ".staging").exists())

    def test_registered_duplicate_ids_and_requested_directory_conflicts_are_refused(self):
        self.archive()
        first = self.registered_site("first")
        second = self.registered_site("second")
        before = snapshot(self.output)
        self.assertIn("Duplicate map IDs", self.cli(expected=1))
        self.assertEqual(before, snapshot(self.output))
        second.rename(self.output / "test_region")
        before = snapshot(self.output)
        self.assertIn("Unlisted output", self.cli(expected=1))
        self.assertEqual(before, snapshot(self.output))
        self.assertTrue((first / "working_dems.json").is_file())

    def test_modified_output_rejected_without_repair(self):
        self.archive()
        self.cli()
        with self.layer("nac").open("ab") as stream:
            stream.write(b"modified")
        before = snapshot(self.output)
        self.assertIn("Modified output", self.cli(expected=1))
        self.assertIn("Modified output", self.cli("--verify", expected=1))
        self.assertEqual(before, snapshot(self.output))

    def test_modified_source_and_config_rejected(self):
        archive = self.archive()
        self.cli()
        before = snapshot(self.output)
        with archive.open("ab") as stream:
            stream.write(b"source changed without changing its member inventory")
        self.assertIn("source/config mismatch", self.cli(expected=1))
        self.assertEqual(before, snapshot(self.output))
        with mock.patch.object(prep, "CONFIG", {**prep.CONFIG, "resampling": "different"}):
            self.assertIn("Unknown output configuration", self.cli(expected=1))

    def test_unknown_nonempty_output_and_raw_reference_overlaps(self):
        self.archive()
        self.output.mkdir()
        (self.output / "keep.txt").write_text("must survive")
        before = snapshot(self.output)
        self.assertIn("Unlisted output", self.cli(expected=1))
        self.assertEqual(before, snapshot(self.output))
        for output in (self.raw, self.raw / "derived", self.root, self.root / "reference_data",
                       self.root / "reference_data/nested", self.root / "other/reference_data/nested",
                       self.root / "data/raw_data/derived", self.root / "data/reference_data/nested"):
            with self.subTest(output=output):
                self.assertIn("overlaps", self.cli("--output", str(output), expected=1))
        self.assertFalse((self.raw / "derived").exists())
        self.assertFalse((self.root / "reference_data").exists())
        link = self.root / "linked-output"
        link.symlink_to(self.output, target_is_directory=True)
        self.assertIn("Symlink", self.cli("--output", str(link), expected=1))

    def test_verify_missing_extra_and_symlink_paths(self):
        self.archive()
        self.cli()
        target = self.layer("nac")
        content = target.read_bytes()
        target.unlink()
        self.assertIn("Unlisted/missing", self.cli("--verify", expected=1))
        target.write_bytes(content)
        extra = target.parent / "extra.txt"
        extra.write_text("unexpected")
        self.assertIn("Unlisted/missing", self.cli("--verify", expected=1))
        extra.unlink()
        extra.symlink_to(target)
        self.assertIn("Symlink", self.cli("--verify", expected=1))

    def test_verify_grid_and_support_equations_beyond_checksums(self):
        self.archive()
        self.cli()
        original = self.layer("nac").read_bytes()
        with rasterio.open(self.layer("nac"), "r+") as src:
            src.transform = rasterio.Affine(5, 0, 105, 0, -5, 200)
        self.refresh_checksum("nac")
        self.assertIn("Grid/type mismatch", self.cli("--verify", expected=1))
        self.layer("nac").write_bytes(original)
        self.refresh_checksum("nac")
        with rasterio.open(self.layer("sfs_support"), "r+") as src:
            values = src.read(1)
            self.assertEqual((values[0, 0], values[0, 3]), (0, 1))
            values[0, 0], values[0, 3] = 1, 0  # Preserve the inventory pixel counts.
            src.write(values, 1)
        self.refresh_checksum("sfs_support")
        self.assertIn("Support mismatch", self.cli("--verify", expected=1))

    def test_subset_never_claims_all_thirteen_regions(self):
        for i in range(13):
            self.archive(i, f"Region_{i}")
        message = self.cli("--region", "2")
        m = self.manifest()
        self.assertEqual(m["status"], "complete")
        self.assertEqual(m["completion_scope"], "requested_regions")
        self.assertEqual(m["included_regions"], ["region_2"])
        self.assertEqual(m["requested_regions"], ["region_2"])
        self.assertEqual(len(m["available_regions"]), 13)
        self.assertFalse(m["all_available_regions_included"])
        self.assertIn("1 of 13", message)
        self.assertIn("1 of 13", self.cli("--verify"))
        self.cli("--verify", "--region", "3", expected=1)
        m["all_available_regions_included"] = True
        (self.output / "manifest.json").write_text(json.dumps(m))
        self.assertIn("Inconsistent completion scope", self.cli("--verify", expected=1))

    def test_bounded_partial_resume_preserves_completed_region(self):
        self.archive(0, "Alpha")
        self.archive(1, "Beta")
        clock, original = [0.0], prep.build_region
        def finish_first(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 11.0
            return result
        with mock.patch.object(prep.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(prep, "build_region", side_effect=finish_first):
            self.cli("--max-seconds", "10", expected=2)
        self.assertEqual(self.manifest()["included_regions"], ["alpha"])
        self.assertEqual(self.manifest()["status"], "partial")
        self.assertFalse(self.manifest()["all_available_regions_included"])
        before = snapshot(self.output)
        self.cli("--verify", expected=1)
        self.assertEqual(before, snapshot(self.output))
        immutable = snapshot(self.output / "alpha")
        self.cli()
        self.assertEqual(immutable, snapshot(self.output / "alpha"))
        self.assertEqual(self.manifest()["included_regions"], ["alpha", "beta"])
        self.assertTrue(self.manifest()["all_available_regions_included"])
        self.cli("--verify")

    def test_interrupted_region_is_unpublished_and_restarts(self):
        self.archive()
        original = prep.align_raster
        def interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise TimeoutError("synthetic deadline during raster work")
        with mock.patch.object(prep, "align_raster", side_effect=interrupt):
            self.cli(expected=2)
        self.assertEqual(self.manifest()["included_regions"], [])
        self.assertFalse((self.output / "test_region").exists())
        self.assertFalse((self.output / ".staging").exists())
        self.cli()
        self.cli("--verify")

    def test_resume_recovers_published_region_before_manifest_update(self):
        self.archive()
        original = prep.write_json
        def interrupt(path, value):
            if path.name == "manifest.json" and value["status"] == "partial" and value["regions"]:
                raise TimeoutError("synthetic interruption after atomic region publication")
            original(path, value)
        with mock.patch.object(prep, "write_json", side_effect=interrupt):
            self.cli(expected=2)
        self.assertEqual(self.manifest()["included_regions"], [])
        before = snapshot(self.output / "test_region")
        with mock.patch.object(prep, "build_region", side_effect=AssertionError("must recover, not rebuild")):
            self.cli()
        self.assertEqual(before, snapshot(self.output / "test_region"))
        self.cli("--verify")

    def test_process_lock_protects_staging_and_recovers_after_holder_is_killed(self):
        self.archive()
        self.cli("--max-seconds", "0.000000000001", expected=2)
        staging = self.output / ".staging"
        staging.mkdir()
        (staging / "first-writer-work").write_text("must not be deleted by a second writer")
        script = ("import runpy,sys,time; from pathlib import Path\n"
                  "p = runpy.run_path(sys.argv[1])\n"
                  "with p['dataset_lock'](Path(sys.argv[2])):\n"
                  " print('locked', flush=True)\n"
                  " time.sleep(30)\n")
        process = subprocess.Popen([sys.executable, "-B", "-c", script, str(PREP_PATH), str(self.output)],
                                   cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert process.stdout is not None
            ready, _, _ = select.select([process.stdout], [], [], 10)
            self.assertTrue(ready, "lock holder failed to start within 10 seconds")
            self.assertEqual(process.stdout.readline(), b"locked\n")
            before = snapshot(self.output)
            self.assertIn("in use", self.cli(expected=1))
            self.assertIn("in use", self.cli("--verify", expected=1))
            self.assertEqual(before, snapshot(self.output))
        finally:
            process.kill()
            process.communicate(timeout=10)
        self.cli()  # Kernel released the dead process's lock; safe staging recovery.
        self.assertFalse(staging.exists())
        self.cli("--verify")

    def test_disk_space_failure_does_not_publish(self):
        self.archive()
        with mock.patch.object(prep.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            self.assertIn("Insufficient extraction disk space", self.cli(expected=1))
        self.assertEqual(self.manifest()["included_regions"], [])
        self.assertFalse((self.output / ".staging").exists())
        self.cli()

    def test_rasterio_error_is_reported_cleanly(self):
        self.archive(changes={"GLDELEV": b"not a GeoTIFF"})
        self.assertIn("Error:", self.cli(expected=1))
        self.assertEqual(self.manifest()["status"], "partial")

    def test_relocated_defaults_and_standalone_help(self):
        self.raw = self.root / "data/raw_data"
        self.output = self.root / "data/processed_data"
        (self.raw / "essentials").mkdir(parents=True)
        self.archive()
        before = snapshot(self.raw)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(prep.main([]), 0)
            self.assertEqual(prep.main(["--verify"]), 0)
        self.assertEqual(before, snapshot(self.raw))
        self.assertEqual(self.manifest()["collection"], "public")
        self.assertEqual(self.catalog()[0]["collection"], "public")
        result = subprocess.run([sys.executable, "-B", str(PREP_PATH), "--help"], cwd=self.root,
                                capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data/processed_data/<region>/", result.stdout)
        self.assertNotIn("data/processed_data/public", result.stdout)
        self.assertFalse((self.output / "public").exists())

    def test_canonical_layout_single_exact_and_variable_edges(self):
        self.assertEqual(prep.TILE_SIZE, 1024)
        for height, width in ((1, 1), (1024, 1024), (1025, 2049), (2048, 2048)):
            with self.subTest(height=height, width=width):
                target = {"height": height, "width": width, "transform": list(GRID)[:6],
                          "crs_wkt": CRS.to_wkt(), "spacing": [5, 5]}
                tiles = list(prep.tile_layout("region", target))
                self.assertEqual(len(tiles), ((height + 1023) // 1024) * ((width + 1023) // 1024))
                coverage = np.zeros((height, width), dtype="uint8")
                for tile in tiles:
                    w, g = tile["window"], tile["grid"]
                    self.assertEqual(tile["tile_id"], f"zenodo-region-r{tile['row']:03d}-c{tile['col']:03d}")
                    self.assertEqual((w["row_off"], w["col_off"]), (tile["row"] * 1024, tile["col"] * 1024))
                    self.assertEqual((g["height"], g["width"]), (w["height"], w["width"]))
                    self.assertEqual(g["transform"], [5, 0, 100 + 5 * w["col_off"], 0, -5, 200 - 5 * w["row_off"]])
                    self.assertTrue(0 < g["height"] <= 1024 and 0 < g["width"] <= 1024)
                    coverage[w["row_off"]:w["row_off"] + w["height"], w["col_off"]:w["col_off"] + w["width"]] += 1
                self.assertTrue((coverage == 1).all())

    def test_multitile_alignment_catalog_and_entire_extent_roundtrip(self):
        height, width = 1027, 1031
        sfs = np.arange(height * width, dtype="float32").reshape(height, width)
        sfs[1024:, 1024:] = -9999  # The corner tile is entirely nodata, but must be published.
        sfs[10, 1023:1026] = -9999
        sfs_mask = np.ones(sfs.shape, dtype="uint8")
        sfs_mask[1023:1026, 10] = 0
        nac = np.arange(515 * 517, dtype="float32").reshape(515, 517)
        nac_mask = np.ones(nac.shape, dtype="uint8")
        nac_mask[510:, 511] = 0
        counts = (np.indices((342, 343))[0] % 4).astype("uint16")
        count_mask = np.ones(counts.shape, dtype="uint8")
        count_mask[340:, 341] = 0
        self.products.update({"GLDELEV": raster_bytes(sfs, nodata=-9999, mask=sfs_mask, scale=2, offset=10, unit="m"),
                              "GLDOMOS": raster_bytes(nac, NAC_GRID, mask=nac_mask),
                              "GLDMASK": raster_bytes(counts, COARSE_GRID, mask=count_mask)})
        archive = self.archive()
        before = snapshot(self.raw)
        self.cli()
        m, catalog = self.metadata(), self.catalog()
        self.assertEqual(m["tile_count"], 4)
        self.assertEqual([r["size"] for r in catalog], [[1024, 1024], [1024, 7], [3, 1024], [3, 7]])
        self.assertEqual([r["tile_id"] for r in catalog], ["zenodo-test_region-r000-c000", "zenodo-test_region-r000-c001",
                                                         "zenodo-test_region-r001-c000", "zenodo-test_region-r001-c001"])
        self.assertEqual(catalog[-1]["valid_count"], 0)
        self.assertEqual(catalog[-1]["valid_data_count"], 0)
        region_root = self.output / "test_region"
        self.assertEqual({p.name for p in region_root.iterdir()}, {"tiles", "metadata", "working_dems.json"})
        self.assertTrue(all(p.relative_to(region_root).parts[0] == "tiles" for p in region_root.rglob("*.tif")))
        self.assertFalse(any("label" in p.name for p in region_root.rglob("*")))
        expected = {}
        for product, content in self.products.items():
            key = prep.PRODUCTS[product]
            original, aligned = self.root / f"native-{key}.tif", self.root / f"expected-{key}.tif"
            original.write_bytes(content)
            prep.align_raster(original, aligned, m["grid"], key)
            with rasterio.open(aligned) as src:
                expected[key] = src.read(1)
        normalized_sfs = np.where((sfs != -9999) & (sfs_mask > 0), sfs, np.nan)
        np.testing.assert_equal(expected["sfs"], normalized_sfs)
        expected["valid_data"] = np.isfinite(expected["sfs"]) & np.isfinite(expected["nac"])
        expected["sfs_support"] = np.isfinite(expected["sfs"]) & np.isfinite(expected["image_count"]) & (expected["image_count"] > 0)
        stitched = {key: np.full((height, width), np.nan, dtype="float32") for key in expected}
        for tile, record in zip(m["tiles"], catalog):
            self.assertEqual((record["group"], record["site"], record["collection"]), ("test_region", "test_region", "public"))
            self.assertEqual(record["working_dem"], record["layers"]["sfs"])
            self.assertEqual(Path(record["working_dem"]).stem, record["tile_id"])
            self.assertEqual(record["transform"], tile["grid"]["transform"])
            self.assertEqual(len(record["transform"]), 6)
            self.assertEqual(set(record["layers"]), set(expected))
            w = tile["window"]
            rows = slice(w["row_off"], w["row_off"] + w["height"])
            cols = slice(w["col_off"], w["col_off"] + w["width"])
            self.assertEqual(record["valid_count"], int(np.isfinite(expected["sfs"][rows, cols]).sum()))
            for key in prep.MASKS:
                self.assertEqual(record[f"{key}_count"], int(expected[key][rows, cols].sum()))
            for key, name in record["layers"].items():
                self.assertFalse(Path(name).is_absolute())
                self.assertNotIn("..", Path(name).parts)
                with rasterio.open(region_root / name) as src:
                    self.assertEqual(list(src.shape), record["size"])
                    self.assertEqual(list(src.transform)[:6], record["transform"])
                    self.assertEqual(src.crs.to_wkt(), m["grid"]["crs_wkt"])
                    stitched[key][rows, cols] = src.read(1)
                    if key == "sfs":
                        self.assertEqual((src.scales, src.offsets, src.units), ((2.0,), (10.0,), ("m",)))
        for key, values in expected.items():
            np.testing.assert_equal(stitched[key], values, err_msg=key)
        manifest = self.manifest()
        self.assertEqual(manifest["tile_count"], 4)
        self.assertEqual(manifest["regions"]["test_region"]["tile_count"], 4)
        self.assertEqual(manifest["requested_archives"], {"test_region": archive.name})
        self.assertEqual(manifest["regions"]["test_region"]["archive"]["sha256"], prep.digest(archive)["sha256"])
        self.cli("--verify")
        self.assertEqual(before, snapshot(self.raw))
        self.assertFalse((self.output / ".staging").exists())

    def test_verify_rejects_changed_coverage_windows_and_tile_ids(self):
        self.archive()
        self.cli()
        path = self.output / "test_region/metadata/test_region_metadata.json"
        original = path.read_text()
        mutations = [lambda m: m["tiles"][0]["window"].update(col_off=1),
                     lambda m: m["tiles"][0]["window"].update(width=5),
                     lambda m: m["tiles"][0].update(row=1),
                     lambda m: m["tiles"][0].update(tile_id="zenodo-test_region-r001-c000"),
                     lambda m: m["tiles"][0]["grid"].update(transform=[5, 0, 105, 0, -5, 200]),
                     lambda m: m.update(tile_count=2), lambda m: m.update(tiles=[]),
                     lambda m: m.update(tiles=m["tiles"] * 2, tile_count=2)]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                m = json.loads(original)
                mutate(m)
                path.write_text(json.dumps(m))
                self.assertIn("Tile coverage", self.cli("--verify", expected=1))
        path.write_text(original)
        self.cli("--verify")

    def test_verify_rejects_catalog_tampering_even_with_updated_checksum(self):
        self.archive()
        self.cli()
        path = self.output / "test_region/working_dems.json"
        original = path.read_text()
        mutations = [lambda c: c[0].update(working_dem="../outside.tif"),
                     lambda c: c[0].update(working_dem=str(self.layer("sfs"))),
                     lambda c: c[0].update(tile_id="different"),
                     lambda c: c[0].update(group="other"), lambda c: c[0].update(site="other"),
                     lambda c: c[0].update(collection="user"), lambda c: c[0].update(size=[7, 6]),
                     lambda c: c[0].update(transform=[5, 0, 105, 0, -5, 200]),
                     lambda c: c[0]["layers"].update(sfs="tiles/renamed.tif"),
                     lambda c: c[0]["layers"].update(nac=c[0]["layers"]["sfs"]),
                     lambda c: c[0]["layers"].pop("valid_data"),
                     lambda c: c[0]["layers"].update(uncertainty="../outside.tif"),
                     lambda c: c[0].update(valid_count=0), lambda c: c.append(c[0]), lambda c: c.clear()]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                catalog = json.loads(original)
                mutate(catalog)
                path.write_text(json.dumps(catalog))
                self.refresh_output_checksum("working_dems.json")
                self.assertIn("Catalog path/record mismatch", self.cli("--verify", expected=1))
        path.write_text(original)
        self.refresh_output_checksum("working_dems.json")
        # Restore canonical metadata formatting as well as content for its pinned hash.
        prep.write_json(self.output / "test_region/metadata/test_region_metadata.json", self.metadata())
        self.cli("--verify")

    def test_verify_rejects_forged_tile_valid_counts_and_manifest_totals(self):
        self.archive()
        self.cli()
        path = self.output / "test_region/metadata/test_region_metadata.json"
        original = path.read_text()
        for key in ("valid_count", "valid_data_count", "sfs_support_count"):
            with self.subTest(key=key):
                m = json.loads(original)
                m["tiles"][0][key] += 1
                path.write_text(json.dumps(m))
                self.assertIn("Tile valid counts mismatch", self.cli("--verify", expected=1))
        path.write_text(original)
        manifest_path = self.output / "manifest.json"
        original = manifest_path.read_text()
        for mutation in (lambda m: m.update(tile_count=2),
                         lambda m: m["requested_archives"].update(test_region="wrong.zip")):
            m = json.loads(original)
            mutation(m)
            manifest_path.write_text(json.dumps(m))
            self.cli("--verify", expected=1)
        manifest_path.write_text(original)
        self.cli("--verify")

    def test_regional_scratch_removed_after_verification_before_publish(self):
        self.archive()
        original_verify, original_rename = prep.verify_region, Path.rename
        verified = []
        def verify(root, *args, **kwargs):
            self.assertTrue((self.output / ".staging/aligned/sfs.tif").is_file())
            self.assertTrue((self.output / ".staging/originals/sfs").is_file())
            result = original_verify(root, *args, **kwargs)
            verified.append(root)
            return result
        def publish(root, destination):
            self.assertEqual(verified, [root])
            self.assertFalse((self.output / ".staging/aligned").exists())
            self.assertFalse((self.output / ".staging/originals").exists())
            self.assertEqual(list((self.output / ".staging").iterdir()), [root])
            self.assertFalse(destination.exists())
            return original_rename(root, destination)
        with mock.patch.object(prep, "verify_region", side_effect=verify), \
                mock.patch.object(Path, "rename", autospec=True, side_effect=publish):
            self.cli()
        self.assertFalse((self.output / ".staging").exists())
        self.cli("--verify")

    def test_deadline_during_tiling_or_verification_never_publishes(self):
        self.archive()
        for function in ("write_tile", "verify_region"):
            with self.subTest(function=function):
                original = getattr(prep, function)
                def interrupt(*args, _original=original, **kwargs):
                    _original(*args, **kwargs)
                    raise TimeoutError("synthetic deadline during tile preparation")
                with mock.patch.object(prep, function, side_effect=interrupt):
                    self.cli(expected=2)
                self.assertEqual(self.manifest()["included_regions"], [])
                self.assertEqual(self.manifest()["tile_count"], 0)
                self.assertFalse((self.output / "test_region").exists())
                self.assertFalse((self.output / ".staging").exists())
        self.cli()
        self.cli("--verify")


if __name__ == "__main__":
    unittest.main()
