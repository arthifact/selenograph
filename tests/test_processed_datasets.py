"""Generic processed datasets only: disposable manifests, annotations and lunar DEMs.

Run: .venv/bin/python -B -m unittest tests.test_processed_datasets -v
"""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from app import core


class ProcessedDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processed = self.root / "processed_data"
        self.processed.mkdir()
        for attr, value in (("DEM_DIR", self.processed),
                            ("OUT_DIR", self.root / "output/paintings"),
                            ("MAP_DIR", self.root / "output/maps"),
                            ("LEGACY_OUT_DIR", self.root / "output/maps"),
                            ("DRAFT_DIR", self.root / "output/painting_drafts"),
                            ("LEGACY_DRAFT_DIR", self.root / "output/drafts"),
                            ("LEGACY_MIRRORED_DIR", self.root / "output")):
            patcher = patch.object(core, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(core._load_catalog.cache_clear)

    def manifest(self, folder, records):
        path = self.processed / folder / "working_dems.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def record(self, name):
        record = core.map_record(name)
        assert record is not None
        return record

    def raster(self, relative, *, shape=(120, 120), left=1000, top=2000,
               px=5, crs: str | None = "ESRI:103878", values=None):
        path = self.processed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if values is None:
            y, x = np.mgrid[:shape[0], :shape[1]]
            values = (20 * np.sin(x / 13) + 8 * np.cos(y / 9) + y / 7).astype(np.float32)
        values = np.asarray(values, dtype=np.float32)
        with rasterio.open(path, "w", driver="GTiff", count=1, dtype="float32",
                           width=values.shape[1], height=values.shape[0], nodata=-9999,
                           crs=crs, transform=from_origin(left, top, px, px)) as dst:
            dst.write(np.where(np.isfinite(values), values, -9999), 1)
        return path

    def array(self, relative, values):
        path = self.processed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)
        return path

    def snapshot(self):
        return {str(p.relative_to(self.processed)): p.read_bytes()
                for p in self.processed.rglob("*") if p.is_file()}

    def bundle(self):
        folder = "reference/location"
        dem = self.raster(f"{folder}/terrain.tif")
        cells = np.zeros((core.GRID, core.GRID), np.uint8)
        cells[10:30, 20:40] = 1
        cells[60:80, 60:80] = 2
        confidence = np.where(cells > 0, core.MOSTLY, 0).astype(np.uint8)
        accepted = cells == 2
        for kind, values in (("painting", cells), ("confidence", confidence),
                             ("accepted", accepted)):
            self.array(f"{folder}/annotations/{kind}.npy", values)
        self.raster(f"{folder}/annotations/labels.tif",
                    values=core.cells_to_labels(cells, (120, 120)))
        record = {"working_dem": "terrain.tif", "site": "geographic-site",
                  "group": "geographic-group", "collection": "any-collection",
                  "annotations": {
                      "painting": "annotations/painting.npy",
                      "confidence": "annotations/confidence.npy",
                      "accepted": "annotations/accepted.npy",
                      "labels": "annotations/labels.tif",
                      "verified": True, "label_status": "user-verified source",
                      "restricted": True, "provenance": {"review": ["explicit request"]}}}
        manifest = self.manifest(folder, [record])
        return dem, manifest, record, cells, confidence, accepted

    def test_catalog_snapshot_discovers_once_for_243_map_annotation_lookups(self):
        _, manifest, record, _, confidence, _ = self.bundle()
        manifest.write_text(json.dumps([
            dict(record, working_dem=f"map-{i}.tif") for i in range(243)]))
        with patch.object(core.os, "walk", wraps=core.os.walk) as walked:
            with core.catalog_snapshot():
                for i in range(243):
                    name = f"map-{i}"
                    self.assertEqual(core.map_site(name), "geographic-group")
                    self.assertEqual(self.record(name)["section"], "reference")
                    self.assertIsNotNone(core.annotation_path(name, "labels"))
                    self.assertTrue(core.meta_get(name)["verified"])
                    np.testing.assert_array_equal(core.load_confidence(name), confidence)
                self.assertEqual(walked.call_count, 1)
            with core.catalog_snapshot():
                self.assertEqual(len(core.records()), 243)
            self.assertEqual(walked.call_count, 2)
        self.assertFalse((self.root / "output").exists())

    def test_catalog_snapshot_preserves_deep_copy_callers(self):
        self.bundle()
        original = self.snapshot()
        with core.catalog_snapshot() as value:
            self.assertIsNone(value)  # never expose the mutable cached catalog
            record = self.record("terrain")
            record["annotations"]["provenance"]["review"].clear()
            core.records()[0]["annotations"]["painting"] = "changed.npy"
            metadata = core.meta_get("terrain")
            metadata["provenance"]["review"].append("changed")
            files = core.dem_files()
            files.clear()
            self.assertEqual(len(core.dem_files()), 1)
            painting = core.annotation_path("terrain", "painting")
            assert painting is not None
            self.assertTrue(painting.endswith("painting.npy"))
            self.assertEqual(core.meta_get("terrain")["provenance"],
                             {"review": ["explicit request"]})
        self.assertEqual(self.snapshot(), original)

    def test_catalog_snapshot_refreshes_changed_added_removed_and_flat_files(self):
        manifest = self.manifest("dataset", [{"working_dem": "map.tif", "group": "before"}])
        with patch.object(core, "manifest_files", wraps=core.manifest_files) as discover:
            with core.catalog_snapshot():
                self.assertEqual(core.map_site("map"), "before")
                manifest.write_text('[{"working_dem":"map.tif","group":"after"}]')
                added = self.manifest("added", [{"working_dem": "new.tif"}])
                flat = self.processed / "flat.tif"
                flat.touch()
                with core.catalog_snapshot():
                    self.assertEqual(core.map_site("map"), "before")
                    self.assertIsNone(core.map_record("new"))
                    self.assertNotIn(str(flat), core.dem_files())
                self.assertEqual(discover.call_count, 1)
            with core.catalog_snapshot():
                self.assertEqual(core.map_site("map"), "after")
                self.assertIsNotNone(core.map_record("new"))
                self.assertIn(str(flat), core.dem_files())
                added.unlink()
                flat.unlink()
                self.assertIsNotNone(core.map_record("new"))
            with core.catalog_snapshot():
                self.assertIsNone(core.map_record("new"))
                self.assertNotIn(str(flat), core.dem_files())
            self.assertEqual(discover.call_count, 3)
            # Ordinary calls still rediscover immediately, with no implicit TTL.
            core.records()
            core.records()
            self.assertEqual(discover.call_count, 5)

    def test_catalog_snapshot_decorator_reuses_nested_scopes_and_refreshes_each_call(self):
        self.manifest("dataset", [{"working_dem": "map.tif", "group": "before"}])

        @core.catalog_snapshot()
        def lookup(depth=0):
            """Nested catalog consumer."""
            if depth:
                return lookup(depth - 1)
            with core.catalog_snapshot():
                return core.map_site("map")

        self.assertEqual(lookup.__name__, "lookup")
        with patch.object(core, "manifest_files", wraps=core.manifest_files) as discover:
            self.assertEqual(lookup(2), "before")
            self.assertEqual(discover.call_count, 1)
            self.manifest("dataset", [{"working_dem": "map.tif", "group": "after"}])
            self.assertEqual(lookup(2), "after")
            self.assertEqual(discover.call_count, 2)
            with core.catalog_snapshot():
                self.assertEqual(lookup(2), "after")
                self.assertEqual(discover.call_count, 3)

    def test_catalog_snapshot_resets_on_exceptions_including_base_exceptions(self):
        for error in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error):
                self.manifest("dataset", [{"working_dem": "map.tif", "group": "before"}])

                @core.catalog_snapshot()
                def fail(error=error):
                    self.assertEqual(core.map_site("map"), "before")
                    raise error("abort scope")

                with patch.object(core, "manifest_files", wraps=core.manifest_files) as discover:
                    with core.catalog_snapshot():
                        with self.assertRaises(error):
                            fail()
                        self.manifest("dataset", [{"working_dem": "map.tif", "group": "after"}])
                        self.assertEqual(core.map_site("map"), "before")
                        self.assertEqual(discover.call_count, 1)
                    with self.assertRaises(error), core.catalog_snapshot():
                        self.assertEqual(core.map_site("map"), "after")
                        raise error("abort outer scope")
                    self.manifest("dataset", [{"working_dem": "map.tif", "group": "latest"}])
                    self.assertEqual(core.map_site("map"), "latest")
                    self.assertEqual(discover.call_count, 3)

    def test_catalog_snapshot_root_mocking_is_isolated_and_restores_outer_snapshot(self):
        self.manifest("dataset", [{"working_dem": "map.tif", "group": "outer"}])
        other = self.root / "other"
        other.mkdir()
        other_manifest = other / "working_dems.json"
        other_manifest.write_text('[{"working_dem":"map.tif","group":"other"}]')
        with (patch.object(core, "manifest_files", wraps=core.manifest_files) as discover,
              core.catalog_snapshot()):
            self.assertEqual(core.map_site("map"), "outer")
            with patch.object(core, "DEM_DIR", str(other)):
                self.assertEqual(core.map_site("map"), "other")
                with core.catalog_snapshot():
                    self.assertEqual(core.dem_path("map"), str(other / "map.tif"))
                    other_manifest.write_text('[{"working_dem":"map.tif","group":"updated"}]')
                    self.assertEqual(core.map_site("map"), "other")
                self.assertEqual(core.map_site("map"), "updated")
            self.assertEqual(core.map_site("map"), "outer")
            with (patch.object(core, "DEM_DIR", self.processed / "dataset" / ".."),
                  core.catalog_snapshot()):
                self.assertEqual(core.map_site("map"), "outer")
            self.assertEqual(discover.call_count, 4)

    def test_catalog_snapshot_failed_entry_does_not_leave_a_snapshot(self):
        self.manifest("dataset", [{"working_dem": "../../escape.tif"}])
        with self.assertRaisesRegex(ValueError, "outside DEM_DIR"), core.catalog_snapshot():
            self.fail("invalid catalog entered")
        self.manifest("dataset", [{"working_dem": "map.tif", "group": "valid"}])
        with core.catalog_snapshot():
            self.assertEqual(core.map_site("map"), "valid")
        self.manifest("dataset", [{"working_dem": "map.tif", "group": "changed"}])
        self.assertEqual(core.map_site("map"), "changed")

    def test_catalog_snapshot_contextvars_are_isolated_and_explicit_copies_inherit(self):
        self.manifest("dataset", [{"working_dem": "map.tif", "group": "before"}])

        @core.catalog_snapshot()
        def lookup():
            return core.map_site("map")

        with patch.object(core, "manifest_files", wraps=core.manifest_files) as discover:
            with core.catalog_snapshot():
                self.manifest("dataset", [{"working_dem": "map.tif", "group": "after"}])
                self.assertEqual(copy_context().run(lookup), "before")
                self.assertEqual(Context().run(lookup), "after")
                self.assertEqual(lookup(), "before")
                self.assertEqual(discover.call_count, 2)
            self.assertEqual(lookup(), "after")
            self.assertEqual(discover.call_count, 3)

    def test_catalog_snapshot_separate_threads_do_not_reuse_or_reset_each_other(self):
        self.manifest("dataset", [{"working_dem": "map.tif", "group": "outer"}])
        entered, release = Event(), Event()

        @core.catalog_snapshot()
        def worker():
            before = core.map_site("map")
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("main thread did not release worker")
            return before, core.map_site("map")

        with (ThreadPoolExecutor(max_workers=1) as pool,
              patch.object(core, "manifest_files", wraps=core.manifest_files) as discover):
            try:
                with core.catalog_snapshot():
                    self.manifest("dataset", [{"working_dem": "map.tif", "group": "worker"}])
                    future = pool.submit(worker)
                    self.assertTrue(entered.wait(timeout=5))
                    self.assertEqual(core.map_site("map"), "outer")
                self.manifest("dataset", [{"working_dem": "map.tif", "group": "latest"}])
                with core.catalog_snapshot():
                    self.assertEqual(core.map_site("map"), "latest")
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5), ("worker", "worker"))
            self.assertEqual(discover.call_count, 3)

    def test_catalog_snapshot_still_rechecks_resolved_path_containment(self):
        self.bundle()
        with core.catalog_snapshot():
            painting_path = core.annotation_path("terrain", "painting")
            assert painting_path is not None
            painting = Path(painting_path)
            outside = self.root / "outside.npy"
            painting.rename(outside)
            painting.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "outside DEM_DIR"):
                core.annotation_path("terrain", "painting")
            with self.assertRaisesRegex(ValueError, "outside DEM_DIR"):
                core.load_painting("terrain")

    def test_sections_use_first_dem_folder_not_inferred_site_or_collection(self):
        self.raster("public/south/tiles/public-map.tif")
        self.raster("reference/sites/north/tiles/bundled-map.tif")
        self.manifest("public/south", [{
            "working_dem": "tiles/public-map.tif", "site": "south",
            "group": "southern-group", "collection": "legacy-name"}])
        self.manifest("reference/sites/north", [{
            "working_dem": "tiles/bundled-map.tif", "site": "north",
            "collection": "not-the-section"}])
        public = self.record("public-map")
        reference = self.record("bundled-map")
        self.assertEqual(public["section"], "public")
        self.assertEqual(reference["section"], "reference")
        self.assertEqual(core.section(public), "public")
        self.assertEqual(public["collection"], "legacy-name")
        self.assertEqual(core.map_site("public-map"), "southern-group")
        self.assertEqual(core.map_site("bundled-map"), "north")
        self.assertNotIn("verified", core.meta_get("bundled-map"))
        self.manifest("", [{"working_dem": "custom/nested/tile.tif", "section": "Explicit"},
                           {"working_dem": "flat.tif", "site": "not-a-section"}])
        self.assertEqual(self.record("tile")["section"], "Explicit")
        self.assertEqual(self.record("flat")["section"], "processed_data")

    def test_section_follows_dem_when_manifest_points_across_dataset_folders(self):
        self.manifest("public/site", [{"working_dem": "../../other/nested/map.tif"}])
        self.assertEqual(self.record("map")["section"], "other")

    def test_only_processed_manifests_and_legacy_companions_are_discovered(self):
        outside = self.root / "reference_data"
        outside.mkdir()
        (outside / "working_dems.json").write_text('[{"working_dem": "hidden.tif"}]')
        self.manifest(".staging", [{"working_dem": "not-published.tif"}])
        self.raster("nested/unlisted.tif")
        dem = self.raster("local.tif")
        self.assertEqual(core.dem_files(), [str(dem)])
        self.assertIsNone(core.companion(str(dem), "nac"))
        image = self.raster("local_nac.tif")
        self.assertEqual(core.companion(str(dem), "nac"), str(image))
        self.assertEqual(core.dem_files(), [str(dem)])

    def test_annotation_paths_metadata_and_records_are_independent_copies(self):
        _, manifest, _, _, _, _ = self.bundle()
        original = self.snapshot()
        record = self.record("terrain")
        self.assertEqual(record["annotations"]["painting"],
                         "reference/location/annotations/painting.npy")
        for kind in core.ANNOTATION_FILES:
            self.assertEqual(core.annotation_path("terrain", kind),
                             str(self.processed / record["annotations"][kind]))
        self.assertIsNone(core.annotation_path("missing", "painting"))
        with self.assertRaisesRegex(ValueError, "Unknown annotation"):
            core.annotation_path("terrain", "provenance")
        record["annotations"]["provenance"]["review"].append("mutated")
        core.records()[0]["annotations"]["painting"] = "mutated.npy"
        metadata = core.meta_get("terrain")
        metadata["provenance"]["review"].clear()
        self.assertEqual(core.meta_get("terrain")["provenance"]["review"], ["explicit request"])
        self.assertTrue(core.meta_get("terrain")["verified"])
        self.assertTrue(core.meta_get("terrain")["restricted"])
        self.assertEqual(self.snapshot(), original)
        self.assertFalse((self.root / "output").exists())
        updated = json.loads(manifest.read_text())
        updated[0]["annotations"]["verified"] = False
        manifest.write_text(json.dumps(updated))
        self.assertFalse(core.meta_get("terrain")["verified"])

    def test_new_path_fields_reject_traversal_hidden_and_symlink_escapes(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.processed / "escape").symlink_to(outside, target_is_directory=True)
        for value, message in (("../../outside/file.npy", "outside DEM_DIR"),
                               (str(outside / "file.npy"), "outside DEM_DIR"),
                               ("../escape/file.npy", "outside DEM_DIR"),
                               ("../.staging/file.npy", "hidden/unpublished")):
            for field in (*core.ANNOTATION_FILES, "feature_sources"):
                with self.subTest(value=value, field=field):
                    record: dict[str, Any] = {"working_dem": "map.tif"}
                    if field == "feature_sources":
                        record[field] = [value]
                    else:
                        record["annotations"] = {field: value}
                    self.manifest("dataset", [record])
                    with self.assertRaisesRegex(ValueError, message):
                        core.records()

    def test_new_path_fields_reject_malformed_values(self):
        for fields in ({"annotations": []}, {"annotations": {"painting": 2}},
                       {"annotations": {"confidence": ""}}, {"feature_sources": "a.tif"},
                       {"feature_sources": [None]}):
            with self.subTest(fields=fields):
                self.manifest("dataset", [{"working_dem": "map.tif", **fields}])
                with self.assertRaises(ValueError):
                    core.records()

    def test_annotation_label_raster_is_not_a_flat_legacy_dem(self):
        dem = self.raster("terrain.tif")
        self.raster("labels.tif")
        self.manifest("", [{"working_dem": "terrain.tif", "annotations": {"labels": "labels.tif"}}])
        self.assertEqual(core.dem_files(), [str(dem)])

    def test_bundled_painting_flags_and_metadata_are_read_only_defaults(self):
        _, _, _, expected, confidence, accepted = self.bundle()
        original = self.snapshot()
        cells, final = core.painting_source("terrain")
        self.assertIs(final, True)
        np.testing.assert_array_equal(cells, expected)
        np.testing.assert_array_equal(core.load_accepted("terrain"), accepted)
        np.testing.assert_array_equal(core.load_confidence("terrain"), confidence)
        np.testing.assert_array_equal(core.load_confidence("terrain", final=True), confidence)
        np.testing.assert_array_equal(core._beside("terrain", True, 1, cells), accepted)
        self.assertFalse(core.load_confidence("terrain", final=False).any())
        self.assertEqual(core.painted_maps(), ({"terrain"}, {"terrain"}))
        cells[:] = 3
        core.load_accepted("terrain")[:] = False
        core.load_confidence("terrain")[:] = 0
        np.testing.assert_array_equal(core.load_painting("terrain"), expected)
        self.assertEqual(self.snapshot(), original)
        self.assertFalse((self.root / "output").exists())

    def test_user_outputs_override_without_inheriting_bundled_flags(self):
        _, _, _, bundled, confidence, _ = self.bundle()
        original = self.snapshot()
        saved = np.where(bundled > 0, 3, 0).astype(np.uint8)
        core.save_painting("terrain", saved, final=True)
        self.assertIs(core.painting_source("terrain")[1], True)
        np.testing.assert_array_equal(core.load_painting("terrain"), saved)
        self.assertFalse(core.load_accepted("terrain").any())
        np.testing.assert_array_equal(core.load_confidence("terrain"), (saved > 0).astype(np.uint8))
        self.assertIsNone(core._beside("terrain", True, 1, saved))
        self.assertEqual(core.painted_maps(), ({"terrain"}, set()))
        core.save_painting("terrain", bundled, final=False,
                           confidence=confidence, accepted=bundled == 1)
        self.assertIs(core.painting_source("terrain")[1], False)
        np.testing.assert_array_equal(core.load_painting("terrain"), bundled)
        np.testing.assert_array_equal(core.load_accepted("terrain"), bundled == 1)
        np.testing.assert_array_equal(core.load_confidence("terrain"), confidence)
        np.testing.assert_array_equal(core.load_confidence("terrain", final=True),
                                      (saved > 0).astype(np.uint8))
        core.save_painting("terrain", np.zeros_like(bundled))
        self.assertFalse(core.load_painting("terrain").any())
        self.assertEqual(core.painted_maps(), (set(), set()))
        core.meta_set("terrain", verified=False, restricted=False, label_status="edited")
        metadata = core.meta_get("terrain")
        self.assertIs(metadata["verified"], False)
        self.assertIs(metadata["restricted"], False)
        self.assertEqual(metadata["label_status"], "edited")
        self.assertEqual(metadata["provenance"], {"review": ["explicit request"]})
        self.assertEqual(self.snapshot(), original)
        self.assertTrue(Path(core.painting_files("terrain", True)[0]).is_file())
        self.assertTrue(Path(core.painting_files("terrain", False)[0]).is_file())

    def test_final_confidence_falls_back_to_bundle_even_when_draft_exists(self):
        _, _, _, cells, confidence, _ = self.bundle()
        core.save_painting("terrain", np.zeros_like(cells))
        self.assertFalse(core.load_confidence("terrain").any())
        np.testing.assert_array_equal(core.load_confidence("terrain", final=True), confidence)
        Path(core.painting_files("terrain")[0]).unlink()
        # Orphan user flags are not flags for a bundled painting.
        output = Path(core.painting_files("terrain", True)[1])
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.zeros_like(cells, dtype=bool))
        np.testing.assert_array_equal(core.load_accepted("terrain"), cells == 2)

    def test_missing_and_wrong_shape_optional_flags_use_painting_defaults(self):
        _, manifest, record, cells, _, _ = self.bundle()
        record["annotations"].pop("confidence")
        record["annotations"].pop("accepted")
        manifest.write_text(json.dumps([record]))
        self.assertFalse(core.load_accepted("terrain").any())
        np.testing.assert_array_equal(core.load_confidence("terrain"), cells > 0)
        record["annotations"].update(confidence="bad.npy", accepted="absent.npy")
        self.array("reference/location/bad.npy", np.ones((2, 2), np.uint8))
        manifest.write_text(json.dumps([record]))
        self.assertFalse(core.load_accepted("terrain").any())
        np.testing.assert_array_equal(core.load_confidence("terrain", final=True), cells > 0)
        self.assertEqual(core.painting_source("missing")[1], None)
        self.assertFalse(core.load_confidence("missing", final=True).any())

    def feature_fixture(self):
        y, x = np.mgrid[:160, :160]
        first = (15 * np.sin(x / 8) + y / 6).astype(np.float32)
        first[35:65, 70:95] = np.nan
        second = (x / 2 + 10 * np.cos(y / 11)).astype(np.float32)
        paths = [self.raster("public/region/first.tif", values=first),
                 self.raster("public/region/second.tif", left=1800, values=second)]
        self.raster("public/region/mask.tif", values=np.zeros_like(first))
        self.manifest("public/region", [
            {"working_dem": p.name, "layers": {"sfs": p.name,
             "valid_data": "mask.tif", "sfs_support": "mask.tif"}}
            for p in paths])
        y, x = np.mgrid[:120, :360]
        local = (50 * np.cos(x / 30) + y).astype(np.float32)
        local[50:60, 160:170] = np.nan
        target = self.raster("reference/site/target.tif", left=900, top=1900, values=local)
        image = self.raster("reference/site/image.tif", left=900, top=1900, values=x + y)
        record = {"working_dem": target.name, "layers": {"nac": image.name},
                  "feature_sources": ["../../public/region/first.tif",
                                      "../../public/region/second.tif"]}
        manifest = self.manifest("reference/site", [record])
        return paths, target, manifest, record, np.isfinite(local)

    def test_feature_sources_normalize_and_remain_caller_owned(self):
        _, _, _, _, _ = self.feature_fixture()
        record = self.record("target")
        self.assertEqual(record["feature_sources"], ["public/region/first.tif", "public/region/second.tif"])
        record["feature_sources"].clear()
        self.assertEqual(len(self.record("target")["feature_sources"]), 2)

    def test_joined_source_features_nearest_transport_preserves_local_display(self):
        sources, target, manifest, record, terrain = self.feature_fixture()
        joined = self.raster("expected/joined.tif", values=np.hstack([core.read(p)[0] for p in sources]))
        original = self.snapshot()
        public_builds = [core.build(str(p)) for p in sources]
        joined_build = core.build(str(joined))
        with patch.object(core, "_build_local", wraps=core._build_local) as local_build:
            linked, names, hs, profile, px = core.build(str(target))
        self.assertEqual(local_build.call_count, 3)
        self.assertEqual(names, core.LAYERS + ["nac"])
        expected = np.full((len(core.BASE_FEATS), *terrain.shape), np.nan, np.float32)
        for stack, source_names, _, grid, _ in [joined_build]:
            footprint = np.full(terrain.shape, np.nan, np.float32)
            reproject(np.ones((grid["height"], grid["width"]), np.float32), footprint,
                      src_transform=grid["transform"], src_crs=grid["crs"],
                      dst_transform=profile["transform"], dst_crs=profile["crs"],
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
            for index, name in enumerate(core.BASE_FEATS):
                warped = np.full(terrain.shape, np.nan, np.float32)
                reproject(stack[source_names.index(name)], warped,
                          src_transform=grid["transform"], src_crs=grid["crs"],
                          dst_transform=profile["transform"], dst_crs=profile["crs"],
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
                expected[index, footprint == 1] = warped[footprint == 1]
        expected[:, ~terrain] = np.nan
        actual = linked[[names.index(n) for n in core.BASE_FEATS]]
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isnan(actual[:, :, :20]).all())
        self.assertTrue(np.isnan(actual[:, :, -20:]).all())
        self.assertTrue(np.isnan(actual[:, 15:45, 90:115]).all())  # source terrain holes
        self.assertTrue(np.isnan(actual[:, ~terrain]).all())
        self.assertTrue(np.isfinite(actual[:, 60:80, 210:240]).all())  # zero support is not a mask
        self.assertEqual(self.snapshot(), original)
        self.assertFalse((self.root / "output").exists())
        record.pop("feature_sources")
        manifest.write_text(json.dumps([record]))
        local, local_names, local_hs, local_profile, local_px = core.build(str(target))
        self.assertEqual(names, local_names)
        self.assertEqual(profile, local_profile)
        self.assertEqual(px, local_px)
        np.testing.assert_array_equal(hs, local_hs)
        for name in set(names) - set(core.BASE_FEATS):
            np.testing.assert_array_equal(linked[names.index(name)], local[names.index(name)])
        self.assertFalse(np.allclose(actual, local[[names.index(n) for n in core.BASE_FEATS]],
                                     equal_nan=True))
        for path, before in zip(sources, public_builds):
            after = core.build(str(path))
            np.testing.assert_array_equal(after[0], before[0])
            np.testing.assert_array_equal(after[2], before[2])

    def test_split_terrain_matches_continuous_terrain_at_both_joins(self):
        y, x = np.mgrid[:256, :256]
        z = (80 * np.exp(-((x - 150)**2 + (y - 130)**2) / 5000)
             + 12 * np.sin(x / 9) + y / 3).astype(np.float32)
        z[30:45, 200:220] = np.nan
        whole = self.raster("dataset/whole.tif", values=z)
        sources = []
        for row in (0, 128):
            for col in (0, 128):
                tile = self.raster(f"dataset/part-{row}-{col}.tif",
                                   left=1000 + col * 5, top=2000 - row * 5,
                                   values=z[row:row+128, col:col+128])
                sources.append(tile.name)
        target = self.raster("dataset/target.tif", values=z)
        manifest = self.manifest("dataset", [{"working_dem": target.name, "feature_sources": sources}])
        expected, names, *_ = core.build(str(whole))
        before = self.snapshot()
        actual = core.build(str(target))[0]
        indices = [names.index(n) for n in core.BASE_FEATS]
        np.testing.assert_array_equal(actual[indices], expected[indices])
        self.assertEqual(before, self.snapshot())
        # Manifest order does not select a different edge condition or overwrite.
        manifest.write_text(json.dumps([{"working_dem": target.name, "feature_sources": sources[::-1]}]))
        np.testing.assert_array_equal(core.build(str(target))[0][indices], expected[indices])

    def test_feature_transport_no_coverage_is_nan_not_local_features(self):
        source = self.raster("dataset/source.tif", left=50000)
        target = self.raster("dataset/target.tif")
        self.manifest("dataset", [{"working_dem": target.name, "feature_sources": [source.name]}])
        X, names, hs, _, _ = core.build(str(target))
        self.assertTrue(np.isnan(X[[names.index(n) for n in core.BASE_FEATS]]).all())
        self.assertTrue(np.isfinite(X[names.index("rel")]).all())
        self.assertTrue(np.isfinite(hs).all())

    def test_overlap_rejected_even_for_all_nodata_sources(self):
        for name in ("first", "second"):
            self.raster(f"dataset/{name}.tif", values=np.full((120, 120), np.nan))
        target = self.raster("dataset/target.tif")
        self.manifest("dataset", [{"working_dem": target.name,
                                  "feature_sources": ["first.tif", "second.tif"]}])
        with self.assertRaisesRegex(ValueError, "Overlapping feature source ownership"):
            core.build(str(target))

    def test_duplicate_source_ownership_is_rejected(self):
        source = self.raster("dataset/source.tif")
        target = self.raster("dataset/target.tif")
        self.manifest("dataset", [{"working_dem": target.name,
                                  "feature_sources": [source.name, source.name]}])
        with self.assertRaisesRegex(ValueError, "Overlapping feature source ownership"):
            core.build(str(target))

    def test_cycles_fail_before_computing_terrain(self):
        for records in ([{"working_dem": "a.tif", "feature_sources": ["a.tif"]}],
                        [{"working_dem": "a.tif", "feature_sources": ["b.tif"]},
                         {"working_dem": "b.tif", "feature_sources": ["a.tif"]}]):
            with self.subTest(records=records):
                self.manifest("dataset", records)
                with (patch.object(core, "_build_local", side_effect=AssertionError("heavy build")),
                      self.assertRaisesRegex(ValueError, "Cyclic feature_sources")):
                    core.build(str(self.processed / "dataset/a.tif"))

    def test_shared_dependency_is_computed_once_without_persistent_cache(self):
        common = self.raster("dataset/common.tif", shape=(120, 240))
        left = self.raster("dataset/left.tif")
        right = self.raster("dataset/right.tif", left=1600)
        target = self.raster("dataset/target.tif", shape=(120, 240))
        self.manifest("dataset", [
            {"working_dem": left.name, "feature_sources": [common.name]},
            {"working_dem": right.name, "feature_sources": [common.name]},
            {"working_dem": target.name, "feature_sources": [left.name, right.name]}])
        with patch.object(core, "_build_local", wraps=core._build_local) as built:
            first = core.build(str(target))
        self.assertEqual(built.call_count, 4)
        baseline = core.build(str(common))
        indices = [first[1].index(n) for n in core.BASE_FEATS]
        np.testing.assert_array_equal(first[0][indices], baseline[0][indices])
        self.raster("dataset/common.tif", values=np.zeros((120, 240)))
        second = core.build(str(target))
        self.assertFalse(np.array_equal(first[0][indices], second[0][indices]))
        second[0][:] = -123
        self.assertFalse((core.build(str(target))[0] == -123).any())

    def test_linked_source_cannot_be_silently_decimated(self):
        self.raster("dataset/source.tif", shape=(160, 160))
        target = self.raster("dataset/target.tif")
        self.manifest("dataset", [{"working_dem": target.name, "feature_sources": ["source.tif"]}])
        with (patch.object(core, "MAX_SIDE", 120),
              self.assertRaisesRegex(ValueError, "changed its native grid")):
            core.build(str(target))

    def test_exact_untagged_grids_work_but_missing_crs_never_gets_earth_fallback(self):
        source = self.raster("dataset/source.tif", crs=None)
        target = self.raster("dataset/target.tif", crs=None, values=np.zeros((120, 120)))
        self.manifest("dataset", [{"working_dem": target.name, "feature_sources": [source.name]}])
        linked = core.build(str(target))
        public = core.build(str(source))
        indices = [linked[1].index(n) for n in core.BASE_FEATS]
        np.testing.assert_array_equal(linked[0][indices], public[0][indices])
        for crs, left in ((None, 1005), ("ESRI:103878", 1000)):
            with self.subTest(crs=crs, left=left):
                self.raster("dataset/target.tif", crs=crs, left=left)
                with (patch.object(core, "reproject", side_effect=AssertionError("guessed CRS")),
                      self.assertRaisesRegex(ValueError, "requires both CRSs")):
                    core.build(str(target))

    def test_feature_warp_validates_shape_and_projects_between_real_crss(self):
        values = np.arange(120 * 120, dtype=np.float32).reshape(120, 120)
        grid = {"height": 120, "width": 120, "transform": from_origin(-1000, 1000, 10, 10),
                "crs": rasterio.CRS.from_string("ESRI:103878")}
        with self.assertRaisesRegex(ValueError, "shape"):
            core._feature_warp(values[:100], grid, grid)
        copied = core._feature_warp(values, grid, grid)
        copied[:] = -1
        self.assertFalse((values == -1).any())
        target = dict(grid, crs=rasterio.CRS.from_string(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=10 +R=1737400 +units=m +no_defs"))
        expected = np.full((120, 120), np.nan, np.float32)
        reproject(values, expected, src_transform=grid["transform"], src_crs=grid["crs"],
                  dst_transform=target["transform"], dst_crs=target["crs"],
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
        actual = core._feature_warp(values, grid, target)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).any())
        self.assertTrue(np.isnan(actual).any())


if __name__ == "__main__":
    unittest.main()
