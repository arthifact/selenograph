"""Generic training and dataset isolation; disposable data, no estimator fitting."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from app import assistant_model as assistant
from app import core


class GenericTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processed = self.root / "processed"
        self.processed.mkdir()
        self.records = []
        for module, key, value in (
            (core, "DEM_DIR", self.processed),
            (core, "OUT_DIR", self.root / "output/paintings"),
            (core, "MAP_DIR", self.root / "output/maps"),
            (core, "LEGACY_OUT_DIR", self.root / "output/maps"),
            (core, "DRAFT_DIR", self.root / "output/painting_drafts"),
            (core, "LEGACY_DRAFT_DIR", self.root / "output/drafts"),
            (core, "LEGACY_MIRRORED_DIR", self.root / "output"),
            (assistant, "MODEL_DIR", self.root / "models"),
        ):
            patcher = patch.object(module, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(core._load_catalog.cache_clear)
        self.cells = np.ones((core.GRID, core.GRID), np.uint8)
        self.cells[:, core.GRID // 2:] = 2
        self.X = np.ones((len(core.BASE_FEATS), core.GRID, core.GRID), np.float32)

    def publish(self):
        (self.processed / "working_dems.json").write_text(json.dumps(self.records), encoding="utf-8")

    def raster(self, path, values=None, *, left=1000, crs: str | None = "ESRI:103878", nodata=-9999):
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.ones((core.GRID, core.GRID), np.float32) if values is None else np.asarray(values)
        with rasterio.open(path, "w", driver="GTiff", count=1, dtype=values.dtype,
                           width=values.shape[1], height=values.shape[0], crs=crs,
                           transform=from_origin(left, 2000, 5, 5), nodata=nodata) as dst:
            dst.write(values, 1)
        return path

    def tile(self, name, group, *, left=1000, **metadata):
        metadata.setdefault("verified", True)
        path = self.raster(self.processed / group / f"{name}.tif", left=left)
        self.records.append(dict(working_dem=str(path.relative_to(self.processed)),
                                 site=group, group=group, **metadata))
        self.publish()
        return path

    def bundle(self, name="tile-001", group="ordinary (reference)", *, painting=True, labels=False):
        path = self.tile(name, group)
        folder = path.parent / "annotations" / name
        folder.mkdir(parents=True)
        annotations: dict[str, Any] = dict(verified=True, restricted=True, provenance={"origin": "offline import"})
        if painting:
            np.save(folder / "painting.npy", self.cells)
            annotations["painting"] = str((folder / "painting.npy").relative_to(self.processed))
        if labels:
            self.raster(folder / "labels.tif", self.cells, nodata=255)
            annotations["labels"] = str((folder / "labels.tif").relative_to(self.processed))
        self.records[-1]["annotations"] = annotations
        self.publish()
        return path, folder

    def snapshot(self, folder):
        return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in folder.rglob("*") if p.is_file()}

    @staticmethod
    def fitted(samples):
        return dict(schema=assistant.schema(), samples=samples, model=None, revision="test-only")

    def update(self, name, **kwargs):
        with patch.object(assistant, "fit", side_effect=self.fitted):
            return assistant.update(name, self.X, core.BASE_FEATS, self.cells, self.X,
                                    name, "new-revision", **kwargs)[0]

    def footprint(self, path) -> dict[str, Any]:
        with rasterio.open(path) as src:
            return dict(crs_wkt=src.crs.to_wkt() if src.crs else None, bounds=list(src.bounds))

    def test_training_is_exclusively_the_supplied_painting_for_every_name(self):
        confidence = np.full(self.cells.shape, core.SURE, np.uint8)
        confidence[:, :20] = core.UNSURE
        accepted = np.zeros(self.cells.shape, bool)
        accepted[:, 80:] = True
        cells = self.cells.copy()
        cells[:10] = 0
        for name in ("tile-001", "MM026", "unregistered"):
            lab, weights, strata = assistant.training_pixels(
                name, (240, 240), cells, confidence, "not-opened.tif", accepted)
            np.testing.assert_array_equal(lab, core.cells_to_labels(cells, (240, 240)))
            np.testing.assert_array_equal(strata, lab)
            self.assertTrue((weights[lab == 255] == 0).all())
            self.assertTrue((weights[core.per_pixel(accepted & (cells > 0), lab.shape)] == core.APPROVED).all())
            self.assertTrue(np.allclose(weights[30:, :40], core.CONFIDENCE[core.UNSURE][1]))
        self.assertFalse(hasattr(assistant, "reference"))

    def test_update_preserves_source_identity_but_replaces_label_dependent_memory(self):
        dem, _ = self.bundle()
        name = dem.stem
        self.records[-1]["feature_sources"] = [str(dem.relative_to(self.processed))]
        self.publish()
        old = dict(source_kind="reference_snapshot", reference_group="old-group",
                   geographic_group="old-group", public_region="old-region", public_tiles=["old-tile"],
                   map_id="stable-map", tile_id=name, snapshot="saved", source_revision="old",
                   footprint={"bounds": [0, 0, 1, 1]}, grid={"bounds": [0, 0, 1, 1]},
                   confidence_counts={"1": 123}, weight_policy="obsolete", painting_hash="old")
        assistant.save(self.fitted({name: old}))
        before = self.snapshot(self.processed)
        with patch.object(core, "manifest_files", wraps=core.manifest_files) as discover:
            result = self.update(name, dem_path=str(dem))
        self.assertEqual(discover.call_count, 1)
        sample = result["samples"][name]
        self.assertEqual(set(result["samples"]), {name})
        for field in ("source_kind", "reference_group", "public_region", "public_tiles", "map_id", "tile_id", "snapshot"):
            self.assertEqual(sample[field], old[field])
        self.assertEqual(sample["geographic_group"], "ordinary (reference)")
        self.assertEqual(sample["feature_sources"], [name])
        self.assertTrue(sample["restricted"])
        self.assertEqual(sample["provenance"], {"origin": "offline import"})
        self.assertEqual(sample["source_revision"], "new-revision")
        self.assertEqual(sample["footprint"]["bounds"], self.footprint(dem)["bounds"])
        self.assertEqual(sample["footprint"]["shape"], [core.GRID, core.GRID])
        self.assertEqual(sample["grid"], sample["footprint"])
        self.assertEqual(sample["painting_hash"], assistant.painting_hash(self.cells))
        self.assertNotIn("confidence_counts", sample)
        self.assertNotIn("weight_policy", sample)
        self.assertEqual(self.snapshot(self.processed), before)
        self.assertFalse(Path(core.OUT_DIR).exists())

    def test_ordinary_sample_gets_actual_grid_and_generic_metadata_without_dem_argument(self):
        dem = self.tile("new", "new-location", restricted=True)
        result = self.update("new")
        sample = result["samples"]["new"]
        self.assertEqual(sample["source_kind"], "processed_map")
        self.assertEqual(sample["geographic_group"], "new-location")
        self.assertEqual(sample["footprint"]["bounds"], self.footprint(dem)["bounds"])
        self.assertNotIn("feature_sources", sample)
        self.assertTrue(sample["restricted"])
        core.meta_set("new", restricted=False)
        self.assertFalse(self.update("new")["samples"]["new"]["restricted"])

    def test_group_aliases_are_transitive_for_all_source_kinds(self):
        self.tile("test", "survey-location", geography_aliases=["old-spelling"])
        self.tile("sibling", "old-spelling", left=20000,
                  geography_aliases={"older-spelling": "old-spelling"})
        samples = {
            "sibling": {"site_id": "site-name"},
            "archive": {"source_kind": "anything", "geographic_group": "canonical",
                        "public_region": "older-spelling", "site_id": "archive-site"},
            "legacy": {"source_kind": "reference_snapshot", "reference_group": "archive-site"},
            "plain": {"site_id": "site-name"},
            "elsewhere": {"geographic_group": "other"},
            "no-metadata": {},
        }
        with patch.object(core, "manifest_files", wraps=core.manifest_files) as discover:
            kept = assistant.retained_samples(samples, {"test"})
        self.assertEqual(discover.call_count, 1)
        self.assertEqual(set(kept), {"elsewhere", "no-metadata"})
        self.assertEqual(assistant.retained_samples(samples, set()), samples)

    def test_direct_key_and_public_tile_hits_remove_whole_generic_groups(self):
        samples = {"old": {"site_id": "alpha", "public_tiles": ["public-test"]},
                   "same": {"geographic_group": "alpha"}, "safe": {"site_id": "beta"}, "unknown": {}}
        for held in ({"old"}, {"public-test"}):
            self.assertEqual(set(assistant.retained_samples(samples, held)), {"safe", "unknown"})

    def test_footprint_hit_excludes_entire_unavailable_group_not_just_one_sample(self):
        dem = self.tile("test", "test-location")
        far = dict(self.footprint(dem), bounds=[20000, 1000, 21000, 2000])
        samples = {"old": {"geographic_group": "alpha", "footprint": self.footprint(dem)},
                   "sibling": {"site_id": "alpha", "footprint": far},
                   "safe": {"site_id": "beta", "footprint": far}, "normal-legacy": {}}
        self.assertEqual(set(assistant.retained_samples(samples, {"test"})), {"safe", "normal-legacy"})

    def test_missing_spatial_metadata_is_required_only_for_spatial_exclusion(self):
        self.tile("test", "test-location")
        samples = {"old": {"geographic_group": "alpha", "public_tiles": ["unavailable"]},
                   "plain": {"site_id": "other"}, "no-metadata": {}}
        with self.assertRaisesRegex(ValueError, "no usable footprint: old"):
            assistant.retained_samples(samples, {"test"})
        self.assertEqual(set(assistant.retained_samples(samples, {"unavailable"})), {"plain", "no-metadata"})
        self.assertEqual(assistant.retained_samples(samples, {"missing-test"}), samples)
        samples.pop("old")
        self.assertEqual(assistant.retained_samples(samples, {"test"}), samples)

    def test_footprints_are_compared_in_their_own_crs(self):
        dem = self.tile("test", "test-location")
        self.raster(dem, left=0, crs="EPSG:3857")
        source = self.footprint(dem)
        footprint = dict(crs_wkt=CRS.from_epsg(4326).to_wkt(),
                         bounds=transform_bounds("EPSG:3857", "EPSG:4326", *source["bounds"]))
        self.assertEqual(assistant.retained_samples({"old": {"footprint": footprint}}, {"test"}), {})

    def test_missing_holdout_crs_is_not_required_for_key_or_group_only_checks(self):
        dem = self.tile("test", "alpha")
        self.raster(dem, crs=None)
        self.assertEqual(assistant.retained_samples({"same": {"site_id": "alpha"}, "plain": {}}, {"test"}), {"plain": {}})
        footprint = dict(crs_wkt=CRS.from_epsg(3857).to_wkt(), bounds=[0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "missing CRS on test"):
            assistant.retained_samples({"old": {"footprint": footprint}}, {"test"})

    def test_new_sample_in_held_group_is_refused_before_fit_or_save(self):
        self.enterContext(patch.object(core, "meta_get", return_value={"verified": True}))
        self.tile("test", "alpha", geography_aliases=["alias-alpha"])
        self.tile("new", "alias-alpha", left=20000)
        original = self.fitted({})
        assistant.save(original)
        artifact = Path(assistant.MODEL_DIR) / f"{core.MODEL}.joblib"
        before = artifact.read_bytes()
        for name in ("new", "not-in-catalog"):
            with self.subTest(name=name), patch.object(core, "held_out_maps", return_value={"test"}), \
                    patch.object(assistant, "fit") as fit, patch.object(assistant, "save") as save:
                with self.assertRaisesRegex(ValueError, "held-out geography"):
                    assistant.update(name, self.X, core.BASE_FEATS, self.cells, self.X,
                                     "alias-alpha", "revision")
                fit.assert_not_called()
                save.assert_not_called()
        self.assertEqual(artifact.read_bytes(), before)

    def test_new_sample_is_checked_against_aliases_of_removed_old_memory(self):
        self.enterContext(patch.object(core, "meta_get", return_value={"verified": True}))
        assistant.save(self.fitted({"old-test": {"geographic_group": "canonical", "site_id": "alias"}}))
        with patch.object(core, "held_out_maps", return_value={"old-test"}), \
                patch.object(assistant, "fit") as fit, patch.object(assistant, "save") as save:
            with self.assertRaisesRegex(ValueError, "held-out geography"):
                assistant.update("new", self.X, core.BASE_FEATS, self.cells, self.X, "alias", "revision")
            fit.assert_not_called()
            save.assert_not_called()

    def test_unavailable_holdout_uses_its_remembered_footprint(self):
        dem = self.tile("terrain", "location")
        footprint = self.footprint(dem)
        dem.unlink()
        samples = {"old-test": {"site_id": "alpha", "footprint": footprint},
                   "old-training": {"site_id": "beta", "footprint": footprint}, "unrelated": {}}
        self.assertEqual(assistant.retained_samples(samples, {"old-test"}), {"unrelated": {}})

    def test_registered_sample_can_supply_its_grid_when_old_memory_lacks_one(self):
        self.tile("test", "alpha")
        self.tile("training", "beta", left=20000)
        samples = {"training": {"public_tiles": ["legacy-tile"]}}
        self.assertEqual(assistant.retained_samples(samples, {"test"}), samples)

    def test_new_overlapping_sample_is_refused_and_refit_never_writes_sources(self):
        self.tile("test", "alpha")
        dem = self.tile("old", "beta")
        assistant.save(self.fitted({"old": {"geographic_group": "beta", "footprint": self.footprint(dem)}}))
        before = self.snapshot(self.processed)
        with patch.object(core, "held_out_maps", return_value={"test"}):
            with patch.object(assistant, "fit") as fit, patch.object(assistant, "save") as save:
                with self.assertRaisesRegex(ValueError, "overlaps a test map"):
                    assistant.update("old", self.X, core.BASE_FEATS, self.cells, self.X,
                                     "beta", "revision", dem_path=str(dem))
                fit.assert_not_called()
                save.assert_not_called()
            with patch.object(assistant, "fit", side_effect=self.fitted):
                self.assertEqual(assistant.refit()["samples"], {})
        self.assertEqual(self.snapshot(self.processed), before)
        self.assertFalse(Path(core.OUT_DIR).exists())


if __name__ == "__main__":
    unittest.main()
