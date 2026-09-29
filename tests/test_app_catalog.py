"""Catalog/path integration without touching the real DEMs, paintings or models."""
import contextlib
import json
import os
import runpy
from pathlib import Path
import tempfile
import unittest
from typing import cast
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest

from app import assistant_model as assistant
from app import core, evaluate, paths
from app import import_reference_maps as reference


class CatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dem_root = self.root / "processed"
        self.dem_root.mkdir()
        for module, attr, value in (
            (core, "DEM_DIR", self.dem_root),
            (core, "OUT_DIR", self.root / "output/paintings"),
            (core, "MAP_DIR", self.root / "output/maps"),
            (core, "LEGACY_OUT_DIR", self.root / "output/maps"),
            (core, "DRAFT_DIR", self.root / "output/painting_drafts"),
            (core, "LEGACY_DRAFT_DIR", self.root / "output/drafts"),
            (core, "LEGACY_MIRRORED_DIR", self.root / "output"),
            (assistant, "MODEL_DIR", self.root / "models"),
            (paths, "OUTPUT_DIR", self.root / "output"),
            (paths, "OUT_DIR", self.root / "output/paintings"),
            (paths, "MAP_DIR", self.root / "output/maps"),
            (paths, "LEGACY_OUT_DIR", self.root / "output/maps"),
            (paths, "DRAFT_DIR", self.root / "output/painting_drafts"),
            (paths, "LEGACY_DRAFT_DIR", self.root / "output/drafts"),
            (paths, "LEGACY_MIRRORED_DIR", self.root / "output"),
            (paths, "POSTER_DIR", self.root / "poster"),
            (reference, "REFERENCE_ROOT", self.root / "references"),
            (reference, "EXTRA", {"Nobile1-MS1": self.root / "missing.gpkg"}),
        ):
            patcher = patch.object(module, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def raster(self, relative, left=1000, image=False):
        path = self.dem_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        y, x = np.mgrid[:120, :120]
        values = (x + y if image else np.sin(x / 10) * 20 + y / 5).astype(np.float32)
        values[:4] = -9999
        with rasterio.open(path, "w", driver="GTiff", count=1, dtype="float32",
                           width=120, height=120, nodata=-9999,
                           crs="ESRI:103878",
                           transform=from_origin(left, 2000, 5, 5)) as dst:
            dst.write(values, 1)
        return path

    def public(self, region="south", name="public-dem", nac=True, left=1000):
        folder = Path(region)
        dem = self.raster(folder / "tiles" / f"{name}.tif", left=left)
        layers = {"sfs": f"tiles/{name}.tif"}
        if nac:
            self.raster(folder / "imagery" / f"{name}-nac.tif", left=left, image=True)
            layers["nac"] = f"imagery/{name}-nac.tif"
        self.raster(folder / "quality" / "not-a-dem.tif", left=left)
        layers["quality"] = "quality/not-a-dem.tif"
        record = dict(working_dem=f"tiles/{name}.tif", tile_id=name, group=region,
                      site=region, collection="public", size=[120, 120],
                      transform=[5, 0, left, 0, -5, 2000], layers=layers)
        manifest = self.json(self.dem_root / folder / "working_dems.json", [record])
        return dem, manifest

    def test_default_paths_are_project_root_relative_not_cwd_relative(self):
        with contextlib.chdir(self.root):
            defaults = runpy.run_path(paths.__file__)
            self.assertEqual(defaults["DEM_DIR"], paths.PROJECT_ROOT / "data/processed_data")
            self.assertEqual(defaults["OUT_DIR"], defaults["OUTPUT_DIR"] / "paintings")
            self.assertEqual(defaults["OUT_DIR"], paths.PROJECT_ROOT / "output/paintings")
            self.assertEqual(defaults["DRAFT_DIR"], paths.PROJECT_ROOT / "output/painting_drafts")
            self.assertEqual(defaults["LEGACY_OUT_DIR"], paths.PROJECT_ROOT / "output/maps")
            self.assertEqual(defaults["LEGACY_DRAFT_DIR"], paths.PROJECT_ROOT / "output/drafts")
            self.assertEqual(defaults["LEGACY_MIRRORED_DIR"], defaults["OUTPUT_DIR"])
            self.assertEqual(defaults["MODEL_DIR"], paths.PROJECT_ROOT / "models")

            with patch.object(core, "DEM_DIR", paths.DEM_DIR):
                self.assertEqual(core.test_maps_path(), paths.PROJECT_ROOT / "output/test_maps.json")

    def test_recursive_catalog_normalizes_paths_and_preserves_legacy_ids(self):
        public, manifest = self.public()
        legacy = self.raster("legacy/sections/mons-mouton/MM026_dem_5m.TIFF", left=20000)
        old_record = dict(working_dem="sections/mons-mouton/MM026_dem_5m.TIFF",
                          site="MM026", group="MM026", collection="legacy",
                          layers={"nac": {"status": "available", "path": "nac.tif"}})
        self.json(self.dem_root / "legacy/working_dems.json", [old_record])
        original = manifest.read_bytes()
        records = core.records()
        self.assertEqual([core.dem_name(p) for p in core.dem_files()],
                         ["MM026_dem_5m", "public-dem"])
        self.assertEqual(core.dem_path("MM026_dem_5m"), str(legacy))
        self.assertEqual(core.dem_path("public-dem"), str(public))
        record = core.map_record("public-dem")
        assert record is not None
        self.assertEqual(record["layers"]["nac"], "south/imagery/public-dem-nac.tif")
        self.assertEqual(records[0]["layers"]["nac"],
                         {"status": "available", "path": "legacy/nac.tif"})
        self.assertEqual(core.record_path(records[1], "sfs"), str(public))
        self.assertEqual(record["section"], "south")
        self.assertEqual(core.site_id(public), "south")
        self.assertEqual(core.map_site("public-dem"), "south")
        self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual(core.held_out_maps(["public-dem"]), {"public-dem"})

    def test_nested_rasters_without_manifests_are_not_maps(self):
        self.raster("unlisted/terrain.tif")
        self.raster("south/quality/reliability.tif")
        self.assertEqual(core.dem_files(), [])
        flat = self.raster("local.TIFF")
        self.raster("local_nac.tif", image=True)
        self.assertEqual(core.dem_files(), [str(flat)])
        self.assertEqual(core.dem_path("local"), str(flat))

    def test_hidden_staging_is_ignored_until_publication(self):
        dem, live = self.public()
        staged = self.dem_root / "south/.staging/build/working_dems.json"
        self.json(staged, [{"working_dem": "same-id/public-dem.tif"},
                           {"working_dem": "new.tif"}])
        self.json(self.dem_root / ".prepare-old/working_dems.json",
                  [{"working_dem": "also-hidden.tif"}])
        self.raster(".hidden.tif")
        with patch.object(core.json, "loads", wraps=json.loads) as parsed:
            self.assertEqual(core.manifest_files(), [live])
            self.assertEqual(core.dem_files(), [str(dem)])
            self.assertIsNone(core.map_record("new"))
            staged.parent.rename(self.dem_root / "south/published")
            # Once visible, even a conflicting ID must be checked immediately.
            with self.assertRaisesRegex(ValueError, "Duplicate map ID 'public-dem'"):
                core.records()
            self.assertEqual(parsed.call_count, 3)

    def test_per_map_lookups_parse_once_and_do_not_expose_cached_mutable_records(self):
        count = 520
        self.json(self.dem_root / "south/working_dems.json", [
            {"working_dem": f"tiles/map-{i}.tif", "site": "south", "group": "south",
             "layers": {"nac": {"path": f"images/map-{i}.tif", "status": "available"}}}
            for i in range(count)])
        with patch.object(core.json, "loads", wraps=json.loads) as parsed:
            record = None
            for i in range(count):
                self.assertEqual(core.map_site(f"map-{i}"), "south")
                record = core.map_record(f"map-{i}")
                assert record is not None
                self.assertEqual(record["working_dem"], f"south/tiles/map-{i}.tif")
            assert record is not None
            record["layers"]["nac"]["path"] = "changed.tif"
            listing = core.records()
            listing[0]["site"] = "changed"
            listing[0]["layers"]["nac"]["status"] = "changed"
            first = core.map_record("map-0")
            last = core.map_record("map-519")
            assert first is not None and last is not None
            self.assertEqual(first["site"], "south")
            self.assertEqual(first["layers"]["nac"]["status"], "available")
            self.assertEqual(last["layers"]["nac"]["path"], "south/images/map-519.tif")
            self.assertEqual(parsed.call_count, 1)

    def test_catalog_refreshes_modified_added_and_removed_manifests(self):
        _, manifest = self.public()
        self.assertEqual(core.map_site("public-dem"), "south")
        stamp = manifest.stat()
        updated = json.loads(manifest.read_text())
        updated[0].update(site="north", group="north")
        self.json(manifest, updated)                       # same length, changed content
        self.assertEqual(manifest.stat().st_size, stamp.st_size)
        os.utime(manifest, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
        self.assertEqual(core.map_site("public-dem"), "north")
        added_dem, added = self.public(region="east", name="added", nac=False)
        self.assertEqual(core.map_site("added"), "east")
        self.assertEqual(core.dem_path("added"), str(added_dem))
        added.unlink()
        self.assertIsNone(core.map_record("added"))
        self.assertNotIn(str(added_dem), core.dem_files())

    def test_cache_separates_patched_dem_roots(self):
        self.public()
        self.assertEqual(core.map_site("public-dem"), "south")
        other = self.root / "other"
        self.json(other / "working_dems.json", [
            {"working_dem": "public-dem.tif", "site": "other"}])
        with patch.object(core, "DEM_DIR", str(other)):
            self.assertEqual(core.map_site("public-dem"), "other")
            self.assertEqual(core.dem_path("public-dem"), str(other / "public-dem.tif"))
        self.assertEqual(core.map_site("public-dem"), "south")

    def test_catalog_rejects_dem_and_layer_paths_outside_dem_root(self):
        outside = self.root / "outside.tif"
        escape = "../../../outside.tif"
        examples = [
            {"working_dem": escape},
            {"working_dem": str(outside)},
            {"working_dem": "dem.tif", "layers": {"nac": escape}},
            {"working_dem": "dem.tif", "layers": {"nac": {"path": str(outside)}}},
            {"working_dem": "dem.tif", "layers": {"quality": {"coverage": escape}}},
        ]
        for record in examples:
            with self.subTest(record=record):
                self.json(self.dem_root / "south/working_dems.json", [record])
                with self.assertRaisesRegex(ValueError, "outside DEM_DIR"):
                    core.records()
        with self.assertRaisesRegex(ValueError, "outside DEM_DIR"):
            core.record_path({"working_dem": "../outside.tif"})

    def test_catalog_rejects_symlink_escapes_and_hidden_raster_references(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.dem_root / "link").symlink_to(outside, target_is_directory=True)
        manifest = self.dem_root / "working_dems.json"
        self.json(manifest, [{"working_dem": "link/dem.tif"}])
        with self.assertRaisesRegex(ValueError, "outside DEM_DIR"):
            core.records()
        self.json(manifest, [{"working_dem": ".staging/dem.tif"}])
        with self.assertRaisesRegex(ValueError, "hidden/unpublished"):
            core.records()
        manifest.unlink()
        source = self.json(outside / "working_dems.json", [{"working_dem": "dem.tif"}])
        manifest.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "outside DEM_DIR"):
            core.records()

    def test_paths_may_cross_manifest_directories_inside_dem_root(self):
        dem = self.raster("legacy/shared/dem.tif")
        self.json(self.dem_root / "south/working_dems.json", [
            {"working_dem": "../legacy/shared/dem.tif",
             "layers": {"nac": "../legacy/shared/nac.tif"}}])
        self.assertEqual(core.dem_path("dem"), str(dem))
        record = core.map_record("dem")
        assert record is not None
        self.assertEqual(record["layers"]["nac"], "legacy/shared/nac.tif")

    def test_duplicate_manifest_ids_never_resolve_to_an_arbitrary_tile(self):
        self.public()
        self.assertEqual(core.map_site("public-dem"), "south")
        _, conflicting = self.public(region="north", name="public-dem")
        for lookup in (core.records, core.dem_files, lambda: core.map_record("public-dem"),
                       lambda: core.map_site("public-dem"), lambda: core.dem_path("public-dem")):
            with self.subTest(lookup=lookup):
                with self.assertRaisesRegex(ValueError, "Duplicate map ID 'public-dem'") as error:
                    lookup()
                self.assertIn("north/working_dems.json", str(error.exception))
                self.assertIn("south/working_dems.json", str(error.exception))
        conflicting.unlink()
        self.assertEqual(core.map_site("public-dem"), "south")

    def test_flat_fallback_cannot_collide_with_registered_or_other_flat_ids(self):
        self.public()
        self.assertEqual(core.map_site("public-dem"), "south")
        flat = self.raster("public-dem.TIFF")
        with self.assertRaisesRegex(ValueError, "Duplicate map ID 'public-dem'"):
            core.dem_path("public-dem")
        flat.unlink()
        self.assertEqual(core.map_site("public-dem"), "south")
        self.raster("local.tif")
        self.raster("local.TIFF")
        with self.assertRaisesRegex(ValueError, "Duplicate map ID 'local'"):
            core.dem_path("local")

    def test_nac_is_display_only_and_keeps_the_model_schema(self):
        dem, _ = self.public()
        schema = assistant.schema()
        X, names, *_ = core.build(str(dem))
        self.assertIn("nac", names)
        self.assertNotIn("quality", names)
        self.assertTrue(np.isnan(X[names.index("nac"), :4]).all())
        aligned = assistant.features(X, names)
        without_nac = [n for n in names if n != "nac"]
        np.testing.assert_array_equal(aligned, assistant.features(
            X[[names.index(n) for n in without_nac]], without_nac))
        self.assertEqual(assistant.schema(), schema)
        self.assertEqual(schema["version"], 1)
        recipe = schema["recipe"]
        assert isinstance(recipe, dict)
        self.assertEqual(cast(dict[str, object], recipe)["version"], 6)
        self.assertEqual(recipe["name"], "terrain_baseline")
        self.assertEqual(schema["features"], ["slope", "rough", "svf", "rel_local",
                                              "curv", "svf_local"])
        self.assertEqual(core.backend("rf").__name__, "app.model_rf")
        self.assertEqual(core.backend("gbm").__name__, "app.model_gbm")
        self.assertEqual(core.model_files(), ["gbm", "rf", "unet"])

    def reference_snapshot(self, name="MM026_dem_5m"):
        self.json(Path(reference.REFERENCE_ROOT) / "sites/mons-mouton/site.json",
                  {"site_id": "mons-mouton", "map_id": "MM026",
                   "original_vector": {"status": "not_bundled"}})
        self.json(Path(reference.REFERENCE_ROOT) /
                  "sites/mons-mouton/tiles/ref-0001/tile.json",
                  {"site_id": "mons-mouton", "legacy_name": name,
                   "layers": {"nac": {"path": "sites/mons-mouton/image.tif"}}})

    def test_source_snapshot_provenance_does_not_lock_the_live_map_or_invent_pixel_truth(self):
        name = "MM026_dem_5m"
        dem, _ = self.public(name=name)
        self.reference_snapshot(name)
        self.assertEqual(reference.reference_for(name), "MM026")
        self.assertIn(name, reference.locked_maps())
        self.assertIsNone(reference.vector_reference_for(name))
        cells = np.ones((core.GRID, core.GRID), np.uint8)
        with patch.object(reference, "reference_pixels", side_effect=AssertionError("no polygons")):
            labels, weights, _ = assistant.training_pixels(name, (120, 120), cells,
                                                           dem_path=str(dem))
            np.testing.assert_array_equal(labels, core.cells_to_labels(cells, (120, 120)))
            self.assertTrue((weights == 1).all())
            self.assertIsNone(evaluate.key_source(name))
            self.assertIsNone(evaluate.answer_key(name))
        # The old site field is also provenance, not a requirement for a vector file.
        self.json(self.dem_root / "working_dems.json", [{"site": "MM026", "working_dem": "a.tif"}])
        self.assertIn("a", reference.locked_maps())
        app = AppTest.from_file(str(paths.PROJECT_ROOT / "selenograph.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state.locked)
        self.assertFalse(app.radio[0].disabled)
        self.assertEqual(app.selectbox(key="selected_map").options, [name])

    def test_source_reference_paths_are_not_live_catalog_companions(self):
        self.reference_snapshot()
        image = Path(reference.REFERENCE_ROOT) / "sites/mons-mouton/image.tif"
        image.touch()
        self.assertEqual(reference.reference_layer("MM026_dem_5m", "nac"), str(image))
        self.assertIsNone(core.companion(core.dem_path("MM026_dem_5m"), "nac"))
        self.assertEqual(core.dem_files(), [])
        self.assertEqual(core.records(), [])
        vector = Path(reference.REFERENCE_ROOT) / "sites/mons-mouton/original.gpkg"
        vector.touch()
        self.json(Path(reference.REFERENCE_ROOT) / "sites/mons-mouton/site.json",
                  {"map_id": "MM026", "original_vector": {
                      "status": "available", "path": "sites/mons-mouton/original.gpkg"}})
        self.assertEqual(reference.source_map("MM026"), vector)
        self.assertEqual(reference.vector_reference_for("MM026_dem_5m"), "MM026")
        dem, _ = self.public(name="MM026_dem_5m")
        self.assertEqual(core.companion(str(dem), "nac"),
                         str(self.dem_root / "south/imagery/MM026_dem_5m-nac.tif"))

    def test_legacy_paintings_and_serialized_memory_keep_ids_after_catalog_move(self):
        name = "MM026_dem_5m"
        self.raster(f"{name}.tif")
        cells = np.ones((core.GRID, core.GRID), np.uint8)
        cells[:, 60:] = 2
        saved = Path(core.LEGACY_OUT_DIR) / name
        drafts = Path(core.LEGACY_DRAFT_DIR)
        saved.mkdir(parents=True)
        drafts.mkdir(parents=True)
        np.save(saved / "painting.npy", cells)
        np.save(saved / "accepted.npy", cells == 1)
        np.save(drafts / f"{name}.npy", cells)
        np.save(drafts / f"{name}.accepted.npy", cells == 2)
        legacy_before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                         for root in (saved, drafts) for p in root.iterdir()}
        bundle = dict(schema=assistant.schema(), samples={name: {
            "painting_hash": assistant.painting_hash(cells, accepted=cells == 2)}},
            model=None, revision="preserved")
        assistant.save(bundle)
        artifact = Path(assistant.MODEL_DIR) / f"{core.MODEL}.joblib"
        before = artifact.read_bytes()
        moved = self.dem_root / f"legacy/sections/mons-mouton/{name}.tif"
        moved.parent.mkdir(parents=True)
        (self.dem_root / f"{name}.tif").rename(moved)
        self.json(self.dem_root / "legacy/working_dems.json", [
            {"working_dem": f"sections/mons-mouton/{name}.tif", "site": "MM026"}])
        self.assertEqual(core.dem_path(name), str(moved))
        np.testing.assert_array_equal(core.load_painting(name), cells)
        np.testing.assert_array_equal(core.load_accepted(name), cells == 2)
        self.assertEqual(Path(core.painting_files(name, existing=True)[0]), drafts / f"{name}.npy")
        self.assertEqual(Path(core.painting_files(name, final=True, existing=True)[0]),
                         saved / "painting.npy")
        np.testing.assert_array_equal(core._painting_source(name, final=True)[0], cells)
        np.testing.assert_array_equal(core._beside(name, True, 1, cells), cells == 1)
        self.assertFalse(Path(core.out_dir(name)).exists())
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns)
                          for root in (saved, drafts) for p in root.iterdir()}, legacy_before)
        self.assertEqual(assistant.load()["revision"], "preserved")
        self.assertFalse(assistant.pending(assistant.load(), name, cells, accepted=cells == 2))
        self.assertEqual(artifact.read_bytes(), before)

    def test_test_map_selection_stays_outside_real_dem_catalog(self):
        self.public()
        core.set_test_maps(["public-dem"])
        self.assertEqual(core.test_maps_path(), self.dem_root / "test_maps.json")
        self.assertEqual(core.test_maps(), ["public-dem"])
        output = self.root / "output/test_maps.json"
        with patch.object(core, "TEST_MAPS_FILE", output):
            core.set_test_maps(["public-dem", "public-dem"])
            self.assertEqual(json.loads(output.read_text()), ["public-dem"])
            self.assertEqual(core.test_maps(), ["public-dem"])

    def test_evaluation_resolves_manifest_dem_not_flat_filename(self):
        dem, _ = self.public()
        labels = np.ones((120, 120), np.uint8)
        features = np.ones((len(core.BASE_FEATS), 120, 120), np.float32)
        with patch.object(evaluate, "answer_key", return_value=(labels, None)), \
                patch.object(evaluate, "_features", return_value=(features, core.BASE_FEATS)) as build:
            result = evaluate.inputs_for(["public-dem"])
        self.assertEqual(result[0][0], "public-dem")
        self.assertEqual(build.call_args.args[0], str(dem))

    def test_empty_catalog_explains_processed_dataset_installation(self):
        app = AppTest.from_file(str(paths.PROJECT_ROOT / "selenograph.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.subheader[0].value, "No DEM found")
        instructions = "\n".join(m.value for m in app.markdown)
        self.assertIn("data/processed_data/", instructions)
        self.assertIn("top-level dataset folder is a Section", instructions)
        self.assertIn("working_dems.json", instructions)
        self.assertNotIn("Put an elevation GeoTIFF", instructions)

    def test_vector_coverage_without_painting_provenance_does_not_lock_public_maps(self):
        self.public()
        self.reference_snapshot("legacy-original")
        reference.EXTRA["Nobile1-MS1"].touch()
        with patch.object(reference, "_covered", return_value=("public-dem",)):
            self.assertIn("public-dem", reference.tiles("Nobile1-MS1"))
            self.assertEqual(reference.locked_maps(), {"legacy-original"})

    def test_all_backgrounds_are_visible_without_changing_training(self):
        self.public()
        schema = assistant.schema()
        with patch.object(core, "save_painting", side_effect=AssertionError("view only")), \
                patch.object(assistant, "update", side_effect=AssertionError("view only")):
            app = AppTest.from_file(str(paths.PROJECT_ROOT / "selenograph.py"), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertFalse(any(toggle.key == "more_layers" for toggle in app.toggle))
            expected = ["Hillshade", "NAC imagery", "Elevation", ":blue[Slope]",
                        ":blue[Roughness]", ":blue[Sky view]", ":blue[Local height]",
                        ":blue[Curvature]", ":blue[Local sky view]", "Relative height"]
            self.assertEqual(app.segmented_control(key="background").options, expected)
            self.assertTrue(any(":blue[Training inputs]" in c.value for c in app.caption))
            self.assertFalse(any("🔹" in c.value for c in app.caption))
            for feature in [*schema["features"], "nac", "zscene", "rel", "hillshade"]:
                app.segmented_control(key="background").set_value(feature).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state.background, feature)
                self.assertEqual(app.segmented_control(key="background").options, expected)
            self.assertFalse(any(widget.key == "background" for widget in app.selectbox))
            self.assertEqual(assistant.schema(), schema)
            self.assertFalse(app.session_state.cells.any())
            self.assertIsNone(app.session_state.assistant["model"])

    def test_visible_training_layers_and_colors_follow_the_active_recipe(self):
        self.public()
        schema = assistant.schema()
        schema["features"] = ["rough"]
        with patch.object(assistant, "schema", return_value=schema):
            app = AppTest.from_file(str(paths.PROJECT_ROOT / "selenograph.py"), default_timeout=60).run()
            self.assertFalse(app.exception)
            options = app.segmented_control(key="background").options
            self.assertEqual(len(options), 10)
            colored = [label for label in options if label.startswith(":blue[")]
            self.assertEqual(colored, [":blue[Roughness]"])
            self.assertIn("Slope", options)
            self.assertFalse(any("🔹" in label for label in options))
            self.assertIsNone(app.session_state.assistant["model"])
            self.assertFalse(app.session_state.cells.any())

    def test_background_selection_survives_rerun(self):
        self.public()
        app = AppTest.from_file(str(paths.PROJECT_ROOT / "selenograph.py"), default_timeout=60).run()
        app.segmented_control(key="background").set_value("rel").run()
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.segmented_control(key="background").value, "rel")

    def test_nac_background_and_section_map_selection_without_location_widget(self):
        dem, manifest = self.public()
        other, _ = self.public(region="north", name="other", nac=False, left=20000)
        local = self.raster("south/another-location/local.tif", left=40000)
        records = json.loads(manifest.read_text())
        records.append({"working_dem": "another-location/local.tif", "site": "local",
                        "group": "local", "collection": "unrelated provenance"})
        self.json(manifest, records)
        source_before = {p: p.read_bytes() for p in self.dem_root.rglob("*") if p.is_file()}
        app = AppTest.from_file(str(paths.PROJECT_ROOT / "selenograph.py"), default_timeout=60).run()

        def check_maps(names, selected=None):
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(any(w.key in ("map_collection", "map_location")
                                 or w.label in ("Collection", "Location") for w in app.selectbox))
            self.assertEqual(set(app.selectbox(key="selected_map").options), set(names))
            if selected is not None:
                self.assertEqual(app.selectbox(key="selected_map").value, str(selected))

        all_names = {"other", "local", "public-dem"}
        check_maps(all_names)
        self.assertEqual(app.selectbox(key="map_section").options, ["All", "north", "south"])
        app.selectbox(key="selected_map").select(str(dem)).run()
        check_maps(all_names, dem)
        app.selectbox(key="map_section").select("south").run()
        check_maps({"local", "public-dem"}, dem)
        self.assertIn("NAC imagery", app.segmented_control(key="background").options)
        app.segmented_control(key="background").set_value("nac").run()
        check_maps({"local", "public-dem"}, dem)
        self.assertEqual(app.segmented_control(key="background").value, "nac")
        captions = "\n".join(c.value for c in app.caption)
        self.assertNotIn("Choosing a background does not change training inputs.", captions)
        self.assertNotIn("NAC imagery is a display-only background", captions)
        self.assertNotIn("illumination and mosaic seams", captions)
        app.selectbox(key="selected_map").select(str(local)).run()
        check_maps({"local", "public-dem"}, local)
        self.assertEqual(app.session_state.background, "hillshade")
        app.selectbox(key="map_section").select("All").run()
        check_maps(all_names, local)
        app.selectbox(key="map_section").select("north").run()
        check_maps({"other"}, other)
        app.selectbox(key="map_section").select("All").run()
        check_maps(all_names, other)
        self.assertEqual({p: p.read_bytes() for p in self.dem_root.rglob("*") if p.is_file()}, source_before)


if __name__ == "__main__":
    unittest.main()
