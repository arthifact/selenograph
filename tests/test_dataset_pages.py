"""Synthetic processed-dataset pages and general evaluation UI regressions.

Run: .venv/bin/python -B -m unittest tests.test_dataset_pages -v
No installed-data smoke or real project artifacts are used.
"""
import hashlib
import json
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import assistant_model as assistant
from app import canvas, core, evaluate, paths, pictures
from app import import_reference_maps as private_catalog
from tests import test_clean_start as clean
from tests.smoke_app_layout import ENTRYPOINT, read_only


def metrics():
    return {"n": 12, "accuracy": 0.75, "macro_f1": 0.7,
            "per_class": {str(c): {"precision": 0.75, "recall": 0.75, "f1": 0.75,
                                  "iou": 0.6, "support": 4} for c in (1, 2, 3)},
            "confusion_matrix": [[3, 1, 0], [0, 3, 1], [1, 0, 3]]}


class PageFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = clean.CleanStartTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.processed = self.root / "data/processed_data"
        self.fixture.stack.enter_context(patch.object(core, "DEM_DIR", self.processed))
        self.fixture.stack.enter_context(patch.object(paths, "DEM_DIR", self.processed))
        self.addCleanup(core._load_catalog.cache_clear)
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        self.engine = types.SimpleNamespace(
            catalog=Mock(return_value={}), plan=Mock(return_value=[]),
            latest_report=Mock(return_value=None),
            run=Mock(side_effect=AssertionError("No evaluation requested")))
        self.fixture.stack.enter_context(patch.dict(sys.modules, {"app.evaluation": self.engine}))

    def dataset(self, name="tile-one", section="atlas", location="west", *,
                bundled=False, verified=True, read_only=False, feature_sources=None):
        folder = self.processed / section / "tiles"
        manifest = folder / "working_dems.json"
        previous = json.loads(manifest.read_text()) if manifest.exists() else []
        dem = self.fixture.public(name, section, location=location,
                                  left=1000 + len(core.dem_files()) * 20000)
        record = json.loads(manifest.read_text())[-1]
        record.update(collection="not-a-section", group=f"{location}-group", read_only=read_only)
        if bundled:
            cells = np.zeros((core.GRID, core.GRID), np.uint8)
            cells[10:30, 20:40] = 1
            cells[60:80, 60:80] = 2
            np.save(folder / f"{name}.painting.npy", cells)
            with rasterio.open(dem) as source:
                profile = source.profile
            core.write_map(str(folder / f"{name}.labels.tif"),
                           core.cells_to_labels(cells, (120, 120)), profile)
            record["annotations"] = {
                "painting": f"{name}.painting.npy", "labels": f"{name}.labels.tif",
                "verified": verified, "label_status": "User flag; review history unknown",
                "restricted": True, "provenance": {"origin": "synthetic source"}}
        if feature_sources is not None:
            record["feature_sources"] = [os.path.relpath(p, folder) for p in feature_sources]
        self.fixture.write_json(manifest, [*previous, record])
        return dem

    def gallery(self, app=None):
        app = self.app() if app is None else app
        app.switch_page("app/app_pages/gallery.py").run()
        self.assert_ok(app)
        return app

    def button(self, app, label):
        if label == "Update model":
            return app.button(key="update_model")
        return next(b for b in app.button if b.label == label)


    def app(self, section=None):
        app = AppTest.from_file(str(ENTRYPOINT), default_timeout=30)
        if section is not None:
            app.session_state["map_section"] = section
        app.run()
        self.assert_ok(app)
        return app

    def progress(self):
        app = self.app()
        app.switch_page("app/app_pages/progress.py").run()
        self.assert_ok(app)
        return app

    def assert_ok(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertFalse(any(w.key == "map_location" or w.label == "Location" for w in app.selectbox))

    def text(self, app):
        return "\n".join(e.value for kind in ("caption", "markdown", "info", "warning", "json")
                         for e in getattr(app, kind))

    def snapshot(self):
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file()}

    @contextmanager
    def browsing(self):
        before = self.snapshot()
        with read_only():
            yield
        self.assertEqual(before, self.snapshot())


class DatasetPagesTests(PageFixture):
    def test_sections_and_maps_list_all_available_tiles_without_location_filter(self):
        first = self.dataset()
        second = self.dataset("tile-two", "atlas", "east")
        third = self.dataset("tile-three", "survey", "ridge", bundled=True)
        outside = self.root / "data/reference_data/working_dems.json"
        self.fixture.write_json(outside, [{"working_dem": first, "section": "private-only"}])
        with self.browsing(), patch.object(st, "navigation", wraps=st.navigation) as navigation, \
                patch.object(private_catalog, "locked_maps", side_effect=AssertionError("No private locks")), \
                patch.object(private_catalog, "reference_records", side_effect=AssertionError("No private catalog")), \
                patch.object(private_catalog, "vector_reference_for", side_effect=AssertionError("No private lookup")):
            app = self.app("Reference data")  # Stale values cannot reopen the removed branch.
            self.assertEqual([p.title for p in navigation.call_args.args[0]], ["Paint", "Gallery", "Progress"])
            self.assertEqual(app.selectbox(key="map_section").options, ["All", "atlas", "survey"])
            self.assertEqual(app.selectbox(key="map_section").value, "All")
            self.assertNotIn("Collection", [s.label for s in app.selectbox])
            self.assertFalse(app.tabs)
            all_labels = {"tile-one", "tile-two", "tile-three — painted, verified"}
            self.assertEqual(set(app.selectbox(key="selected_map").options), all_labels)
            app.selectbox(key="selected_map").select(second).run()
            self.assert_ok(app)
            app.selectbox(key="map_section").select("atlas").run()
            self.assert_ok(app)
            self.assertEqual(set(app.selectbox(key="selected_map").options), {"tile-one", "tile-two"})
            self.assertEqual(app.selectbox(key="selected_map").value, second)
            app.selectbox(key="selected_map").select(first).run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").value, first)
            app.selectbox(key="map_section").select("survey").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").options, ["tile-three — painted, verified"])
            self.assertEqual(app.selectbox(key="selected_map").value, third)
            np.testing.assert_array_equal(app.session_state.cells, core.load_painting("tile-three"))
            self.assertFalse(app.radio[0].disabled)
            app.selectbox(key="map_section").select("All").run()
            self.assert_ok(app)
            self.assertEqual(set(app.selectbox(key="selected_map").options), all_labels)
            self.assertEqual(app.selectbox(key="selected_map").value, third)
            app.selectbox(key="selected_map").select(second).run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").value, second)
            app.selectbox(key="map_section").select("atlas").run()
            self.assert_ok(app)
            self.assertEqual(set(app.selectbox(key="selected_map").options), {"tile-one", "tile-two"})
            self.assertEqual(app.selectbox(key="selected_map").value, second)
            with self.assertRaises(ValueError):
                app.switch_page("app/app_pages/reference.py")
        self.fixture.assert_no_artifacts()

    def test_empty_and_single_section_never_inject_a_special_option(self):
        with self.browsing():
            app = self.app()
            self.assertEqual(app.selectbox(key="map_section").options, ["All"])
            self.assertIn("data/processed_data/", self.text(app))
            self.gallery(app)
            self.assertIn("No installed maps yet", self.text(app))
        dem = self.dataset()
        with self.browsing():
            app = self.app()
            self.assertEqual(app.selectbox(key="map_section").options, ["All", "atlas"])
            self.assertEqual(app.selectbox(key="selected_map").value, dem)
            self.assertNotIn("Location", [s.label for s in app.selectbox])
            self.gallery(app)
            self.assertFalse(app.checkbox(key="v_tile-one").value)
            self.assertIn("No labels yet", self.text(app))
        self.fixture.assert_no_artifacts()

    def test_stale_location_cannot_filter_group_or_site_metadata(self):
        dem = self.dataset()
        second = self.dataset("tile-two", "atlas", "east")
        self.dataset("tile-three", "survey", "ridge")
        manifest = Path(dem).parent / "working_dems.json"
        records = json.loads(manifest.read_text())
        records[0].pop("group")
        self.fixture.write_json(manifest, records)
        self.assertEqual(core.map_site("tile-one"), "west")
        self.assertEqual(core.map_site("tile-two"), "east-group")
        with self.browsing():
            app = AppTest.from_file(str(ENTRYPOINT), default_timeout=30)
            app.session_state["map_section"] = "atlas"
            app.session_state["selected_map"] = second
            for section, names in (("atlas", {"tile-one", "tile-two"}),
                                   ("All", {"tile-one", "tile-two", "tile-three"})):
                for stale in ("west", "east-group", "missing-location"):
                    with self.subTest(section=section, stale=stale):
                        if app.selectbox:
                            app.selectbox(key="map_section").select(section)
                        app.session_state["map_location"] = stale
                        app.run()
                        self.assert_ok(app)
                        self.assertNotIn("map_location", app.session_state)
                        self.assertEqual(app.selectbox(key="map_section").value, section)
                        self.assertEqual(set(app.selectbox(key="selected_map").options), names)
                        self.assertEqual(app.selectbox(key="selected_map").value, second)
            self.assertEqual(app.selectbox(key="map_section").options, ["All", "atlas", "survey"])
        self.fixture.assert_no_artifacts()

    def test_all_sections_still_exclude_geographic_holdouts(self):
        self.dataset()
        self.dataset("held-out", "atlas", "east")
        third = self.dataset("tile-three", "survey", "ridge")
        core.set_test_maps(["held-out"])
        self.assertEqual(core.held_out_maps(), {"held-out"})
        with self.browsing():
            app = self.app()
            self.assertEqual(set(app.selectbox(key="selected_map").options), {"tile-one", "tile-three"})
            app.selectbox(key="map_section").select("atlas").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").options, ["tile-one"])
            app.selectbox(key="map_section").select("All").run()
            self.assert_ok(app)
            self.assertEqual(set(app.selectbox(key="selected_map").options), {"tile-one", "tile-three"})
            app.selectbox(key="selected_map").select(third).run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").value, third)

    def test_bundled_verified_painting_uses_standard_view_without_source_notes(self):
        source = self.dataset()
        dem = self.dataset("tile-two", "survey", "ridge", bundled=True, feature_sources=[source])
        expected = core.load_painting("tile-two")
        metadata = core.meta_get("tile-two")
        record = core.map_record("tile-two")
        with self.browsing():
            app = self.app("survey")
            self.assertEqual(app.selectbox(key="selected_map").value, dem)
            self.assertIn("verified", app.selectbox(key="selected_map").options[0])
            np.testing.assert_array_equal(app.session_state.cells, expected)
            self.assertTrue(app.checkbox(key="verified_tile-two").value)
            self.assertFalse(app.session_state.locked)
            self.assertFalse(app.radio[0].disabled)
            self.assertFalse(self.button(app, "Save labels + map").disabled)
            self.assertNotIn("Metadata and source notes", [e.label for e in app.expander])
            self.assertFalse(app.json)
            for text in ("User flag; review history unknown", "restricted", "feature_sources",
                         "atlas/tiles/tile-one.tif", "unknown confidence", "not independent certification",
                         "verification metadata"):
                self.assertNotIn(text, self.text(app))
            self.assertIn("save", app.checkbox(key="verified_tile-two").proto.help.lower())
            labels = app.segmented_control(key="background").options
            self.assertEqual(sum(":blue[" in label for label in labels), 6)
        self.assertEqual(core.meta_get("tile-two"), metadata)
        self.assertEqual(core.map_record("tile-two"), record)
        self.fixture.assert_no_artifacts()
        source_before = {p: p.read_bytes() for p in self.processed.rglob("*") if p.is_file()}
        app.checkbox(key="verified_tile-two").uncheck().run()
        self.assert_ok(app)
        self.assertFalse(core.meta_get("tile-two")["verified"])
        self.assertEqual(core.meta_get("tile-two", final=True), metadata)
        self.assertTrue(Path(core.output_file("tile-two", "meta.json", final=False)).is_file())
        self.assertFalse(Path(core.painting_files("tile-two")[0]).exists())
        self.assertFalse(paths.OUT_DIR.exists())
        self.assertEqual({p: p.read_bytes() for p in self.processed.rglob("*") if p.is_file()}, source_before)

    def test_gallery_lists_all_installed_maps_and_combines_filters(self):
        self.dataset("unpainted", "atlas", "east")
        self.dataset("bundled", "survey", "ridge", bundled=True)
        saved = self.dataset("saved", "atlas", "west")
        with rasterio.open(saved) as source:
            profile = source.profile
        core.write_map(core.output_file("saved", "labels.tif"),
                       core.cells_to_labels(core.load_painting("bundled"), (120, 120)), profile)
        core.meta_set("saved", verified=False, saved_at="2026-09-28")
        Image.new("RGB", (12, 12), "blue").save(core.output_file("saved", "thumb.png"))
        core.save_painting("not-installed", np.ones((core.GRID, core.GRID), np.uint8), final=True)
        core.meta_set("not-installed", verified=True)
        app = self.app()
        with self.browsing(), patch.object(core, "build", side_effect=AssertionError("No features in Gallery")), \
                patch.object(assistant, "load", side_effect=AssertionError("No models in Gallery")), \
                patch.object(core, "read", wraps=core.read) as read:
            self.gallery(app)
            self.assertEqual({c.key for c in app.checkbox}, {"v_bundled", "v_saved", "v_unpainted"})
            self.assertTrue(app.checkbox(key="v_bundled").value)
            self.assertFalse(app.checkbox(key="v_saved").value)
            self.assertNotIn("Metadata and source notes", [e.label for e in app.expander])
            self.assertFalse(app.json)
            self.assertNotIn("User flag; review history unknown", self.text(app))
            self.assertNotIn("Bundled annotations; output edits take precedence", self.text(app))
            for text in ("Verification is metadata", "scientific certification", "Bundled inputs remain",
                         "verification overrides"):
                self.assertNotIn(text, self.text(app))
            self.assertIn("3 maps · 1 verified", self.text(app))
            self.assertIn("save in paint", app.checkbox(key="v_bundled").proto.help.lower())
            self.assertEqual(len(app.get("image")), 3)
            self.assertEqual(read.call_count, 3)  # DEM masks also correct saved thumbnails.
            app.run()
            self.assert_ok(app)
            self.assertEqual(read.call_count, 3)  # Terrain previews are cached in memory.
            self.assertNotIn("0.0 %", self.text(app))
            app.selectbox(key="gallery_section").select("survey").run()
            self.assertEqual([c.key for c in app.checkbox], ["v_bundled"])
            self.assertIn("Saved labels", self.text(app))
            app.selectbox(key="gallery_section").select("All").run()
            app.segmented_control(key="gallery_status").set_value("Verified only").run()
            self.assert_ok(app)
            self.assertEqual([c.key for c in app.checkbox], ["v_bundled"])
        self.assertFalse(Path(core.out_dir("bundled")).exists())

    def test_gallery_pagination_resets_when_filters_change(self):
        for index in range(26):
            self.dataset(f"tile-{index:02}", "atlas" if index < 25 else "survey",
                         f"site-{index}", bundled=index == 25)
        with self.browsing():
            app = self.gallery()
            self.assertEqual(app.selectbox(key="gallery_section").value, "All")
            self.assertEqual(app.segmented_control(key="gallery_status").value, "All maps")
            first = {c.key for c in app.checkbox}
            self.assertEqual(len(first), 24)
            app.selectbox(key="gallery_page").select(2).run()
            self.assert_ok(app)
            second = {c.key for c in app.checkbox}
            self.assertEqual(len(second), 2)
            self.assertEqual(len(first | second), 26)
            self.assertFalse(first & second)
            app.selectbox(key="gallery_section").select("atlas").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="gallery_page").value, 1)
            app.segmented_control(key="gallery_status").set_value("Verified only").run()
            self.assert_ok(app)
            self.assertFalse(app.checkbox)
            self.assertIn("No verified maps in this selection", self.text(app))
            app.selectbox(key="gallery_section").select("All").run()
            self.assert_ok(app)
            self.assertEqual([c.key for c in app.checkbox], ["v_tile-25"])
            app.segmented_control(key="gallery_status").set_value("All maps").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="gallery_page").value, 1)
            self.assertEqual(len(app.checkbox), 24)
        self.fixture.assert_no_artifacts()

    def test_gallery_edge_tile_preview_fits_without_cropping_or_stretching(self):
        for width, height in ((128, 1024), (1024, 128), (1024, 1024)):
            with self.subTest(size=(width, height)):
                original = Image.new("RGB", (width, height), "blue")
                preview = pictures.gallery_preview(original)
                self.assertEqual(preview.size, (240, 240))
                left, top, right, bottom = preview.getbbox()
                self.assertAlmostEqual((right - left) / (bottom - top), width / height)
                self.assertEqual(max(right - left, bottom - top), 240)
                self.assertEqual((left, top), (0, 0))
                if width != height:
                    self.assertEqual(preview.getpixel((239, 239)), (0, 0, 0))
                self.assertEqual(original.size, (width, height))

    def test_gallery_verification_stages_draft_metadata_without_committing_saved_labels(self):
        self.dataset(bundled=True)
        before = {str(p): p.read_bytes() for p in self.processed.rglob("*") if p.is_file()}
        app = self.gallery()
        app.checkbox(key="v_tile-one").uncheck().run()
        self.assert_ok(app)
        self.assertFalse(core.meta_get("tile-one")["verified"])
        self.assertTrue(core.meta_get("tile-one", final=True)["verified"])
        self.assertIsNotNone(evaluate.answer_key("tile-one"))
        self.assertEqual(list(Path(core.out_dir("tile-one", final=False)).iterdir()),
                         [Path(core.output_file("tile-one", "meta.json", final=False))])
        app.switch_page("app/app_pages/paint.py").run()
        self.assert_ok(app)
        self.assertFalse(app.checkbox(key="verified_tile-one").value)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.processed.rglob("*") if p.is_file()})
        self.assertFalse(Path(core.OUT_DIR).exists())
        self.assertFalse(Path(core.painting_files("tile-one")[0]).exists())

    def test_cleared_seed_without_metadata_remains_savable(self):
        dem = self.dataset(bundled=True)
        manifest = Path(dem).parent / "working_dems.json"
        records = json.loads(manifest.read_text())
        records[0]["annotations"] = {key: value for key, value in records[0]["annotations"].items()
                                      if key in ("painting", "labels")}
        self.fixture.write_json(manifest, records)
        core.seed_paintings()
        saved_path = Path(core.painting_files("tile-one", final=True)[0])
        original = saved_path.read_bytes()
        app = self.app()
        self.assertEqual(core.meta_get("tile-one"), {})
        self.button(app, "Clear").click().run()
        self.assert_ok(app)
        self.assertFalse(core.load_painting("tile-one").any())
        self.assertEqual(saved_path.read_bytes(), original)
        self.assertFalse(self.button(app, "Save labels + map").disabled)
        self.button(app, "Save labels + map").click().run()
        self.assert_ok(app)
        self.assertFalse(np.load(saved_path, allow_pickle=False).any())
        self.assertFalse(paths.MODEL_DIR.exists())

    def test_gallery_cannot_stage_verification_for_held_out_maps_or_neighbours(self):
        for name in ("held", "neighbour", "editable"):
            self.dataset(name, bundled=True)
        with patch.object(core, "held_out_maps", return_value={"held", "neighbour"}):
            with self.browsing():
                app = self.gallery()
                for name in ("held", "neighbour"):
                    self.assertTrue(app.checkbox(key=f"v_{name}").disabled)
                self.assertFalse(app.checkbox(key="v_editable").disabled)
                original = st.checkbox

                def force_held(label, *args, **kwargs):
                    value = original(label, *args, **kwargs)
                    return False if kwargs.get("key") in ("v_held", "v_neighbour") else value

                with patch.object(st, "checkbox", side_effect=force_held):
                    app.run()
                    self.assert_ok(app)
            app.checkbox(key="v_editable").uncheck().run()
            self.assert_ok(app)
            self.assertFalse(core.meta_get("editable")["verified"])
            self.assertTrue(core.meta_get("editable", final=True)["verified"])
            self.assertFalse(Path(core.OUT_DIR).exists())
            for name in ("held", "neighbour"):
                self.assertFalse(Path(core.output_file(name, "meta.json", final=False)).exists())

    def test_paint_edit_undo_clear_and_save_leave_all_source_bytes_unchanged(self):
        self.dataset(bundled=True)
        # Mirror hard-linked publications: writes through either name would corrupt the source.
        for path in list(self.processed.rglob("*")):
            if path.is_file():
                original = self.root / "data/reference_data" / path.relative_to(self.processed)
                original.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, original)
        source_files = [p for p in (self.root / "data").rglob("*") if p.is_file()]
        before = {str(p): p.read_bytes() for p in source_files}
        cells = core.load_painting("tile-one")
        app = self.app()
        with patch.object(canvas, "map_canvas", return_value={"id": "edit", "cells": [0], "click": True}):
            app.run()
        self.assert_ok(app)
        self.assertEqual(core.load_painting("tile-one")[0, 0], 1)
        self.button(app, "Undo").click().run()
        self.assert_ok(app)
        np.testing.assert_array_equal(core.load_painting("tile-one"), cells)
        self.button(app, "Clear").click().run()
        self.assert_ok(app)
        self.assertFalse(core.load_painting("tile-one").any())
        self.button(app, "Undo").click().run()
        self.assert_ok(app)
        app.checkbox(key="verified_tile-one").uncheck().run()
        self.button(app, "Save labels + map").click().run()
        self.assert_ok(app)
        self.assertTrue(Path(core.output_file("tile-one", "labels.tif")).is_file())
        self.assertFalse(core.meta_get("tile-one")["verified"])
        np.testing.assert_array_equal(core.load_painting("tile-one"), cells)
        self.assertEqual(before, {str(p): p.read_bytes() for p in source_files})
        self.assertFalse(paths.MODEL_DIR.exists())

    def test_restore_saved_replaces_draft_flags_and_can_be_undone(self):
        self.dataset(bundled=True)
        saved = np.zeros((core.GRID, core.GRID), np.uint8)
        saved[20:40, 30:60] = 3
        saved_accepted = saved > 0
        saved_confidence = np.where(saved > 0, core.MOSTLY, 0).astype(np.uint8)
        core.save_painting("tile-one", saved, final=True,
                           accepted=saved_accepted, confidence=saved_confidence)
        draft = np.ones_like(saved)
        draft_accepted = np.zeros_like(saved_accepted)
        draft_confidence = np.full_like(saved, core.UNSURE)
        core.save_painting("tile-one", draft, accepted=draft_accepted, confidence=draft_confidence)
        # Neither the saved arrays nor bundled inputs may be rewritten by Restore.
        protected = [p for p in self.root.rglob("*") if p.is_file()
                     and not p.is_relative_to(Path(core.DRAFT_DIR))]
        before = {str(p): p.read_bytes() for p in protected}
        app = self.app()
        self.assertFalse(self.button(app, "Restore saved").disabled)
        with patch.object(assistant, "retrain", side_effect=AssertionError("No training requested")):
            self.button(app, "Restore saved").click().run()
        self.assert_ok(app)
        for key, expected, reader in (("cells", saved, core.load_painting),
                                     ("accepted", saved_accepted, core.load_accepted),
                                     ("confidence", saved_confidence, core.load_confidence)):
            np.testing.assert_array_equal(app.session_state[key], expected)
            np.testing.assert_array_equal(reader("tile-one"), expected)
        self.assertTrue(self.button(app, "Restore saved").disabled)
        restarted = self.app()
        np.testing.assert_array_equal(restarted.session_state.cells, saved)
        self.button(app, "Undo").click().run()
        self.assert_ok(app)
        np.testing.assert_array_equal(core.load_painting("tile-one"), draft)
        np.testing.assert_array_equal(core.load_accepted("tile-one"), draft_accepted)
        np.testing.assert_array_equal(core.load_confidence("tile-one"), draft_confidence)
        self.assertEqual(before, {str(p): p.read_bytes() for p in protected})
        self.assertFalse(paths.MODEL_DIR.exists())
        self.assertFalse(Path(core.output_file("tile-one", "labels.tif")).exists())

    def test_restore_saved_handles_bundled_and_empty_saved_paintings(self):
        self.dataset(bundled=True)
        bundled = core.load_painting("tile-one")
        draft = np.full_like(bundled, 3)
        for empty_save in (False, True):
            with self.subTest(empty_save=empty_save):
                expected = np.zeros_like(bundled) if empty_save else bundled
                if empty_save:
                    core.save_painting("tile-one", expected, final=True)
                core.save_painting("tile-one", draft, accepted=draft > 0,
                                   confidence=np.full_like(draft, core.UNSURE))
                app = self.app()
                self.button(app, "Restore saved").click().run()
                self.assert_ok(app)
                np.testing.assert_array_equal(core.load_painting("tile-one"), expected)
                self.assertFalse(core.load_accepted("tile-one").any())
                np.testing.assert_array_equal(core.load_confidence("tile-one"), expected > 0)
                self.assertTrue(self.button(app, "Restore saved").disabled)

    def test_restore_saved_requires_a_saved_painting(self):
        self.dataset()
        core.save_painting("tile-one", np.ones((core.GRID, core.GRID), np.uint8))
        with self.browsing():
            app = self.app()
            self.assertTrue(self.button(app, "Restore saved").disabled)

    def test_gallery_preview_cache_refreshes_for_output_paintings_and_dem_changes(self):
        dem = self.dataset(bundled=True)
        app = self.app()
        with patch.object(core, "read", wraps=core.read) as read, \
                patch.object(core, "load_painting", wraps=core.load_painting) as painting:
            with self.browsing():
                self.gallery(app)
                app.run()
                self.assert_ok(app)
                self.assertEqual(read.call_count, 1)
            edited = core.load_painting("tile-one")
            edited[0, 0] = 3
            core.save_painting("tile-one", edited)
            painting.reset_mock()
            with self.browsing():
                app.run()
                self.assert_ok(app)
                self.assertEqual(read.call_count, 2)
                painting.assert_called_with("tile-one")
            stat = os.stat(dem)
            os.utime(dem, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            with self.browsing():
                app.run()
                self.assert_ok(app)
                self.assertEqual(read.call_count, 3)

    def test_generic_read_only_flag_rejects_forced_actions_and_canvas_events(self):
        self.dataset(bundled=True, read_only=True)
        original = st.button
        forced = {"Save labels + map", "Approve prediction", "Clear", "Undo", "Restore saved"}

        def force(label, *args, **kwargs):
            value = original(label, *args, **kwargs)
            return True if label in forced else value

        with self.browsing(), patch.object(st, "button", side_effect=force), \
                patch.object(canvas, "map_canvas", return_value={"id": "forced", "cells": [0], "click": True}):
            app = self.app()
            self.assertTrue(app.session_state.locked)
            self.assertTrue(app.radio[0].disabled)
            for label in forced:
                self.assertTrue(self.button(app, label).disabled)
            self.assertEqual(app.session_state.cells[0, 0], 0)
            self.gallery(app)
            self.assertTrue(app.checkbox(key="v_tile-one").disabled)
            with patch.object(st, "checkbox", return_value=False):
                app.run()
                self.assert_ok(app)
        self.fixture.assert_no_artifacts()

    def assert_feature_refresh_preserves_painting(self, dem, changes):
        names = [*core.LAYERS, "nac"]
        shape = (120, 120)
        preview_shape = (30, 30)
        with rasterio.open(dem) as source:
            transform = source.transform

        def features(phase):
            slope = np.roll(np.tile(np.arange(shape[1], dtype=np.float32), (shape[0], 1)),
                            phase * 17, axis=1)
            return (np.stack([slope] * len(names)), names, np.full(shape, 0.5, np.float32),
                    {"height": shape[0], "width": shape[1], "transform": transform}, 5)

        cells = np.zeros((core.GRID, core.GRID), np.uint8)
        cells[4:8, 9:13] = 3
        accepted = cells > 0
        confidence = np.where(accepted, core.MOSTLY, 0).astype(np.uint8)
        history = [(np.zeros_like(cells), np.zeros_like(accepted), np.zeros_like(confidence))]
        with patch.object(core, "build", return_value=features(0)) as build, \
                patch.object(assistant, "predict", return_value=np.ones(preview_shape, np.uint8)) as predict, \
                patch.object(assistant, "confidence", return_value=np.full(preview_shape, 0.5)) as score, \
                patch.object(canvas, "map_canvas", return_value=None) as shown:
            with self.browsing():
                app = self.app("survey")
                self.assertEqual(app.selectbox(key="selected_map").value, dem)
                app.session_state["cells"] = cells.copy()
                app.session_state["accepted"] = accepted.copy()
                app.session_state["confidence"] = confidence.copy()
                app.session_state["hist"] = history
                app.segmented_control(key="background").set_value("slope").run()
                self.assert_ok(app)
                terrain = shown.call_args.kwargs["terrain"].tobytes()
                app.pills(key="layers").set_value(["Painting", "Confidence"]).run()
                self.assert_ok(app)
                app.pills(key="layers").set_value(["Painting"]).run()
                self.assert_ok(app)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(score.call_count, 1)
                dem_revision = app.session_state.dem_revision
                feature_revision = app.session_state.feature_revision
            for phase, change in enumerate(changes, 1):
                with self.subTest(change=phase):
                    change()
                    build.return_value = features(phase)
                    prediction = np.full(preview_shape, 1 + phase % 3, np.uint8)
                    predict.return_value = prediction
                    score.return_value = np.full(preview_shape, 0.5 + phase * 0.1)
                    with self.browsing():
                        app.run()
                        self.assert_ok(app)
                        self.assertEqual(build.call_count, phase + 1)
                        self.assertEqual(predict.call_count, phase + 1)
                        self.assertEqual(app.session_state.dem_revision, dem_revision)
                        self.assertNotEqual(app.session_state.feature_revision, feature_revision)
                        feature_revision = app.session_state.feature_revision
                        np.testing.assert_array_equal(app.session_state.cells, cells)
                        np.testing.assert_array_equal(app.session_state.accepted, accepted)
                        np.testing.assert_array_equal(app.session_state.confidence, confidence)
                        self.assertEqual(len(app.session_state.hist), 1)
                        for actual, expected in zip(app.session_state.hist[0], history[0]):
                            np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(app.session_state.pred, prediction)
                        refreshed = shown.call_args.kwargs["terrain"].tobytes()
                        self.assertNotEqual(refreshed, terrain)
                        terrain = refreshed
                        app.pills(key="layers").set_value(["Painting", "Confidence"]).run()
                        self.assert_ok(app)
                        self.assertEqual(score.call_count, phase + 1)
                        np.testing.assert_array_equal(app.session_state.conf, score.return_value)
                        app.pills(key="layers").set_value(["Painting"]).run()
                        app.run()
                        self.assert_ok(app)
                        self.assertEqual(build.call_count, phase + 1)
                        self.assertEqual(predict.call_count, phase + 1)
                        self.assertEqual(score.call_count, phase + 1)
        self.fixture.assert_no_artifacts()

    def test_source_manifest_recipe_refreshes_features_without_reloading_painting(self):
        first = self.dataset("source-one")
        second = self.dataset("source-two")
        context = self.dataset("context", feature_sources=[first])
        dem = self.dataset("target", "survey", feature_sources=[context])

        def change_recipe():
            manifest = Path(context).parent / "working_dems.json"
            records = json.loads(manifest.read_text())
            next(r for r in records if r["working_dem"] == Path(context).name)["feature_sources"] = [Path(second).name]
            self.fixture.write_json(manifest, records)

        self.assert_feature_refresh_preserves_painting(dem, [change_recipe])

    def test_recursive_source_and_companion_changes_refresh_without_reloading_painting(self):
        source = self.dataset("source")
        context = self.dataset("context", feature_sources=[source])
        dem = self.dataset("target", "survey", feature_sources=[context])
        source_layer = core.companion(source, "nac")
        target_layer = core.companion(dem, "nac")
        assert source_layer is not None and target_layer is not None

        def touch(path):
            stat = os.stat(path)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        self.assert_feature_refresh_preserves_painting(
            dem, [lambda: touch(source), lambda: touch(source_layer), lambda: touch(target_layer)])

    def test_filters_keep_dem_paths_and_refresh_published_model_without_resetting_painting(self):
        self.dataset()
        second = self.dataset("tile-two", "survey", "ridge", bundled=True)
        bundle = assistant.load()
        cells = core.load_painting("tile-two")
        bundle["samples"]["tile-two"] = {"painting_hash": assistant.painting_hash(
            cells, core.load_confidence("tile-two"), core.load_accepted("tile-two"))}
        with self.browsing(), patch.object(assistant, "artifact_stamp", return_value=("first",)) as stamp, \
                patch.object(assistant, "load", return_value=bundle) as load:
            app = self.app()
            app.selectbox(key="map_section").select("survey").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").value, second)
            self.assertEqual(app.selectbox(key="selected_map").options, ["tile-two — painted, verified"])
            before = load.call_count
            app.run()
            self.assertEqual(load.call_count, before)
            stamp.return_value = ("published",)
            app.run()
            self.assert_ok(app)
            self.assertGreater(load.call_count, before)
            self.assertEqual(app.session_state.assistant_stamp, ("published",))
            np.testing.assert_array_equal(app.session_state.cells, cells)
            app.selectbox(key="map_section").select("All").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").value, second)


class ProgressPageTests(PageFixture):
    def evaluation_catalog(self):
        catalog = {"public-unlabeled": {"source_kind": "public", "section": "north", "group": "north",
                                       "label_source": None, "available": False, "reason": "No labels"},
                   "public-trained": {"source_kind": "public", "section": "south", "group": "south",
                                      "label_source": "verified labels", "available": True, "verified": True}}
        for site, count in (("west", 1), ("ridge", 1), ("east", 1), ("valley", 3)):
            for index in range(count):
                catalog[f"{site}-{index}"] = {"source_kind": "bundled", "section": "survey",
                    "group": site, "label_source": "bundled annotations", "available": True}
        self.engine.catalog.return_value = catalog
        return catalog

    def report(self):
        report_path = paths.OUTPUT_DIR / "evaluations/evaluation-20260928T120000Z.json"
        report = {"run_id": "fixture-run", "created_at": "2026-09-28",
            "selected_maps": ["west-0"], "skipped": [{"map": "old-unlabeled", "reason": "No labels"}],
            "validation": {"aggregate": metrics(), "baseline_aggregate": metrics(),
                "folds": [{"group": "west", "maps": ["west-0"],
                           "train_keys": ["other"], "metrics": metrics(), "baseline": metrics()}]},
            "provenance": {"maps": {}}}
        self.fixture.write_json(report_path, report)
        self.engine.latest_report.return_value = (report_path, report)
        return report_path, report

    def test_json_reports_restore_selection_without_images(self):
        self.evaluation_catalog()
        self.report()
        with read_only():
            app = self.progress()
            self.assertEqual(app.multiselect(key="evaluation_maps").value, ["west-0"])
            self.assertEqual(len(app.multiselect(key="evaluation_maps").options), 7)
            self.assertFalse(any("public-unlabeled" in option for option in app.multiselect(key="evaluation_maps").options))
            self.assertEqual(next(m.value for m in app.metric if m.label == "Evaluation accuracy"), "75.0%")
            self.assertEqual(len(app.get("download_button")), 1)
            self.assertEqual(app.get("download_button")[0].label, "Download evaluation JSON")
            self.assertFalse(app.get("image"))
            app.multiselect(key="evaluation_maps").set_value(["public-trained"]).run()
            self.assert_ok(app)
            self.assertIn("not unsaved chooser changes", self.text(app))
            self.assertFalse(paths.TEST_MAPS_FILE.exists())
        self.engine.run.assert_not_called()

    def test_report_listing_refreshes_progress_without_a_latest_pointer(self):
        self.evaluation_catalog()
        _, report = self.report()
        with self.browsing():
            app = self.progress()
            calls = self.engine.latest_report.call_count
            app.run()
            self.assert_ok(app)
            self.assertEqual(self.engine.latest_report.call_count, calls)
        newer = paths.OUTPUT_DIR / "evaluations/evaluation-20260929T120000Z.json"
        report = dict(report, run_id="newer-run", created_at="2026-09-29")
        report["validation"] = dict(report["validation"], aggregate=dict(metrics(), accuracy=0.25))
        self.fixture.write_json(newer, report)
        self.engine.latest_report.return_value = (newer, report)
        with self.browsing():
            app.run()
            self.assert_ok(app)
            self.assertGreater(self.engine.latest_report.call_count, calls)
            self.assertEqual(next(m.value for m in app.metric if m.label == "Evaluation accuracy"), "25.0%")
        self.assertFalse((paths.OUTPUT_DIR / "evaluation/latest.json").exists())

    def test_evaluate_only_calls_engine_on_explicit_ready_request(self):
        self.evaluation_catalog()
        self.engine.plan.return_value = [{"group": "south", "maps": ["public-trained"],
            "train_keys": ["other"], "excluded_keys": ["public-trained"], "reason": None}]
        result = self.report()
        self.engine.run.side_effect = None
        self.engine.run.return_value = result
        bundle = {"model": None, "samples": {"public-trained": {}, "other": {}}, "revision": "unchanged"}
        with read_only(), patch.object(assistant, "load", return_value=bundle):
            app = self.progress()
            app.multiselect(key="evaluation_maps").set_value(["public-trained"]).run()
            self.assert_ok(app)
            self.engine.run.assert_not_called()
            self.assertFalse(next(b for b in app.button if b.label == "Evaluate").disabled)
            next(b for b in app.button if b.label == "Evaluate").click().run()
            self.assert_ok(app)
            self.engine.run.assert_called_once()
            self.assertEqual(self.engine.run.call_args.args, (bundle, ["public-trained"]))
            self.assertFalse(paths.TEST_MAPS_FILE.exists())
            self.assertEqual(bundle["revision"], "unchanged")

    def test_disabled_evaluation_cannot_be_forced_with_no_samples_or_no_usable_fold(self):
        self.evaluation_catalog()
        original = st.button
        for samples, selected, reason in (({}, ["public-trained"], None),
                                           ({"one": {}}, [], None),
                                           ({"one": {}}, ["public-trained"], "No training samples remain")):
            with self.subTest(samples=samples, selected=selected, reason=reason):
                self.engine.plan.return_value = [{"group": "south", "maps": selected,
                    "train_keys": [], "excluded_keys": ["one"], "reason": reason}]
                with read_only(), patch.object(assistant, "load", return_value={"model": None, "samples": samples}):
                    app = self.progress()
                    app.multiselect(key="evaluation_maps").set_value(selected).run()
                    self.assert_ok(app)
                    self.assertTrue(next(b for b in app.button if b.label == "Evaluate").disabled)
                    with patch.object(st, "button", side_effect=lambda label, *a, **kw:
                                      True if label == "Evaluate" else original(label, *a, **kw)):
                        app.run()
                        self.assert_ok(app)
                self.engine.run.assert_not_called()

    def test_only_evaluation_is_shown_and_stale_unpainted_selection_is_cleared(self):
        self.evaluation_catalog()
        with read_only(), patch.object(assistant, "refit", side_effect=AssertionError("No model writes")):
            app = self.progress()
            app.session_state["evaluation_maps"] = ["public-unlabeled", "west-0"]
            app.run()
            self.assert_ok(app)
            self.assertEqual(app.multiselect(key="evaluation_maps").value, ["west-0"])
            self.assertEqual([h.value for h in app.subheader], ["Evaluation"])
            self.assertEqual([w.key for w in app.multiselect], ["evaluation_maps"])
            self.assertIn("public-trained · south — painted, verified", app.multiselect(key="evaluation_maps").options)
            self.assertIn("west-0 · survey — painted", app.multiselect(key="evaluation_maps").options)
            self.assertNotIn("Use these test maps", [b.label for b in app.button])
        self.assertFalse(paths.TEST_MAPS_FILE.exists())

    def test_individual_results_from_the_same_location_stay_separate(self):
        self.evaluation_catalog()
        _, report = self.report()
        report["validation"]["folds"] = [
            {"group": "valley", "test_tiles": [name], "train_tiles": ["other"],
             "metrics": metrics(), "baseline": metrics()}
            for name in ("valley-0", "valley-1")]
        with read_only():
            app = self.progress()
            self.assertEqual(app.selectbox(key="evaluation_score_detail").options,
                             ["All selected maps", "Majority baseline", "valley-0", "valley-1"])

    def test_old_export_metadata_does_not_open_any_images(self):
        self.evaluation_catalog()
        _, report = self.report()
        report["exports"] = {"files": [{"path": "../escape.png"}], "bundle": "../escape.zip"}
        with read_only():
            app = self.progress()
            self.assert_ok(app)
            self.assertFalse(app.get("image"))
            self.assertEqual(len(app.get("download_button")), 1)


if __name__ == "__main__":
    unittest.main()
