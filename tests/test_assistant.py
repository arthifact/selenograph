"""Regression checks for explicit model updates and dataset isolation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
import streamlit as st
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest

from app import assistant_model as assistant
from app import core, evaluation, paths, pictures
from app.paths import PROJECT_ROOT


def open_app(page="paint"):
    app = AppTest.from_file(str(PROJECT_ROOT / "selenograph.py"), default_timeout=60).run()
    if page != "paint":
        app.switch_page(f"app/app_pages/{page}.py").run()
    return app


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module, key, value in ((core, "DEM_DIR", str(self.root / "dem")),
                                    (core, "OUT_DIR", str(self.root / "output/paintings")),
                                    (core, "MAP_DIR", str(self.root / "output/maps")),
                                    (core, "LEGACY_OUT_DIR", str(self.root / "output/maps")),
                                    (core, "DRAFT_DIR", str(self.root / "output/painting_drafts")),
                                    (core, "LEGACY_DRAFT_DIR", str(self.root / "output/drafts")),
                                    (core, "LEGACY_MIRRORED_DIR", str(self.root / "output")),
                                    (paths, "OUTPUT_DIR", self.root / "output"),
                                    (paths, "OUT_DIR", self.root / "output/paintings"),
                                    (paths, "MAP_DIR", self.root / "output/maps"),
                                    (paths, "LEGACY_OUT_DIR", self.root / "output/maps"),
                                    (paths, "DRAFT_DIR", self.root / "output/painting_drafts"),
                                    (paths, "LEGACY_DRAFT_DIR", self.root / "output/drafts"),
                                    (paths, "LEGACY_MIRRORED_DIR", self.root / "output"),
                                    (paths, "POSTER_DIR", self.root / "poster"),
                                    (core, "MODEL", "rf"),
                                    (assistant, "MODEL_DIR", str(self.root / "models"))):
            context = patch.object(module, key, value)
            context.start()
            self.addCleanup(context.stop)
        rng = np.random.default_rng(5)
        self.X = rng.normal(size=(len(core.BASE_FEATS), 240, 240)).astype(np.float32)
        self.X[:, :8] = np.nan
        self.cells = np.zeros((core.GRID, core.GRID), np.uint8)
        self.cells[10:40, 10:40] = 1
        self.cells[70:100, 70:100] = 2

    def update(self, name, cells):
        with patch.object(core, "meta_get", return_value={"verified": True}):
            return assistant.update(name, self.X, core.BASE_FEATS, cells,
                                self.X[:, ::4, ::4], name, (1, 2))

    def artifact(self):
        return (Path(assistant.MODEL_DIR) / f"{core.MODEL}.joblib").read_bytes()

    def test_persisted_prediction_for_both_pixel_backends(self):
        backends = ["rf"] + (["gbm"] if importlib.util.find_spec("lightgbm") else [])
        for backend in backends:
            with self.subTest(backend=backend), patch.object(core, "MODEL", backend):
                bundle, preview = self.update("a", self.cells)
                self.assertIsNotNone(bundle["model"])
                self.assertEqual(bundle["model"].n_features_in_, 6)
                self.assertEqual(bundle["samples"]["a"]["X"].shape[1], 6)
                assert preview is not None
                restored = assistant.load()
                np.testing.assert_array_equal(preview, assistant.predict(
                    restored, self.X[:, ::4, ::4], core.BASE_FEATS))
                self.assertTrue((preview[:2] == 255).all())
                self.assertTrue(set(np.unique(preview)).issubset({1, 2, 255}))
                self.assertFalse(Path(core.OUT_DIR).exists())

    def test_corrections_replace_samples_and_clear_unlearns(self):
        first, _ = self.update("a", self.cells)
        b = np.where(self.cells == 1, 3, 0).astype(np.uint8)
        second, _ = self.update("b", b)
        np.testing.assert_array_equal(first["samples"]["a"]["y"], second["samples"]["a"]["y"])
        corrected = np.where(self.cells != 0, 1, 0).astype(np.uint8)
        third, _ = self.update("a", corrected)
        self.assertEqual(set(third["model"].classes_), {1, 3})
        self.assertFalse(assistant.pending(third, "a", corrected))
        self.assertTrue(assistant.pending(third, "a", self.cells))
        cleared, preview = self.update("a", np.zeros_like(self.cells))
        self.assertEqual(set(cleared["samples"]), {"b"})
        self.assertIsNone(cleared["model"])
        self.assertIsNone(preview)
        self.assertFalse(Path(core.OUT_DIR).exists())

    def test_one_class_updates_are_remembered_and_unknown_is_excluded(self):
        single = np.where(self.cells == 1, 1, 0).astype(np.uint8)
        single[:4] = 3                 # entirely missing terrain, never a training label
        bundle, preview = self.update("a", single)
        self.assertIsNone(preview)
        self.assertEqual(set(bundle["samples"]["a"]["y"]), {1})
        self.assertLessEqual(len(bundle["samples"]["a"]["y"]), assistant.PER_STRATUM)
        other = np.where(self.cells == 2, 2, 0).astype(np.uint8)
        bundle, preview = self.update("b", other)
        self.assertIsNotNone(preview)
        self.assertEqual(set(bundle["model"].classes_), {1, 2})

    def test_terrain_features_align_and_other_layers_are_excluded(self):
        aligned = assistant.features(self.X, core.BASE_FEATS)
        self.assertEqual(aligned.shape[0], 6)
        np.testing.assert_array_equal(aligned, self.X)
        extras = ["nac", "psr", "cpr", "radar_s1_log", "image_count", "solar_bins",
                  "best_resolution", "uncertainty", "lola_slope", "rel", "zscene"]
        names = [*extras, *reversed(core.BASE_FEATS)]
        other = np.full((len(extras), *self.X.shape[1:]), np.nan, np.float32)
        reordered = np.concatenate([other, self.X[::-1]])
        np.testing.assert_array_equal(assistant.features(reordered, names), self.X)
        reordered[:len(extras)] = 100000
        np.testing.assert_array_equal(assistant.features(reordered, names), self.X)

    def test_training_and_inference_use_the_same_six_features(self):
        names = ["nac", "cpr", "psr", *reversed(core.BASE_FEATS)]
        X = np.concatenate([np.ones((3, *self.X.shape[1:]), np.float32), self.X[::-1]])
        self.enterContext(patch.object(core, "meta_get", return_value={"verified": True}))
        bundle, preview = assistant.update("a", X, names, self.cells, X[:, ::4, ::4],
                                            "a", (1, 2))
        self.assertEqual(bundle["model"].n_features_in_, 6)
        expected_X, _, _ = assistant.sample(self.X, *assistant.training_pixels(
            "a", self.X.shape[1:], self.cells))
        np.testing.assert_array_equal(bundle["samples"]["a"]["X"], expected_X)
        np.testing.assert_array_equal(preview, assistant.predict(
            bundle, self.X[:, ::4, ::4], core.BASE_FEATS))
        X[:3] = np.nan
        np.testing.assert_array_equal(preview, assistant.predict(bundle, X[:, ::4, ::4], names))
        np.testing.assert_array_equal(assistant.confidence(bundle, X, names),
                                      assistant.confidence(bundle, self.X, core.BASE_FEATS))

    def test_invalid_or_missing_terrain_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing: svf_local"):
            assistant.features(self.X[:-1], core.BASE_FEATS[:-1])
        for X, names in ((self.X, core.BASE_FEATS[:-1]),
                         (self.X[0], core.BASE_FEATS),
                         (self.X, [*core.BASE_FEATS[:-1], core.BASE_FEATS[0]])):
            with self.subTest(shape=X.shape, names=names):
                with self.assertRaisesRegex(ValueError, "uniquely named"):
                    assistant.features(X, names)

    def test_previous_optional_channel_recipe_is_rejected_without_modifying_it(self):
        bundle = assistant.load()
        bundle["schema"]["features"] += ["psr", "cpr"]
        bundle["schema"]["recipe"]["version"] = 4
        bundle["schema"]["recipe"].pop("name")
        assistant.save(bundle)
        previous = self.artifact()
        with self.assertRaisesRegex(ValueError, "different features or units"):
            assistant.load()
        self.assertEqual(self.artifact(), previous)

    def test_seamed_recipe_waits_for_explicit_update_without_writing(self):
        self.write_dem("a", 0)
        core.save_painting("a", self.cells)
        core.meta_set("a", verified=True)
        old, _ = self.update("a", self.cells)
        old["schema"]["recipe"]["version"] = 5
        assistant.save(old)
        before = self.artifact()
        paint_before = Path(core.painting_files("a")[0]).read_bytes()
        loaded = assistant.load()
        self.assertTrue(loaded["needs_retrain"])
        self.assertEqual(loaded["schema"], assistant.schema())
        self.assertEqual(loaded["samples"], {})
        self.assertIsNone(assistant.predict(loaded, self.X, core.BASE_FEATS))
        app = open_app()
        self.assertFalse(app.exception)
        self.assertFalse(any("Terrain seams" in item.value for item in app.info))
        self.assertIsNone(app.session_state.pred)
        self.assertEqual(self.artifact(), before)
        np.testing.assert_array_equal(app.session_state.cells, self.cells)
        app.button(key="update_model").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(self.artifact(), before)
        self.assertFalse(assistant.load().get("needs_retrain", False))
        self.assertIsNotNone(app.session_state.pred)
        self.assertFalse(any("Terrain seams" in item.value for item in app.info))
        self.assertEqual(Path(core.painting_files("a")[0]).read_bytes(), paint_before)

    def test_changed_class_schema_is_rejected(self):
        self.update("a", self.cells)
        with patch.object(core, "CODES", {1: "changed", 2: "unit"}):
            with self.assertRaisesRegex(ValueError, "different features or units"):
                assistant.load()

    def test_failed_training_keeps_previous_artifact(self):
        self.update("a", self.cells)
        previous = self.artifact()
        with patch.object(core.backend(), "fit_rows", side_effect=ValueError("fit failed")):
            with self.assertRaisesRegex(ValueError, "fit failed"):
                self.update("b", self.cells)
        self.assertEqual(previous, self.artifact())

    def test_confidence_is_kept_and_weights_training(self):
        unsure = np.where(self.cells == 2, core.UNSURE, np.where(self.cells > 0, core.SURE, 0))
        core.save_painting("a", self.cells)
        np.testing.assert_array_equal(core.load_confidence("a"),
                                      np.where(self.cells > 0, core.SURE, 0))  # the default
        self.assertFalse(Path(core.painting_files("a")[2]).exists())
        core.save_painting("a", self.cells, confidence=unsure)
        np.testing.assert_array_equal(core.load_confidence("a"), unsure)
        self.assertNotEqual(assistant.painting_hash(self.cells),
                            assistant.painting_hash(self.cells, unsure))
        self.enterContext(patch.object(core, "meta_get", return_value={"verified": True}))
        bundle, _ = assistant.update("a", self.X, core.BASE_FEATS, self.cells,
                                     self.X[:, ::4, ::4], "a", (1, 2), unsure)
        s = bundle["samples"]["a"]
        self.assertTrue(np.allclose(s["w"][s["y"] == 2], 0.2))
        self.assertTrue(np.allclose(s["w"][s["y"] == 1], 1.0))
        self.assertFalse(assistant.pending(bundle, "a", self.cells, unsure))
        self.assertTrue(assistant.pending(bundle, "a", self.cells))   # now all sure: changed

    def test_imported_maps_teach_only_their_painting_without_vector_lookup(self):
        mine = self.cells.copy()
        mine[0:10, 0:10] = 3
        lab, w, strata = assistant.training_pixels("a", (240, 240), mine, dem_path="a.tif")
        np.testing.assert_array_equal(lab, core.cells_to_labels(mine, (240, 240)))
        np.testing.assert_array_equal(strata, lab)
        np.testing.assert_array_equal(w, (lab != 255).astype(np.float32))

    def test_painted_maps_are_found_for_the_map_list(self):
        core.save_painting("a", self.cells)
        core.save_painting("b", self.cells, accepted=self.cells == 2)
        core.save_painting("c", np.zeros_like(self.cells))    # cleared: not painted
        core.save_painting("d", self.cells, final=True)       # saved only
        self.assertEqual(core.painted_maps(), ({"a", "b", "d"}, {"b"}))

    def test_approved_predictions_teach_at_half_weight(self):
        approved = self.cells == 2
        lab, w, _ = assistant.training_pixels("a", (240, 240), self.cells, accepted=approved)
        self.assertTrue(np.allclose(w[core.per_pixel(approved, (240, 240))], core.APPROVED))
        self.assertTrue(np.allclose(w[core.per_pixel(self.cells == 1, (240, 240))], 1.0))
        self.assertNotEqual(assistant.painting_hash(self.cells),
                            assistant.painting_hash(self.cells, accepted=approved))

    def test_confidence_is_the_probability_of_the_predicted_unit(self):
        bundle, _ = self.update("a", self.cells)
        conf = assistant.confidence(bundle, self.X, core.BASE_FEATS)
        self.assertEqual(conf.dtype, np.float32)
        self.assertTrue(np.isnan(conf[:8]).all())                # no terrain
        ok = conf[8:]
        self.assertTrue(((ok >= pictures.floor() - 1e-6) & (ok <= 1)).all())

    def test_roughness_window_is_round(self):
        steep = np.zeros((101, 101), np.float32)
        steep[50, 50] = 60                                     # one steep pixel
        r = core.local_std(steep, core.window(core.ROUGH_M, 5))
        self.assertAlmostEqual(r[50, 60] / r[57, 57], 1, delta=0.05)   # same distance
        spot = r > r.max() / 4
        ys, xs = np.nonzero(spot)
        self.assertLess(spot.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)), 0.85)  # no square

    def test_held_out_maps_are_never_trained_on(self):
        self.enterContext(patch.object(assistant, "verified_samples", side_effect=dict))
        with patch.object(core, "held_out_maps", return_value={"t"}):
            with self.assertRaises(ValueError):
                self.update("t", self.cells)
            bundle, _ = self.update("a", self.cells)
            bundle["samples"]["t"] = dict(bundle["samples"]["a"], site_id="t", geographic_group="t")
            assistant.save(bundle)
            bundle, _ = self.update("b", self.cells)           # an older held-out sample
            self.assertEqual(sorted(bundle["samples"]), ["a", "b"])  # is dropped
            bundle["samples"]["t"] = dict(bundle["samples"]["a"], site_id="t", geographic_group="t")
            assistant.save(bundle)
            refitted = assistant.refit()                       # without reading any map
            self.assertEqual(sorted(refitted["samples"]), ["a", "b"])
            self.assertNotEqual(refitted["revision"], bundle["revision"])
            self.assertEqual(assistant.refit()["revision"], refitted["revision"])

    def test_smart_fill_takes_the_model_outline(self):
        g = core.GRID
        suggested = np.ones((g, g), np.uint8)
        suggested[:, 60:] = 2
        patch_ = np.zeros((g, g), bool)
        patch_[10:20, 10:20] = True
        suggested[patch_] = 2                                  # an island of unit 2
        suggested[30, 30] = 2                                  # a one-cell speck
        suggested[:, 118:] = 0                                 # no terrain
        region = core.smart_fill(suggested, seed=(50, 5))      # click in unit 1
        self.assertTrue(region[30, 30])                        # small hole filled
        self.assertFalse(region[patch_].any() or region[:, 60:].any())
        self.assertEqual(int(region.sum()), g * 60 - 100)
        np.testing.assert_array_equal(core.smart_fill(suggested, seed=(15, 15)), patch_)
        self.assertFalse(core.smart_fill(suggested, seed=(5, 119)).any())
        inside = np.zeros((g, g), bool)
        inside[:40, :80] = True                                # a loop around both units
        region = core.smart_fill(suggested, inside=inside, unit=2)
        self.assertTrue(region[patch_].all() and region[:40, 60:80].all())
        self.assertFalse(region[30, 30])                       # the speck is left out
        self.assertEqual(int(region.sum()), 100 + 40 * 20)

    def test_accepted_cells_follow_their_painting(self):
        core.save_painting("a", self.cells, accepted=self.cells == 1)
        np.testing.assert_array_equal(core.load_accepted("a"), self.cells == 1)
        core.save_painting("a", self.cells)             # e.g. re-imported reference labels
        self.assertFalse(core.load_accepted("a").any())
        self.assertFalse(Path(core.painting_files("a")[1]).exists())
        core.save_painting("a", self.cells, final=True, accepted=self.cells == 2)
        self.assertFalse(core.load_accepted("a").any())  # the draft is what opens
        Path(core.painting_files("a")[0]).unlink()
        np.testing.assert_array_equal(core.load_accepted("a"), self.cells == 2)
        for shape in ((240, 240), (1024, 1024)):
            lab = core.cells_to_labels(self.cells, shape)
            np.testing.assert_array_equal(core.labels_to_cells(lab), self.cells)

    def test_drafts_recover_without_changing_saved_painting(self):
        core.save_painting("a", self.cells, final=True)
        saved = Path(core.output_file("a", "painting.npy")).read_bytes()
        blank = np.zeros_like(self.cells)
        core.save_painting("a", blank)
        np.testing.assert_array_equal(core.load_painting("a"), blank)
        self.assertEqual(saved, Path(core.output_file("a", "painting.npy")).read_bytes())
        core.save_painting("draft_only", self.cells)
        np.testing.assert_array_equal(core.load_painting("draft_only"), self.cells)

    def write_dem(self, name, shift, left=1000):
        """A 240 px, 1.2 km synthetic DEM with its left edge at `left` metres."""
        path = Path(core.DEM_DIR) / f"{name}.tif"
        path.parent.mkdir(exist_ok=True)
        y, x = np.mgrid[:240, :240]
        values = (np.sin(x / 17 + shift) * 30 + np.cos(y / 23) * 15 + x * .2).astype(np.float32)
        values[:8] = -9999
        with rasterio.open(path, "w", driver="GTiff", count=1, dtype="float32",
                           width=240, height=240, crs="ESRI:103878",
                           transform=from_origin(left, 2000, 5, 5), nodata=-9999) as dst:
            dst.write(values, 1)
            dst.update_tags(site_id=name)
        return str(path)

    def test_prediction_approval_depends_on_readonly_metadata_not_site_or_verification(self):
        self.write_dem("a", 0)
        self.update("a", self.cells)                      # a model, so there is a prediction
        for site, verified, read_only in (("regional", False, False), ("MM026", False, False),
                                           ("MM026", True, False), ("regional", True, True)):
            with self.subTest(site=site, verified=verified, read_only=read_only):
                Path(core.DEM_DIR, "working_dems.json").write_text(json.dumps([{
                    "site": site, "working_dem": "a.tif", "verified": verified,
                    "read_only": read_only}]), encoding="utf-8")
                app = open_app()
                self.assertFalse(app.exception)
                self.assertIsNotNone(app.session_state.pred)
                self.assertEqual(app.session_state.locked, read_only)
                accept = next(b for b in app.button if b.label == "Approve prediction")
                self.assertEqual(accept.disabled, read_only)
                self.assertEqual(app.segmented_control(key="tool").disabled, read_only)

    def test_explicit_readonly_maps_reject_strokes_and_painting_writes(self):
        self.write_dem("a", 0)
        self.update("a", self.cells)
        core.save_painting("a", self.cells)
        Path(core.DEM_DIR, "working_dems.json").write_text(json.dumps([{
            "site": "ordinary-location", "working_dem": "a.tif", "verified": True,
            "read_only": True}]), encoding="utf-8")
        artifact = self.artifact()
        strokes = []
        with patch("app.canvas.map_canvas", side_effect=lambda *a, **k: strokes.pop() if strokes
                   else None) as shown:
            app = open_app()
            self.assertFalse(app.exception)
            self.assertEqual(shown.call_args.kwargs["tool"], "locked")
            self.assertEqual(app.selectbox(key="selected_map").options,
                             ["a — painted, verified"])
            for label in ("Clear", "Undo", "Approve prediction", "Save labels + map"):
                self.assertTrue(next(b for b in app.button if b.label == label).disabled, label)
            self.assertTrue(app.radio[0].disabled)

            # A stroke from the page is refused, and Update leaves the saved painting alone.
            strokes.append({"id": "s1", "cells": list(range(50)), "click": False})
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.cells.tobytes(), self.cells.tobytes())
            original = st.button

            def force_button(label, *args, **kwargs):
                value = original(label, *args, **kwargs)
                return True if label in {"Save labels + map"} else value

            with patch.object(st, "button", side_effect=force_button), \
                    patch.object(assistant, "update", side_effect=AssertionError("Read-only map")), \
                    patch.object(core, "save_painting", side_effect=AssertionError("Read-only map")):
                app.run()
            self.assertFalse(app.exception)
        self.assertEqual(core.load_painting("a").tobytes(), self.cells.tobytes())
        self.assertEqual(self.artifact(), artifact)
        self.assertFalse(Path(core.output_file("a", "painting.npy")).exists())
        self.assertFalse(Path(core.output_file("a", "labels.tif")).exists())

    def write_records(self, *maps):
        """working_dems.json with (map name, site) pairs."""
        Path(core.DEM_DIR, "working_dems.json").write_text(json.dumps(
            [{"site": site, "group": site, "working_dem": f"{name}.tif"}
             for name, site in maps]), encoding="utf-8")

    def test_training_status_tracks_all_verified_paintings_without_fitting(self):
        a = self.write_dem("a", 0)
        b = self.write_dem("b", 1, left=20000)
        self.write_dem("draft-only", 2, left=40000)
        self.write_records(("a", "a"), ("b", "b"), ("draft-only", "draft-only"))
        for name in ("a", "b"):
            core.save_painting(name, self.cells)
            core.meta_set(name, verified=True)
        self.assertTrue(assistant.training_changed(assistant.load()))
        bundle = assistant.retrain()
        previous = self.artifact()
        with patch.object(core, "build", side_effect=AssertionError("Status must not build features")):
            self.assertFalse(assistant.training_changed(bundle))
            # The user's already updated version-6 model must start green too.
            legacy = {key: value for key, value in bundle.items() if key != "training_inputs"}
            self.assertFalse(assistant.training_changed(legacy))
            core.save_painting("draft-only", self.cells)
            self.assertFalse(assistant.training_changed(bundle))
            edited = np.where(self.cells == 2, 3, self.cells).astype(np.uint8)
            core.save_painting("b", edited)
            self.assertTrue(assistant.training_changed(bundle))
            core.save_painting("b", self.cells)
            self.assertFalse(assistant.training_changed(bundle))
            core.save_painting("b", self.cells, accepted=self.cells > 0)
            self.assertTrue(assistant.training_changed(bundle))
            core.save_painting("b", self.cells, confidence=np.where(self.cells > 0, core.MOSTLY, 0))
            self.assertTrue(assistant.training_changed(bundle))
            core.save_painting("b", self.cells)
            core.meta_set("b", final=False, verified=False)
            self.assertTrue(assistant.training_changed(bundle))
            core.meta_set("b", final=False, verified=True)
            self.assertFalse(assistant.training_changed(bundle))
            core.meta_set("draft-only", final=False, verified=True)
            self.assertTrue(assistant.training_changed(bundle))
            core.meta_set("draft-only", final=False, verified=False)
            core.save_painting("b", np.zeros_like(self.cells))
            self.assertTrue(assistant.training_changed(bundle))
            core.save_painting("b", self.cells)
            self.write_dem("b", 3, left=20000)
            self.assertTrue(assistant.training_changed(bundle))
        self.assertEqual(self.artifact(), previous)
        bundle = assistant.retrain()
        self.assertFalse(assistant.training_changed(bundle))
        core.set_test_maps(["b"])
        self.assertFalse(assistant.training_changed(bundle))

    def test_training_status_remembers_a_painting_with_no_usable_samples(self):
        self.write_dem("a", 0)
        core.meta_set("a", verified=True)
        cells = np.zeros_like(self.cells)
        cells[:4] = 1  # the DEM's first eight pixel rows are all missing
        core.save_painting("a", cells)
        bundle = assistant.retrain()
        self.assertEqual(bundle["samples"], {})
        self.assertIn("a", bundle["training_inputs"])
        self.assertFalse(assistant.training_changed(bundle))
        changed = cells.copy(); changed[10:20, 20:30] = 2
        core.save_painting("a", changed)
        self.assertTrue(assistant.training_changed(bundle))

    def test_legacy_test_maps_and_neighbours_remain_available(self):
        for name, shift, x in (("a", 0, 1000), ("b", 1, 20000), ("c", 2, 21200),
                               ("d", 3, 23600)):
            self.write_dem(name, shift, x)                    # c touches b; d is further
        self.write_records(("a", "A"), ("b", "B"), ("c", "B"), ("d", "B"))
        self.assertEqual(core.held_out_maps(["b"]), {"b", "c"})
        core.set_test_maps(["b"])
        app = open_app()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="selected_map").options, ["a", "b", "c", "d"])

    def test_model_update_does_not_run_evaluation(self):
        self.write_dem("a", 0)
        self.write_dem("unverified", 2, left=40000)
        self.write_dem("t", 1, left=20000)
        self.write_records(("a", "a"), ("t", "t"), ("unverified", "unverified"))
        core.set_test_maps(["t"])
        core.save_painting("a", self.cells)
        core.meta_set("a", final=False, verified=True)
        core.save_painting("unverified", self.cells)
        app = open_app()
        app.selectbox(key="selected_map").select(core.dem_path("unverified")).run()
        self.assertFalse(app.checkbox(key="verified_unverified").value)
        with patch.object(evaluation, "run", side_effect=AssertionError("Training must not evaluate")), \
                patch.object(core, "save_painting", side_effect=AssertionError("Training must not save paintings")), \
                patch.object(assistant, "retrain", wraps=assistant.retrain) as retrain:
            for _ in range(2):
                old_revision = assistant.load()["revision"]
                app.button(key="update_model").click().run()
                self.assertFalse(app.exception)
                self.assertNotEqual(assistant.load()["revision"], old_revision)
                self.assertEqual(set(assistant.load()["samples"]), {"a"})
                self.assertIsNotNone(app.session_state.pred)
                self.assertIn("Prediction", app.session_state.layers)
                self.assertEqual(app.button(key="update_model").label, ":green[Update model]")
                self.assertFalse(app.button(key="update_model").disabled)
                self.assertFalse(app.success)
            self.assertEqual(retrain.call_count, 2)
        self.assertFalse((paths.OUTPUT_DIR / "evaluations").exists())
        self.assertFalse(Path(core.output_file("a", "labels.tif")).exists())

    def test_evaluation_selection_and_run_leave_active_training_unchanged(self):
        for name, shift, x in (("a", 0, 1000), ("b", 1, 20000)):
            self.write_dem(name, shift, x)
        self.write_records(("a", "a"), ("b", "b"))
        for name in ("a", "b"):
            self.update(name, self.cells)
            core.save_painting(name, self.cells, final=True)
            core.meta_set(name, verified=True)
        progress = open_app("progress")
        before = self.artifact()
        progress.multiselect(key="evaluation_maps").set_value(["b"]).run()
        self.assertFalse(progress.exception)
        self.assertEqual([w.key for w in progress.multiselect], ["evaluation_maps"])
        self.assertEqual(core.test_maps(), [])
        self.assertEqual(self.artifact(), before)
        next(b for b in progress.button if b.label == "Evaluate").click().run()
        self.assertFalse(progress.exception)
        self.assertFalse(progress.error)
        self.assertEqual(progress.multiselect(key="evaluation_maps").value, ["b"])
        self.assertEqual(core.test_maps(), [])
        self.assertEqual(sorted(assistant.load()["samples"]), ["a", "b"])
        self.assertEqual(self.artifact(), before)
        reports = list((paths.OUTPUT_DIR / "evaluations").glob("evaluation-*.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertEqual(report["validation"]["folds"][0]["train_tiles"], ["a"])
        self.assertEqual(report["validation"]["folds"][0]["test_tiles"], ["b"])
        for name in ("a", "b"):
            np.testing.assert_array_equal(core.painting_source(name, final=True)[0], self.cells)

    def test_live_recipe_change_preserves_painting_and_refreshes_assistant(self):
        self.write_dem("a", 0)
        app = open_app()
        self.assertFalse(app.exception)
        legacy = assistant.load()
        legacy["schema"]["recipe"]["version"] = 4
        legacy["schema"]["features"] += ["psr", "cpr"]
        app.session_state.assistant = legacy
        app.session_state.cells = self.cells.copy()
        app.session_state.pred = np.ones((120, 120), np.uint8)
        app.session_state.conf_for = ("a", "old-recipe")
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.assistant["schema"], assistant.schema())
        self.assertIsNone(app.session_state.pred)
        self.assertNotIn("conf_for", app.session_state)
        np.testing.assert_array_equal(app.session_state.cells, self.cells)
        self.assertFalse(Path(assistant.MODEL_DIR).exists())

    def test_live_model_install_preserves_painting_and_refreshes_prediction(self):
        self.write_dem("a", 0)
        app = open_app()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state.pred)
        app.session_state.cells = self.cells.copy()
        before = assistant.artifact_stamp()
        bundle, _ = self.update("another-map", self.cells)
        self.assertNotEqual(before, assistant.artifact_stamp())
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.assistant["revision"], bundle["revision"])
        self.assertIsNotNone(app.session_state.pred)
        np.testing.assert_array_equal(app.session_state.cells, self.cells)

    def reference_samples(self):
        def seed(group, region, public):
            return dict(source_kind="reference_snapshot", geographic_group=group, site_id=group,
                        public_region=region, public_tiles=[public])
        return {"N1014": seed("nobile-1", "nobile_rim_1", "public-n1014"),
                "MS1": seed("nobile-1", "nobile_rim_1", "public-ms1"),
                "N2005": seed("nobile-2", "nobile_rim_2", "public-n2005"),
                "correction": {}}

    def test_reference_holdout_excludes_whole_geography_by_region_or_source_tile(self):
        samples = self.reference_samples()
        with patch.object(core, "map_record", return_value=None):
            for held in ({"public-ms1"}, {"N1014"}):
                self.assertEqual(set(assistant.retained_samples(samples, held)),
                                 {"N2005", "correction"})
        with patch.object(core, "map_record", return_value={
                        "site": "nobile_rim_1", "working_dem": "distant-tile-in-same-region.tif"}):
            self.assertEqual(set(assistant.retained_samples(samples, {"distant-tile-in-same-region"})),
                             {"N2005", "correction"})
        self.assertEqual(assistant.retained_samples(samples, set()), samples)

    def test_reference_holdout_excludes_new_catalog_overlap_by_footprint(self):
        dem = self.write_dem("custom", 0)
        with rasterio.open(dem) as src:
            footprint = dict(crs_wkt=src.crs.to_wkt(), bounds=list(src.bounds))
        samples = self.reference_samples()
        for key, sample in samples.items():
            if key != "correction":
                sample["footprint"] = dict(footprint)
        samples["N2005"]["footprint"]["bounds"] = [20000, 1000, 21000, 2000]
        self.assertEqual(set(assistant.retained_samples(samples, {"custom"})),
                         {"N2005", "correction"})
        samples["N1014"].pop("footprint")
        with self.assertRaisesRegex(ValueError, "no usable footprint"):
            assistant.retained_samples(samples, {"custom"})

    def test_refit_and_update_share_reference_exclusion(self):
        self.enterContext(patch.object(assistant, "verified_samples", side_effect=dict))
        self.update("a", self.cells)
        bundle = assistant.load()
        for key, provenance in self.reference_samples().items():
            if key != "correction":
                bundle["samples"][key] = dict(bundle["samples"]["a"], **provenance)
        assistant.save(bundle)
        with patch.object(core, "held_out_maps", return_value={"public-ms1"}):
            refitted = assistant.refit()
            self.assertEqual(set(refitted["samples"]), {"a", "N2005"})
            assistant.save(bundle)
            updated, _ = self.update("b", self.cells)
            self.assertEqual(set(updated["samples"]), {"a", "b", "N2005"})

    def test_background_falls_back_on_a_map_without_it(self):
        self.write_dem("a", 0)
        self.write_dem("a_cpr", 2)                        # radar only for map a
        second = self.write_dem("b", 1)
        app = open_app()
        self.assertIn("Radar (CPR)", app.segmented_control(key="background").options)
        app.segmented_control(key="background").set_value("cpr").run()
        self.assertFalse(app.exception)
        app.selectbox(key="selected_map").select(second).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.background, "hillshade")

    def test_app_update_save_new_map_and_restart(self):
        self.write_dem("a", 0)
        second = self.write_dem("b", 1)
        app = open_app()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state.pred)
        self.assertFalse(any("Self-check" in h.value for h in app.subheader))

        def click(label):
            button = (app.button(key="update_model") if label == "Update model" else
                      next(b for b in app.button if b.label == label))
            button.click().run()
            self.assertFalse(app.exception)

        # Deliver one stroke, repeated on the rerun: it must be applied only once.
        stroke = {"id": "1", "cells": [1210]}
        calls = iter([stroke, stroke])
        with patch("app.canvas.map_canvas", side_effect=lambda *a, **kw: next(calls, None)):
            app.run()
        self.assertEqual(app.session_state.cells[10, 10], 1)
        self.assertEqual(len(app.session_state.hist), 1)
        self.assertFalse(Path(assistant.MODEL_DIR).exists())
        self.assertTrue(Path(core.painting_files("a")[0]).is_file())
        self.assertFalse(Path(core.output_file("a", "painting.npy")).exists())
        self.assertFalse(Path(core.output_file("a", "labels.tif")).exists())
        click("Undo")
        self.assertFalse(app.session_state.cells.any())

        # Painting while "Mostly" is chosen records it, and Undo takes it back too.
        app.segmented_control(key="certainty").set_value(core.MOSTLY).run()
        calls = iter([{"id": "m", "cells": [1210]}])
        with patch("app.canvas.map_canvas", side_effect=lambda *a, **kw: next(calls, None)):
            app.run()
        self.assertEqual(app.session_state.confidence[10, 10], core.MOSTLY)
        self.assertEqual(core.load_confidence("a")[10, 10], core.MOSTLY)
        self.assertIn("Training weights (defaults included): full weight 0 · mostly 1 · unsure 0 cells.",
                      [c.value for c in app.caption])
        click("Undo")
        self.assertFalse(app.session_state.confidence.any())
        self.assertFalse(Path(core.painting_files("a")[2]).exists())
        app.segmented_control(key="certainty").set_value(core.SURE).run()

        # Seed a larger hand painting to fit a useful classifier in this short test.
        app.session_state.cells = self.cells.copy()
        core.save_painting("a", self.cells)
        app.run()
        self.assertIsNone(app.session_state.pred)
        app.checkbox(key="verified_a").check().run()
        click("Update model")
        self.assertIsNotNone(app.session_state.pred)
        self.assertFalse(Path(core.output_file("a", "painting.npy")).exists())
        self.assertFalse(Path(core.output_file("a", "labels.tif")).exists())

        # Prediction and painting switch on and off independently; Flip swaps them.
        self.assertIn("Prediction", app.session_state.layers)
        app.pills(key="layers").set_value(["Painting"]).run()
        shots, extras = [], []
        with patch("app.canvas.map_canvas", side_effect=lambda image, **kw: (
                shots.append(image.tobytes()), extras.append(kw))[-1]):
            app.run()
            layers = app.pills(key="layers")
            self.assertEqual(layers.value, ["Painting"])
            painting = shots[-1]
            click("Flip")
            self.assertEqual(app.session_state.layers, ["Prediction"])
            prediction = shots[-1]
            click("Flip")
            self.assertEqual(app.session_state.layers, ["Painting"])
            self.assertEqual(shots[-1], painting)
            app.pills(key="layers").set_value(["Prediction", "Painting"]).run()
            both = shots[-1]
            self.assertEqual(len({both, prediction, painting}), 3)
            self.assertEqual(extras[-1]["legend"], "Yellow: disagreement")
            app.pills(key="layers").set_value([]).run()
            self.assertFalse(app.exception)
            self.assertNotEqual(shots[-1], prediction)
            app.pills(key="layers").set_value(["Prediction", "Painting"]).run()
            self.assertEqual(shots[-1], both)

            # Any model input can replace the hillshade under the units, and Space shows it.
            hillshade = extras[-1]["terrain"].tobytes()
            self.assertEqual(extras[-1]["legend"], "Yellow: disagreement")
            app.segmented_control(key="background").set_value("svf").run()
            self.assertFalse(app.exception)
            self.assertNotEqual(extras[-1]["terrain"].tobytes(), hillshade)
            self.assertIn("Sky view", extras[-1]["legend"])
            self.assertNotEqual(shots[-1], both)
            app.run()
            self.assertEqual(app.session_state.background, "svf")
            app.segmented_control(key="background").set_value("hillshade").run()
            self.assertEqual(shots[-1], both)

            # Confidence replaces the background and survives a Flip.
            app.pills(key="layers").set_value(["Prediction", "Confidence"]).run()
            self.assertFalse(app.exception)
            self.assertEqual(extras[-1]["legend"],
                             "Model score: black is low or missing confidence, white is high; "
                             "not certainty of correctness")
            self.assertNotEqual(extras[-1]["terrain"].tobytes(), hillshade)
            self.assertTrue(app.segmented_control(key="background").disabled)
            click("Flip")
            self.assertEqual(app.session_state.layers, ["Painting", "Confidence"])
            app.pills(key="layers").set_value(["Prediction", "Painting"]).run()
            self.assertEqual(shots[-1], both)
        self.assertTrue(app.session_state.cells.any())   # hiding never erases painting

        # Accept fills only unpainted cells and is undoable; the assistant does not learn
        # accepted cells until they are painted over.
        painted = app.session_state.cells.copy()
        click("Approve prediction")
        cells, accepted = app.session_state.cells, app.session_state.accepted
        np.testing.assert_array_equal(cells[painted > 0], painted[painted > 0])
        np.testing.assert_array_equal(accepted, (painted == 0) & (cells > 0))
        self.assertTrue(accepted[60, 60])
        self.assertTrue(Path(core.painting_files("a")[1]).exists())
        self.assertEqual(app.button(key="update_model").label, ":yellow[Update model]")
        calls = iter([{"id": "2", "cells": [60 * core.GRID + 60]}])
        with patch("app.canvas.map_canvas", side_effect=lambda *a, **kw: next(calls, None)):
            app.run()
        self.assertFalse(app.session_state.accepted[60, 60])
        self.assertEqual(app.session_state.cells[60, 60], 1)
        self.assertEqual(app.button(key="update_model").label, ":yellow[Update model]")
        click("Undo")
        click("Undo")
        np.testing.assert_array_equal(app.session_state.cells, painted)
        self.assertFalse(app.session_state.accepted.any())
        self.assertFalse(Path(core.painting_files("a")[1]).exists())

        # Smart fill: a click paints the model's region around it as the chosen unit, only
        # over unpainted cells; agreeing cells are accepted, relabelled ones are yours.
        app.segmented_control(key="tool").set_value("smart").run()
        suggested = core.labels_to_cells(app.session_state.pred)
        seed = tuple(np.argwhere((painted == 0) & (suggested > 0))[0])
        expected = core.smart_fill(suggested, seed=seed) & (painted == 0)
        calls = iter([{"id": "3", "cells": [seed[0] * core.GRID + seed[1]], "click": True}])
        with patch("app.canvas.map_canvas", side_effect=lambda *a, **kw: next(calls, None)):
            app.run()
        self.assertFalse(app.exception)
        np.testing.assert_array_equal(app.session_state.cells != painted, expected)
        self.assertTrue((app.session_state.cells[expected] == 1).all())   # smooth highlands
        np.testing.assert_array_equal(app.session_state.accepted, expected & (suggested == 1))
        click("Undo")
        np.testing.assert_array_equal(app.session_state.cells, painted)
        app.segmented_control(key="tool").set_value("paint").run()
        before = self.artifact()
        before_pred = app.session_state.pred.copy()
        click("Clear")
        self.assertEqual(before, self.artifact())
        np.testing.assert_array_equal(before_pred, app.session_state.pred)
        click("Undo")

        buttons = ["Update model" if b.key == "update_model" else b.label for b in app.button]
        self.assertLess(buttons.index("Update model"), buttons.index("Save labels + map"))
        save_button = next(b for b in app.button if b.label == "Save labels + map")
        self.assertEqual(save_button.proto.type, "primary")
        click("Save labels + map")
        saved_files = [Path(core.output_file("a", filename)) for filename in
                       ("painting.npy", "labels.tif", "map.tif", "meta.json", "thumb.png")]
        saved_schema = core.meta_get("a")["model_schema"]
        self.assertEqual(saved_schema["features"], list(core.BASE_FEATS))
        self.assertEqual(saved_schema["recipe"], assistant.schema()["recipe"])
        self.assertEqual(before, self.artifact())  # Save must not retrain
        with rasterio.open(core.output_file("a", "labels.tif")) as labels:
            actual = labels.read(1)
            self.assertEqual(actual[120, 120], 255)  # unpainted, even though predicted
            self.assertTrue((actual[:8] == 255).all())
            self.assertEqual(actual[30, 30], 1)
        self.assertTrue(Path(core.output_file("a", "map.tif")).exists())
        snapshot = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in saved_files}
        app.session_state.cells = np.where(self.cells == 2, 3, self.cells).astype(np.uint8)
        core.save_painting("a", app.session_state.cells)
        app.run()
        self.assertEqual(before, self.artifact())
        app.checkbox(key="verified_a").check().run()
        click("Update model")
        self.assertEqual(snapshot, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in saved_files})
        app.selectbox(key="selected_map").select(second).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state.cells.any())
        self.assertIsNotNone(app.session_state.pred)
        self.assertEqual(set(np.unique(app.session_state.pred)), {1, 3, 255})
        fresh = open_app()
        self.assertFalse(fresh.exception)
        self.assertIsNotNone(fresh.session_state.pred)
        self.assertEqual(fresh.session_state.assistant["revision"], assistant.load()["revision"])
        gallery = open_app("gallery")
        self.assertFalse(gallery.exception)


if __name__ == "__main__":
    unittest.main()
