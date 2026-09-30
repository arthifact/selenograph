"""Clean-start UI and smoke regressions, using only disposable catalogs and artifacts.

Run: python -B -m unittest tests.test_clean_start -v
"""
from contextlib import ExitStack, redirect_stdout
import io
import json

from pathlib import Path
from typing import Any
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import assistant_model as assistant
from app import canvas, core, evaluation, paths
from tests import smoke_app_layout as smoke


class CleanStartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for attr, value in {
            "PROJECT_ROOT": self.root, "DATA_DIR": self.root / "data",
            "DEM_DIR": self.root / "data/processed_data", "OUTPUT_DIR": self.root / "output",
                        "POSTER_DIR": self.root / "poster",
            "OUT_DIR": self.root / "output/paintings", "LEGACY_OUT_DIR": self.root / "output/maps",
            "MAP_DIR": self.root / "output/maps",
            "DRAFT_DIR": self.root / "output/painting_drafts",
            "LEGACY_DRAFT_DIR": self.root / "output/drafts",
            "LEGACY_MIRRORED_DIR": self.root / "output",
            "MODEL_DIR": self.root / "models", "REFERENCE_DIR": self.root / "data/reference",
            "PROFESSOR_MAPS_DIR": self.root / "data/reference/professor",
            "TEST_MAPS_FILE": self.root / "output/test_maps.json",
        }.items():
            self.stack.enter_context(patch.object(paths, attr, value))
        for module, attr, value in (
            (core, "DEM_DIR", paths.DEM_DIR), (core, "OUT_DIR", paths.OUT_DIR),
            (core, "MAP_DIR", paths.MAP_DIR),
            (core, "LEGACY_OUT_DIR", paths.LEGACY_OUT_DIR),
            (core, "LEGACY_DRAFT_DIR", paths.LEGACY_DRAFT_DIR),
            (core, "LEGACY_MIRRORED_DIR", paths.LEGACY_MIRRORED_DIR),
            (core, "DRAFT_DIR", paths.DRAFT_DIR), (core, "TEST_MAPS_FILE", paths.TEST_MAPS_FILE),
            (core, "MODEL", "rf"), (assistant, "MODEL_DIR", paths.MODEL_DIR),
        ):
            self.stack.enter_context(patch.object(module, attr, value))

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def public(self, name="public-a", section="south", *, location=None, shape=(120, 120), left=1000):
        folder = paths.DEM_DIR / section / "tiles"
        dem = folder / f"{name}.tif"
        nac = folder / f"{name}_nac.tif"
        folder.mkdir(parents=True, exist_ok=True)
        y, x = np.mgrid[:shape[0], :shape[1]]
        for path, values in ((dem, np.sin(x / 10) * 20 + y / 5), (nac, x + y)):
            with rasterio.open(path, "w", driver="GTiff", count=1, dtype="float32",
                               width=shape[1], height=shape[0], nodata=-9999,
                               crs="ESRI:103878", transform=from_origin(left, 2000, 5, 5)) as dst:
                dst.write(values.astype(np.float32), 1)
        manifest = folder / "working_dems.json"
        records = json.loads(manifest.read_text()) if manifest.exists() else []
        records.append({
            "working_dem": dem.name, "site": location or f"{section}-location",
            "group": location or f"{section}-location", "collection": "unused provenance",
            "size": list(shape), "transform": [5, 0, left, 0, -5, 2000],
            "layers": {"nac": nac.name},
        })
        self.write_json(manifest, records)
        return str(dem)

    def test_unhydrated_lfs_data_or_model_shows_download_instructions(self):
        dem = Path(self.public())
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 1000\n"
        original = dem.read_bytes()
        for missing in (dem, paths.MODEL_DIR / "rf.joblib"):
            with self.subTest(missing=missing.name):
                dem.write_bytes(original)
                missing.parent.mkdir(parents=True, exist_ok=True)
                missing.write_bytes(pointer)
                with patch.object(core, "build", side_effect=AssertionError("Do not read LFS pointers")):
                    app = AppTest.from_file(str(smoke.ENTRYPOINT), default_timeout=45).run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any("git lfs pull" in error.value for error in app.error))

    def bundled(self, name="bundled-map", section="mons-mouton(reference)", *,
                left=40000, read_only=False):
        dem = Path(self.public(name, section, left=left))
        folder = dem.parent / f"{name}-annotations"
        folder.mkdir()
        cells = np.zeros((core.GRID, core.GRID), np.uint8)
        cells[10:40, 10:40] = 1
        cells[70:100, 70:100] = 2
        np.save(folder / "painting.npy", cells)
        with rasterio.open(dem) as src:
            profile: dict[str, Any] = dict(src.profile, dtype="uint8", nodata=255)
        with rasterio.open(folder / "labels.tif", "w", **profile) as dst:
            dst.write(core.cells_to_labels(cells, (120, 120)), 1)
        manifest = dem.parent / "working_dems.json"
        records = json.loads(manifest.read_text())
        records[-1].update(read_only=read_only, annotations={
            "painting": f"{folder.name}/painting.npy", "labels": f"{folder.name}/labels.tif",
            "verified": True, "restricted": True, "label_status": "user-verified source"})
        self.write_json(manifest, records)
        return str(dem), cells

    def reference_snapshot(self, name="private-reference", *, map_id="MM026",
                           site_id="mons-mouton", tile_id="ref-1", variants=()):
        site = paths.PROFESSOR_MAPS_DIR / "sites" / site_id
        self.write_json(site / "site.json", {"map_id": map_id})
        folder = site / "tiles" / tile_id
        record: dict[str, object] = {"legacy_name": name, "site_id": site_id}
        if variants:
            folder.mkdir(parents=True, exist_ok=True)
            dem = folder / "dem.tif"
            layers: list[tuple[Path, np.ndarray]] = [(dem, np.arange(576, dtype=np.float32).reshape(24, 24))]
            annotations = []
            for variant in variants:
                labels = folder / f"{variant}.tif"
                layers.append((labels, np.full((24, 24), 1 if variant == "saved" else 2, np.uint16)))
                provenance = folder / f"{variant}.json"
                self.write_json(provenance, {"review_status": "unreviewed_snapshot"})
                annotations.append({"snapshot_variant": variant,
                                    "path": str(labels.relative_to(paths.PROFESSOR_MAPS_DIR)),
                                    "provenance": str(provenance.relative_to(paths.PROFESSOR_MAPS_DIR))})
            for path, values in layers:
                with rasterio.open(path, "w", driver="GTiff", count=1, dtype=values.dtype,
                                   width=24, height=24, crs="EPSG:4326",
                                   transform=from_origin(1000, 2000, 5, 5)) as dst:
                    dst.write(values, 1)
            record.update(tile_id=tile_id, annotations=annotations,
                          layers={"dem": {"path": str(dem.relative_to(paths.PROFESSOR_MAPS_DIR))}})
        self.write_json(folder / "tile.json", record)

    def app(self):
        app = AppTest.from_file(str(smoke.ENTRYPOINT), default_timeout=30).run()
        self.assert_ok(app)
        return app

    def assert_ok(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertFalse(any(w.key == "map_location" or w.label == "Location" for w in app.selectbox))

    def button(self, app, label):
        if label == "Update model":
            return app.button(key="update_model")
        return next(b for b in app.button if b.label == label)

    def assert_no_artifacts(self):
        for name in ("output", "models", "reports", "poster"):
            self.assertFalse((self.root / name).exists(), name)

    def test_browsing_clean_pages_never_saves_or_trains(self):
        self.public()
        edge = self.public("edge", "north", shape=(40, 32), left=20000)
        self.reference_snapshot()
        with smoke.read_only():
            app = self.app()
            self.assertEqual(set(app.selectbox(key="selected_map").options), {"public-a", "edge"})
            self.assertIsNone(app.session_state.pred)
            self.assertFalse(app.session_state.cells.any())
            self.assertFalse(app.radio[0].disabled)
            for label in ("Clear", "Undo", "Approve prediction", "Save labels + map"):
                self.assertTrue(self.button(app, label).disabled, label)

            app.selectbox(key="selected_map").select(edge).run()
            self.assert_ok(app)
            app.selectbox(key="map_section").select("north").run()
            self.assert_ok(app)
            self.assertEqual(app.selectbox(key="selected_map").options, ["edge"])
            self.assertEqual(app.selectbox(key="selected_map").value, edge)
            app.selectbox(key="map_section").select("All").run()
            self.assert_ok(app)
            self.assertEqual(set(app.selectbox(key="selected_map").options), {"public-a", "edge"})
            self.assertEqual(app.selectbox(key="selected_map").value, edge)
            self.assertEqual(app.session_state.cells.shape, (core.GRID, core.GRID))
            app.segmented_control(key="background").set_value("nac").run()
            self.assert_ok(app)
            app.switch_page("app/app_pages/gallery.py").run()
            self.assert_ok(app)
            self.assertEqual({c.key for c in app.checkbox}, {"v_public-a", "v_edge"})
            app.switch_page("app/app_pages/progress.py").run()
            self.assert_ok(app)
            self.assertTrue(self.button(app, "Evaluate").disabled)
            self.assertTrue(any("No training samples" in info.value for info in app.info))
            app.run()
            self.assert_ok(app)
        self.assert_no_artifacts()

    def test_smoke_discovers_all_section_tiles_despite_different_site_metadata(self):
        self.public()
        self.public("edge", "north", shape=(40, 32), left=20000)
        self.public("local-map", "south", location="another-location", left=40000)
        self.reference_snapshot(variants=("saved", "draft"))
        self.bundled(left=60000)
        output = io.StringIO()
        with redirect_stdout(output):
            summary = smoke.main(expected_sections=3, expected_verified_bundles=1)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["catalog"]["installed"], 4)
        self.assertEqual(summary["sections"], {"mons-mouton(reference)": 1, "north": 1, "south": 2})
        self.assertNotIn("locations", summary)
        self.assertEqual(summary["edge_tile"], [40, 32])
        self.assertTrue(summary["nac_display_checked"])
        self.assertEqual(summary["annotations"], {"bundled": 1, "verified": 1, "browsed": ["bundled-map"]})
        self.assertEqual(summary["gallery"]["bundled"], 1)
        self.assertEqual(summary["navigation"], ["Paint", "Gallery", "Progress"])
        self.assertFalse(summary["evaluation"]["report_available"])
        self.assertEqual(summary["persistence"]["changed"], [])
        self.assertTrue(json.loads(output.getvalue())["passed"])
        self.assert_no_artifacts()

    def test_smoke_covers_seventeen_sections_and_six_verified_bundles(self):
        sections = [f"site-{i}" for i in range(13)]
        for index, section in enumerate(sections):
            self.public(f"tile-{index}", section, shape=(40, 32), left=1000 + index * 20000)
        bundle_sections = ["mons-mouton(reference)", "nobile1(reference)",
                           "nobile1-ms1(reference)", "nobile2(reference)"]
        counts = dict.fromkeys(sections, 1)
        for index in range(6):
            section = bundle_sections[index % len(bundle_sections)]
            self.bundled(f"bundled-{index}", section, left=300000 + index * 20000)
            counts[section] = counts.get(section, 0) + 1
        with redirect_stdout(io.StringIO()):
            summary = smoke.main(expected_sections=17, expected_verified_bundles=6)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["catalog"]["installed"], sum(counts.values()))
        self.assertEqual(summary["sections"], counts)
        self.assertEqual(set(summary["annotations"]["browsed"]), {f"bundled-{i}" for i in range(6)})
        self.assertEqual(summary["gallery"]["verification_controls"], sum(counts.values()))
        self.assertEqual(summary["persistence"]["changed"], [])
        self.assert_no_artifacts()

    def test_smoke_accepts_one_section_without_a_location_widget(self):
        self.public()
        with redirect_stdout(io.StringIO()):
            summary = smoke.main()
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["sections"], {"south": 1})
        self.assertNotIn("locations", summary)
        self.assert_no_artifacts()

    def test_smoke_accepts_an_empty_catalog(self):
        with redirect_stdout(io.StringIO()):
            summary = smoke.main()
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["catalog"]["installed"], 0)
        self.assertEqual(summary["sections"], {})
        self.assertEqual(summary["annotations"], {"bundled": 0, "verified": 0, "browsed": []})
        self.assertFalse(summary["evaluation"]["report_available"])
        self.assertEqual(summary["navigation"], ["Paint", "Gallery", "Progress"])
        self.assertIn("Return to Paint", [t["action"] for t in summary["timings"]])
        self.assert_no_artifacts()

    def test_smoke_browses_bundled_annotations_and_json_evaluation(self):
        self.reference_snapshot(variants=("saved", "draft"))
        self.reference_snapshot("second-reference", tile_id="ref-2", variants=("saved", "draft"))
        self.reference_snapshot("nobile-reference", map_id="N1014", site_id="nobile1",
                                tile_id="ref-3", variants=("draft",))
        self.bundled("bundled-1")
        self.bundled("bundled-2", left=60000)
        self.bundled("bundled-3", "nobile1(reference)", left=80000)
        source_before = {str(p): p.read_bytes() for p in paths.DATA_DIR.rglob("*") if p.is_file()}
        run_id = "20260928T120000.000000Z-01234567"
        report_path = paths.OUTPUT_DIR / "evaluations" / f"evaluation-{run_id}.json"
        report = {
            "run_id": run_id, "created_at": "2026-09-28T12:00:00Z",
            "provenance": {"maps": {}},
            "selected_maps": ["bundled-1", "bundled-2", "bundled-3"],
            "validation": {"aggregate": {"n": 9, "accuracy": 0.75, "macro_f1": 0.7}},
        }
        self.write_json(report_path, report)
        with redirect_stdout(io.StringIO()):
            summary = smoke.main()
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["sections"], {"mons-mouton(reference)": 2, "nobile1(reference)": 1})
        self.assertEqual(summary["annotations"], {"bundled": 3, "verified": 3,
                                                "browsed": ["bundled-1", "bundled-2", "bundled-3"]})
        self.assertEqual(summary["gallery"]["verification_controls"], 3)
        self.assertTrue(summary["evaluation"]["report_available"])
        self.assertTrue(summary["evaluation"]["json_download"])
        self.assertEqual(summary["persistence"]["changed"], [])
        self.assertFalse(paths.MODEL_DIR.exists())
        self.assertEqual(source_before, {str(p): p.read_bytes()
                                         for p in paths.DATA_DIR.rglob("*") if p.is_file()})

    def test_erase_on_blank_canvas_is_not_an_autosave(self):
        self.public()
        with smoke.read_only():
            app = self.app()
            app.radio[0].set_value("erase").run()
            with patch.object(canvas, "map_canvas", return_value={
                    "id": "blank-erase", "cells": [0], "click": True}):
                app.run()
            self.assert_ok(app)
            self.assertFalse(app.session_state.cells.any())
            self.assertEqual(app.session_state.hist, [])
        self.assert_no_artifacts()

    def test_explicit_readonly_metadata_rejects_canvas_events(self):
        _, cells = self.bundled(read_only=True)
        with smoke.read_only(), patch.object(canvas, "map_canvas", return_value={
                "id": "locked-stroke", "cells": [0], "click": True}):
            app = self.app()
            self.assertTrue(app.session_state.locked)
            self.assertTrue(app.radio[0].disabled)
            np.testing.assert_array_equal(app.session_state.cells, cells)
            for label in ("Clear", "Undo", "Approve prediction", "Save labels + map"):
                self.assertTrue(self.button(app, label).disabled, label)
        self.assert_no_artifacts()

    def test_forced_disabled_actions_cannot_write_on_a_clean_start(self):
        self.public()
        original = st.button
        forced = {"Save labels + map", "Update model", "Approve prediction",
                  "Evaluate", "Use these test maps"}

        def force_button(label, *args, **kwargs):
            value = original(label, *args, **kwargs)
            return True if label in forced or kwargs.get("key") == "update_model" else value

        with smoke.read_only(), patch.object(st, "button", side_effect=force_button):
            app = self.app()
            app.switch_page("app/app_pages/progress.py").run()
            self.assert_ok(app)
        self.assert_no_artifacts()

    def test_progress_has_only_evaluation_even_with_legacy_independent_history(self):
        self.public()
        self.write_json(paths.TEST_MAPS_FILE, ["public-a"])
        score = dict(accuracy=0.7, baseline=0.5, iou={u: 0.5 for u in core.CODES.values()},
                     small_craters=None)
        entry = dict(at="2026-01-01T12:00:00", backend="rf", trained_on=["old-map"],
                     maps={"public-a": 0.7}, **score)

        self.write_json(paths.OUTPUT_DIR / "test_scores.json", [entry])
        with smoke.read_only():
            app = self.app()
            app.switch_page("app/app_pages/progress.py").run()
            self.assert_ok(app)
            self.assertTrue(self.button(app, "Evaluate").disabled)
            self.assertEqual(len(app.metric), 0)
            self.assertNotIn("Original-vector checks", [h.value for h in app.subheader])
            self.assertNotIn("Every score", [e.label for e in app.expander])
            self.assertNotIn("Independent checks", [h.value for h in app.subheader])
            self.assertFalse(any(s.label == "Evaluation map" for s in app.selectbox))
        self.assertFalse(paths.MODEL_DIR.exists())

    def test_progress_cannot_configure_persistent_test_maps(self):
        self.public()
        with smoke.read_only():
            app = self.app()
            app.switch_page("app/app_pages/progress.py").run()
            self.assert_ok(app)
            self.assertEqual([w.key for w in app.multiselect], ["evaluation_maps"])
            self.assertEqual(app.multiselect(key="evaluation_maps").options, [])
            self.assertNotIn("Use these test maps", [b.label for b in app.button])
        self.assertFalse(paths.TEST_MAPS_FILE.exists())
        self.assertFalse(paths.MODEL_DIR.exists())

    def test_explicit_paint_save_train_and_gallery_verification_still_work(self):
        self.public()
        app = self.app()
        for unit, cells, event_id in (
            ("smooth highlands", range(core.GRID ** 2 // 2), "first-unit"),
            ("rough highlands", range(core.GRID ** 2 // 2, core.GRID ** 2), "second-unit"),
        ):
            app.radio[0].set_value(unit).run()
            with patch.object(canvas, "map_canvas", return_value={
                    "id": event_id, "cells": list(cells), "click": False}):
                app.run()
            self.assert_ok(app)
        self.assertTrue(Path(core.painting_files("public-a")[0]).is_file())
        self.assertFalse(paths.OUT_DIR.exists())
        self.assertFalse(paths.LEGACY_DRAFT_DIR.exists())
        self.assertFalse(paths.MODEL_DIR.exists())
        self.button(app, "Save labels + map").click().run()
        self.assert_ok(app)
        self.assertTrue(Path(core.output_file("public-a", "labels.tif")).is_file())
        self.assertFalse(Path(core.output_file("public-a", "map.tif")).exists())
        self.assertFalse(core.meta_get("public-a")["prediction_saved"])
        self.assertIsNone(assistant.load()["model"])
        app.checkbox(key="verified_public-a").check().run()
        self.button(app, "Update model").click().run()
        self.assert_ok(app)
        self.assertIsNotNone(assistant.load()["model"])
        self.assertIsNotNone(app.session_state.pred)
        self.assertFalse(Path(core.output_file("public-a", "map.tif")).exists())
        self.button(app, "Save labels + map").click().run()
        self.assert_ok(app)
        self.assertTrue(Path(core.output_file("public-a", "map.tif")).is_file())
        self.assertTrue(core.meta_get("public-a")["prediction_saved"])
        app.switch_page("app/app_pages/gallery.py").run()
        self.assert_ok(app)
        app.checkbox(key="v_public-a").uncheck().run()
        self.assert_ok(app)
        self.assertFalse(core.meta_get("public-a")["verified"])
        self.assertTrue(core.meta_get("public-a", final=True)["verified"])
        self.assertTrue(Path(core.output_file("public-a", "meta.json", final=False)).is_file())
        app.switch_page("app/app_pages/progress.py").run()
        self.assert_ok(app)
        # There are no verified training samples after revoking the only training map.
        self.assertTrue(self.button(app, "Evaluate").disabled)
        app.multiselect(key="evaluation_maps").set_value(["public-a"]).run()
        self.assert_ok(app)
        self.assertTrue(self.button(app, "Evaluate").disabled)
        self.assertTrue(any("No evaluation is ready" in m.value for m in app.info))
        with redirect_stdout(io.StringIO()):
            summary = smoke.main()
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["model"]["trained"])
        self.assertEqual(summary["persistence"]["changed"], [])

    def test_bundled_verification_is_editable_and_gallery_overrides_leave_sources_unchanged(self):
        _, cells = self.bundled()
        source_before = {str(p): p.read_bytes() for p in paths.DEM_DIR.rglob("*") if p.is_file()}
        with smoke.read_only():
            app = self.app()
            np.testing.assert_array_equal(app.session_state.cells, cells)
            self.assertFalse(app.session_state.locked)
            self.assertFalse(app.radio[0].disabled)
            self.assertEqual(app.selectbox(key="selected_map").options,
                             ["bundled-map — painted, verified"])
            app.switch_page("app/app_pages/gallery.py").run()
            self.assert_ok(app)
            self.assertTrue(app.checkbox(key="v_bundled-map").value)
            self.assertFalse(app.checkbox(key="v_bundled-map").disabled)
            self.assertIn("**bundled-map**", [m.value for m in app.markdown])
        self.assert_no_artifacts()
        app.checkbox(key="v_bundled-map").uncheck().run()
        self.assert_ok(app)
        self.assertFalse(core.meta_get("bundled-map")["verified"])
        self.assertTrue(Path(core.output_file("bundled-map", "meta.json", final=False)).is_file())
        self.assertTrue(core.meta_get("bundled-map", final=True)["verified"])
        self.assertFalse(paths.OUT_DIR.exists())
        self.assertFalse(paths.MODEL_DIR.exists())
        self.assertEqual(source_before, {str(p): p.read_bytes()
                                         for p in paths.DEM_DIR.rglob("*") if p.is_file()})

    def test_gallery_includes_standard_bundled_labels_without_a_painting_array(self):
        dem, _ = self.bundled()
        manifest = Path(dem).parent / "working_dems.json"
        records = json.loads(manifest.read_text())
        del records[0]["annotations"]["painting"]
        self.write_json(manifest, records)
        self.assertEqual(core.painted_maps(), (set(), set()))
        with smoke.read_only():
            app = self.app()
            app.switch_page("app/app_pages/gallery.py").run()
            self.assert_ok(app)
            self.assertTrue(app.checkbox(key="v_bundled-map").value)
            self.assertTrue(any("Saved labels" in c.value for c in app.caption))
            self.assertFalse(any("Bundled annotations" in c.value for c in app.caption))
            self.assertFalse(any("No saved maps yet" in m.value for m in app.info))
        self.assert_no_artifacts()

    def test_poster_snapshot_and_guard_follow_a_relocated_root(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as outside, \
                patch.object(paths, "POSTER_DIR", Path(outside) / "poster"):
            paths.POSTER_DIR.mkdir()
            (paths.POSTER_DIR / "empty").mkdir()
            figure = paths.POSTER_DIR / "figure.png"
            Image.new("RGB", (12, 12), "red").save(figure)
            before = smoke.snapshot()
            self.assertEqual(before[str(paths.POSTER_DIR)], "directory")
            self.assertEqual(before[str(paths.POSTER_DIR / "empty")], "directory")
            self.assertIn(str(figure), before)
            for action in (lambda: figure.write_bytes(b"changed"), figure.unlink,
                           lambda: (paths.POSTER_DIR / "new").mkdir()):
                with self.subTest(action=action), self.assertRaisesRegex(AssertionError, "Read-only"):
                    with smoke.read_only():
                        action()
            self.assertEqual(smoke.snapshot(), before)
            Image.new("RGB", (12, 12), "blue").save(figure)
            self.assertNotEqual(smoke.snapshot()[str(figure)], before[str(figure)])
        self.assert_no_artifacts()

    def test_readonly_guard_blocks_direct_writes_and_deletion(self):
        self.public()
        existing = paths.DEM_DIR / "keep.txt"
        existing.write_text("keep", encoding="utf-8")
        for action in (lambda: paths.TEST_MAPS_FILE.parent.mkdir(parents=True, exist_ok=True),
                       lambda: existing.write_text("changed", encoding="utf-8"),
                       existing.unlink,
                       lambda: core.save_painting("public-a", np.zeros((core.GRID, core.GRID))),
                       lambda: assistant.fit({}),
                       lambda: evaluation.run(assistant.load(), ["public-a"]),
                       lambda: paths.POSTER_DIR.mkdir()):
            with self.subTest(action=action), self.assertRaisesRegex(AssertionError, "Read-only"):
                with smoke.read_only():
                    action()
        self.assertEqual(existing.read_text(), "keep")
        self.assert_no_artifacts()


if __name__ == "__main__":
    unittest.main()
