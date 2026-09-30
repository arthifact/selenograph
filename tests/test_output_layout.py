"""Separate saved/draft mirrors and read-only legacy compatibility, using disposable data.

Run: .venv/bin/python -B -m unittest tests.test_output_layout -v
"""
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from app import assistant_model as assistant
from app import canvas, core, evaluation, paths, pictures
from tests.test_dataset_pages import PageFixture


def snapshot(root):
    """Include empty directories, file bytes and mtimes, without recording read atimes."""
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            if p.is_file() else None for p in root.rglob("*")}


class OutputLayoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processed = self.root / "data/processed_data"
        self.processed.mkdir(parents=True)
        self.output = self.root / "output"
        self.saved = self.output / "paintings"
        self.maps = self.output / "maps"
        self.drafts = self.output / "painting_drafts"
        self.legacy = self.output / "maps"
        self.legacy_drafts = self.output / "drafts"
        stack = ExitStack()
        self.addCleanup(stack.close)
        for module in (core, paths):
            for attr, value in (("DEM_DIR", self.processed), ("OUT_DIR", self.saved),
                                ("MAP_DIR", self.maps),
                                ("LEGACY_OUT_DIR", self.legacy), ("DRAFT_DIR", self.drafts),
                                ("LEGACY_DRAFT_DIR", self.legacy_drafts),
                                ("LEGACY_MIRRORED_DIR", self.output)):
                stack.enter_context(patch.object(module, attr, value))
        stack.enter_context(patch.object(paths, "OUTPUT_DIR", self.output))
        stack.enter_context(patch.object(paths, "MODEL_DIR", self.root / "models"))
        stack.enter_context(patch.object(assistant, "MODEL_DIR", self.root / "models"))
        self.addCleanup(core._load_catalog.cache_clear)
        self.records = []
        self.name = "tile-one"
        self.parent = Path("survey/region/tiles/zone-a")
        self.cells = np.zeros((core.GRID, core.GRID), np.uint8)
        self.cells[10:40, 10:40] = 1
        self.cells[60:90, 60:90] = 2

    def array(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)
        return path

    def json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def raster(self, path, values=None, *, nodata=255):
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.ones(self.cells.shape, np.uint8) if values is None else values
        with rasterio.open(path, "w", driver="GTiff", count=1, dtype=values.dtype,
                           width=values.shape[1], height=values.shape[0], nodata=nodata,
                           crs="ESRI:103878", transform=from_origin(1000, 2000, 5, 5)) as dst:
            dst.write(values, 1)
        return path

    def install(self, name=None, parent=None, *, bundled=False):
        name = self.name if name is None else name
        parent = self.parent if parent is None else Path(parent)
        dem = self.raster(self.processed / parent / f"{name}.tif")
        record: dict[str, Any] = {
            "working_dem": str(dem.relative_to(self.processed)),
            "section": "not-the-output-folder", "site": "not-the-site-folder",
            "group": "not-the-group-folder", "collection": "not-the-collection-folder"}
        if bundled:
            folder = dem.parent / "annotations" / name
            annotations: dict[str, Any] = {
                "verified": True, "restricted": True, "label_status": "bundled",
                "provenance": {"origin": ["fixture"]}}
            for kind, values in (("painting", self.cells), ("accepted", self.cells == 2),
                                 ("confidence", np.where(self.cells > 0, core.MOSTLY, 0))):
                path = self.array(folder / f"{kind}.npy", values)
                annotations[kind] = str(path.relative_to(self.processed))
            labels = self.raster(folder / "labels.tif", core.cells_to_labels(self.cells, self.cells.shape))
            annotations["labels"] = str(labels.relative_to(self.processed))
            record["annotations"] = annotations
        self.records.append(record)
        self.json(self.processed / "working_dems.json", self.records)
        return dem

    def canonical(self, filename, name=None, parent=None, *, final=True):
        name = self.name if name is None else name
        parent = self.parent if parent is None else Path(parent)
        root = self.maps if final and filename in ("labels.tif", "map.tif") else self.saved if final else self.drafts
        return root / parent / f"{name}_{filename}"

    def old_mirror(self, filename, name=None):
        return self.output / self.parent / f"{name or self.name}_{filename}"

    def legacy_painting(self, *, final=False, cells=None, flags=True):
        cells = self.cells if cells is None else cells
        files = ((self.legacy / self.name / "painting.npy",
                  self.legacy / self.name / "accepted.npy",
                  self.legacy / self.name / "confidence.npy") if final else
                 (self.legacy_drafts / f"{self.name}.npy", self.legacy_drafts / f"{self.name}.accepted.npy",
                  self.legacy_drafts / f"{self.name}.confidence.npy"))
        self.array(files[0], cells)
        if flags:
            self.array(files[1], cells == 2)
            self.array(files[2], np.where(cells > 0, core.UNSURE, 0))
        return files

    def assert_source(self, expected, *, final, bundled):
        source = core._painting_source(self.name)
        self.assertEqual(len(source), 3)
        cells, saved, from_bundle = source
        np.testing.assert_array_equal(cells, expected)
        self.assertIs(saved, final)
        self.assertIs(from_bundle, bundled)

    def test_exact_dem_parent_hierarchy_not_section_site_or_collection(self):
        dem = self.install()
        before = snapshot(self.root)
        self.assertEqual(Path(core.out_dir(self.name)), self.saved / self.parent)
        for final, root in ((True, self.saved), (False, self.drafts)):
            self.assertEqual(Path(core.out_dir(self.name, final=final)), root / self.parent)
            for filename in ("painting.npy", "accepted.npy", "confidence.npy", "labels.tif",
                             "map.tif", "thumb.png", "meta.json"):
                with self.subTest(final=final, filename=filename):
                    self.assertEqual(Path(core.output_file(self.name, filename, final=final)),
                                     self.canonical(filename, final=final))
                    self.assertEqual(Path(core.output_file(self.name, filename, existing=True, final=final)),
                                     self.canonical(filename, final=final))
        self.assertEqual(core.dem_path(self.name), str(dem))
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.output.exists())

    def test_manifest_location_does_not_replace_dem_parent(self):
        dem = self.raster(self.processed / "actual/deep/tiles/cross-folder.TIFF")
        self.json(self.processed / "catalog/working_dems.json",
                  [{"working_dem": "../actual/deep/tiles/cross-folder.TIFF", "section": "elsewhere"}])
        self.assertEqual(Path(core.out_dir("cross-folder")), self.saved / "actual/deep/tiles")
        self.assertEqual(Path(core.output_file("cross-folder", "labels.tif")),
                         self.maps / "actual/deep/tiles/cross-folder_labels.tif")
        self.assertEqual(core.dem_path("cross-folder"), str(dem))
        self.assertFalse(self.output.exists())

    def test_reference_like_names_follow_the_same_rule_without_special_folders(self):
        name = "MM026_dem_5m"
        parent = Path("mons-mouton(reference)/sites/MM026/tiles")
        self.install(name, parent)
        core.save_painting(name, self.cells)
        self.assertEqual(Path(core.out_dir(name, final=False)), self.drafts / parent)
        self.assertTrue((self.drafts / parent / f"{name}_painting.npy").is_file())
        self.assertEqual({p.relative_to(self.output) for p in self.output.rglob("*") if p.is_file()},
                         {Path("painting_drafts") / parent / f"{name}_painting.npy"})
        self.assertFalse(self.saved.exists())

    def test_flat_unregistered_dem_and_missing_id_keep_flat_canonical_support(self):
        self.raster(self.processed / "flat.TIFF")
        self.assertIsNone(core.map_record("flat"))
        for name in ("flat", "not-installed"):
            with self.subTest(name=name):
                self.assertEqual(Path(core.out_dir(name)), self.saved)
                self.assertEqual(Path(core.output_file(name, "labels.tif")),
                                 self.maps / f"{name}_labels.tif")
                core.save_painting(name, self.cells, final=True)
                np.testing.assert_array_equal(core.load_painting(name), self.cells)
        self.assertEqual(core.painted_maps(), ({"flat", "not-installed"}, set()))
        self.assertFalse(self.legacy.exists())
        self.assertFalse(self.drafts.exists())

    def test_dotted_catalog_id_is_exact_even_when_its_stem_is_another_map(self):
        versioned = self.install("alpha.v1", "site")
        plain = self.install("alpha", "other-site")
        self.records[0]["site"] = "versioned-site"
        self.records[1]["site"] = "plain-site"
        self.json(self.processed / "working_dems.json", self.records)
        before = snapshot(self.processed)
        for name, dem, site, code in (("alpha.v1", versioned, "versioned-site", 3),
                                      ("alpha", plain, "plain-site", 1)):
            with self.subTest(name=name):
                record = core.map_record(name)
                assert record is not None
                self.assertEqual(record["working_dem"], str(dem.relative_to(self.processed)))
                self.assertEqual(record["site"], site)
                self.assertEqual(core.dem_path(name), str(dem))
                expected_dir = self.saved / dem.parent.relative_to(self.processed)
                expected_labels = self.maps / dem.parent.relative_to(self.processed) / f"{name}_labels.tif"
                self.assertEqual(Path(core.out_dir(name)), expected_dir)
                self.assertEqual(Path(core.output_file(name, "labels.tif")), expected_labels)
                labels = np.full_like(self.cells, code)
                with rasterio.open(dem) as source:
                    profile = source.profile
                core.write_map(core.output_file(name, "labels.tif"), labels, profile)
                core.meta_set(name, verified=True)
                self.assertEqual(Path(core.output_file(name, "labels.tif", existing=True)), expected_labels)
                with rasterio.open(expected_labels) as saved_labels:
                    np.testing.assert_array_equal(saved_labels.read(1), labels)
        self.assertEqual(snapshot(self.processed), before)
        self.assertEqual({p.relative_to(self.saved) for p in self.saved.rglob("*") if p.is_file()},
                         {Path("site/alpha.v1_meta.json"), Path("other-site/alpha_meta.json")})
        self.assertEqual({p.relative_to(self.maps) for p in self.maps.rglob("*") if p.is_file()},
                         {Path("site/alpha.v1_labels.tif"), Path("other-site/alpha_labels.tif")})
        self.assertFalse(self.drafts.exists())

    def test_two_dems_share_parent_without_colliding_and_only_derived_files_are_written(self):
        names = (self.name, "tile-two")
        for name in names:
            self.install(name, bundled=True)
        before = snapshot(self.processed)
        expected = set()
        for index, name in enumerate(names, start=1):
            cells = np.full_like(self.cells, index)
            accepted = np.zeros(cells.shape, bool)
            accepted[0, 0] = True
            confidence = np.full_like(cells, core.MOSTLY)
            filenames = ("painting.npy", "accepted.npy", "confidence.npy")
            for final, bucket in ((False, "painting_drafts"), (True, "paintings")):
                self.assertEqual(tuple(map(Path, core.painting_files(name, final))),
                                 tuple(self.canonical(f, name, final=final) for f in filenames))
                core.save_painting(name, cells, final=final, accepted=accepted, confidence=confidence)
                for filename in filenames:
                    expected.add(Path(bucket) / self.parent / f"{name}_{filename}")
            with rasterio.open(core.dem_path(name)) as source:
                profile = source.profile
            for filename in ("labels.tif", "map.tif"):
                core.write_map(core.output_file(name, filename), cells, profile)
                expected.add(Path("maps") / self.parent / f"{name}_{filename}")
            Image.new("RGB", (8, 8), "blue").save(core.output_file(name, "thumb.png"))
            core.meta_set(name, verified=False)
            expected.update(Path("paintings") / self.parent / f"{name}_{f}" for f in ("thumb.png", "meta.json"))
        actual = {p.relative_to(self.output) for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(actual, expected)
        for index, name in enumerate(names, start=1):
            np.testing.assert_array_equal(core.load_painting(name), np.full_like(self.cells, index))
        self.assertEqual(snapshot(self.processed), before)
        self.assertFalse((self.legacy / self.name).exists())
        self.assertFalse(self.legacy_drafts.exists())

    def test_existing_saved_cohort_uses_markers_not_metadata_or_individual_files(self):
        for canonical_marker in (None, "painting.npy", "labels.tif"):
            for legacy_marker in (None, "painting.npy", "labels.tif"):
                with self.subTest(canonical=canonical_marker, legacy=legacy_marker):
                    name = f"tile-{str(canonical_marker).split('.')[0]}-{str(legacy_marker).split('.')[0]}"
                    self.install(name)
                    canonical = self.canonical("meta.json", name)
                    self.json(canonical, {"verified": False})
                    legacy = self.legacy / name
                    self.json(legacy / "meta.json", {"verified": True})
                    (legacy / "map.tif").write_bytes(b"stale prediction")
                    (legacy / "thumb.png").write_bytes(b"stale thumbnail")
                    if canonical_marker:
                        self.canonical(canonical_marker, name).parent.mkdir(parents=True, exist_ok=True)
                        self.canonical(canonical_marker, name).touch()
                    if legacy_marker:
                        (legacy / legacy_marker).touch()
                    before = snapshot(self.root)
                    for filename in ("painting.npy", "accepted.npy", "confidence.npy",
                                     "labels.tif", "map.tif", "thumb.png"):
                        expected = (legacy / filename if legacy_marker and not canonical_marker
                                    else self.canonical(filename, name))
                        self.assertEqual(Path(core.output_file(name, filename, existing=True)), expected)
                        self.assertEqual(Path(core.output_file(name, filename)), self.canonical(filename, name))
                    self.assertEqual(tuple(map(Path, core.painting_files(name, True, existing=True))),
                                     tuple(Path(core.output_file(name, f, existing=True))
                                           for f in ("painting.npy", "accepted.npy", "confidence.npy")))
                    self.assertEqual(snapshot(self.root), before)

    def test_old_mirrored_saved_cohort_precedes_flat_legacy_without_migrating(self):
        self.install(bundled=True)
        self.legacy_painting(final=True)
        saved = np.where(self.cells > 0, 3, 0).astype(np.uint8)
        files = []
        for kind, values in (("painting", saved), ("accepted", saved > 0),
                             ("confidence", np.where(saved > 0, core.MOSTLY, 0))):
            files.append(self.array(self.old_mirror(f"{kind}.npy"), values))
        self.json(self.old_mirror("meta.json"), {"verified": False})
        before = snapshot(self.root)
        self.assertEqual(tuple(map(Path, core.painting_files(self.name, True, existing=True))), tuple(files))
        self.assert_source(saved, final=True, bundled=False)
        np.testing.assert_array_equal(core.load_accepted(self.name), saved > 0)
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(saved > 0, core.MOSTLY, 0))
        self.assertFalse(core.meta_get(self.name, final=True)["verified"])
        self.assertEqual(snapshot(self.root), before)
        core.save_painting(self.name, self.cells, final=True)
        self.assert_source(self.cells, final=True, bundled=False)
        self.assertFalse(core.load_accepted(self.name).any())
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(self.cells > 0, core.SURE, 0))
        for path in files:
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before[str(path.relative_to(self.root))])

    def test_old_painting_tiffs_remain_readable_until_new_map_export_takes_over(self):
        self.install(bundled=True)
        old_folder = self.saved / self.parent
        old_labels = self.raster(old_folder / f"{self.name}_labels.tif", np.full_like(self.cells, 3))
        old_prediction = self.raster(old_folder / f"{self.name}_map.tif")
        saved_before = snapshot(self.saved)
        before = snapshot(self.root)
        self.assertEqual(Path(core.output_file(self.name, "labels.tif", existing=True)), old_labels)
        self.assertEqual(Path(core.output_file(self.name, "map.tif", existing=True)), old_prediction)
        self.assertIsNone(core.painting_source(self.name, final=True)[1])
        self.assertTrue(core.has_saved_labels(self.name))
        self.assertEqual(core.seed_paintings(), [])
        self.assertEqual(snapshot(self.root), before)
        self.raster(self.canonical("labels.tif"), np.full_like(self.cells, 2))
        self.assertEqual(Path(core.output_file(self.name, "labels.tif", existing=True)), self.canonical("labels.tif"))
        self.assertTrue(core.has_saved_labels(self.name))
        prediction = Path(core.output_file(self.name, "map.tif", existing=True))
        self.assertEqual(prediction, self.canonical("map.tif"))
        self.assertFalse(prediction.exists())  # No stale prediction from the former folder.
        self.assertEqual(snapshot(self.saved), saved_before)

    def test_map_export_symlink_cannot_redirect_a_save(self):
        self.install()
        outside = self.root / "outside"
        outside.mkdir()
        self.maps.mkdir(parents=True)
        (self.maps / self.parent.parts[0]).symlink_to(outside, target_is_directory=True)
        before = snapshot(self.root)
        for filename in ("labels.tif", "map.tif"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                core.output_file(self.name, filename)
        self.assertEqual(snapshot(self.root), before)

    def test_old_mirrored_draft_flags_stay_with_source_until_canonical_blank_override(self):
        self.install(bundled=True)
        self.legacy_painting()
        draft = np.where(self.cells > 0, 3, 0).astype(np.uint8)
        files = [self.array(self.old_mirror(filename), values) for filename, values in
                 (("draft.npy", draft), ("draft.accepted.npy", draft > 0),
                  ("draft.confidence.npy", np.where(draft > 0, core.MOSTLY, 0)))]
        before = snapshot(self.root)
        self.assertEqual(tuple(map(Path, core.painting_files(self.name, existing=True))), tuple(files))
        self.assert_source(draft, final=False, bundled=False)
        np.testing.assert_array_equal(core.load_accepted(self.name), draft > 0)
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(draft > 0, core.MOSTLY, 0))
        self.assertEqual(snapshot(self.root), before)
        core.save_painting(self.name, np.zeros_like(self.cells))
        self.assert_source(np.zeros_like(self.cells), final=False, bundled=False)
        self.assertFalse(core.load_accepted(self.name).any())
        self.assertFalse(core.load_confidence(self.name).any())
        self.assertFalse(self.saved.exists())
        for path in files:
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before[str(path.relative_to(self.root))])

    def test_draft_metadata_is_effective_but_only_explicit_commit_changes_saved_metadata(self):
        self.install(bundled=True)
        core.save_painting(self.name, self.cells, final=True)
        core.meta_set(self.name, verified=True, label_status="saved")
        saved_before = snapshot(self.saved)
        core.meta_set(self.name, final=False, verified=False, label_status="draft")
        self.assertEqual(snapshot(self.saved), saved_before)
        self.assertTrue(core.meta_get(self.name, final=True)["verified"])
        self.assertEqual(core.meta_get(self.name, final=True)["label_status"], "saved")
        self.assertFalse(core.meta_get(self.name)["verified"])
        self.assertEqual(core.meta_get(self.name)["label_status"], "draft")
        self.assertTrue(self.canonical("meta.json", final=False).is_file())
        self.assertFalse(self.canonical("painting.npy", final=False).exists())
        core.meta_set(self.name, final=True, verified=False, label_status="reviewed")
        self.assertFalse(self.canonical("meta.json", final=False).exists())
        self.assertFalse(core.meta_get(self.name, final=True)["verified"])
        self.assertEqual(core.meta_get(self.name)["label_status"], "reviewed")
        self.assertEqual(core.meta_get(self.name), core.meta_get(self.name, final=True))

    def test_partial_metadata_overlays_legacy_then_canonical_over_bundled_defaults(self):
        self.install(bundled=True)
        self.json(self.legacy / self.name / "meta.json",
                  {"verified": True, "label_status": "legacy", "legacy_only": "retained"})
        self.json(self.canonical("meta.json"), {"verified": False, "restricted": False})
        before = snapshot(self.root)
        metadata = core.meta_get(self.name)
        self.assertIs(metadata["verified"], False)
        self.assertIs(metadata["restricted"], False)
        self.assertEqual(metadata["label_status"], "legacy")
        self.assertEqual(metadata["legacy_only"], "retained")
        self.assertEqual(metadata["provenance"], {"origin": ["fixture"]})
        self.assertEqual(snapshot(self.root), before)

    def test_metadata_layers_preserve_false_and_do_not_hide_legacy_painting_or_labels(self):
        self.install(bundled=True)
        legacy_files = self.legacy_painting(final=True)
        labels = self.raster(self.legacy / self.name / "labels.tif", self.cells)
        self.json(self.legacy / self.name / "meta.json",
                  {"verified": True, "restricted": True, "label_status": "legacy", "legacy_only": 7})
        legacy_before = snapshot(self.legacy)
        source_before = snapshot(self.processed)
        core.meta_set(self.name, verified=False, restricted=False, label_status="edited")
        self.assertTrue(self.canonical("meta.json").is_file())
        self.assertEqual(Path(core.output_file(self.name, "labels.tif", existing=True)), labels)
        self.assertEqual(Path(core.painting_files(self.name, True, existing=True)[0]), legacy_files[0])
        self.assert_source(self.cells, final=True, bundled=False)
        metadata = core.meta_get(self.name)
        self.assertIs(metadata["verified"], False)
        self.assertIs(metadata["restricted"], False)
        self.assertEqual(metadata["label_status"], "edited")
        self.assertEqual(metadata["legacy_only"], 7)
        self.assertEqual(metadata["provenance"], {"origin": ["fixture"]})
        metadata["provenance"]["origin"].clear()
        self.assertEqual(core.meta_get(self.name)["provenance"], {"origin": ["fixture"]})
        self.assertEqual(snapshot(self.legacy), legacy_before)
        self.assertEqual(snapshot(self.processed), source_before)

    def test_painting_precedence_retains_three_value_contract_and_same_source_flags(self):
        self.install(bundled=True)
        self.assert_source(self.cells, final=True, bundled=True)
        saved = np.where(self.cells > 0, 3, 0).astype(np.uint8)
        self.legacy_painting(final=True, cells=saved, flags=False)
        self.assert_source(saved, final=True, bundled=False)
        self.assertFalse(core.load_accepted(self.name).any())
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(saved > 0, core.SURE, 0))
        legacy_draft = self.legacy_painting()
        self.assert_source(self.cells, final=False, bundled=False)
        self.assertEqual(tuple(map(Path, core.painting_files(self.name, existing=True))), legacy_draft)
        np.testing.assert_array_equal(core.load_accepted(self.name), self.cells == 2)
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(self.cells > 0, core.UNSURE, 0))
        core.save_painting(self.name, saved)
        self.assert_source(saved, final=False, bundled=False)
        self.assertEqual(tuple(map(Path, core.painting_files(self.name, existing=True))),
                         tuple(self.canonical(f, final=False) for f in ("painting.npy", "accepted.npy", "confidence.npy")))
        self.assertFalse(core.load_accepted(self.name).any())
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(saved > 0, core.SURE, 0))
        np.testing.assert_array_equal(core.load_confidence(self.name, final=True), np.where(saved > 0, core.SURE, 0))
        self.assertEqual(core.painted_maps(), ({self.name}, set()))

    def test_blank_canonical_draft_overrides_legacy_and_bundle_without_flag_fallthrough(self):
        self.install(bundled=True)
        self.legacy_painting(final=True)
        self.legacy_painting()
        before = snapshot(self.legacy), snapshot(self.legacy_drafts), snapshot(self.processed)
        core.save_painting(self.name, np.zeros_like(self.cells))
        self.assert_source(np.zeros_like(self.cells), final=False, bundled=False)
        self.assertFalse(core.load_accepted(self.name).any())
        self.assertFalse(core.load_confidence(self.name).any())
        self.assertEqual(core.painted_maps(), (set(), set()))
        np.testing.assert_array_equal(core.load_confidence(self.name, final=True),
                                      np.where(self.cells > 0, core.UNSURE, 0))
        self.assertEqual((snapshot(self.legacy), snapshot(self.legacy_drafts), snapshot(self.processed)), before)

    def test_blank_canonical_final_overrides_legacy_and_bundle_answer_keys(self):
        self.install(bundled=True)
        self.legacy_painting(final=True)
        self.raster(self.legacy / self.name / "labels.tif", self.cells)
        core.save_painting(self.name, np.zeros_like(self.cells), final=True)
        self.assert_source(np.zeros_like(self.cells), final=True, bundled=False)
        self.assertFalse(core.load_accepted(self.name).any())
        self.assertFalse(core.load_confidence(self.name, final=True).any())
        self.assertEqual(core.painted_maps(), (set(), set()))

    def test_canonical_final_does_not_inherit_legacy_or_bundled_flags(self):
        self.install(bundled=True)
        self.legacy_painting(final=True)
        core.save_painting(self.name, self.cells, final=True)
        self.assert_source(self.cells, final=True, bundled=False)
        self.assertFalse(core.load_accepted(self.name).any())
        np.testing.assert_array_equal(core.load_confidence(self.name, final=True),
                                      np.where(self.cells > 0, core.SURE, 0))
        cells, eligible, _ = evaluation._painting(self.name)
        np.testing.assert_array_equal(cells, self.cells)
        np.testing.assert_array_equal(eligible, self.cells > 0)

    def test_orphan_canonical_draft_flags_do_not_attach_to_legacy_draft(self):
        self.install(bundled=True)
        self.legacy_painting(flags=False)
        self.array(self.canonical("accepted.npy", final=False), np.ones(self.cells.shape, bool))
        self.array(self.canonical("confidence.npy", final=False), np.full_like(self.cells, core.UNSURE))
        self.assert_source(self.cells, final=False, bundled=False)
        self.assertFalse(core.load_accepted(self.name).any())
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(self.cells > 0, core.SURE, 0))

    def test_saving_without_flags_removes_only_canonical_flags(self):
        self.install()
        for final in (False, True):
            self.legacy_painting(final=final)
            before = snapshot(self.legacy), snapshot(self.legacy_drafts)
            core.save_painting(self.name, self.cells, final=final, accepted=self.cells > 0,
                               confidence=np.where(self.cells > 0, core.MOSTLY, 0))
            files = tuple(map(Path, core.painting_files(self.name, final)))
            self.assertTrue(all(p.is_file() for p in files))
            core.save_painting(self.name, self.cells, final=final)
            self.assertTrue(files[0].is_file())
            self.assertFalse(files[1].exists())
            self.assertFalse(files[2].exists())
            self.assertEqual((snapshot(self.legacy), snapshot(self.legacy_drafts)), before)
        self.assertFalse(core.load_accepted(self.name).any())
        np.testing.assert_array_equal(core.load_confidence(self.name), np.where(self.cells > 0, core.SURE, 0))

    def check_tiff_only_saved_override(self, *, legacy):
        self.install(bundled=True)
        labels_path = (self.legacy / self.name / "labels.tif" if legacy
                       else self.canonical("labels.tif"))
        labels = np.full_like(self.cells, 3)
        self.raster(labels_path, labels)
        accepted = np.zeros(self.cells.shape, bool)
        accepted[10:20] = True
        confidence = np.full_like(self.cells, core.SURE)
        confidence[30:40] = core.MOSTLY
        for kind, flags in (("accepted", accepted), ("confidence", confidence)):
            path = labels_path.parent / f"{kind}.npy" if legacy else self.canonical(f"{kind}.npy")
            self.array(path, flags)
        before = snapshot(self.root)
        self.assert_source(np.zeros_like(self.cells), final=None, bundled=False)
        cells, final, bundled = core._painting_source(self.name, final=True)
        self.assertFalse(cells.any())
        self.assertIsNone(final)
        self.assertIs(bundled, False)
        self.assertFalse(core.load_painting(self.name).any())
        self.assertFalse(core.load_accepted(self.name).any())
        for final in (None, False, True):
            self.assertFalse(core.load_confidence(self.name, final=final).any())
        self.assertEqual(core.painted_maps(), (set(), set()))
        cells, eligible, source = evaluation._painting(self.name)
        self.assertFalse(cells.any())
        self.assertFalse(eligible.any())
        self.assertEqual(source, "none")
        self.assertEqual(snapshot(self.root), before)

    def test_canonical_tiff_only_override_suppresses_bundled_cell_painting(self):
        self.check_tiff_only_saved_override(legacy=False)

    def test_legacy_tiff_only_override_suppresses_bundled_cell_painting(self):
        self.check_tiff_only_saved_override(legacy=True)

    def test_read_only_resolution_never_migrates_legacy_or_copies_inputs(self):
        self.install(bundled=True)
        self.legacy_painting(final=True)
        self.legacy_painting()
        self.json(self.legacy / self.name / "meta.json", {"verified": True})
        before = snapshot(self.root)
        for _ in range(2):
            with core.catalog_snapshot():
                core.dem_files()
                core.out_dir(self.name)
                core.meta_get(self.name)
                core.painting_source(self.name)
                core.load_accepted(self.name)
                core.load_confidence(self.name)
                core.painted_maps()
                core.output_file(self.name, "thumb.png", existing=True)
                core.painting_files(self.name, existing=True)
                evaluation.catalog()
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.output / self.parent).exists())
        self.assertFalse(self.saved.exists())
        self.assertFalse(self.drafts.exists())

    def test_site_names_maps_and_drafts_are_safe_inside_both_buckets(self):
        cases = (("drafts", "a"), ("drafts/nested", "a-nested"),
                 ("maps", "b"), ("maps/nested", "b-nested"))
        for parent, name in cases:
            self.install(name, parent)
        self.array(self.legacy_drafts / "a_draft.npy", self.cells)
        self.array(self.legacy_drafts / "a_draft.accepted.npy", self.cells == 2)
        self.array(self.legacy / "b" / "painting.npy", self.cells)
        self.json(self.legacy / "b" / "meta.json", {"verified": True})
        before = snapshot(self.legacy), snapshot(self.legacy_drafts), snapshot(self.processed)
        for parent, name in cases:
            for final, root in ((False, self.drafts), (True, self.saved)):
                with self.subTest(parent=parent, final=final):
                    self.assertEqual(Path(core.out_dir(name, final=final)), root / parent)
                    core.save_painting(name, self.cells, final=final)
                    core.meta_set(name, final=final, verified=False)
                    self.assertTrue((root / parent / f"{name}_painting.npy").is_file())
                    self.assertTrue((root / parent / f"{name}_meta.json").is_file())
                    self.assertEqual(Path(core.output_file(name, "labels.tif", final=final)),
                                     (self.maps if final else root) / parent / f"{name}_labels.tif")
        self.assertEqual((snapshot(self.legacy), snapshot(self.legacy_drafts), snapshot(self.processed)), before)

    def test_output_filenames_cannot_escape_or_create_subfolders(self):
        self.install()
        before = snapshot(self.root)
        for final in (False, True):
            for filename in ("../escape.npy", "../../escape.npy", str(self.root / "escape.npy"),
                             "nested/labels.tif"):
                with self.subTest(final=final, filename=filename), self.assertRaises(ValueError):
                    core.output_file(self.name, filename, final=final)
        self.assertEqual(snapshot(self.root), before)

    def test_map_name_traversal_cannot_escape_output(self):
        before = snapshot(self.root)
        for name in ("../escape", "../../escape", str(self.root / "escape")):
            with self.subTest(name=name):
                try:
                    path = Path(core.output_file(name, "labels.tif"))
                except ValueError:
                    continue
                self.assertTrue(path.resolve().is_relative_to(self.output.resolve()))
        self.assertEqual(snapshot(self.root), before)

    def test_catalog_dem_traversal_and_symlink_escapes_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.processed / "escape").symlink_to(outside, target_is_directory=True)
        for relative in ("../outside/tile-one.tif", "escape/tile-one.tif"):
            self.json(self.processed / "working_dems.json", [{"working_dem": relative}])
            before = snapshot(self.root)
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                core.output_file(self.name, "labels.tif")
            self.assertEqual(snapshot(self.root), before)

    def test_mirrored_output_parent_symlink_cannot_redirect_writes_outside_output(self):
        self.install()
        outside = self.root / "outside"
        outside.mkdir()
        for root in (self.saved, self.drafts):
            root.mkdir(parents=True)
            (root / self.parent.parts[0]).symlink_to(outside, target_is_directory=True)
        before = snapshot(self.root)
        with self.assertRaises(ValueError):
            core.save_painting(self.name, self.cells)
        with self.assertRaises(ValueError):
            core.meta_set(self.name, verified=True)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(list(outside.iterdir()), [])

    def test_output_file_symlink_cannot_overwrite_external_metadata(self):
        self.install()
        outside = self.json(self.root / "external.json", {"do_not_change": True})
        for final in (False, True):
            target = self.canonical("meta.json", final=final)
            target.parent.mkdir(parents=True)
            target.symlink_to(outside)
            before = snapshot(self.root)
            with self.subTest(final=final), self.assertRaises(ValueError):
                core.meta_set(self.name, final=final, verified=True)
            self.assertEqual(snapshot(self.root), before)

    def test_explicit_seed_copies_six_bundles_to_both_buckets_independently_and_idempotently(self):
        names = [f"bundle-{index}" for index in range(6)]
        for name in names:
            self.install(name, bundled=True)
        self.install("unpainted")
        source_before = snapshot(self.processed)
        for name in names:
            core.load_painting(name)
            core.load_accepted(name)
            core.load_confidence(name)
            core.meta_get(name)
        self.assertFalse(self.output.exists())
        core.seed_paintings()
        for name in names:
            for kind in ("painting", "accepted", "confidence"):
                source = core.annotation_path(name, kind)
                assert source is not None
                bundled = Path(source)
                saved = self.canonical(f"{kind}.npy", name)
                draft = self.canonical(f"{kind}.npy", name, final=False)
                self.assertTrue(saved.is_file())
                self.assertTrue(draft.is_file())
                np.testing.assert_array_equal(np.load(saved, allow_pickle=False), np.load(bundled, allow_pickle=False))
                np.testing.assert_array_equal(np.load(draft, allow_pickle=False), np.load(bundled, allow_pickle=False))
                self.assertFalse(saved.samefile(draft))
                self.assertFalse(saved.samefile(bundled))
                self.assertFalse(draft.samefile(bundled))
            self.assertTrue(core.meta_get(name, final=True)["verified"])
        self.assertEqual(len(list(self.saved.rglob("*_painting.npy"))), 6)
        self.assertEqual(len(list(self.drafts.rglob("*_painting.npy"))), 6)
        self.assertFalse(self.canonical("painting.npy", "unpainted").exists())
        self.assertFalse(self.canonical("painting.npy", "unpainted", final=False).exists())
        self.assertEqual(snapshot(self.processed), source_before)
        seeded = snapshot(self.root)
        core.seed_paintings()
        self.assertEqual(snapshot(self.root), seeded)
        saved_before = snapshot(self.saved)
        core.save_painting(names[0], np.zeros_like(self.cells))
        core.meta_set(names[0], final=False, verified=False)
        edited = snapshot(self.root)
        core.seed_paintings(names)
        self.assertEqual(snapshot(self.root), edited)
        self.assertEqual(snapshot(self.saved), saved_before)
        self.assertEqual(snapshot(self.processed), source_before)
        self.assertFalse((self.root / "models").exists())

    def test_seed_preserves_existing_canonical_legacy_and_metadata_only_work(self):
        names = ("saved-edit", "draft-edit", "legacy-saved", "legacy-draft",
                 "mirror-saved", "mirror-draft", "metadata-edit")
        for name in names:
            self.install(name, bundled=True)
        blank = np.zeros_like(self.cells)
        core.save_painting("saved-edit", blank, final=True)
        core.save_painting("draft-edit", blank)
        self.array(self.legacy / "legacy-saved/painting.npy", blank)
        self.array(self.legacy_drafts / "legacy-draft.npy", blank)
        self.array(self.old_mirror("painting.npy", "mirror-saved"), blank)
        self.array(self.old_mirror("draft.npy", "mirror-draft"), blank)
        core.meta_set("metadata-edit", final=False, verified=False)
        before = snapshot(self.root)
        core.seed_paintings(names)
        self.assertEqual(snapshot(self.root), before)

    def test_seed_selection_is_explicit_and_labels_only_or_invalid_bundles_create_no_work(self):
        for name in ("selected", "untouched"):
            self.install(name, bundled=True)
        self.install("labels-only", bundled=True)
        self.records[-1]["annotations"].pop("painting")
        self.json(self.processed / "working_dems.json", self.records)
        core.seed_paintings(["selected", "labels-only"])
        for final in (False, True):
            self.assertTrue(self.canonical("painting.npy", "selected", final=final).is_file())
            self.assertFalse(self.canonical("painting.npy", "untouched", final=final).exists())
            self.assertFalse(self.canonical("painting.npy", "labels-only", final=final).exists())
            folder = self.canonical("painting.npy", "labels-only", final=final).parent
            self.assertEqual(list(folder.glob("labels-only_*")), [])
            self.assertEqual(list(folder.glob("untouched_*")), [])
        self.install("invalid", bundled=True)
        source = core.annotation_path("invalid", "painting")
        assert source is not None
        self.array(Path(source), np.ones((2, 2), np.uint8))
        before = snapshot(self.root)
        try:
            core.seed_paintings(["invalid"])
        except ValueError:
            pass  # A rejected invalid bundle may be reported or skipped, but never copied.
        self.assertEqual(snapshot(self.root), before)


class GalleryOutputLayoutTests(PageFixture):
    def test_save_replaces_gis_exports_separately_from_paintings(self):
        dem = self.dataset(bundled=True)
        with rasterio.open(dem) as source:
            transform, crs = source.transform, source.crs
        source_before = snapshot(self.processed)
        app = self.app()
        self.assertEqual(app.checkbox(key="verified_tile-one").label, "Verified")
        self.button(app, "Update model").click().run()
        self.assert_ok(app)
        model_before = snapshot(paths.MODEL_DIR)
        folder = paths.MAP_DIR / "atlas/tiles"
        labels_path = folder / "tile-one_labels.tif"
        prediction_path = folder / "tile-one_map.tif"
        for code in (1, 3):
            app.radio[0].set_value("smooth highlands" if code == 1 else "shadowed floor").run()
            with patch.object(canvas, "map_canvas", return_value={
                    "id": f"export-stroke-{code}", "cells": [0], "click": True}):
                app.run()
            self.assert_ok(app)
            cells = app.session_state.cells.copy()
            expected_prediction = np.full_like(cells, code)
            expected_prediction[0, 0] = 255
            with patch.object(assistant, "predict", return_value=expected_prediction):
                self.button(app, "Save labels + map").click().run()
            self.assert_ok(app)
            self.assertEqual(set(folder.iterdir()), {labels_path, prediction_path})
            with rasterio.open(labels_path) as labels:
                np.testing.assert_array_equal(labels.read(1), core.cells_to_labels(cells, cells.shape))
                self.assertEqual(labels.crs, crs)
                self.assertEqual(labels.transform, transform)
                self.assertEqual(labels.nodata, 255)
                self.assertEqual(labels.dtypes, ("uint8",))
                self.assertEqual(labels.colormap(1)[code], core.CMAP[code])
            with rasterio.open(prediction_path) as prediction:
                np.testing.assert_array_equal(prediction.read(1), expected_prediction)
                self.assertEqual(prediction.crs, crs)
                self.assertEqual(prediction.transform, transform)
                self.assertEqual(prediction.nodata, 255)
            np.testing.assert_array_equal(np.load(core.output_file("tile-one", "painting.npy")), cells)
            self.assertFalse(list(paths.OUT_DIR.rglob("*.tif")))
            self.assertEqual(snapshot(paths.MODEL_DIR), model_before)
            self.assertEqual(snapshot(self.processed), source_before)
        # A save without a usable model must not leave the previous prediction.
        app.session_state.assistant["model"] = None
        self.button(app, "Save labels + map").click().run()
        self.assert_ok(app)
        self.assertTrue(labels_path.is_file())
        self.assertFalse(prediction_path.exists())
        self.assertFalse(core.meta_get("tile-one", final=True)["prediction_saved"])
        self.assertEqual(snapshot(paths.MODEL_DIR), model_before)

    def test_browsing_empty_painting_never_creates_empty_draft_or_saved_bucket(self):
        self.dataset()
        with self.browsing():
            app = self.app()
            self.assertFalse(app.session_state.cells.any())
            self.gallery(app)
            app.switch_page("app/app_pages/paint.py").run()
            self.assert_ok(app)
            app.run()
            self.assert_ok(app)
        self.assertFalse(paths.OUT_DIR.exists())
        self.assertFalse(paths.DRAFT_DIR.exists())
        self.fixture.assert_no_artifacts()

    def test_edits_training_and_verification_leave_saved_cohort_unchanged_until_save(self):
        dem = self.dataset(bundled=True)
        cells = core.load_painting("tile-one")
        core.save_painting("tile-one", cells, final=True)
        core.meta_set("tile-one", verified=True)
        with rasterio.open(dem) as source:
            profile = source.profile
        for filename in ("labels.tif", "map.tif"):
            core.write_map(core.output_file("tile-one", filename),
                           core.cells_to_labels(cells, cells.shape), profile)
        Image.new("RGB", (12, 12), "blue").save(core.output_file("tile-one", "thumb.png"))
        saved_before = snapshot(paths.OUT_DIR)
        maps_before = snapshot(paths.MAP_DIR)
        source_before = snapshot(self.processed)

        def unchanged():
            self.assertEqual(snapshot(paths.OUT_DIR), saved_before)
            self.assertEqual(snapshot(paths.MAP_DIR), maps_before)
            self.assertEqual(snapshot(self.processed), source_before)

        app = self.app()
        unchanged()
        self.assertFalse(paths.DRAFT_DIR.exists())
        with patch.object(canvas, "map_canvas", return_value={"id": "draft-stroke", "cells": [0], "click": True}):
            app.run()
        self.assert_ok(app)
        unchanged()
        draft = Path(core.painting_files("tile-one")[0])
        self.assertEqual(draft, paths.DRAFT_DIR / "atlas/tiles/tile-one_painting.npy")
        self.assertEqual(np.load(draft, allow_pickle=False)[0, 0], 1)
        self.button(app, "Undo").click().run()
        self.assert_ok(app)
        unchanged()
        np.testing.assert_array_equal(core.load_painting("tile-one"), cells)
        self.button(app, "Clear").click().run()
        self.assert_ok(app)
        unchanged()
        self.assertFalse(core.load_painting("tile-one").any())
        self.button(app, "Undo").click().run()
        self.assert_ok(app)
        unchanged()
        app.checkbox(key="verified_tile-one").uncheck().run()
        self.assert_ok(app)
        unchanged()
        self.assertFalse(core.meta_get("tile-one")["verified"])
        self.assertTrue(core.meta_get("tile-one", final=True)["verified"])
        self.gallery(app)
        self.assertFalse(app.checkbox(key="v_tile-one").value)
        app.checkbox(key="v_tile-one").check().run()
        self.assert_ok(app)
        unchanged()
        app.checkbox(key="v_tile-one").uncheck().run()
        self.assert_ok(app)
        unchanged()
        self.assertFalse(paths.MODEL_DIR.exists())
        app.switch_page("app/app_pages/paint.py").run()
        self.assert_ok(app)
        self.assertFalse(app.checkbox(key="verified_tile-one").value)
        self.assertTrue(self.button(app, "Update model").disabled)
        app.checkbox(key="verified_tile-one").check().run()
        self.button(app, "Update model").click().run()
        self.assert_ok(app)
        unchanged()
        self.assertIsNotNone(assistant.load()["model"])
        model_before_save = snapshot(paths.MODEL_DIR)
        app.checkbox(key="verified_tile-one").uncheck().run()
        with patch.object(canvas, "map_canvas", return_value={"id": "final-stroke", "cells": [0], "click": True}):
            app.run()
        self.assert_ok(app)
        unchanged()
        expected = core.load_painting("tile-one").copy()
        self.button(app, "Save labels + map").click().run()
        self.assert_ok(app)
        self.assertNotEqual(snapshot(paths.OUT_DIR), saved_before)
        final_path = Path(core.output_file("tile-one", "painting.npy"))
        self.assertEqual(final_path, paths.OUT_DIR / "atlas/tiles/tile-one_painting.npy")
        np.testing.assert_array_equal(np.load(final_path, allow_pickle=False), expected)
        with rasterio.open(core.output_file("tile-one", "labels.tif")) as labels:
            np.testing.assert_array_equal(labels.read(1), core.cells_to_labels(expected, cells.shape))
        self.assertFalse(core.meta_get("tile-one", final=True)["verified"])
        self.assertFalse(Path(core.output_file("tile-one", "meta.json", final=False)).exists())
        self.assertEqual(snapshot(paths.MODEL_DIR), model_before_save)
        self.assertEqual(snapshot(self.processed), source_before)
        self.assertFalse((paths.LEGACY_OUT_DIR / "tile-one").exists())
        self.assertFalse(paths.LEGACY_DRAFT_DIR.exists())

    def test_legacy_tiff_thumbnail_remains_visible_after_metadata_only_override(self):
        dem = self.dataset()
        legacy = Path(core.LEGACY_OUT_DIR) / "tile-one"
        with rasterio.open(dem) as source:
            profile = source.profile
        core.write_map(str(legacy / "labels.tif"), np.ones((core.GRID, core.GRID), np.uint8), profile)
        Image.new("RGB", (12, 12), "blue").save(legacy / "thumb.png")
        self.fixture.write_json(legacy / "meta.json", {"verified": True})
        core.meta_set("tile-one", verified=False)
        app = self.app()
        before = snapshot(self.root)
        with self.browsing(), patch.object(pictures, "render", side_effect=AssertionError("Use saved legacy thumbnail")):
            self.gallery(app)
            self.assertFalse(app.checkbox(key="v_tile-one").value)
            self.assertIn("Saved labels", self.text(app))
            app.run()
            self.assert_ok(app)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(list(Path(core.out_dir("tile-one")).iterdir()),
                         [Path(core.output_file("tile-one", "meta.json"))])

    def test_new_saved_cohort_does_not_display_stale_legacy_thumbnail(self):
        self.dataset(bundled=True)
        legacy = Path(core.LEGACY_OUT_DIR) / "tile-one"
        legacy.mkdir(parents=True)
        cells = core.load_painting("tile-one")
        np.save(legacy / "painting.npy", cells)
        Image.new("RGB", (12, 12), "blue").save(legacy / "thumb.png")
        app = self.app()
        with self.browsing():
            self.gallery(app)
        self.assertIn("Saved labels", self.text(app))
        core.save_painting("tile-one", np.where(cells > 0, 3, 0).astype(np.uint8), final=True)
        with self.browsing(), patch.object(core, "read", wraps=core.read) as read:
            app.run()
            self.assert_ok(app)
            self.assertNotIn("Saved map preview", self.text(app))
            self.assertEqual(read.call_count, 1)
            app.run()
            self.assert_ok(app)
            self.assertEqual(read.call_count, 1)
        self.assertTrue((legacy / "thumb.png").is_file())
        self.assertFalse(Path(core.output_file("tile-one", "thumb.png", existing=True)).exists())

    def test_gallery_cache_tracks_legacy_draft_revisions_and_canonical_takeover(self):
        self.dataset(bundled=True)
        cells = core.load_painting("tile-one")
        draft = Path(core.LEGACY_DRAFT_DIR) / "tile-one.npy"
        draft.parent.mkdir(parents=True)
        np.save(draft, cells)
        app = self.app()
        with patch.object(core, "read", wraps=core.read) as read:
            with self.browsing():
                self.gallery(app)
                app.run()
                self.assert_ok(app)
                self.assertEqual(read.call_count, 1)
            cells[0, 0] = 3
            np.save(draft, cells)
            with self.browsing():
                app.run()
                self.assert_ok(app)
                self.assertEqual(read.call_count, 2)
            core.save_painting("tile-one", np.full_like(cells, 2))
            with self.browsing():
                app.run()
                self.assert_ok(app)
                self.assertEqual(read.call_count, 3)
                app.run()
                self.assert_ok(app)
                self.assertEqual(read.call_count, 3)


if __name__ == "__main__":
    unittest.main()
