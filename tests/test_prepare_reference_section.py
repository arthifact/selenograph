"""Synthetic/offline importer tests; every data write is under TemporaryDirectory.

uv run --no-project python -B -m unittest tests.test_prepare_reference_section
The small fixture still uses the real hash/TIFF validator; only its reported
artifact count is mocked, so no real 88-artifact collection is required.
"""

import errno
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio

from app import assistant_model as assistant
from app import dataset_utils as data
from app import core
from app import prepare_reference_section as prep
from tests.raster_fixtures import MOON, TRANSFORM, raster, sha, write_json


def snapshot(root):
    """Hard links change nlink/ctime, but must not change bytes, mode, inode or mtime."""
    return {p.relative_to(root).as_posix():
            (p.stat().st_ino, p.stat().st_mtime_ns, p.stat().st_mode,
             p.read_bytes() if p.is_file() else None)
            for p in (root, *sorted(root.rglob("*"))) if p.exists()}


class ReferenceSectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="selenograph-reference-section-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.processed = self.root / "processed"
        self.reference = self.root / "preserved"

        self.output = self.root / "output"
        self.reference.mkdir()
        self.output.mkdir()
        (self.output / "keep.txt").write_text("Existing user edits must not change")
        self.site = "mons-mouton"
        self.target = self.processed / f"{self.site}(reference)"
        self.tile_id = "mons-mouton-0001"
        self.region = data.REGIONS[self.site]
        self.regions, self.tiles, self.sites = {}, {}, {}
        self.enterContext(rasterio.Env(GDAL_PAM_ENABLED="NO", PROJ_NETWORK="OFF"))
        self.make_public(self.region)
        self.make_reference()
        real_references = data._references

        def synthetic_references(root, hashes):
            refs, _ = real_references(root, hashes)
            return refs, prep.EXPECTED_REFERENCE_ARTIFACTS

        self.references = self.enterContext(patch.object(data, "_references", side_effect=synthetic_references))

    def make_public(self, region):
        folder = self.processed / region
        records, outputs = [], {}
        # Insert in reverse order to test reference_training's sorted tile order.
        for name, offset in (("right", 120), ("left", 0), ("far", 1000)):
            tile_id = f"public-{region}-{name}"
            layers = {}
            for role in ("sfs", "valid_data", "sfs_support"):
                relative = f"tiles/{tile_id}/{tile_id}{'' if role == 'sfs' else '_' + role}.tif"
                values = np.ones((120, 120), np.float32)
                if role == "sfs":
                    y, x = np.mgrid[:120, :120]
                    values = (np.sin((x + offset) / 15) * 20 + y / 5).astype(np.float32)
                raster(folder / relative, values,
                       transform=TRANSFORM @ rasterio.Affine.translation(offset, 0))
                layers[role] = relative
                outputs[relative] = {"checksum": {"sha256": sha(folder / relative)}}
            records.append({"tile_id": tile_id, "site": region, "collection": "public",
                            "working_dem": layers["sfs"], "layers": layers})
        write_json(folder / "working_dems.json", records)
        outputs["working_dems.json"] = {"checksum": {"sha256": sha(folder / "working_dems.json")}}
        metadata = folder / "metadata" / f"{region}_metadata.json"
        write_json(metadata, {"outputs": outputs})
        self.regions[region] = {"metadata_sha256": sha(metadata)}
        write_json(self.processed / "manifest.json", {"regions": self.regions})
        return records

    def refresh_public(self, records, region=None):
        region = region or self.region
        folder = self.processed / region
        write_json(folder / "working_dems.json", records)
        meta_path = folder / "metadata" / f"{region}_metadata.json"
        meta = data._json(meta_path)
        for relative in meta["outputs"]:
            meta["outputs"][relative]["checksum"]["sha256"] = sha(folder / relative)
        write_json(meta_path, meta)
        self.regions[region]["metadata_sha256"] = sha(meta_path)
        write_json(self.processed / "manifest.json", {"regions": self.regions})

    def make_reference(self, tile_id=None, site=None, variants=("saved", "draft"),
                       accepted=True, confidence=False):
        tile_id, site = tile_id or self.tile_id, site or self.site
        folder = self.reference / tile_id
        if folder.exists():
            shutil.rmtree(folder)
        layers = {}
        for role in ("dem", "nac", "radar-cpr", "radar-s1"):
            relative = f"{tile_id}/{role}.tif"
            values = np.full((120, 240), 10, np.float32)
            values[0, 0] = -9999
            raster(self.reference / relative, values, nodata=-9999)
            layers[role] = {"path": relative, "sha256": sha(self.reference / relative)}
        annotations = []
        for variant in variants:
            cells = np.full((120, 120), 1 if variant == "saved" else 2, np.uint8)
            files = []
            arrays = {"painting": cells}
            if accepted:
                flags = np.zeros(cells.shape, bool)
                flags[0, 0 if variant == "saved" else 1] = True
                arrays["accepted"] = flags
            if confidence:
                arrays["confidence"] = np.full(cells.shape, 2, np.uint8)
            for role, value in arrays.items():
                path = folder / f"{variant}_{role}.npy"
                np.save(path, value, allow_pickle=False)
                files.append({"role": role, "path": path.relative_to(self.reference).as_posix(),
                              "sha256": sha(path)})
            labels = data._expand(cells, (120, 240)).astype(np.uint16)
            labels[0, 0] = 65535
            raster(folder / f"{variant}.tif", labels, nodata=65535)
            provenance = folder / f"{variant}.json"
            write_json(provenance, {"schema_id": "selenograph-geologic-units-v1", "files": files,
                                    "review_status": "unreviewed_snapshot"})
            annotations.append({"snapshot_variant": variant, "path": f"{tile_id}/{variant}.tif",
                "sha256": sha(folder / f"{variant}.tif"), "provenance": f"{tile_id}/{variant}.json"})
        record = {"tile_id": tile_id, "site_id": site, "layers": layers,
                  "annotations": annotations, "legacy_name": f"old-{tile_id}-dem",
                  "legacy": {"preferred_seed": "draft"}}
        write_json(folder / "tile.json", record)
        self.tiles[tile_id] = {"tile_id": tile_id, "site_id": site, "path": f"{tile_id}/tile.json"}
        map_ids = {"mons-mouton": "MM026", "nobile1": "N1014", "nobile1-ms1": "Nobile1-MS1",
                   "nobile2": "N2005"}
        site_path = f"sites/{site}/site.json"
        write_json(self.reference / site_path, {"site_id": site, "map_id": map_ids.get(site, site),
                                              "title": f"Professor map — {map_ids.get(site, site)}"})
        self.sites[site] = {"site_id": site, "path": site_path}
        self.publish_reference()
        return record

    def make_all_sites(self):
        self.make_public("nobile_rim_1")
        self.make_public("nobile_rim_2")
        self.make_reference("nobile1-0001", "nobile1")
        for tile_id in ("nobile1-0002", "nobile1-0003", "nobile1-0005"):
            self.make_reference(tile_id, "nobile1-ms1")
        self.make_reference("nobile2-0001", "nobile2")

    def publish_reference(self):
        artifacts = [{"path": p.relative_to(self.reference).as_posix(), "sha256": sha(p)}
                     for p in sorted(self.reference.rglob("*")) if p.is_file() and p.name != "dataset.json"]
        write_json(self.reference / "dataset.json", {"tiles": list(self.tiles.values()),
                   "sites": list(self.sites.values()), "artifacts": artifacts})

    def run_import(self, **kwargs):
        return prep.run(self.processed, self.reference, **kwargs)

    def catalog(self):
        return data._json(self.target / "working_dems.json")

    def test_publish_generic_manifest_saved_priority_union_hashes_and_no_source_mutation(self):
        before = {root: snapshot(root) for root in (self.reference, self.processed / self.region,
                                                   self.processed / "manifest.json", self.output)}
        with patch.object(core, "build", side_effect=AssertionError("Importer must not compute features")):
            result = self.run_import()
        self.assertEqual(result["status"], "published")
        self.assertEqual((result["tile_count"], result["reference_artifacts_verified"]), (1, 88))
        self.references.assert_called_once()
        record, = self.catalog()
        self.assertEqual(record["working_dem"], f"tiles/{self.tile_id}/{self.tile_id}.tif")
        self.assertEqual((record["tile_id"], record["site"], record["group"]),
                         (self.tile_id, self.site, data.GROUPS[self.site]))
        self.assertEqual((record["section"], record["collection"]), (self.target.name, "reference"))
        self.assertEqual(record["map_id"], "MM026")
        self.assertEqual(record["title"], "Professor map — MM026")
        self.assertTrue(record["restricted"])
        self.assertEqual(record["size"], [120, 240])
        self.assertEqual(record["transform"], list(TRANSFORM)[:6])
        self.assertEqual(set(record["layers"]), {"dem", "nac", "cpr", "radar_s1"})
        for alias in (self.region, self.site, "MM026", f"old-{self.tile_id}-dem"):
            self.assertIn(alias, record["geography_aliases"])
        annotation = record["annotations"]
        self.assertTrue(annotation["verified"])
        self.assertEqual(annotation["label_status"], prep.LABEL_STATUS)
        provenance = annotation["provenance"]
        self.assertEqual(provenance["snapshot"], "saved")
        self.assertEqual(provenance["verification"]["basis"], "user_requested")
        self.assertFalse(provenance["verification"]["independent_certification"])
        self.assertEqual(provenance["confidence_status"], "unknown")
        self.assertNotIn("confidence", annotation)
        self.assertTrue((np.load(self.target / annotation["painting"]) == 1).all())
        accepted = np.load(self.target / annotation["accepted"])
        self.assertEqual(int(accepted.sum()), 2)
        self.assertTrue(accepted[0, 0] and accepted[0, 1])
        self.assertEqual(provenance["original_provenance"]["provenance_sha256"],
                         sha(self.reference / self.tile_id / "saved.json"))
        self.assertFalse(record["source_refs"]["download_allowed"])
        dataset = data._json(self.target / "dataset.json")
        self.assertEqual(dataset["materialized_inputs"],
                         result["sections"][self.target.name]["materialized_inputs"])
        for relative, item in dataset["materialized_inputs"].items():
            path = self.target / relative
            self.assertEqual(sha(path), item["sha256"])
            self.assertFalse(path.is_symlink())
            if item["storage"] == "hardlink":
                source = self.reference / item["source"].removeprefix("reference/")
                self.assertTrue(os.path.samefile(source, path))
                self.assertEqual((source.stat().st_dev, source.stat().st_ino),
                                 (path.stat().st_dev, path.stat().st_ino))
                self.assertGreaterEqual(source.stat().st_nlink, 2)
        self.assertEqual(result["hardlink_count"], 5)
        self.assertIn("HARD LINKS", (self.target / "README.md").read_text())
        for root, state in before.items():
            self.assertEqual(snapshot(root), state)
        self.assertFalse(list(self.processed.glob(".reference-stage-*")))
        with patch.object(core, "DEM_DIR", str(self.processed)):
            generic = core.map_record(self.tile_id)
            assert generic is not None
            dem_path = core.record_path(generic)
            assert dem_path is not None
            self.assertEqual(Path(dem_path), self.target / record["working_dem"])
            self.assertEqual(core.companion(dem_path, "cpr"),
                             str(self.target / record["layers"]["cpr"]))

    def test_feature_sources_match_whole_public_tile_intersections(self):
        record, = self.run_import()["records"]
        folder = self.processed / self.region
        grid = data._Grid((120, 240), TRANSFORM, MOON)
        expected = []
        for public in sorted(data._json(folder / "working_dems.json"), key=lambda r: r["tile_id"]):
            dem = folder / public["layers"]["sfs"]
            with rasterio.open(dem) as src:
                if data._intersects(data._Grid.read(src), grid):
                    expected.append(dem)
        actual = [(self.target / p).resolve() for p in record["feature_sources"]]
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 2)
        self.assertTrue(all(p.startswith(f"../{self.region}/") for p in record["feature_sources"]))
        self.assertTrue(all(p.is_relative_to(self.processed) for p in actual))
        self.assertNotIn((self.target / record["working_dem"]).resolve(), actual)
        source_hashes = data._json(self.target / "dataset.json")["source_hashes"]
        for path in actual:
            key = "processed/" + path.relative_to(self.processed).as_posix()
            self.assertEqual(source_hashes[key]["sha256"], sha(path))
            self.assertTrue(source_hashes[key]["verified_expected"])

    def test_draft_fallback_missing_flags_and_explicit_no_verified(self):
        self.make_reference(variants=("draft",), accepted=False)
        record, = self.run_import(verified=False)["records"]
        annotation = record["annotations"]
        self.assertFalse(annotation["verified"])
        self.assertEqual(annotation["label_status"], prep.UNVERIFIED_STATUS)
        self.assertEqual(annotation["provenance"]["snapshot"], "draft")
        self.assertEqual(annotation["provenance"]["verification"]["basis"], "not_requested")
        self.assertNotIn("accepted", annotation)
        self.assertNotIn("confidence", annotation)
        self.assertTrue((np.load(self.target / annotation["painting"]) == 2).all())

    def test_confidence_preserved_and_zero_accepted_array_still_observed(self):
        self.make_reference(variants=("saved",), confidence=True)
        path = self.reference / self.tile_id / "saved_accepted.npy"
        np.save(path, np.zeros((120, 120), bool), allow_pickle=False)
        provenance_path = self.reference / self.tile_id / "saved.json"
        provenance = data._json(provenance_path)
        next(item for item in provenance["files"] if item["role"] == "accepted")["sha256"] = sha(path)
        write_json(provenance_path, provenance)
        self.publish_reference()
        record, = self.run_import()["records"]
        annotation = record["annotations"]
        self.assertEqual(annotation["provenance"]["confidence_status"], "preserved")
        self.assertTrue((np.load(self.target / annotation["confidence"]) == 2).all())
        self.assertFalse(np.load(self.target / annotation["accepted"]).any())

    def test_n1014_and_ms1_share_canonical_geographic_group_not_map_id(self):
        self.make_public("nobile_rim_1")
        self.make_reference("nobile1-0001", "nobile1")
        self.make_reference("nobile1-0002", "nobile1-ms1")
        records = {r["tile_id"]: r for r in self.run_import()["records"]}
        a, b = records["nobile1-0001"], records["nobile1-0002"]
        self.assertEqual((a["group"], b["group"]), ("nobile-1", "nobile-1"))
        self.assertNotEqual(a["site"], b["site"])
        self.assertIn("N1014", a["geography_aliases"])
        self.assertIn("Nobile1-MS1", b["geography_aliases"])
        self.assertEqual(a["feature_sources"], b["feature_sources"])

    def test_inventory_and_cli_dry_run_write_nothing_and_defaults(self):
        before = snapshot(self.root)
        result = prep.inventory(self.processed, self.reference)
        self.assertEqual(result["status"], "dry_run")
        with patch.object(prep.paths, "DEM_DIR", self.processed), \
             patch.object(prep.paths, "PROFESSOR_MAPS_DIR", self.reference), \
             redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(prep.main(["--dry-run", "--no-verified"]), 0)
        self.assertFalse(json.loads(stdout.getvalue())["records"][0]["annotations"]["verified"])
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.target.exists())

    def test_cli_help_requires_no_data(self):
        with patch.object(prep, "_plans", side_effect=AssertionError("Help must not inspect data")), \
             redirect_stdout(io.StringIO()) as stdout, self.assertRaises(SystemExit) as exc:
            prep.main(["--help"])
        self.assertEqual(exc.exception.code, 0)
        self.assertIn("uv run", stdout.getvalue())
        self.assertIn("--dry-run", stdout.getvalue())
        self.assertIn("--no-verified", stdout.getvalue())
        self.assertIn("--site", stdout.getvalue())
        self.assertNotIn("--output-root", stdout.getvalue())

    def test_idempotent_rerun_validates_without_writes_or_links(self):
        self.target.mkdir()
        self.run_import()
        before = snapshot(self.root)
        with patch.object(prep.os, "link", side_effect=AssertionError("Must not relink")):
            self.assertEqual(self.run_import()["status"], "already_published")
            self.assertEqual(prep.inventory(self.processed, self.reference)["status"], "already_published")
        self.assertEqual(snapshot(self.root), before)

    def test_nonempty_or_changed_destination_refused_without_overwriting(self):
        self.target.mkdir()
        (self.target / "unrelated.txt").write_text("do not replace")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.run_import()
        self.assertEqual(snapshot(self.root), before)
        (self.target / "unrelated.txt").unlink()
        self.run_import()
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.run_import(verified=False)
        self.assertEqual(snapshot(self.root), before)

    def test_modified_materialized_array_and_raster_rejected(self):
        record, = self.run_import()["records"]
        painting = self.target / record["annotations"]["painting"]
        original = painting.read_bytes()
        painting.write_bytes(b"bad painting")
        with self.assertRaisesRegex(ValueError, "materialized content differs"):
            self.run_import()
        painting.write_bytes(original)
        # Unlink first: never mutate the hard-linked source, even in a corruption test.
        path = self.target / record["working_dem"]
        path.unlink()
        path.write_bytes(b"bad raster")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_import()

    def test_portable_regular_file_copy_is_validated_by_content(self):
        record, = self.run_import()["records"]
        path = self.target / record["working_dem"]
        content = path.read_bytes()
        path.unlink()
        path.write_bytes(content)
        self.assertEqual(self.run_import()["status"], "already_published")

    def test_hash_mismatch_count_and_tiff_painting_disagreement_refuse_before_writes(self):
        dem = self.reference / self.tile_id / "dem.tif"
        content = dem.read_bytes()
        dem.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_import()
        self.assertFalse(self.target.exists())
        dem.write_bytes(content)
        labels = self.reference / self.tile_id / "saved.tif"
        with rasterio.open(labels, "r+") as dst:
            values = dst.read(1)
            values[1, 1] = 3
            dst.write(values, 1)
        tile_path = self.reference / self.tile_id / "tile.json"
        record = data._json(tile_path)
        record["annotations"][0]["sha256"] = sha(labels)
        write_json(tile_path, record)
        self.publish_reference()
        with self.assertRaisesRegex(ValueError, "does not exactly expand"):
            self.run_import()
        self.assertFalse(self.target.exists())
        self.make_reference()
        with patch.object(data, "_references", return_value=([], 87)), \
             self.assertRaisesRegex(ValueError, "Expected 88 reference artifacts"):
            self.run_import()
        self.assertFalse(self.target.exists())

    def test_public_collision_in_unrelated_region_refused(self):
        unrelated = self.processed / "unrelated/working_dems.json"
        write_json(unrelated, [{"working_dem": f"tiles/{self.tile_id}.tif"}])
        with self.assertRaisesRegex(ValueError, "map ID collision"):
            self.run_import()
        self.assertFalse(self.target.exists())

    def test_no_intersection_and_overlapping_public_grids_refused(self):
        folder = self.processed / self.region
        records = data._json(folder / "working_dems.json")
        for record in records:
            for relative in record["layers"].values():
                with rasterio.open(folder / relative, "r+") as dst:
                    dst.transform = TRANSFORM @ rasterio.Affine.translation(1000, 0)
        self.refresh_public(records)
        with self.assertRaisesRegex(ValueError, "No matching public feature tiles"):
            self.run_import()
        for record in records:
            for relative in record["layers"].values():
                with rasterio.open(folder / relative, "r+") as dst:
                    dst.transform = TRANSFORM
        self.refresh_public(records)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.run_import()
        self.assertFalse(self.target.exists())

    def test_hardlink_failure_cleans_hidden_stage_and_never_copies(self):
        before = snapshot(self.reference)
        with patch.object(prep.os, "link", side_effect=OSError(errno.EXDEV, "cross-device link")), \
             patch.object(shutil, "copyfile", side_effect=AssertionError("No byte copy allowed")), \
             self.assertRaisesRegex(RuntimeError, "Hard link unavailable.*no large-copy fallback"):
            self.run_import()
        self.assertEqual(snapshot(self.reference), before)
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.processed.glob(".reference-stage-*")))

    def test_atomic_staging_is_not_discovered_and_late_failure_leaves_no_target(self):
        real_link = os.link
        calls = []

        def interrupted_link(source, destination):
            with patch.object(core, "DEM_DIR", str(self.processed)):
                self.assertFalse(any(".reference-stage-" in str(p) for p in core.manifest_files()))
                self.assertNotIn(self.tile_id, {Path(p).stem for p in core.dem_files()})
            self.assertFalse(self.target.exists())
            calls.append(destination)
            if len(calls) == 3:
                raise OSError(errno.EIO, "synthetic interruption")
            real_link(source, destination)

        before = snapshot(self.reference)
        with patch.object(prep.os, "link", side_effect=interrupted_link), self.assertRaises(RuntimeError):
            self.run_import()
        self.assertEqual(len(calls), 3)
        self.assertEqual(snapshot(self.reference), before)
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.processed.glob(".reference-stage-*")))

    def test_output_source_and_public_symlink_escapes_refused(self):
        for site in ("unknown-site", "../escape", "nobile1(reference)", ".hidden"):
            with self.subTest(site=site), self.assertRaises(ValueError):
                self.run_import(site=site)
        for root in (self.processed, self.processed / self.region):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "must not overlap"):
                prep.run(self.processed, root)
        self.target.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_import()
        self.target.unlink()
        dem = self.reference / self.tile_id / "dem.tif"
        outside = self.root / "outside.tif"
        shutil.copyfile(dem, outside)
        dem.unlink()
        dem.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes root"):
            self.run_import()
        dem.unlink()
        shutil.copyfile(outside, dem)
        folder = self.processed / self.region
        record = data._json(folder / "working_dems.json")[0]
        dem = folder / record["working_dem"]
        dem.unlink()
        dem.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes root"):
            self.run_import()
        self.assertFalse(self.target.exists())

    def test_existing_symlink_rejected_even_when_pointing_to_correct_source(self):
        record, = self.run_import()["records"]
        path = self.target / record["working_dem"]
        path.unlink()
        path.symlink_to(self.reference / self.tile_id / "dem.tif")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_import()

    def test_unsafe_seed_id_rejected(self):
        original = self.references.side_effect

        def unsafe(root, hashes):
            refs, count = original(root, hashes)
            refs[0]["record"]["tile_id"] = "../escape"
            return refs, count

        self.references.side_effect = unsafe
        with self.assertRaisesRegex(ValueError, "Unsafe tile"):
            self.run_import()
        self.assertFalse(self.target.exists())

    def test_generic_annotation_fallback_and_editing_only_user_outputs(self):
        self.run_import()
        before = {root: snapshot(root) for root in (self.reference, self.processed)}
        with patch.object(core, "DEM_DIR", self.processed), \
             patch.object(core, "OUT_DIR", self.output / "paintings"), \
             patch.object(core, "MAP_DIR", self.output / "maps"), \
             patch.object(core, "LEGACY_OUT_DIR", self.output / "maps"), \
             patch.object(core, "DRAFT_DIR", self.output / "painting_drafts"), \
             patch.object(core, "LEGACY_DRAFT_DIR", self.output / "drafts"), \
             patch.object(core, "LEGACY_MIRRORED_DIR", self.output):
            np.testing.assert_array_equal(core.load_painting(self.tile_id),
                                          np.ones((120, 120), np.uint8))
            self.assertEqual(int(core.load_accepted(self.tile_id).sum()), 2)
            self.assertTrue(core.meta_get(self.tile_id)["verified"])
            self.assertIn(self.tile_id, core.painted_maps()[0])
            self.assertFalse((self.output / "maps").exists())
            self.assertFalse((self.output / "drafts").exists())
            edited = np.full((120, 120), 3, np.uint8)
            core.save_painting(self.tile_id, edited)
            core.meta_set(self.tile_id, final=False, verified=False)
            np.testing.assert_array_equal(core.load_painting(self.tile_id), edited)
            self.assertFalse(core.meta_get(self.tile_id)["verified"])
            self.assertFalse(core.load_accepted(self.tile_id).any())
            self.assertTrue(Path(core.painting_files(self.tile_id)[0]).is_file())
            self.assertTrue(core.meta_get(self.tile_id, final=True)["verified"])
            self.assertFalse((self.output / "paintings").exists())
            self.assertFalse((self.output / "drafts").exists())
        for root, state in before.items():
            self.assertEqual(snapshot(root), state)
        self.assertEqual(self.run_import()["status"], "already_published")

    def test_changed_source_during_staging_aborts_publication(self):
        real_link = os.link
        changed = False

        def change_source(source, destination):
            nonlocal changed
            real_link(source, destination)
            if not changed:
                changed = True
                # Simulate another process replacing a source, without mutating its link.
                source.unlink()
                source.write_bytes(b"external source replacement")

        with patch.object(prep.os, "link", side_effect=change_source), \
             self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_import()
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.processed.glob(".reference-stage-*")))

    def test_concurrent_nonempty_target_is_never_replaced(self):
        real_link = os.link

        def concurrent_target(source, destination):
            real_link(source, destination)
            self.target.mkdir(exist_ok=True)
            (self.target / "keep.txt").write_text("concurrent writer")

        with patch.object(prep.os, "link", side_effect=concurrent_target), \
             self.assertRaisesRegex(ValueError, "became nonempty"):
            self.run_import()
        self.assertEqual((self.target / "keep.txt").read_text(), "concurrent writer")
        self.assertFalse((self.target / "working_dems.json").exists())
        self.assertFalse(list(self.processed.glob(".reference-stage-*")))

    def test_six_maps_in_four_direct_site_sections_beside_thirteen_public_regions(self):
        self.make_all_sites()
        for index in range(10):
            self.make_public(f"unrelated-region-{index}")
        before = {root: snapshot(root) for root in
                  (self.reference, self.output, self.processed / "manifest.json",
                   *(self.processed / region for region in self.regions))}
        inventory = prep.inventory(self.processed, self.reference)
        expected = {"mons-mouton(reference)": 1, "nobile1(reference)": 1,
                    "nobile2(reference)": 1, "nobile1-ms1(reference)": 3}
        self.assertEqual(inventory["tile_count"], 6)
        self.assertEqual(set(inventory["sections"]), set(expected))
        self.assertTrue(all(not (self.processed / name).exists() for name in expected))
        result = self.run_import()
        self.assertEqual(result["tile_count"], 6)
        self.assertEqual(result["hardlink_count"], 30)
        self.assertEqual(set(result["sections"]), set(expected))
        self.assertEqual({p.name for p in self.processed.iterdir()},
                         set(self.regions) | set(expected) | {"manifest.json"})
        for section, count in expected.items():
            folder = self.processed / section
            report = result["sections"][section]
            self.assertEqual(report["output_root"], str(folder))
            self.assertEqual(report["tile_count"], count)
            self.assertEqual(report["status"], "published")
            self.assertEqual(report["records"], data._json(folder / "working_dems.json"))
            self.assertEqual(data._json(folder / "dataset.json")["section"], section)
            self.assertIn(f"# {section}", (folder / "README.md").read_text())
            for record in report["records"]:
                self.assertEqual(record["section"], section)
                self.assertEqual(section, f"{record['site']}(reference)")
                self.assertTrue(record["map_id"])
                self.assertTrue((folder / record["working_dem"]).is_file())
                region = data.REGIONS[record["site"]]
                self.assertTrue(all(p.startswith(f"../{region}/") for p in record["feature_sources"]))
                for relative in record["layers"].values():
                    self.assertFalse((folder / relative).is_symlink())
                    self.assertGreaterEqual((folder / relative).stat().st_nlink, 2)
        with patch.object(core, "DEM_DIR", self.processed):
            records = [record for record in core.records() if record.get("collection") == "reference"]
            self.assertEqual(len(records), 6)
            self.assertEqual({r["section"] for r in records}, set(expected))
        for root, state in before.items():
            self.assertEqual(snapshot(root), state)
        published = snapshot(self.root)
        self.assertEqual(self.run_import()["status"], "already_published")
        self.assertEqual(snapshot(self.root), published)

    def test_site_cli_selection_then_all_sites_and_selected_reruns_are_idempotent(self):
        self.make_all_sites()
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(prep.main(["--processed-root", str(self.processed),
                "--reference-root", str(self.reference), "--site", "nobile1-ms1"]), 0)
        selected = json.loads(stdout.getvalue())
        self.assertEqual(selected["tile_count"], 3)
        self.assertEqual(set(selected["sections"]), {"nobile1-ms1(reference)"})
        self.assertFalse(self.target.exists())
        folder = self.processed / "nobile1-ms1(reference)"
        before = snapshot(folder)
        complete = self.run_import()
        self.assertEqual(complete["tile_count"], 6)
        self.assertEqual(complete["status"], "published")
        self.assertEqual(complete["sections"][folder.name]["status"], "already_published")
        self.assertEqual(snapshot(folder), before)
        before_all = snapshot(self.root)
        for site in ("mons-mouton", "nobile1", "nobile1-ms1", "nobile2"):
            result = self.run_import(site=site)
            self.assertEqual(result["status"], "already_published")
            self.assertEqual(set(result["sections"]), {f"{site}(reference)"})
        self.assertEqual(snapshot(self.root), before_all)

    def test_all_existing_destinations_are_validated_before_any_staging(self):
        self.make_all_sites()
        last = self.processed / "nobile2(reference)"
        last.mkdir()
        (last / "keep.txt").write_text("Unrelated data must not be replaced")
        before = snapshot(self.root)
        with patch.object(prep.os, "link", side_effect=AssertionError("No staging before validation")), \
             self.assertRaisesRegex(ValueError, "nonempty"):
            self.run_import()
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.target.exists())

    def test_later_site_plan_failure_leaves_all_destinations_unpublished(self):
        self.make_all_sites()
        (self.processed / "nobile_rim_2/metadata/nobile_rim_2_metadata.json").unlink()
        before = snapshot(self.root)
        with patch.object(prep.os, "link", side_effect=AssertionError("All plans must be ready")), \
             self.assertRaises(FileNotFoundError):
            self.run_import()
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(list(self.processed.glob("*(reference)")))

    def test_later_site_hardlink_failure_cleans_all_stages_before_any_publication(self):
        self.make_all_sites()
        real_link = os.link
        sources = []

        def interrupted_link(source, destination):
            self.assertFalse(list(self.processed.glob("*(reference)")))
            sources.append(source)
            if source.parent.name == "nobile2-0001":
                raise OSError(errno.EXDEV, "later site's cross-device link")
            real_link(source, destination)

        before = snapshot(self.reference)
        with patch.object(prep.os, "link", side_effect=interrupted_link), self.assertRaises(RuntimeError):
            self.run_import()
        self.assertGreater(len(sources), 5)
        self.assertEqual(snapshot(self.reference), before)
        self.assertFalse(list(self.processed.glob("*(reference)")))
        self.assertFalse(list(self.processed.glob(".reference-stage-*")))

    def test_other_site_ids_remain_reserved_when_selecting_a_single_site(self):
        self.make_all_sites()
        self.run_import(site="mons-mouton")
        manifest = self.target / "working_dems.json"
        records = data._json(manifest)
        records.append({"working_dem": "tiles/nobile1-0001.tif"})
        write_json(manifest, records)
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "map ID collision"):
            self.run_import(site="nobile1")
        self.assertEqual(snapshot(self.root), before)


if __name__ == "__main__":
    unittest.main()
