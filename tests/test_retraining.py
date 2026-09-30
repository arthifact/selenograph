"""Global model updates read current verified paintings and publish only a model."""
import unittest
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import assistant_model as assistant, core
from tests import test_generic_training as fixtures


class RetrainingTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.GenericTrainingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.enterContext(patch.object(core, "MODEL", "rf"))
        self.enterContext(patch.object(core, "held_out_maps", return_value=set()))
        self.enterContext(patch.object(core.backend(), "fit_rows", return_value=None))
        self.build = self.enterContext(patch.object(core, "build", side_effect=self.features))

    def features(self, path):
        value = 10 if Path(path).stem == "first" else 20
        return np.full_like(self.f.X, value), list(core.BASE_FEATS), None, None, 5

    def map(self, name, *, verified=True, offset=0):
        path = self.f.tile(name, name, left=1000 + offset * 20000, verified=verified)
        core.save_painting(name, self.f.cells, final=False)
        return path

    def test_rebuilds_all_current_verified_paintings_and_publishes_only_model(self):
        self.map("first")
        self.map("second", offset=1)
        self.map("unverified", verified=False, offset=2)
        before = self.f.snapshot(self.f.root / "output"), self.f.snapshot(self.f.processed)
        assistant.save(self.f.fitted({"deleted-map": {"stale": True}}))
        with patch.object(assistant, "predict", side_effect=AssertionError("Only training")), \
                patch.object(assistant, "fit", wraps=assistant.fit) as fit:
            first = assistant.retrain()
            self.assertEqual(set(first["samples"]), {"first", "second"})
            self.assertEqual({Path(c.args[0]).stem for c in self.build.call_args_list}, {"first", "second"})
            self.assertTrue((first["samples"]["first"]["X"] == 10).all())
            second = assistant.retrain()
            self.assertEqual(fit.call_count, 2)  # Even with unchanged inputs, every click fits again.
            self.assertNotEqual(first["revision"], second["revision"])
        self.assertEqual(before, (self.f.snapshot(self.f.root / "output"), self.f.snapshot(self.f.processed)))
        self.assertEqual({p.name for p in Path(assistant.MODEL_DIR).iterdir()}, {"rf.joblib"})

    def test_changes_on_other_maps_are_read_and_revoked_cleared_removed_maps_are_dropped(self):
        for index, name in enumerate(("first", "second", "revoked", "cleared", "removed")):
            self.map(name, offset=index)
        assistant.retrain()
        changed = np.where(self.f.cells == 2, 3, self.f.cells).astype(np.uint8)
        core.save_painting("second", changed, final=False)
        core.meta_set("revoked", final=False, verified=False)
        core.save_painting("cleared", np.zeros_like(self.f.cells), final=False)
        self.f.records = [r for r in self.f.records if Path(r["working_dem"]).stem != "removed"]
        self.f.publish()
        result = assistant.retrain()
        self.assertEqual(set(result["samples"]), {"first", "second"})
        self.assertEqual(set(result["samples"]["second"]["y"]), {1, 3})
        self.assertEqual(set(result["samples"]["first"]["y"]), {1, 2})

    def test_legacy_test_selection_does_not_exclude_verified_maps_or_aliases(self):
        self.map("first")
        self.map("test", offset=1)
        self.map("test-peer", offset=2)
        self.f.records[-1]["group"] = "test"
        self.f.publish()
        core.set_test_maps(["test"])
        with patch.object(core, "held_out_maps", side_effect=AssertionError("No persistent holdouts")):
            result = assistant.retrain()
            self.assertFalse(assistant.training_changed(result))
        expected = {"first", "test", "test-peer"}
        self.assertEqual(set(result["samples"]), expected)
        self.assertEqual({Path(c.args[0]).stem for c in self.build.call_args_list}, expected)

    def test_failure_preserves_previous_model_and_all_paintings(self):
        self.map("first")
        assistant.retrain()
        before = self.f.snapshot(self.f.root)
        with patch.object(core, "build", side_effect=ValueError("broken terrain")), \
                self.assertRaisesRegex(ValueError, "broken terrain"):
            assistant.retrain()
        self.assertEqual(self.f.snapshot(self.f.root), before)
        with patch.object(assistant, "fit", side_effect=RuntimeError("fit failed")), \
                self.assertRaisesRegex(RuntimeError, "fit failed"):
            assistant.retrain()
        self.assertEqual(self.f.snapshot(self.f.root), before)

    def test_training_status_survives_relocation_and_new_checkout_timestamps(self):
        self.map("first")
        bundle = assistant.retrain()
        moved = self.f.root / "new-computer"
        shutil.copytree(core.DEM_DIR, moved)
        for path in moved.rglob("*.tif"):
            os.utime(path, (1, 1))
        with patch.object(core, "DEM_DIR", moved):
            self.assertFalse(assistant.training_changed(bundle))
            dem = Path(core.dem_files()[0])
            import rasterio
            with rasterio.open(dem, "r+") as raster:
                raster.write(raster.read(1) + 1, 1)
            self.assertTrue(assistant.training_changed(bundle))

    def test_no_verified_maps_clears_old_training_memory(self):
        self.map("first")
        assistant.retrain()
        core.meta_set("first", final=False, verified=False)
        result = assistant.retrain()
        self.assertEqual(result["samples"], {})
        self.assertIsNone(result["model"])
        self.assertEqual(assistant.load()["revision"], result["revision"])

    def test_current_verification_overrides_saved_and_missing_verification_is_rejected(self):
        self.map("first")
        self.map("second", offset=1)
        core.meta_set("first", final=False, verified=False)
        self.f.records[-1].pop("verified")
        self.f.publish()
        self.assertEqual(assistant.verified_maps(), {})
        self.assertEqual(assistant.verified_samples({"first": {}, "second": {}, "unknown": {}}), {})
        with patch.object(assistant, "fit", side_effect=AssertionError("must not train")), \
                self.assertRaisesRegex(ValueError, "Verify first before training"):
            assistant.update("first", self.f.X, core.BASE_FEATS, self.f.cells, self.f.X, "first", "revision")
        self.assertFalse(Path(assistant.MODEL_DIR).exists())

    def test_refit_drops_revoked_memory_even_without_test_maps(self):
        self.map("first")
        self.map("second", offset=1)
        assistant.retrain()
        core.meta_set("second", final=False, verified=False)
        result = assistant.refit()
        self.assertEqual(set(result["samples"]), {"first"})


if __name__ == "__main__":
    unittest.main()
