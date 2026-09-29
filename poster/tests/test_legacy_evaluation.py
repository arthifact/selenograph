import unittest
from unittest.mock import patch
import numpy as np
from app import core, assistant_model as assistant
from poster import legacy_evaluation as evaluate, storage
from tests import test_assistant as fixtures

class LegacyEvaluationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.AssistantTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.__dict__.update({k: v for k, v in fixture.__dict__.items() if not k.startswith('_')})
        self.update = fixture.update
        self.enterContext(patch.object(storage, "ROOT", self.root / "poster"))

    def test_validation_leaves_each_site_out_in_turn(self):
        for name in ("a", "b", "c"):
            self.update(name, self.cells)
        key = np.where(np.isfinite(self.X[0]), 1, 255).astype(np.uint8)
        seen = {}

        def fit(samples):
            seen.setdefault("folds", []).append(sorted(samples))
            return real_fit(samples)
        real_fit = assistant.fit
        with patch.object(evaluate, "folds", return_value={"A": ["a"], "B": ["b", "c"]}), \
                patch.object(evaluate, "inputs_for", side_effect=lambda names: [
                    (n, self.X, core.BASE_FEATS, key, None) for n in names]), \
                patch.object(assistant, "fit", side_effect=fit), \
                patch.object(core, "held_out_maps", side_effect=lambda names: set(names)):
            entry = evaluate.validate(assistant.load())
        self.assertEqual(seen["folds"], [["b", "c"], ["a"]])      # never the site scored
        self.assertEqual(set(entry["sites"]), {"A", "B"})
        self.assertEqual(entry["mean"]["accuracy"], round(
            (entry["sites"]["A"]["accuracy"] + entry["sites"]["B"]["accuracy"]) / 2, 3))
        self.assertEqual(len(evaluate.validation_history()), 1)
        saved = evaluate.saved_maps()                          # kept for poster figures
        self.assertEqual(sorted(saved), ["a", "b", "c"])
        self.assertEqual(saved["a"]["site"], "A")
        self.assertEqual(saved["a"]["pred"].shape, self.X.shape[1:])
        self.assertTrue(np.isnan(saved["a"]["conf"][:8]).all())
        self.assertEqual(saved["a"]["trained_on"], ["b", "c"])
