"""The optional poster extractor agrees with app terrain features."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from app import core, assistant_model as assistant
from poster import reference_training as training
from tests.raster_fixtures import TRANSFORM
from tests import test_prepare_reference_section as fixtures

class FeatureEquivalenceTests(unittest.TestCase):
    def test_features(self):
        fixture = fixtures.ReferenceSectionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self = fixture
        record, = self.run_import()["records"]
        before = fixtures.snapshot(self.root)
        with patch.object(core, "DEM_DIR", self.processed):
            extracted = training.load_data(self.processed, self.reference,
                                           self.root / "absent-raw")
            stack, names, _, profile, _ = core.build(str(self.target / record["working_dem"]))
        expected, = extracted["tiles"]
        np.testing.assert_allclose(assistant.features(stack, names), expected["X"],
                                   rtol=0, atol=0, equal_nan=True)
        self.assertEqual(profile["transform"], TRANSFORM)
        self.assertEqual((profile["height"], profile["width"]), (120, 240))
        self.assertEqual(expected["provenance"]["public_tiles"],
                         [Path(p).stem for p in record["feature_sources"]])
        self.assertEqual(fixtures.snapshot(self.root), before)
