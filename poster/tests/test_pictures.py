import unittest
import numpy as np
from poster import pictures

class PictureTests(unittest.TestCase):
    def test_poster_figure_has_every_panel(self):
        base = np.linspace(0, 1, 200 * 200, dtype=np.float32).reshape(200, 200)
        key = np.full((200, 200), 1, np.uint8)
        key[50:120, 50:120] = 3
        pred = key.copy()
        pred[:40] = 2
        conf = np.full((200, 200), 0.8, np.float32)
        conf[:10] = np.nan
        panels = [("key", pictures.units(base, key)), ("model", pictures.units(base, pred)),
                  ("differ", pictures.differences(base, key, pred)),
                  ("sure", pictures.confidence_picture(conf))]
        fig = pictures.figure("a title far too long to fit on a figure this narrow " * 3,
                              panels, 5.0, note="note")
        self.assertEqual(fig.width, 2 * 60 + 4 * 200 + 3 * 40)
        diff = np.array(panels[2][1])
        self.assertTrue((diff[:40, :, 0] > diff[:40, :, 1]).all())  # magenta where they differ
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(pictures.zipped({"a.png": panels[0][1],
                                                          "b.png": panels[3][1]}))) as z:
            self.assertEqual(sorted(z.namelist()), ["a.png", "b.png"])
