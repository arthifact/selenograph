"""Square geographic display frames without changing rasters or painting cell IDs."""
import io
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from rasterio import Affine
import rasterio
import streamlit as st

from app import canvas, core, pictures
from tests import test_clean_start as fixtures


class TileDisplayTests(unittest.TestCase):
    def test_disagreement_only_highlights_different_known_painted_units(self):
        base = np.full((120, 120), .5, np.float32)
        cells = np.zeros((core.GRID, core.GRID), np.uint8)
        cells[0, [0, 2, 4, 6, 8]] = [1, 2, 3, 2, 1]
        prediction = np.ones((60, 60), np.uint8)
        prediction[0, 3:5] = [255, 0]
        before = cells.copy(), prediction.copy()
        result = np.asarray(pictures.disagreement(base, prediction, cells, .5))
        coloured = result[:, :, 0] != result[:, :, 2]
        expected = np.zeros(base.shape, bool)
        expected[0, [2, 4]] = True
        np.testing.assert_array_equal(coloured, expected)
        self.assertTrue((result[coloured, 0] > result[coloured, 2]).all())
        self.assertEqual(pictures.disagreement(base, prediction, cells, 0).tobytes(),
                         pictures.render(base, []).tobytes())
        np.testing.assert_array_equal(cells, before[0])
        np.testing.assert_array_equal(prediction, before[1])

    def test_disagreement_preserves_floor_cell_edges_on_non_square_grids(self):
        base = np.full((123, 241), .5, np.float32)
        cells = np.zeros((core.GRID, core.GRID), np.uint8)
        cells[1, 1] = 2
        result = np.asarray(pictures.disagreement(base, np.ones((1, 1), np.uint8), cells, 1))
        expected = np.zeros(base.shape, bool)
        expected[1:2, 2:4] = True
        np.testing.assert_array_equal(result[:, :, 0] != result[:, :, 2], expected)

    def bounds(self, size, origin, *, peer=True):
        record = dict(working_dem="edge.tif", section="survey", size=size,
                      transform=list(Affine(5, 0, origin[0], 0, -5, origin[1]))[:6])
        full = dict(working_dem="full.tif", section="survey", size=[1024, 1024],
                    transform=[5, 0, 0, 0, -5, 0])
        with patch.object(core, "map_record", return_value=record), \
                patch.object(core, "records", return_value=[full, record] if peer else [record]):
            return pictures.tile_bounds("edge", size, Affine(*record["transform"]))

    def test_geographic_placement_of_each_edge_and_corner(self):
        for size, origin, expected in (
            ([1024, 128], (3 * 5120, 0), (0, 0, .125, 1)),
            ([128, 1024], (0, -3 * 5120), (0, 0, 1, .125)),
            ([128, 128], (3 * 5120, -3 * 5120), (0, 0, .125, .125)),
            ([1024, 128], (-640, 0), (.875, 0, .125, 1)),
            ([128, 1024], (0, 640), (0, .875, 1, .125)),
            ([128, 128], (-640, 640), (.875, .875, .125, .125)),
            ([1024, 1024], (5120, -5120), (0, 0, 1, 1)),
        ):
            with self.subTest(size=size, origin=origin):
                self.assertEqual(self.bounds(size, origin), expected)

    def test_unaligned_or_standalone_maps_are_not_moved_to_an_invented_grid(self):
        self.assertEqual(self.bounds([1024, 128], (100, 100)), (0, 0, .125, 1))
        self.assertEqual(self.bounds([128, 128], (0, 0), peer=False), (0, 0, 1, 1))

    def test_frames_use_black_missing_pixels_and_preserve_offset_and_scale(self):
        source = Image.new("RGB", (128, 128), "blue")
        for bounds in ((0, 0, .125, .125), (.875, .875, .125, .125)):
            result = pictures.tile_frame(source, bounds, size=256)
            x, y = round(bounds[0] * 256), round(bounds[1] * 256)
            self.assertEqual(result.size, (256, 256))
            self.assertEqual(result.getbbox(), (x, y, x + 32, y + 32))
            self.assertEqual(result.getpixel((x, y)), (0, 0, 255))
            self.assertEqual(result.getpixel((128, 128)), (0, 0, 0))
        self.assertEqual(source.size, (128, 128))

    def test_old_square_thumbnail_is_restored_to_its_raster_footprint(self):
        old = Image.new("RGB", (360, 360), "blue")
        result = pictures.gallery_preview(old, bounds=(0, 0, .125, 1))
        self.assertEqual(result.getbbox(), (0, 0, 30, 240))

    def test_nodata_stays_black_over_labels_and_saved_white_borders(self):
        missing = np.zeros((120, 120), bool)
        missing[-30:] = True
        missing[30:40, 20:30] = True
        image = Image.new("RGB", (360, 360), "white")
        result = np.asarray(pictures.gallery_preview(image, bounds=(0, 0, 1, 1), missing=missing))
        self.assertTrue((result[180:] == 0).all())
        self.assertTrue((result[60:80, 40:60] == 0).all())
        self.assertTrue((result[:50] == 255).all())

    @unittest.skipUnless(shutil.which("node"), "Node runs the actual canvas selection code")
    def test_canvas_clicks_and_lassos_keep_original_cell_ids_and_ignore_padding(self):
        script = canvas.JS + r'''
import assert from 'node:assert/strict';
const original = [[2.5, 2.5], [62.5, 37.5], [117.5, 117.5]];
for (const bounds of [[0, 0, .125, 1], [0, 0, 1, .125], [.875, .875, .125, .125]]) {
  const mapPoint = ([x, y]) => [(bounds[0] + x / 120 * bounds[2]) * 1024,
                               (bounds[1] + y / 120 * bounds[3]) * 1024];
  for (const point of original) {
    assert.deepEqual(selectCells([mapPoint(point)], 120, 1024, 1024, bounds),
                     selectCells([point], 120, 120, 120));
  }
  const loop = [[10,10],[20,10],[20,20],[10,20],[10,10]];
  assert.deepEqual(selectCells(loop.map(mapPoint), 120, 1024, 1024, bounds),
                   selectCells(loop, 120, 120, 120));
}
assert.deepEqual(selectCells([[500,500]], 120, 1024, 1024, [0,0,.125,1]), []);
assert.deepEqual(selectCells([[2,2]], 120, 1024, 1024, [.875,.875,.125,.125]), []);
assert.deepEqual(selectCells([[2.5,2.5]], 120, 120, 120, [0,0,1,1], []), []);
assert.deepEqual(selectCells([[2.5,2.5]], 120, 120, 120, [0,0,1,1], [242]), [242]);
'''
        result = subprocess.run([shutil.which("node"), "--input-type=module"], input=script,
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


class TileDisplayAppTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CleanStartTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)

    def test_paint_renders_geographic_frame_and_keeps_existing_painting_cells(self):
        self.f.public("full", shape=(120, 120), left=1000)
        path = self.f.public("edge", shape=(120, 15), left=1600)
        cells = np.zeros((core.GRID, core.GRID), np.uint8)
        cells[50, 60] = 1
        core.save_painting("edge", cells)
        app = self.f.app()
        with patch.object(canvas, "map_canvas", return_value=None) as shown:
            app.selectbox(key="selected_map").select(path).run()
            self.f.assert_ok(app)
        call = shown.call_args
        self.assertEqual(call.args[0].size, (120, 120))
        self.assertEqual(call.kwargs["bounds"], (0, 0, .125, 1))
        self.assertTrue((np.asarray(call.args[0])[:, 15:] == 0).all())
        self.assertTrue((np.asarray(call.kwargs["terrain"])[:, 15:] == 0).all())
        np.testing.assert_array_equal(app.session_state.cells, cells)
        np.testing.assert_array_equal(core.load_painting("edge"), cells)

    def test_nodata_is_black_in_paint_gallery_and_saved_thumbnail(self):
        path = self.f.public("hole")
        with rasterio.open(path, "r+") as src:
            values = src.read(1)
            values[-30:] = src.nodata
            src.write(values, 1)
        core.save_painting("hole", np.ones((core.GRID, core.GRID), np.uint8))
        with patch.object(canvas, "map_canvas", return_value=None) as shown:
            app = self.f.app()
        self.assertTrue((np.asarray(shown.call_args.args[0])[-30:] == 0).all())
        self.assertFalse(shown.call_args.kwargs["valid_cells"][-30:].any())
        original = core.load_painting("hole").copy()
        with patch.object(canvas, "map_canvas", return_value={"id": "black", "cells": [14399], "click": True}):
            app.run()
        self.f.assert_ok(app)
        np.testing.assert_array_equal(core.load_painting("hole"), original)
        self.f.button(app, "Save labels + map").click().run()
        self.f.assert_ok(app)
        thumb = core.output_file("hole", "thumb.png")
        with Image.open(thumb) as img:
            self.assertTrue((np.asarray(img)[-30:] == 0).all())
        # Older saves used white missing areas. Current paintings replace that
        # stale preview in memory, with missing terrain still black.
        Image.new("RGB", (360, 360), "white").save(thumb)
        before = Path(thumb).read_bytes()
        images = []
        display = st.image
        def capture(image, *args, **kwargs):
            images.append(image)
            return display(image, *args, **kwargs)
        with patch.object(st, "image", side_effect=capture):
            app.switch_page("app/app_pages/gallery.py").run()
        self.f.assert_ok(app)
        with Image.open(io.BytesIO(images[-1])) as img:
            self.assertTrue((np.asarray(img)[180:] == 0).all())
            pixels = np.asarray(img)
            self.assertTrue((pixels[:170, :, 2] > pixels[:170, :, 0]).all())  # current blue painting
        self.assertEqual(Path(thumb).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
