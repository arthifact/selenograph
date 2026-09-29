"""Local poster assets from supplied map-label/OOF arrays; no fitting or I/O of inputs.

Public API: export_figures(tiles, report, Path("poster") / run_id / "figures"). Class colours here
are deliberately poster-specific; the application's palette/schema is untouched.
"""
import csv
import io
import json
import math
import re
import zipfile
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coordinates

from app.core import GRID, per_pixel

DPI = 300
PAPER = (249, 248, 244)
INK = (28, 48, 59)
MUTED = (94, 111, 116)
TEAL = (28, 125, 126)
RULE = (217, 224, 221)
WHITE = (255, 255, 255)
UNKNOWN = (235, 231, 219)
UNSCORED = (231, 235, 234)
AGREE = (75, 153, 141)
DISAGREE = (190, 81, 79)
CLASS_COLOURS = {1: (74, 139, 103), 2: (216, 144, 65), 3: (45, 65, 98)}
CLASS_LABELS = {1: "Smooth highlands", 2: "Rough highlands", 3: "Crater interiors / floor"}
RAMP = ((232, 244, 238), TEAL, INK)
FEATURE_LABELS = {
    "slope": "Slope", "rough": "Terrain roughness", "svf": "Sky-view factor",
    "rel_local": "Local relative elevation", "curv": "Curvature",
    "svf_local": "Local sky-view factor",
}
RESTRICTION = "CUI / LOCAL-ONLY  |  Publication permission required"
PUBLIC_NOTICE = "LOCAL EXPORT / Review before publication"


def _legacy_reference_report(report):
    # Original reference reports predate the explicit restriction/provenance contract.
    # Restricted model seeds do not imply that new user-map labels are unreviewed.
    return report is None or "restricted" not in report


def _restriction(report):
    return RESTRICTION if report is None or report.get("restricted", True) else PUBLIC_NOTICE


def _review_note(report):
    if _legacy_reference_report(report):
        return "Unreviewed reference snapshots"
    return "Label sources and review status: see per-tile provenance and supplied run limitations"


def _validation_note(report):
    if _legacy_reference_report(report):
        return "Earlier feature selection means development validation, not an untouched final test."
    return "Spatial held-out evaluation; interpret with supplied run limitations. No independent final-test claim."


def _source_label(tile, report):
    record = next((r for r in (report or {}).get("tiles", []) if r["tile_id"] == tile["tile_id"]), {})
    default = "Reference snapshot" if _legacy_reference_report(report) else "Map labels"
    source = tile.get("source_label", record.get("source_label", default))
    review = tile.get("review_status", record.get("review_status"))
    if review is None:
        review = "unreviewed" if _legacy_reference_report(report) else "review status not supplied"
    return f"{source} / {tile['snapshot']}", str(review)


def _group_note(report):
    count = len(report["validation"]["folds"])
    return f"{count} held-out spatial groups. Spatially correlated cells; no tight confidence intervals are claimed."


def _edges(shape):
    """Original floor-cell edges, including empty intervals on sub-GRID rasters."""
    return tuple(np.arange(GRID + 1) * n // GRID for n in shape)


@lru_cache(maxsize=1)
def _font_path():
    # Only system font directories: no network, home-directory crawl or font install.
    roots = (Path("/System/Library/Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
             Path("/Library/Fonts"))
    for name in ("Avenir Next.ttc", "DejaVuSans.ttf", "Helvetica.ttc", "Arial.ttf"):
        for root in roots:
            path = root / name
            if path.is_file():
                return path
    return None


@lru_cache(maxsize=96)
def _font(size):
    path = _font_path()
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _fitted(draw, text, size, width):
    text = str(text)
    while width is not None and size > 18 and draw.textlength(text, font=_font(size)) > width:
        size -= 1
    if width is not None:
        while text and draw.textlength(text, font=_font(size)) > width:
            text = text[:-4] + "..." if len(text) > 4 else ""
    return text, size


def _text(draw, xy, text, size=34, fill=INK, width=None):
    text, size = _fitted(draw, text, size, width)
    draw.text(xy, text, font=_font(size), fill=fill, anchor="lt")


def _paragraph(draw, xy, text, width, size=30, fill=MUTED):
    x, y = xy
    line = ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=_font(size)) > width:
            _text(draw, (x, y), line, size, fill, width)
            y += int(size * 1.45)
            line = word
        else:
            line = trial
    if line:
        _text(draw, (x, y), line, size, fill, width)
        y += int(size * 1.45)
    return y


def _page(size, kicker, title, subtitle, report=None):
    image = Image.new("RGB", size, PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 65, 150, 74), fill=TEAL)
    _text(draw, (175, 53), kicker.upper(), 30, TEAL, size[0] - 255)
    _text(draw, (80, 123), title, 76, INK, size[0] - 160)
    _paragraph(draw, (80, 225), subtitle, size[0] - 160, 32)
    draw.line((80, size[1] - 155, size[0] - 80, size[1] - 155), fill=RULE, width=3)
    _text(draw, (80, size[1] - 115), _restriction(report), 30, TEAL)
    footer = (_review_note(report) + " / Development validation, not an untouched final test"
              if _legacy_reference_report(report) else _review_note(report))
    _text(draw, (80, size[1] - 65), footer, 27, MUTED, size[0] - 160)
    return image


def _spacing(tile):
    a, b, _, d, e, _ = tile["grid"]["transform"][:6]
    return math.hypot(a, d), math.hypot(b, e)


def _distance(metres):
    return f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"


def _cell_size_text(tile):
    rows, cols = _edges(tile["reference"].shape)
    sx, sy = _spacing(tile)
    col_widths, row_heights = np.diff(cols), np.diff(rows)
    widths = sorted(set(col_widths[col_widths > 0] * sx))
    heights = sorted(set(row_heights[row_heights > 0] * sy))
    def span(values):
        return "/".join(f"{v:g}" for v in values)
    return f"{span(widths)} m across x {span(heights)} m along rows"


def _scale_bar(source_width, display_width, pixel_metres):
    """Projected distance and display length, accounting for the actual resize."""
    maximum = source_width * pixel_metres * 0.27
    power = 10 ** math.floor(math.log10(maximum))
    metres = max(n * power for n in (0.1, 0.2, 0.5, 1, 2, 5) if n * power <= maximum)
    return metres, metres / pixel_metres * display_width / source_width


def _panel(page, box, title, note, picture, tile):
    draw = ImageDraw.Draw(page)
    x, y, right, bottom = box
    draw.rounded_rectangle(box, radius=24, fill=WHITE, outline=RULE, width=2)
    _text(draw, (x + 28, y + 25), title, 43, INK, right - x - 56)
    _text(draw, (x + 28, y + 86), note, 28, MUTED, right - x - 56)
    sx, sy = _spacing(tile)
    source_w, source_h = picture.size
    available_w, available_h = right - x - 48, bottom - y - 205
    factor = min(available_w / (source_w * sx), available_h / (source_h * sy))
    size = (max(1, round(source_w * sx * factor)), max(1, round(source_h * sy * factor)))
    left, top = x + (right - x - size[0]) // 2, y + 135 + (available_h - size[1]) // 2
    # Nearest neighbours preserve categorical/cell boundaries and missing masks.
    page.paste(picture.resize(size, Image.Resampling.NEAREST), (left, top))
    metres, length = _scale_bar(source_w, size[0], sx)
    bar_right, bar_y = right - 35, bottom - 30
    draw.line((bar_right - length, bar_y, bar_right, bar_y), fill=INK, width=7)
    for bx in (bar_right - length, bar_right):
        draw.line((bx, bar_y - 8, bx, bar_y + 8), fill=INK, width=3)
    _text(draw, (bar_right - length, bar_y - 42), _distance(metres), 26)


def _cell_predictions(tile):
    values = np.asarray(tile["cell_prediction"])
    if values.shape == (GRID, GRID):
        return values
    if values.ndim == 1 and len(values) == int(np.count_nonzero(tile["eligible"])):
        full = np.full((GRID, GRID), 255, np.uint8)
        full[tile["eligible"]] = values
        return full
    raise ValueError(f"{tile['tile_id']}: cell_prediction must be a {GRID} x {GRID} grid or an eligible-cell vector")


def _cell_comparison(tile):
    predicted = _cell_predictions(tile)
    scored = tile["eligible"] & np.isin(tile["cells"], (1, 2, 3)) & np.isin(predicted, (1, 2, 3))
    equal = tile["cells"] == predicted
    n = int(np.count_nonzero(scored))
    agreement = float(np.count_nonzero(scored & equal) / n) if n else None
    return scored, equal, n, agreement


def _hatch(rgb, mask):
    rows, cols = np.indices(mask.shape)
    stripe = mask & ((rows + cols) % max(8, min(mask.shape) // 45) < 2)
    rgb[stripe] = (0.65 * rgb[stripe] + 0.35 * np.array(MUTED)).astype(np.uint8)


def _class_picture(tile, values, present):
    shade = np.nan_to_num(tile["hillshade"], nan=0.65, posinf=1, neginf=0)
    shade = 0.78 + 0.22 * np.clip(shade, 0, 1)
    rgb = np.full((*values.shape, 3), UNKNOWN, dtype=np.uint8)
    for code, colour in CLASS_COLOURS.items():
        mask = values == code
        rgb[mask] = (np.array(colour) * shade[mask, None]).astype(np.uint8)
    rgb[~present] = WHITE
    return rgb


def _ramp(values):
    values = np.clip(np.nan_to_num(values, nan=0.0), 0, 1)
    low, middle, high = (np.array(c, dtype=float) for c in RAMP)
    left = low + np.minimum(values * 2, 1)[..., None] * (middle - low)
    right = middle + np.maximum(values * 2 - 1, 0)[..., None] * (high - middle)
    return np.where((values <= 0.5)[..., None], left, right).astype(np.uint8)


def _map_images(tile):
    """Native-grid RGB panels; reference coverage is independent of public coverage."""
    shape = tile["reference"].shape
    scored, equal, _, _ = _cell_comparison(tile)
    scored_pixels = per_pixel(scored, shape)
    equal_pixels = per_pixel(equal, shape)
    present = tile["valid"] & np.isin(tile["prediction"], (1, 2, 3))
    reference = _class_picture(tile, tile["reference"], np.ones(shape, bool))
    prediction = _class_picture(tile, tile["prediction"], present)
    _hatch(prediction, present & ~scored_pixels)
    difference = np.full((*shape, 3), UNSCORED, dtype=np.uint8)
    difference[scored_pixels & equal_pixels] = AGREE
    difference[scored_pixels & ~equal_pixels] = DISAGREE
    _hatch(difference, present & ~scored_pixels)
    difference[~present] = WHITE
    confidence = _ramp((tile["confidence"] - 1 / 3) / (1 - 1 / 3))
    finite = present & np.isfinite(tile["confidence"])
    _hatch(confidence, finite & ~scored_pixels)
    confidence[~finite] = WHITE
    return tuple(Image.fromarray(rgb) for rgb in (reference, prediction, difference, confidence))


def _swatch(draw, xy, colour, text, size=30, hatch=False):
    x, y = xy
    draw.rounded_rectangle((x, y, x + 36, y + 36), radius=5, fill=colour, outline=RULE)
    if hatch:
        for step in (8, 20, 32):
            draw.line((x + step, y + 2, x + 2, y + step), fill=MUTED, width=2)
    _text(draw, (x + 52, y + 1), text, size)


def _tile_figure(tile, report=None):
    _, _, n, agreement = _cell_comparison(tile)
    value = f"{agreement:.1%}" if agreement is not None else "not available"
    source, review = _source_label(tile, report)
    image = _page((2800, 3300), "Map labels / out-of-fold comparison", str(tile["tile_id"]),
                  f"Map {tile['map_id']}  |  Spatial group {tile['group']} held out  |  Snapshot {tile['snapshot']}  |  CELL agreement {value} (n = {n:,})", report)
    panels = _map_images(tile)
    notes = (
        (f"A  {source}", f"{review}; full label extent retained"),
        ("B  Out-of-fold prediction", "Supplied held-out output; hatching = outside cell scoring"),
        ("C  CELL agreement / disagreement", f"Original {GRID} x {GRID} cells; not geological error"),
        ("D  Model confidence", "Maximum class probability; uncalibrated"),
    )
    for i, (picture, (title, note)) in enumerate(zip(panels, notes)):
        x, y = 80 + (i % 2) * 1350, 350 + (i // 2) * 1270
        _panel(image, (x, y, x + 1290, y + 1210), title, note, picture, tile)
    draw = ImageDraw.Draw(image)
    for x, code in zip((80, 835, 1590), (1, 2, 3)):
        _swatch(draw, (x, 2870), CLASS_COLOURS[code], f"{code}  {CLASS_LABELS[code]}", 33)
    for x, colour, label, hatch in (
        (80, AGREE, "Cell agrees", False), (520, DISAGREE, "Cell differs", False),
        (975, UNSCORED, "Unscored", True), (1390, WHITE, "Missing output", False),
        (1970, UNKNOWN, "Unknown label", False),
    ):
        _swatch(draw, (x, 2940), colour, label, 29, hatch)
    gradient = Image.fromarray(_ramp(np.linspace(0, 1, 440)[None, :])).resize((440, 28))
    image.paste(gradient, (80, 3020))
    _text(draw, (80, 3060), "1/3", 26)
    _text(draw, (460, 3060), "1", 26)
    _text(draw, (560, 3017), "Fixed probability scale; not a reliability measure", 29, MUTED)
    _text(draw, (80, 3110), f"Prediction grid: {_distance(_spacing(tile)[0])} pixels, not verified boundary accuracy. Class 3 is not PSR / ice.", 29, INK, 2640)
    return image


def _representative(tiles):
    # This deterministic choice cannot cherry-pick agreement or confidence.
    return min(tiles, key=lambda t: (-float(np.mean(t["valid"] & np.isfinite(t["X"]).all(axis=0))), str(t["tile_id"])))


def _feature_picture(values, valid):
    finite = valid & np.isfinite(values)
    if not finite.any():
        return Image.new("RGB", values.shape[::-1], WHITE), None
    lo, hi = map(float, np.percentile(values[finite], (2, 98)))
    scaled = np.full(values.shape, 0.5, np.float32)
    if hi > lo:
        scaled = (values - lo) / (hi - lo)
    rgb = _ramp(scaled)
    rgb[~finite] = WHITE
    return Image.fromarray(rgb), (lo, hi)


def _feature_figure(tile, names, report=None):
    image = _page((3000, 2600), "Six-feature terrain view", f"Feature montage / {tile['tile_id']}",
                  "Illustrative tile selected by greatest finite six-feature public coverage; ties by tile ID. No score-based selection.", report)
    limits = []
    for i, name in enumerate(names):
        picture, extent = _feature_picture(tile["X"][i], tile["valid"])
        limits.append(extent)
        x, y = 80 + i % 3 * 960, 365 + i // 3 * 930
        note = "No finite public data" if extent is None else f"P02 {extent[0]:.4g}   /   P98 {extent[1]:.4g}   (input units)"
        _panel(image, (x, y, x + 920, y + 875), f"{i + 1}  {FEATURE_LABELS.get(name, name)}", note, picture, tile)
    draw = ImageDraw.Draw(image)
    gradient = Image.fromarray(_ramp(np.linspace(0, 1, 570)[None, :])).resize((570, 34))
    image.paste(gradient, (80, 2260))
    _text(draw, (690, 2258), "Low to high, independently scaled per feature; 2nd-98th percentiles clipped for display only", 30, MUTED, 2210)
    _paragraph(draw, (80, 2340), "White = missing public terrain or nonfinite feature. Constant channels use the ramp midpoint. These are supplied feature values, not reconstructed geological truth.", 2840, 30)
    return image, limits


def _number(value, percent=False):
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def _metric_value(metrics, key):
    if not metrics or not metrics.get("n", 0):
        return None
    return metrics.get(key)


class _MetricCanvas:
    """Small shared drawing vocabulary for identical PNG and true vector SVG scores."""
    def __init__(self, width, height, report):
        self.image = Image.new("RGB", (width, height), PAPER)
        self.draw = ImageDraw.Draw(self.image)
        description = f"{_restriction(report)}. {_review_note(report)}. {_validation_note(report)} No confidence intervals are shown."
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width / DPI}in" height="{height / DPI}in" viewBox="0 0 {width} {height}" role="img">',
                    '<title>Cell-level spatial held-out evaluation by group and class</title>',
                    f'<desc>{escape(description)}</desc>']
        self.rect((0, 0, width, height), PAPER)

    @staticmethod
    def colour(rgb):
        return "#" + "".join(f"{channel:02x}" for channel in rgb)

    def rect(self, box, fill, radius=0):
        x, y, right, bottom = box
        self.draw.rounded_rectangle(box, radius=radius, fill=fill)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{right - x}" height="{bottom - y}" rx="{radius}" fill="{self.colour(fill)}"/>')

    def text(self, x, y, text, size=32, fill=INK, width=None):
        text, size = _fitted(self.draw, text, size, width)
        _text(self.draw, (x, y), text, size, fill)
        self.svg.append(f'<text x="{x}" y="{y}" dominant-baseline="text-before-edge" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{self.colour(fill)}">{escape(text)}</text>')


def _score_figures(report):
    validation = report["validation"]
    aggregate = validation["aggregate"]
    folds = sorted(validation["folds"], key=lambda f: str(f["group"]))
    rows = [(str(f["group"]), f["metrics"], f.get("baseline")) for f in folds]
    rows.append(("Pooled OOF cells", aggregate, validation.get("baseline_aggregate")))
    table_bottom = 735 + len(rows) * 148
    detail_top = table_bottom + 95
    height = max(2400, detail_top + 990)
    canvas = _MetricCanvas(2800, height, report)
    canvas.text(80, 65, "CELL-LEVEL / SPATIAL HELD-OUT EVALUATION", 31, TEAL)
    canvas.text(80, 135, "Scores by spatial group + class", 78)
    canvas.text(80, 248, str(validation["method"]), 33, MUTED, 2640)
    for i, (label, value, percent) in enumerate((
        ("POOLED CELL AGREEMENT", _metric_value(aggregate, "accuracy"), True),
        ("POOLED MACRO F1", _metric_value(aggregate, "macro_f1"), False),
        ("EQUAL-GROUP MEAN MACRO F1", validation.get("group_mean_macro_f1"), False),
    )):
        x = 80 + i * 900
        canvas.rect((x, 340, x + 850, 570), WHITE, 22)
        canvas.text(x + 30, 370, label, 28, TEAL, 790)
        canvas.text(x + 30, 427, _number(value, percent), 81)
    columns = (100, 605, 880, 1215, 1540, 1880, 2160, 2440)
    headings = ("Held-out spatial group", "Cells", "Agreement", "Macro F1", "Baseline F1", "Class 1 F1", "Class 2 F1", "Class 3 F1")
    for x, heading in zip(columns, headings):
        canvas.text(x, 650, heading, 29, MUTED)
    for index, (group, metrics, baseline) in enumerate(rows):
        y = 730 + index * 148
        canvas.rect((80, y - 15, 2720, y + 116), WHITE if index % 2 == 0 else (240, 243, 239), 12)
        values = [group, f"{metrics.get('n', 0):,}" if metrics else "--",
                  _number(_metric_value(metrics, "accuracy"), True),
                  _number(_metric_value(metrics, "macro_f1")),
                  _number(_metric_value(baseline, "macro_f1"))]
        for code in (1, 2, 3):
            score = (metrics or {}).get("per_class", {}).get(str(code), {})
            values.append(_number(score.get("f1") if score.get("support", 0) else None))
        for i, (x, value) in enumerate(zip(columns, values)):
            canvas.text(x, y + 15, value, 34, CLASS_COLOURS[i - 4] if i >= 5 else INK,
                        (columns[i + 1] - x - 25) if i < 7 else 250)
        macro = _metric_value(metrics, "macro_f1")
        if macro is not None and np.isfinite(macro):
            canvas.rect((1215, y + 75, 1470, y + 84), RULE, 4)
            if macro > 0:
                canvas.rect((1215, y + 75, 1215 + 255 * np.clip(macro, 0, 1), y + 84), TEAL, 4)
    canvas.text(80, detail_top, "Pooled class metrics", 46)
    headers = ((80, "Reference class"), (680, "Precision"), (905, "Recall"), (1120, "F1"), (1305, "IoU"), (1510, "Support"))
    for x, text in headers:
        canvas.text(x, detail_top + 90, text, 27, MUTED)
    for index, code in enumerate((1, 2, 3)):
        y = detail_top + 158 + index * 133
        values = (aggregate or {}).get("per_class", {}).get(str(code), {})
        canvas.rect((80, y, 94, y + 70), CLASS_COLOURS[code], 4)
        canvas.text(115, y + 12, CLASS_LABELS[code], 31, INK, 525)
        for x, key in ((680, "precision"), (905, "recall"), (1120, "f1"), (1305, "iou")):
            canvas.text(x, y + 12, _number(values.get(key) if values.get("support", 0) else None), 31)
        canvas.text(1510, y + 12, f"{values['support']:,}" if "support" in values else "--", 31)
    canvas.text(1850, detail_top, "Confusion / cell counts", 41, INK, 870)
    canvas.text(1850, detail_top + 65, "Rows: reference  |  Columns: prediction", 26, MUTED)
    cm = np.asarray((aggregate or {}).get("confusion_matrix", np.zeros((3, 3))), dtype=float)
    maximum = float(cm.max()) if cm.size else 0
    for col in range(3):
        canvas.text(2015 + col * 225, detail_top + 122, str(col + 1), 31, CLASS_COLOURS[col + 1])
    for row in range(3):
        canvas.text(1880, detail_top + 210 + row * 115, str(row + 1), 31, CLASS_COLOURS[row + 1])
        for col in range(3):
            x, y = 1975 + col * 225, detail_top + 180 + row * 115
            t = float(cm[row, col] / maximum) if maximum else 0
            colour = tuple(round(a + (b - a) * t) for a, b in zip((236, 241, 236), TEAL))
            canvas.rect((x, y, x + 207, y + 99), colour, 8)
            canvas.text(x + 24, y + 29, f"{int(cm[row, col]):,}" if aggregate else "--", 32, WHITE if t > 0.6 else INK, 163)
    notes = [
        _group_note(report),
        "Baseline = training-majority class in each fold. Zero-support classes display --; exact supplied values remain in the CSV/JSON.",
        "Class 3: Crater interiors / floor (legacy shadowed_floor), NOT PSR / ice. Recall is label recall, not crater-object recall.",
        _validation_note(report),
    ]
    for i, note in enumerate(notes):
        canvas.text(80, detail_top + 610 + i * 52, note, 28, MUTED, 2640)
    canvas.rect((80, height - 145, 2720, height - 142), RULE)
    canvas.text(80, height - 104, _restriction(report), 31, TEAL)
    canvas.text(80, height - 57, f"{_review_note(report)}. Metrics come from the report, not rendered pixels.", 27, MUTED, 2640)
    return canvas.image, "\n".join(canvas.svg + ["</svg>"])


def _lunar_crs(grid):
    crs = CRS.from_wkt(grid["crs_wkt"])
    sphere = re.search(r'(?:SPHEROID|ELLIPSOID)\s*\[\s*"[^"]+"\s*,\s*([\d.eE+-]+)', crs.to_wkt())
    params = crs.to_dict()
    if (not crs.is_projected or crs.linear_units != "metre" or sphere is None
            or not 1_700_000 < float(sphere[1]) < 1_800_000
            or params.get("proj") != "stere" or abs(float(params.get("lat_0", 0))) != 90):
        raise ValueError("Reference footprints require a metre-based lunar polar stereographic CRS; no Earth/global fallback")
    return crs


def _footprints(tiles):
    target = _lunar_crs(tiles[0]["grid"])
    footprints = []
    with rasterio.Env(PROJ_NETWORK="OFF"):
        for tile in tiles:
            grid = tile["grid"]
            source = _lunar_crs(grid)
            h, w = grid["shape"]
            a, b, c, d, e, f = grid["transform"][:6]
            # Densify raster edges before transporting between different polar grids.
            corners = ((0, 0), (w, 0), (w, h), (0, h), (0, 0))
            pixel = [((1 - t) * x0 + t * x1, (1 - t) * y0 + t * y1)
                     for (x0, y0), (x1, y1) in pairwise(corners)
                     for t in np.linspace(0, 1, 17)]
            xs = [a * x + b * y + c for x, y in pixel]
            ys = [d * x + e * y + f for x, y in pixel]
            if source != target:
                transported = transform_coordinates(source, target, xs, ys)
                xs, ys = transported[0], transported[1]
            points = np.column_stack((xs, ys))
            if not np.isfinite(points).all():
                raise ValueError(f"Nonfinite lunar footprint: {tile['tile_id']}")
            footprints.append(points)
    return target, footprints


def _plot_footprints(draw, box, footprints, indices, colours, ticks=True):
    x, y, right, bottom = box
    all_points = np.concatenate([footprints[i] for i in indices])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    middle = (low + high) / 2
    span = max(float(np.max(high - low)), 1.0) * 1.22
    scale = min(right - x, bottom - y) / span
    cx, cy = (x + right) / 2, (y + bottom) / 2
    draw.rectangle(box, fill=WHITE, outline=RULE, width=2)
    if ticks:
        decimals = max(2, math.ceil(-math.log10(span / 5000)))
        for fraction in np.linspace(-0.4, 0.4, 5):
            px, py = cx + fraction * span * scale, cy - fraction * span * scale
            draw.line((px, y, px, bottom), fill=RULE, width=2)
            draw.line((x, py, right, py), fill=RULE, width=2)
            _text(draw, (px - 62, bottom + 22), f"{(middle[0] + fraction * span) / 1000:.{decimals}f}", 25)
            _text(draw, (x - 128, py - 14), f"{(middle[1] + fraction * span) / 1000:.{decimals}f}", 25, MUTED, 116)
    for i in indices:
        points = [(cx + (p[0] - middle[0]) * scale, cy - (p[1] - middle[1]) * scale) for p in footprints[i]]
        draw.polygon(points, fill=tuple(round(0.15 * c + 0.85 * 255) for c in colours[i]))
        draw.line(points + [points[0]], fill=colours[i], width=4)
        # Numbered leaders remain visible even for tiny raster windows in the overview.
        px = sum(p[0] for p in points) / len(points)
        py = sum(p[1] for p in points) / len(points)
        offset = 26 + 32 * (i % 3)
        draw.line((px, py, px + offset, py - offset), fill=colours[i], width=2)
        draw.rounded_rectangle((px + offset - 5, py - offset - 5, px + offset + 42, py - offset + 35), radius=6, fill=WHITE)
        _text(draw, (px + offset, py - offset), str(i + 1), 27, colours[i])
    return span


def _location_figure(tiles, geometry, report=None):
    crs, footprints = geometry
    groups = sorted({str(t["group"]) for t in tiles})
    height = max(2600, 560 + len(groups) * 600)
    image = _page((3000, height), "Selected-location footprint atlas / no basemap", "Local lunar polar coordinates",
                  f"{len(tiles)} raster windows  /  {len({t['map_id'] for t in tiles})} map groups  /  {len(groups)} spatial groups. Outlines are raster footprints, not geological vectors.", report)
    draw = ImageDraw.Draw(image)
    palette = (TEAL, (71, 96, 133), (153, 111, 80), (106, 117, 85))
    colours = [palette[groups.index(str(tile["group"])) % len(palette)] for tile in tiles]
    _plot_footprints(draw, (225, 465, 1845, 2085), footprints, list(range(len(tiles))), colours)
    _text(draw, (225, 365), "Overview  /  equal axis scale", 40)
    _text(draw, (225, 2170), "Polar easting (km); left axis: polar northing (km)", 32, MUTED, 1660)
    params = crs.to_dict()
    _paragraph(draw, (225, 2240), f"Lunar polar stereographic; central meridian {params.get('lon_0', 0):g} degrees; latitude of origin {params['lat_0']:g} degrees. Actual affine-grid edges. No invented north arrow or global basemap.", 1630, 29)
    card_height = min(590, (height - 600) // max(1, len(groups)))
    for index, group in enumerate(groups):
        y = 365 + index * card_height
        indices = [i for i, tile in enumerate(tiles) if str(tile["group"]) == group]
        _text(draw, (2000, y), group, 40, palette[index % len(palette)], 900)
        _plot_footprints(draw, (2030, y + 70, 2770, y + 330), footprints, indices, colours, ticks=False)
        label = "; ".join(f"{i + 1}: {tiles[i]['tile_id']}" for i in indices)
        _paragraph(draw, (2000, y + 355), label, 910, 26)
        centres = np.array([(footprints[i].min(axis=0) + footprints[i].max(axis=0)) / 2 for i in indices])
        _text(draw, (2000, y + 495), f"Window-centre mean E {centres[:, 0].mean() / 1000:.2f} / N {centres[:, 1].mean() / 1000:.2f} km", 26, MUTED, 910)
    return image


def _validate(tiles, report):
    if not tiles:
        raise ValueError("At least one selected-location tile is required")
    if "restricted" in report and not isinstance(report["restricted"], (bool, np.bool_)):
        raise ValueError("report['restricted'] must be boolean when supplied")
    if len(report["feature_names"]) != 6:
        raise ValueError("The poster montage requires six ordered feature_names")
    ids = [str(t["tile_id"]) for t in tiles]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate tile_id")
    folds = report["validation"]["folds"]
    if len({f["group"] for f in folds}) != len(folds):
        raise ValueError("Expected one held-out fold per geography group")
    for fold in folds:
        held_out = {t["tile_id"] for t in tiles if t["group"] == fold["group"]}
        if held_out.intersection(fold["train_tiles"]):
            raise ValueError("A tile from the held-out geography appears in training")
    for tile in tiles:
        name = tile["tile_id"]
        shape = np.asarray(tile["reference"]).shape
        if len(shape) != 2 or min(shape) < 1 or tuple(tile["grid"]["shape"]) != shape:
            raise ValueError(f"{name}: raster shape must match grid and be nonempty on both axes")
        for key in ("hillshade", "prediction", "confidence", "valid", "support"):
            if np.asarray(tile[key]).shape != shape:
                raise ValueError(f"{name}: {key} shape does not match reference")
        if np.asarray(tile["X"]).shape != (6, *shape):
            raise ValueError(f"{name}: X must have shape (6, H, W)")
        for key in ("cells", "eligible"):
            if np.asarray(tile[key]).shape != (GRID, GRID):
                raise ValueError(f"{name}: {key} must be a {GRID} x {GRID} grid")
        for key in ("valid", "support", "eligible"):
            if np.asarray(tile[key]).dtype.kind != "b":
                raise ValueError(f"{name}: {key} must be boolean")
        if not np.isfinite(tile["grid"]["transform"][:6]).all() or min(_spacing(tile)) <= 0:
            raise ValueError(f"{name}: invalid affine pixel spacing")
        finite_confidence = tile["confidence"][np.isfinite(tile["confidence"])]
        if np.any((finite_confidence < 0) | (finite_confidence > 1)):
            raise ValueError(f"{name}: finite confidence must be in [0, 1]")
        _cell_predictions(tile)
        matches = [f for f in folds if name in f["test_tiles"]]
        if (len(matches) != 1 or matches[0]["group"] != tile["group"]
                or tile["group"] in matches[0]["train_groups"]
                or name in matches[0]["train_tiles"]):
            raise ValueError(f"{name}: missing or inconsistent held-out geography fold")

    for metrics in [report["validation"].get("aggregate"), *[f.get("metrics") for f in folds]]:
        if metrics is not None:
            cm = np.asarray(metrics["confusion_matrix"])
            if cm.shape != (3, 3) or not np.isfinite(cm).all() or (cm < 0).any():
                raise ValueError("Expected a finite, nonnegative 3 x 3 cell confusion matrix")


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 1000:
            raise ValueError("Report data must be metadata/metrics, not duplicated raster arrays")
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scores_csv(report):
    stream = io.StringIO(newline="")
    fields: list[str] = ["scope", "key", "estimate", "n", "accuracy", "macro_f1", "class_id", "class_label",
                         "precision", "recall", "f1", "iou", "support"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    validation = report["validation"]
    rows = [("aggregate", "all", "OOF", validation.get("aggregate")),
            ("aggregate", "all", "training-majority", validation.get("baseline_aggregate"))]
    for fold in validation["folds"]:
        rows += [("group", fold["group"], "OOF", fold.get("metrics")),
                 ("group", fold["group"], "training-majority", fold.get("baseline"))]
    rows += [("tile", tile["tile_id"], "OOF", tile.get("metrics")) for tile in report["tiles"]]
    for scope, key, estimate, metrics in rows:
        if not metrics:
            continue
        for code in (1, 2, 3):
            writer.writerow(dict(scope=scope, key=key, estimate=estimate,
                                 **{k: metrics.get(k) for k in ("n", "accuracy", "macro_f1")},
                                 class_id=code, class_label=CLASS_LABELS[code],
                                 **{k: metrics.get("per_class", {}).get(str(code), {}).get(k)
                                    for k in ("precision", "recall", "f1", "iou", "support")}))
    return stream.getvalue()


def _captions(tiles, report, names, representative, feature_limits, geometry):
    validation = report["validation"]
    restricted = report.get("restricted", True)
    permission = ("Do not publish, upload, or share these figures, captions, or report data without the required permission. Restrictions may arise from inputs or model seeds."
                  if restricted else "Review source licences, label provenance, and supplied limitations before publication. A local export does not certify publication rights or scientific validity.")
    lines = ["# Poster assets — selected-location maps and spatial evaluation", "", f"**{_restriction(report)}.**",
             permission + " PNGs are tagged 300 dpi; resizing a figure does not create scientific spatial resolution.", "",
             f"Run: `{report['run_id']}`. Created: {report['created_at']}. Model revision: `{report.get('model_revision') or 'fresh fold models; see fold revisions'}`.",
             f"Inventory supplied to exporter: {len(tiles)} tiles, {len({t['map_id'] for t in tiles})} map groups, {len({t['group'] for t in tiles})} spatial groups.", "",
             "## Interpretation and provenance", "",
             ("- All reference snapshots are unreviewed, provisional coarse-cell annotations, not an authoritative geological map."
              if _legacy_reference_report(report) else "- Label snapshots and any supplied source/review status are listed per tile. Saved, draft, or manual does not by itself certify independent review or geological correctness."),
             "- " + _validation_note(report),
             "- " + _group_note(report),
             "- Fold membership follows the supplied group field, not map names or hardcoded site aliases. Each tile has one matching held-out fold; tiles from that group cannot occur in its training set.",
             ("- There is no full-resolution geological truth. Missing reference vectors are not reconstructed. Footprints describe raster windows, not geologic boundaries."
              if _legacy_reference_report(report) else "- These figures do not establish full-resolution geological truth. Missing reference vectors are not reconstructed. Footprints describe raster windows, not geologic boundaries."),
             "- Prediction pixel spacing is not verified boundary accuracy. Cell disagreement is not geological error, and class recall is not crater-object recall. No timing, crater-detection, or ice-detection claim is made.",
             "- Confidence means supplied maximum class probability; probabilities are uncalibrated. The fixed ramp spans 1/3 to 1, not empirical reliability.",
             "- Class 1: Smooth highlands (green); class 2: Rough highlands (orange); class 3: **Crater interiors / floor** (navy), legacy `shadowed_floor`, **NOT PSR / ice**. This is a poster-specific palette, not a schema change.",
             "- Reference labels remain visible outside public terrain coverage. Unknown reference labels are warm grey. Prediction, difference and confidence are white where public terrain or the relevant output is missing. Hatching marks available output outside scored cells; unscored is not agreement. Missing confidence is independently white.",
             "- Low-support public terrain is not silently removed: the supplied support mask is not an extra evaluation or display exclusion. Eligibility is supplied by the run.",
             f"- Cell comparisons use the supplied {GRID} x {GRID} target and cell_prediction on eligible known cells, rendered by core.per_pixel with exact floor-cell boundaries. This matches benchmark_data._expand on rasters at least 120 pixels per axis and also supports smaller rasters. Raster predictions are never used to invent cell agreement.",
             "- On tiny rasters, some original painting-cell intervals have zero width or height. They are not shifted or merged; size summaries list positive-width rasterized intervals only. Supplied cell geometry, eligibility and metrics are not changed.",
             "- Map panels preserve aspect ratio and use nearest-neighbour display resampling. Scale bars use affine column spacing and actual displayed width: projected-grid distances, not geodesic distances. There are no north arrows on the polar projection.",
             "- Only supplied arrays are rendered; no models are fitted, no pipeline is run, and no external data, basemap, service, or network resource is used.", "", "## Spatial holdouts", ""]
    for fold in sorted(validation["folds"], key=lambda f: str(f["group"])):
        maps = sorted({str(t["map_id"]) for t in tiles if t["group"] == fold["group"]})
        lines.append(f"- Group `{fold['group']}`: maps {' + '.join(maps)} held out together; test tiles: {', '.join(map(str, fold['test_tiles']))}; training groups: {', '.join(map(str, fold['train_groups'])) or '(none)'}. ")
    lines += ["", "## Tile comparisons", ""]
    for tile, name in zip(tiles, names):
        _, _, n, agreement = _cell_comparison(tile)
        source, review = _source_label(tile, report)
        lines += [f"### {name}", "",
                  f"**{tile['tile_id']}** — map `{tile['map_id']}`, site `{tile['site_id']}`, held-out spatial group `{tile['group']}`, snapshot `{tile['snapshot']}`.",
                  f"Label source: {source}; review status: {review}.",
                  f"A: full supplied labels. B: supplied OOF pixel prediction. C: CELL agreement {_number(agreement, True)} across {n:,} scored cells (teal agrees, muted red differs). D: uncalibrated maximum class probability.",
                  f"Grid {tile['reference'].shape[0]} x {tile['reference'].shape[1]}; affine pixel spacing {_spacing(tile)[0]:g} x {_spacing(tile)[1]:g} m. Original {GRID} x {GRID} painting grid: positive-width rasterized cells are {_cell_size_text(tile)}. Prediction pixels are not independently verified boundaries.",
                  f"Public-valid grid pixels: {int(np.count_nonzero(tile['valid'])):,}; public-valid pixels outside supplied support: {int(np.count_nonzero(tile['valid'] & ~tile['support'])):,}. These counts describe coverage, not accuracy.", ""]
    lines += ["## feature-montage.png", "",
              f"Illustrative representative tile: **{representative['tile_id']}**. Selection rule: greatest fraction of grid pixels with public-valid terrain and all six finite features, ties by lexical tile ID; no metrics or confidence enter selection. It is not a claim of statistically representative geology.",
              "Each panel shows the actual supplied feature array in report.feature_names order. Independent 2nd–98th percentile display ranges use finite public terrain only, in the input feature's units. Outliers are clipped for display, constant channels use the midpoint, and all-missing channels stay white. Values are not modified.", ""]
    for name, limits in zip(report["feature_names"], feature_limits):
        lines.append(f"- `{name}`: " + ("no finite public values." if limits is None else f"P02 {limits[0]:.8g}; P98 {limits[1]:.8g}."))
    lines += ["", "## scores-by-group-class.png / .svg", "",
              f"Method: {validation['method']}. Pooled accuracy is labelled CELL agreement. Pooled macro F1 and equal-group mean macro F1 are distinct quantities, read directly from the report. Baseline is each fold's training-majority class. Class support and confusion entries count original cells, not raster pixels. Zero-support classes are marked -- in the figure; supplied metric values are preserved in scores.csv and report.json.", "",
              "## reference-footprints.png", "",
              "Actual affine-grid perimeter coordinates, including rotation, are transported offline to the first tile's lunar polar stereographic CRS if necessary. The overview has equal x/y scale; group insets are independently zoomed and should not be compared for size. Coordinates are projected easting/northing in kilometres, not an Earth CRS, invented globe, or reconstructed vector map.", "",
              f"Coordinate system: `{geometry[0].to_string()}`. The full WKT remains in report.json.", ""]
    for tile, points in zip(tiles, geometry[1]):
        low, high = points.min(axis=0), points.max(axis=0)
        lines.append(f"- `{tile['tile_id']}`: E {low[0]:.3f} to {high[0]:.3f} m; N {low[1]:.3f} to {high[1]:.3f} m.")
    lines += ["", "## Supplied run limitations", ""]
    lines += [f"- {item}" for item in report.get("limitations", [])]
    if "sources" in report:
        lines += ["", "## Supplied sources", "", "```json", json.dumps(_json_value(report["sources"]), indent=2, ensure_ascii=False), "```"]
    lines += ["", "## Supplied policies", "", "```json", json.dumps(_json_value(report.get("policies", {})), indent=2, ensure_ascii=False), "```", "",
              "## Bundle contents", "", "Each generated figure appears once. captions.md, report.json and scores.csv contain captions and supplied report metadata/metrics only. No source raster, feature tensor, prediction array, model artifact, or pre-existing run file is copied into the bundle.", ""]
    return "\n".join(lines)


def export_figures(tiles, report, directory):
    """Write poster PNGs (300 dpi), vector scores and a local ZIP.

    Returns {"files": [{"path": "figures/<name>", "label": ...}, ...],
             "bundle": "poster-assets.zip"}, with paths relative to directory.parent.
    Files: tile-NN-<safe-tile-id>.png (all tiles, lexical order), feature-montage.png,
    scores-by-group-class.png/.svg, reference-footprints.png; captions.md, report.json
    and scores.csv are also written in directory and included at the ZIP root.
    Pass poster/<run>/figures to keep assets under the top-level poster folder;
    the exporter neither redirects paths nor moves/copies existing run artifacts.

    report['restricted'] controls publication notices (default True for legacy
    private-reference reports). Explicit True also covers restricted model seeds;
    it does not imply that user labels are unreviewed. Optional source_label and
    review_status on tiles or report tile records are printed verbatim alongside
    snapshot; otherwise generic reports make no review-status assumption.

    Inputs are read-only. Existing named outputs are never overwritten, and unrelated
    files are neither touched nor swept into the archive. The caller supplies OOF
    predictions; fold metadata is checked, but this exporter does not fit a model.
    """
    tiles = sorted(tiles, key=lambda t: str(t["tile_id"]))
    _validate(tiles, report)
    geometry = _footprints(tiles)
    representative = _representative(tiles)
    directory = Path(directory)
    names = [f"tile-{i:02d}-{re.sub(r'[^A-Za-z0-9_-]+', '-', str(t['tile_id'])).strip('-')[:90] or 'tile'}.png"
             for i, t in enumerate(tiles, 1)]
    extras = ["feature-montage.png", "scores-by-group-class.png", "scores-by-group-class.svg", "reference-footprints.png"]
    metadata = ["captions.md", "report.json", "scores.csv"]
    bundle = directory.parent / "poster-assets.zip"
    for path in [directory / name for name in names + extras + metadata] + [bundle]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing poster asset: {path}")
    report_json = json.dumps(_json_value(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name, label, image):
        image.save(directory / name, format="PNG", dpi=(DPI, DPI))
        files.append({"path": (directory / name).relative_to(directory.parent).as_posix(), "label": label})

    for tile, name in zip(tiles, names):
        save(name, f"{tile['tile_id']} — {tile['snapshot']} labels / OOF / cell agreement / confidence", _tile_figure(tile, report))
    montage, limits = _feature_figure(representative, report["feature_names"], report)
    save(extras[0], f"Six features — {representative['tile_id']}", montage)
    score_image, score_svg = _score_figures(report)
    save(extras[1], "Cell scores by held-out spatial group and class", score_image)
    (directory / extras[2]).write_text(score_svg, encoding="utf-8")
    files.append({"path": (directory / extras[2]).relative_to(directory.parent).as_posix(), "label": "Cell scores — editable vector SVG"})
    save(extras[3], "Selected-location raster footprints in lunar polar coordinates", _location_figure(tiles, geometry, report))
    (directory / "captions.md").write_text(_captions(tiles, report, names, representative, limits, geometry), encoding="utf-8")
    (directory / "report.json").write_text(report_json, encoding="utf-8")
    (directory / "scores.csv").write_text(_scores_csv(report), encoding="utf-8")
    with zipfile.ZipFile(bundle, "w") as archive:
        for item in files:
            path = directory.parent / item["path"]
            archive.write(path, item["path"], compress_type=zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED)
        for name in metadata:
            archive.write(directory / name, name, compress_type=zipfile.ZIP_DEFLATED)
    return {"files": files, "bundle": bundle.name}
