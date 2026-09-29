"""Paint maps from the professor's geologic maps.

Reads each reference map's geounits polygons, converts the source units to Selenograph
codes with UNITS, and writes the draft painting the painter opens. The three
ground-truth maps in data/reference_data/professor_geologic_maps each paint their own
tile when original vectors are available; an extra map (EXTRA) paints the tiles it covers. Dataset
exports are not touched: review each map in the app, then press Save labels + map. A
different painting already on a map is backed up first.
"""

import argparse
import functools
import json
import os
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine

from app import core, paths

REFERENCE_ROOT = paths.PROFESSOR_MAPS_DIR

# Source units (domuMS2.xlsx, DOMU v06272025jmh) -> Selenograph codes. Per site,
# because the numbers mean different things at different sites. Elephant hide is
# the inner wall of large degraded craters, so it joins their floors as the crater
# interior (3). The floors and inner slopes of single small (>30 m) fresh craters are
# crater interiors too: at full resolution the model learns them, and the test scores
# how many it finds. Their rims and ejecta are not converted. Unconverted units are
# never painted or scored.
SMALL = {"crater floor": 3, "crater inner slope": 3}
UNITS = {
    "MM026": {"1": 3, "2": 1, "3": 2, "4": 3, **SMALL},  # elephant hide, smooth, rough, shadowed
    "N1014": {"1": 3, "2": 1, "3": 2, **SMALL,           # degraded crater interior;
              "crate rfloor": 3},                        # "4" is not in the DOMU
    "N2005": {"1": 3, "2": 1, "3": 2, "4": 3, **SMALL},
    # The older Nobile-1 map (MS1, 2024; domuMS1.docx), just south-west of N1014. Degraded
    # crater interiors, Nobile's inner wall and Blackbird's interior are crater interiors,
    # and so are its small fresh craters (fcU); highland types C and B have low (to
    # intermediate) slope and ruggedness, type A intermediate to high. Blackbird's ejecta
    # is not converted.
    "Nobile1-MS1": {"dciA": 3, "dciB": 3, "dciU": 3, "dciN": 3, "fcBiA": 3, "fcBiB": 3,
                    "fcU": 3, "hC": 1, "hB": 1, "hA": 2},
}
SMALL_CRATERS = {*SMALL, "crate rfloor", "fcU"}    # units that are single small craters
# Extra vector coverage can be used on working tiles, but never locks them. Keep
# the fallback inside the self-contained reference collection, relative to its root.
EXTRA = {"Nobile1-MS1": Path("sites/nobile1-ms1/references/nobile1_ms1.gpkg")}
MIN_MAPPED = 0.5   # paint a cell when converted units cover at least half its terrain
MIN_TILE = 0.05    # an extra map paints the tiles it covers at least this share of
SUB = 5            # rasterize polygons at 1 m inside each 5 m pixel


def wkb_polygons(blob, offset=0):
    """(Multi)Polygon WKB, with or without Z/M, as GeoJSON polygon coordinates."""
    order = "<" if blob[offset] == 1 else ">"
    kind, = struct.unpack_from(order + "I", blob, offset + 1)
    base, dims = kind % 1000, (2, 3, 3, 4)[kind // 1000]
    count, = struct.unpack_from(order + "I", blob, offset + 5)
    offset += 9
    if base == 6:
        polygons = []
        for _ in range(count):
            part, offset = wkb_polygons(blob, offset)
            polygons += part
        return polygons, offset
    if base != 3:
        raise ValueError(f"Unsupported geometry type {kind}; expected polygons")
    rings = []
    for _ in range(count):
        n, = struct.unpack_from(order + "I", blob, offset)
        points = np.frombuffer(blob, order + "f8", n * dims, offset + 4).reshape(n, dims)
        rings.append(points[:, :2].tolist())
        offset += 4 + 8 * n * dims
    return [rings], offset


def geounits(gpkg):
    """(unit label, GeoJSON geometry) for every non-empty geounits polygon."""
    with sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT unit, geom FROM geounits").fetchall()
    for unit, blob in rows:
        if blob is None or blob[3] & 0x10:               # GeoPackage empty geometry
            continue
        header = 8 + (0, 32, 48, 48, 64)[(blob[3] >> 1) & 7]
        yield (unit or "").strip(), dict(type="MultiPolygon",
                                         coordinates=wkb_polygons(blob, header)[0])


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def reference_records():
    """Preserved reference provenance, including snapshots whose vectors are absent."""
    root = Path(REFERENCE_ROOT)
    sites = {p.parent.name: _read_json(p) for p in sorted(root.glob("sites/*/site.json"))}
    out = []
    for path in sorted(root.glob("sites/*/tiles/*/tile.json")):
        record = _read_json(path)
        site_id = record.get("site_id")
        if not isinstance(site_id, str):
            site_id = path.parents[2].name
        site = sites.get(site_id, sites.get(path.parents[2].name, {}))
        key = site.get("map_id")
        if key in UNITS and record.get("legacy_name"):
            out.append(dict(record, reference_map=key))
    return out


def reference_layer(name, kind):
    """A preserved raster layer, resolved against the reference dataset root."""
    aliases = {"cpr": "radar-cpr"}
    for record in reference_records():
        if record["legacy_name"] != name:
            continue
        layers = record.get("layers", {})
        layer = layers.get(kind, layers.get(aliases.get(kind)))
        relative = layer.get("path") if isinstance(layer, dict) else layer
        if relative:
            path = Path(REFERENCE_ROOT) / relative
            if path.is_file():
                return str(path.resolve())
    return None


def source_map(key):
    """An original vector, never a reconstructed polygon from a painting snapshot."""
    for path in sorted(Path(REFERENCE_ROOT).glob("sites/*/site.json")):
        site = _read_json(path)
        if site.get("map_id") == key:
            vector = site.get("original_vector", {}).get("path")
            if vector and (Path(REFERENCE_ROOT) / vector).is_file():
                return Path(REFERENCE_ROOT) / vector
    if key in EXTRA:
        return Path(REFERENCE_ROOT) / EXTRA[key]
    # Missing originals remain missing; callers must check before using pixel truth.
    return Path(REFERENCE_ROOT) / "unavailable_vectors" / f"{key}.gpkg"


def tiles(key):
    """The prepared maps a reference map paints: a ground-truth map its own tile, an
    extra map every tile whose terrain it covers at least MIN_TILE of."""
    names = [r["legacy_name"] for r in reference_records() if r["reference_map"] == key]
    names += [core.dem_name(r["working_dem"]) for r in core.records() if r.get("site") == key]
    gpkg = source_map(key)
    if key in EXTRA and gpkg.is_file():
        listed = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                       for p in core.manifest_files())
        names += _covered(str(gpkg), gpkg.stat().st_mtime_ns, str(core.DEM_DIR), listed)
    return list(dict.fromkeys(names))


@functools.lru_cache(maxsize=8)
def _covered(gpkg, mtime, dem_dir, listed):
    shapes = [geom for _, geom in geounits(gpkg)]
    if not shapes:
        return ()
    xy = np.concatenate([np.asarray(ring) for g in shapes for poly in g["coordinates"]
                         for ring in poly])
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    out = []
    for r in core.records():
        path = core.record_path(r)
        if not path or not os.path.isfile(path):
            continue
        if "transform" in r and "size" in r:            # skip far tiles without opening them
            a, (H, W) = r["transform"], r["size"]
            left, top, right, bottom = a[2], a[5], a[2] + a[0] * W, a[5] + a[4] * H
        elif os.path.exists(path):
            with rasterio.open(path) as src:
                left, bottom, right, top = src.bounds
        else:
            continue
        if left > x1 or right < x0 or top < y0 or bottom > y1:
            continue
        with rasterio.open(path) as src:
            terrain = src.read_masks(1) > 0
            hit = rasterize(((g, 1) for g in shapes), out_shape=src.shape,
                            transform=src.transform, fill=0, dtype="uint8").astype(bool)
        if terrain.any() and (hit & terrain).sum() >= MIN_TILE * terrain.sum():
            out.append(core.dem_name(r["working_dem"]))
    return tuple(out)


def reference_for(name):
    """The map's reference provenance, independent of original-vector availability."""
    return next((key for key in UNITS if name in tiles(key)), None)


def locked_maps():
    """Lock preserved reference paintings, not new tiles merely under a vector footprint."""
    return ({r["legacy_name"] for r in reference_records()}
            | {core.dem_name(r["working_dem"]) for r in core.records()
               if r.get("site") in UNITS})


def vector_reference_for(name):
    """A reference suitable for polygon-based training/scoring, or None."""
    key = reference_for(name)
    return key if key and source_map(key).is_file() else None


def reference_pixels(dem_path, key):
    """A reference map on the tile's own pixels: unit codes (255 where it gives none: an
    unconverted unit, no polygon, or no terrain), the source unit of every pixel (an
    index, 0 where none) and where it maps a single small crater. Pixels count by their
    centres."""
    with rasterio.open(dem_path) as src:
        shape, transform = src.shape, src.transform
        terrain = src.read_masks(1) > 0
    features = list(geounits(source_map(key)))
    names = sorted({unit for unit, _ in features})
    source = rasterize(((geom, names.index(unit) + 1) for unit, geom in features),
                       out_shape=shape, transform=transform, fill=0, dtype="uint16")
    source[~terrain] = 0
    codes = np.array([255] + [UNITS[key].get(n, 255) for n in names], np.uint8)[source]
    small = np.array([False] + [n in SMALL_CRATERS for n in names])[source]
    return codes, source, small


def per_cell(mask, shape):
    """Count true 1 m pixels inside each painting cell, on the painter's own cell edges."""
    H, W = shape
    rows = [i * H // core.GRID * SUB for i in range(core.GRID)]
    cols = [j * W // core.GRID * SUB for j in range(core.GRID)]
    return np.add.reduceat(np.add.reduceat(mask, rows, 0, dtype=np.int32), cols, 1)


def paint(dem_path, gpkg, units):
    """Painting cells for one tile, plus the km2 of every source unit on it."""
    with rasterio.open(dem_path) as src:
        shape, transform, px = src.shape, src.transform, abs(src.transform.a)
        terrain = src.read_masks(1) > 0
    features = list(geounits(gpkg))
    names = sorted({unit for unit, _ in features})
    index = rasterize(((geom, names.index(unit) + 1) for unit, geom in features),
                      out_shape=(shape[0] * SUB, shape[1] * SUB), fill=0, dtype="uint8",
                      transform=transform * Affine.scale(1 / SUB))
    terrain = np.repeat(np.repeat(terrain, SUB, 0), SUB, 1)
    area = np.bincount(index[terrain], minlength=len(names) + 1)[1:] * (px / SUB) ** 2 / 1e6
    code = np.array([0] + [units.get(n, 0) for n in names], np.uint8)[index]
    counts = np.stack([per_cell((code == c) & terrain, shape) for c in core.CODES])
    ground = per_cell(terrain, shape)
    mapped = counts.sum(0)
    cells = np.where((ground > 0) & (mapped >= MIN_MAPPED * ground),
                     np.array(list(core.CODES), np.uint8)[counts.argmax(0)], 0)
    return cells.astype(np.uint8), dict(zip(names, area.round(3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be painted without writing drafts")
    parser.add_argument("--map", action="append", choices=list(UNITS), dest="maps",
                        help="Only this reference map (repeatable); all of them by default")
    args = parser.parse_args()

    jobs = [(key, name) for key in (args.maps or UNITS) for name in tiles(key)]
    for key, name in jobs:
        gpkg = source_map(key)
        if not gpkg.exists():
            print(f"{name}: skipped, {gpkg} not found", flush=True)
            continue
        dem = core.dem_path(name)
        if not dem or not Path(dem).is_file():
            print(f"{name}: skipped, no working DEM installed", flush=True)
            continue
        cells, areas = paint(dem, gpkg, UNITS[key])
        painted = ", ".join(f"{core.CODES[c]} {int((cells == c).sum())}" for c in core.CODES)
        skipped = ", ".join(f"{u or '(no label)'} {a} km2" for u, a in areas.items()
                            if u not in UNITS[key] and a > 0)
        print(f"{name} ({key}): cells {painted}, unpainted {int((cells == 0).sum())}",
              flush=True)
        print(f"  not converted: {skipped or 'none'}", flush=True)
        previous = core.load_painting(name)
        if np.array_equal(previous, cells):
            print("  unchanged", flush=True)
            continue
        if previous.any():
            both = (previous > 0) & (cells > 0)
            agree = (previous[both] == cells[both]).mean() if both.any() else 0
            backup = core.output_file(name, f"before-reference-{time.strftime('%Y%m%dT%H%M%S')}.npy",
                                      final=False)
            print(f"  replaces a painting that agrees on {agree:.0%} of shared cells; "
                  f"backup {backup}", flush=True)
            if not args.dry_run:
                Path(backup).parent.mkdir(parents=True, exist_ok=True)
                np.save(backup, previous)
        if not args.dry_run:
            core.save_painting(name, cells)


if __name__ == "__main__":
    main()
