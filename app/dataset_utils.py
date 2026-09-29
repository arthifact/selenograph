"""Read-only raster, annotation and provenance helpers for dataset preparation."""
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds

GRID = 120
MIN_COVERAGE = 0.9
REGIONS = {"mons-mouton": "leibnitz_beta_plateau", "nobile1": "nobile_rim_1",
           "nobile1-ms1": "nobile_rim_1", "nobile2": "nobile_rim_2"}
GROUPS = {"mons-mouton": "mons-mouton", "nobile1": "nobile-1",
          "nobile1-ms1": "nobile-1", "nobile2": "nobile-2"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(root, relative):
    """Operational paths must remain inside their declared dataset, including symlinks."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


class _Hashes:
    def __init__(self):
        self.cache = {}
        self.records = {}

    @staticmethod
    def stream(stream):
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()

    def file(self, path, key, expected=None):
        path = Path(path).resolve()
        if path not in self.cache:
            with path.open("rb") as stream:
                self.cache[path] = self.stream(stream)
        actual = self.cache[path]
        if expected is not None and actual != expected:
            raise ValueError(f"SHA256 mismatch: {key}")
        previous = self.records.get(key, {})
        self.records[key] = {"sha256": actual,
                             "verified_expected": expected is not None or previous.get("verified_expected", False)}
        return actual


@contextmanager
def _timed(timings, group):
    start = perf_counter()
    try:
        yield
    finally:
        timings[group] = timings.get(group, 0.0) + perf_counter() - start


@dataclass(frozen=True)
class _Grid:
    shape: tuple[int, int]
    transform: rasterio.Affine
    crs: CRS

    @classmethod
    def read(cls, src):
        if src.crs is None:
            raise ValueError(f"Missing CRS: {src.name}; no Earth-CRS fallback is allowed")
        return cls(src.shape, src.transform, src.crs)

    @property
    def bounds(self):
        return array_bounds(self.shape[0], self.shape[1], self.transform)

    def metadata(self):
        return {"shape": list(self.shape), "transform": list(self.transform)[:6],
                "crs_wkt": self.crs.to_wkt(), "bounds": list(self.bounds)}


def _clean(values, mask):
    values = np.asarray(values, dtype=np.float32).copy()
    values[~mask | ~np.isfinite(values) | (np.abs(values) >= 1e30)] = np.nan
    return values


def _read(path):
    with rasterio.open(path) as src:
        return _clean(src.read(1), src.read_masks(1) > 0), _Grid.read(src)


def _metric_grid(grid):
    t = grid.transform
    if (not grid.crs.is_projected or grid.crs.linear_units != "metre" or
            t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0 or not np.isclose(t.a, -t.e)):
        raise ValueError("Terrain/image features require a square, north-up, metre-based grid")
    return t.a


def _intersects(source, target):
    a = transform_bounds(source.crs, target.crs, *source.bounds, densify_pts=21)
    b = target.bounds
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _warp(values, source, target):
    """Nearest feature/mask transport; means are computed on the reference pixels."""
    if source == target:
        return values
    out = np.full(target.shape, np.nan, np.float32)
    reproject(values.astype(np.float32, copy=False), out,
              src_transform=source.transform, src_crs=source.crs,
              dst_transform=target.transform, dst_crs=target.crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest,
              warp_mem_limit=64, num_threads=1)
    return out


def _edges(shape):
    if min(shape) < GRID:
        raise ValueError("Reference raster must have at least 120 pixels on each axis")
    return tuple(np.arange(GRID + 1) * n // GRID for n in shape)


def _cell_sum(values):
    rows, cols = _edges(values.shape)
    return np.add.reduceat(np.add.reduceat(values, rows[:-1], axis=0, dtype=np.float64),
                           cols[:-1], axis=1, dtype=np.float64)


def _expand(cells, shape):
    rows, cols = _edges(shape)
    ri = np.searchsorted(rows[1:-1], np.arange(shape[0]), side="right")
    ci = np.searchsorted(cols[1:-1], np.arange(shape[1]), side="right")
    return cells[np.ix_(ri, ci)]


def _cell_mean(values, terrain, minimum=MIN_COVERAGE):
    valid = np.isfinite(values) & terrain
    counts = _cell_sum(valid)
    ground = _cell_sum(terrain)
    out = np.full((GRID, GRID), np.nan, np.float32)
    ok = (ground > 0) & (counts >= minimum * ground)
    np.divide(_cell_sum(np.where(valid, values, 0)), counts, out=out, where=ok)
    return out


def _class_counts(labels, mask=None):
    if mask is None:
        mask = np.ones(labels.shape, bool)
    return {str(c): int(((labels == c) & mask).sum()) for c in (1, 2, 3)}


def _snapshot(root, annotation, grid, terrain, hashes):
    provenance_path = _path(root, annotation["provenance"])
    provenance = _json(provenance_path)
    if provenance.get("schema_id") != "selenograph-geologic-units-v1":
        raise ValueError(f"Unknown annotation schema: {provenance_path}")
    arrays = {}
    for item in provenance["files"]:
        role = item["role"]
        if role in arrays:
            raise ValueError(f"Duplicate annotation role: {role}")
        path = _path(root, item["path"])
        hashes.file(path, "reference/" + item["path"], item["sha256"])
        if role not in ("painting", "accepted", "confidence"):
            raise ValueError(f"Unexpected annotation array role: {role}")
        value = np.load(path, allow_pickle=False)
        limit = 1 if role == "accepted" else 3
        if (value.shape != (GRID, GRID) or value.dtype.kind not in "biu" or
                np.any(value < 0) or np.any(value > limit)):
            raise ValueError(f"Invalid {role} cell array: {path}")
        arrays[role] = value
    cells = arrays["painting"].astype(np.uint8)
    path = _path(root, annotation["path"])
    hashes.file(path, "reference/" + annotation["path"], annotation["sha256"])
    with rasterio.open(path) as src:
        if src.dtypes != ("uint16",) or src.nodata != 65535 or _Grid.read(src) != grid:
            raise ValueError(f"Invalid label dtype/nodata/grid: {path}")
        expected = _expand(cells, grid.shape).astype(np.uint16)
        expected[~terrain] = 65535
        if not np.array_equal(src.read(1), expected):
            raise ValueError(f"Label TIFF does not exactly expand painting cells: {path}")
    arrays["painting"] = cells
    arrays["provenance"] = provenance
    return arrays


def _references(root, hashes):
    manifest = _json(root / "dataset.json")
    hashes.file(root / "dataset.json", "reference/dataset.json")
    listed = set()
    for item in manifest["artifacts"]:
        if item["path"] in listed:
            raise ValueError("Duplicate reference artifact path")
        listed.add(item["path"])
        hashes.file(_path(root, item["path"]), "reference/" + item["path"], item["sha256"])
    refs, ids = [], set()
    for item in manifest["tiles"]:
        record = _json(_path(root, item["path"]))
        tile_id, site = record["tile_id"], record["site_id"]
        if tile_id in ids or tile_id != item["tile_id"] or site != item["site_id"]:
            raise ValueError(f"Duplicate/inconsistent reference tile: {tile_id}")
        ids.add(tile_id)
        if site not in REGIONS:
            raise ValueError(f"No predeclared public region/group for {site}")
        dem = record["layers"]["dem"]
        hashes.file(_path(root, dem["path"]), "reference/" + dem["path"], dem["sha256"])
        z, grid = _read(_path(root, dem["path"]))
        if max(grid.shape) > 1024:
            raise ValueError("Reference windows must be bounded to 1024 pixels per axis")
        _metric_grid(grid)
        terrain = np.isfinite(z)
        snapshots = {}
        for annotation in record["annotations"]:
            variant = annotation["snapshot_variant"]
            if variant not in ("saved", "draft") or variant in snapshots:
                raise ValueError(f"Ambiguous snapshot selection: {tile_id}")
            snapshots[variant] = _snapshot(root, annotation, grid, terrain, hashes)
        if not snapshots:
            raise ValueError(f"No target painting: {tile_id}")
        saved = snapshots["saved"] if "saved" in snapshots else snapshots["draft"]
        draft = snapshots.get("draft", saved)
        accepted = np.zeros((GRID, GRID), bool)
        for snapshot in snapshots.values():
            accepted |= snapshot.get("accepted", np.zeros_like(accepted)).astype(bool)
        rows, cols = _edges(grid.shape)
        refs.append({"record": record, "grid": grid, "terrain": terrain,
                     "ground": _cell_sum(terrain), "area": np.outer(np.diff(rows), np.diff(cols)),
                     "saved": saved["painting"], "draft": draft["painting"],
                     "snapshots": snapshots, "accepted": accepted,
                     "occupied": np.zeros(grid.shape, bool), "joint": np.zeros(grid.shape, bool),
                     "support": np.zeros(grid.shape, bool), "public_tiles": []})
    return refs, len(listed)


def _claim(ref, footprint, tile_id):
    if np.any(ref["occupied"] & footprint):
        raise ValueError(f"Ambiguous overlapping public tiles for {ref['record']['tile_id']}: {tile_id}")
    ref["occupied"] |= footprint
