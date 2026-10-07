"""
Terrain features and the model behind the painter. No Streamlit in here, so it can be
run and tested on its own.

Prepared DEMs are registered in data/processed_data/**/working_dems.json. Manifest
paths are relative to their own JSON file; map IDs remain the DEM basename stems.
Flat DEMs and their _psr/_cpr companions remain supported for local legacy viewing.
NAC, PSR and CPR are display-only backgrounds, not assistant training features.
Companions may be on any grid or CRS; they are reprojected onto the DEM.

A record's section defaults to its DEM's first folder under DEM_DIR; group/site
remain geographic metadata. Optional annotations contain manifest-relative
painting/confidence/accepted NPY paths, a labels TIFF path, and metadata defaults
(verified, label_status, restricted, provenance). These sources are read-only;
painting_files/save_painting/meta_set always target user outputs, never the bundle.
Optional feature_sources are manifest-relative DEM paths: only BASE_FEATS are
computed on joined, aligned source terrain before transport; display geometry stays local.
"""
import glob
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi

from app import paths
from app.paths import DEM_DIR, DRAFT_DIR, LEGACY_DRAFT_DIR, LEGACY_MIRRORED_DIR, LEGACY_OUT_DIR, MAP_DIR, OUT_DIR

SUFFIXES = ("_psr", "_cpr", "_nac", "_quality", "_mask")  # companions, not DEMs
ANNOTATION_FILES = ("painting", "confidence", "accepted", "labels")

CODES = {1: "smooth_highlands", 2: "rough_highlands", 3: "shadowed_floor"}
CMAP = {0: (150, 150, 150, 255), 1: (70, 110, 200, 255),
        2: (190, 215, 60, 255), 3: (170, 50, 60, 255)}

GRID = 120           # painting cells per side
# How sure you were of a painted cell, and how much it counts when the model learns.
SURE, MOSTLY, UNSURE = 1, 2, 3
CONFIDENCE = {SURE: ("sure", 1.0),       # the whole area is this unit
              MOSTLY: ("mostly", 0.5),   # mostly this unit; may hide smaller bits of others
              UNSURE: ("unsure", 0.2)}   # a best guess
MAX_SIDE = 4000      # read bigger DEMs decimated; keeps memory and redraws sane
ROUGH_M = 135        # roughness window, in metres
REL_M = 4000         # relative-elevation window, in metres
LOCAL_M = 150        # crater-scale relative-height window, in metres
CURV_M = 60          # curvature smoothing window, in metres
SKY_LOCAL_M = 300    # how far the crater-scale sky view looks, in metres
FEATURE_VERSION = 6 # join adjoining source terrain before computing neighbourhood features
# The windows are round (Gaussian), with the same spread as a square this wide. A square
# window turned every steep spot into a hard-edged square aligned with the grid.

# Historical offline evaluation helpers only. Paint, Gallery, Update model and
# Progress do not use this persistent selection; Progress rotates individual maps.
TEST_MAPS_FILE = paths.TEST_MAPS_FILE
DEFAULT_TEST_MAPS = []                             # until test maps are chosen
APPROVED = 0.5     # how much a prediction you approved counts when the model learns

# Display layers and the fixed training baseline are deliberately separate. The
# development benchmark did not support adding scene-relative height (rel/zscene)
# or automatically including every available sensor. Keep these views for painting.
LAYERS = ["slope", "rough", "rel", "zscene", "svf", "rel_local", "curv", "svf_local"]
BASE_FEATS = ["slope", "rough", "svf", "rel_local", "curv", "svf_local"]

# ---------------------------------------------------------------- which model

# The one switch. "gbm" | "rf" | "unet", each a model_<name>.py next to this file.
# Override without editing:  SELENOGRAPH_MODEL=rf streamlit run selenograph.py
MODEL = os.environ.get("SELENOGRAPH_MODEL", "rf")

# A model file offers a NAME and these functions:
#     fit(X, cells) -> model or None      X: (n_feats, H, W), cells: (GRID, GRID)
#     fit_rows(Xtr, ytr, w) -> model or None   pooled samples across maps, weighted
#     predict(model, X) -> (H, W) uint8   unit codes, 255 where there is no terrain
#     confidence(model, X) -> (H, W) float32   probability of the predicted unit, NaN
#                                              where there is no terrain
# `cells` is the painting: 0 means unpainted, i.e. unknown, not background.


def model_files():
    """The backends on offer: every model_<name>.py sitting next to core.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    return sorted(os.path.basename(p)[len("model_"):-len(".py")]
                  for p in glob.glob(f"{here}/model_*.py"))


def backend(name=None):
    """Import the chosen model file. Imported here rather than at the top so switching
    MODEL costs nothing and an unused backend's dependencies stay unimported -- which is
    what keeps torch out of the way until someone actually builds the U-Net."""
    import importlib
    return importlib.import_module(f"app.model_{name or MODEL}")


# ---------------------------------------------------------------- raster helpers

def read(path, max_side=None):
    """Read band 1 as float32 with nodata as NaN, optionally decimated."""
    with rasterio.open(path) as s:
        h, w = s.height, s.width
        if max_side and max(h, w) > max_side:
            k = max(h, w) / max_side
            h, w = max(1, int(round(h / k))), max(1, int(round(w / k)))
            a = s.read(1, out_shape=(h, w), resampling=Resampling.average)
            tr = s.transform * rasterio.Affine.scale(s.width / w, s.height / h)
        else:
            a = s.read(1)
            tr = s.transform
        a = a.astype(np.float32)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        # PDS MISSING_CONSTANT is float64; GDAL cannot tag it on a float32 band, so it
        # arrives as ordinary data and has to be caught by magnitude.
        a[np.abs(a) > 1e30] = np.nan
        return a, tr, s.crs, abs(tr.a)


def warp(arr, tr, crs, ref, resampling=Resampling.bilinear):
    """Put a companion raster on the DEM's grid.

    If either side carries no CRS the only sane reading is that both are in the same
    coordinate system, so give them a common one and let the transforms do the work --
    rather than refusing to open an untagged GeoTIFF.
    """
    dst_crs = ref["crs"]
    if crs is None or dst_crs is None:
        crs = dst_crs = crs or dst_crs or rasterio.CRS.from_epsg(4326)
    out = np.full((ref["height"], ref["width"]), np.nan, np.float32)
    reproject(arr.astype(np.float32), out, src_transform=tr, src_crs=crs,
              dst_transform=ref["transform"], dst_crs=dst_crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return out


def fill_nan(a):
    """Nearest-neighbour fill, so gradients do not see the holes. Returns the hole mask
    too: every feature is set back to NaN there."""
    m = np.isnan(a)
    if not m.any():
        return a, m
    idx = ndi.distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)], m


def window(metres, px):
    """Gaussian sigma in pixels with the same spread as a square window `metres` wide."""
    return metres / np.sqrt(12) / px


def local_std(a, sigma):
    """Standard deviation around each pixel, in a round Gaussian window."""
    m = ndi.gaussian_filter(a, sigma)
    m2 = ndi.gaussian_filter(a * a, sigma)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def sky_view(z, px, n_az=16, max_m=10000, n_ray=48, ds=4):
    """Mean sky-view factor from horizon angles: how much sky a pixel can see. Low inside
    a bowl at any scale, so unlike scene-relative height it is not tied to one crater.
    Computed on a ds-downsampled grid with log-spaced rays, then resampled back."""
    zc = ndi.zoom(z, 1.0 / ds, order=1)
    pxc = px * ds
    d = np.unique(np.round(np.logspace(0, np.log10(max(max_m / pxc, 2)), n_ray)).astype(int))
    acc = np.zeros_like(zc, np.float32)
    for k in range(n_az):
        a = 2 * np.pi * k / n_az
        hmax = np.zeros_like(zc, np.float32)
        for dd in d[d >= 1]:
            dr, dc = int(round(dd * np.cos(a))), int(round(dd * np.sin(a)))
            if dr == 0 and dc == 0:
                continue
            sh = np.roll(np.roll(zc, -dr, 0), -dc, 1)
            if dr > 0:   sh[-dr:, :] = zc[-dr:, :]      # clamp at the edge, don't wrap
            elif dr < 0: sh[:-dr, :] = zc[:-dr, :]
            if dc > 0:   sh[:, -dc:] = zc[:, -dc:]
            elif dc < 0: sh[:, :-dc] = zc[:, :-dc]
            np.maximum(hmax, (sh - zc) / (dd * pxc), out=hmax)
        acc += 1.0 - np.sin(np.arctan(hmax))
    out = ndi.zoom(acc / n_az, (z.shape[0] / zc.shape[0], z.shape[1] / zc.shape[1]), order=1)
    return out[:z.shape[0], :z.shape[1]]


# ---------------------------------------------------------------- the data on disk

def manifest_files():
    """Published manifests only: do not even descend into hidden staging directories."""
    root = Path(DEM_DIR).resolve()
    found = []
    for directory, children, files in os.walk(root):
        children[:] = [name for name in children if not name.startswith(".")]
        if "working_dems.json" in files:
            found.append(Path(directory) / "working_dems.json")
    return sorted(found, key=lambda p: (len(p.relative_to(root).parts), str(p)))


def _within(path, root):
    """path.is_relative_to(root) for resolved paths, without building every parent."""
    path, root = Path(path).parts, Path(root).parts
    return path[:len(root)] == root


def _catalog_path(root, value, parent=None):
    """Resolve paths before checking containment, including symlink and ../ escapes."""
    path = ((parent or root) / value).resolve()
    if not _within(path, root):
        raise ValueError(f"Catalog path {value!r} resolves outside DEM_DIR: {path}")
    if any(part.startswith(".") for part in path.relative_to(root).parts):
        raise ValueError(f"Catalog path {value!r} refers to hidden/unpublished data: {path}")
    return path


_CATALOG_SNAPSHOT: ContextVar[tuple[Path, Any] | None] = ContextVar(
    "selenograph_catalog_snapshot", default=None)


@contextmanager
def catalog_snapshot():
    """Discover the catalog once for a synchronous operation or app rerun.

    Use ``with catalog_snapshot():`` or ``@catalog_snapshot()``. Nested scopes
    reuse the current snapshot when the resolved DEM_DIR matches. Otherwise
    entry performs normal discovery/stat checks; the next outer scope discovers
    anew, without a TTL. Context-local tokens isolate concurrent callers and
    restore the previous scope even on exceptions or Streamlit rerun/stop.

    Only catalog discovery is frozen, not raster/output contents or path guards.
    Public record APIs still return deep copies; no mutable catalog is yielded.
    """
    root = Path(DEM_DIR).resolve()
    catalog = _catalog()
    token = _CATALOG_SNAPSHOT.set((root, catalog))
    try:
        yield
    finally:
        _CATALOG_SNAPSHOT.reset(token)


def _catalog():
    root = Path(DEM_DIR).resolve()
    snapshot = _CATALOG_SNAPSHOT.get()
    if snapshot is not None and snapshot[0] == root:
        return snapshot[1]
    stamps = []
    for manifest in manifest_files():
        _catalog_path(root, manifest)
        try:
            stat = manifest.stat()
        except FileNotFoundError:                 # a publisher may replace a directory
            continue
        stamps.append((manifest, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino))
    flat = tuple(str(_catalog_path(root, p)) for p in sorted(root.glob("*"))
                 if not p.name.startswith(".") and p.is_file()
                 and p.suffix.lower() in (".tif", ".tiff")
                 and not p.stem.lower().endswith(SUFFIXES))
    return _load_catalog(root, tuple(stamps), flat)


@lru_cache(maxsize=8)
def _load_catalog(root, stamps, flat):
    """Parse/normalize once per catalog revision; index stable IDs for per-map lookups."""
    out, by_name, locations, companions = [], {}, {}, set()
    for manifest, *_ in stamps:
        try:
            listing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(listing, list):
            continue

        def relative(value, parent=manifest.parent):
            return str(_catalog_path(root, value, parent).relative_to(root))

        def layer_paths(value):
            if isinstance(value, str):
                path = relative(value)
                companions.add(str(root / path))
                return path
            if isinstance(value, dict):
                if "path" in value:
                    return dict(value, path=layer_paths(value["path"])) if value["path"] else dict(value)
                if "status" in value:
                    return dict(value)
                return {k: layer_paths(v) for k, v in value.items()}
            return value

        for record in listing:
            if not isinstance(record, dict) or not isinstance(record.get("working_dem"), str):
                continue
            normalized: dict[str, Any] = dict(record, working_dem=relative(record["working_dem"]))
            normalized["section"] = section(normalized)
            if "annotations" in record:
                if not isinstance(record["annotations"], dict):
                    raise ValueError(f"annotations must be a dict: {manifest}")
                annotations = dict(record["annotations"])
                for kind in ANNOTATION_FILES:
                    value = annotations.get(kind)
                    if value is not None:
                        if not isinstance(value, str) or not value:
                            raise ValueError(f"annotations.{kind} must be a path: {manifest}")
                        annotations[kind] = relative(value)
                        companions.add(str(root / annotations[kind]))
                normalized["annotations"] = annotations
            if "feature_sources" in record:
                sources = record["feature_sources"]
                if not isinstance(sources, list) or any(
                        not isinstance(p, str) or not p for p in sources):
                    raise ValueError(f"feature_sources must be a list of DEM paths: {manifest}")
                normalized["feature_sources"] = [relative(p) for p in sources]
            if isinstance(record.get("layers"), dict):
                normalized["layers"] = layer_paths(record["layers"])
            name = dem_name(normalized["working_dem"])
            if name in by_name:
                raise ValueError(f"Duplicate map ID {name!r}: {locations[name]} and "
                                 f"{manifest} ({normalized['working_dem']}). "
                                 "Map IDs must be unique to protect paintings and model samples.")
            locations[name] = f"{manifest} ({normalized['working_dem']})"
            by_name[name] = normalized
            out.append(normalized)

    # Flat fallback shares the same ID namespace, including when resolving by name.
    registered = {name: str(root / r["working_dem"]) for name, r in by_name.items()}
    extra = []
    for path in flat:
        name = dem_name(path)
        if path == registered.get(name) or path in companions:
            continue
        if name in registered:
            raise ValueError(f"Duplicate map ID {name!r}: {registered[name]} and {path}. "
                             "Map IDs must be unique to protect paintings and model samples.")
        registered[name] = path
        extra.append(path)
    return out, by_name, extra


def section(record):
    """Explicit section or the DEM's top-level dataset folder, never inferred site.

    Accepts a normalized record. Flat legacy DEMs use DEM_DIR's directory name.
    """
    if record.get("section"):
        return record["section"]
    root = Path(DEM_DIR).resolve()
    parts = _catalog_path(root, record["working_dem"]).relative_to(root).parts
    return parts[0] if len(parts) > 1 else root.name


def records():
    """Published records in stable order, with all file paths relative to DEM_DIR.

    Copies keep callers from modifying the cached catalog or its serialized records.
    """
    return deepcopy(_catalog()[0])


def record_path(record, layer=None):
    """Resolve a normalized catalog record's DEM or named layer to an absolute path."""
    value = record.get("working_dem") if layer is None else record.get("layers", {}).get(layer)
    if isinstance(value, dict):
        value = value.get("path")
    return str(_catalog_path(Path(DEM_DIR).resolve(), value)) if value else None


def annotation_path(name, kind):
    """Absolute bundled annotation path, or None if undeclared (not an output path).

    Existence is not implied. Gallery callers can enumerate painted_maps(), use
    meta_get() for merged metadata, and resolve bundled labels with kind='labels'.
    """
    if kind not in ANNOTATION_FILES:
        raise ValueError(f"Unknown annotation kind: {kind!r}")
    record = map_record(name)
    value = record.get("annotations", {}).get(kind) if record else None
    return str(_catalog_path(Path(DEM_DIR).resolve(), value)) if value else None


def dem_files():
    """Manifest DEMs in import order, then unregistered flat legacy DEMs by name."""
    listing, _, flat = _catalog()
    out = []
    for record in listing:
        path = record_path(record)
        if path and Path(path).is_file():
            out.append(path)
    return out + list(flat)


def map_record(name):
    by_name = _catalog()[1]
    # IDs may contain dots; strip a file extension only when the exact ID is absent.
    record = by_name.get(os.fspath(name))
    return deepcopy(record if record is not None else by_name.get(dem_name(name)))


def dem_path(name):
    """Resolve a stable map ID without assuming its directory or TIFF extension."""
    record = map_record(name)
    if record:
        return record_path(record)
    files = {dem_name(p): p for p in dem_files()}
    value = os.fspath(name)
    if value in files:
        return files[value]
    if dem_name(value) in files:
        return files[dem_name(value)]
    key = (dem_name(value) if Path(value).name != value or
           Path(value).suffix.lower() in (".tif", ".tiff") else value)
    return str((Path(DEM_DIR) / f"{key}.tif").resolve())


def map_site(name):
    """The site a map belongs to. Every tile of one physical site shares it, and train and
    test are split by site, never by tile."""
    r = map_record(name)
    if r:
        return r.get("group", r.get("site"))
    path = next((p for p in dem_files() if dem_name(p) == name), None)
    return site_id(path) if path else None


def test_maps_path():
    """Keep test fixtures/local catalogs self-contained unless explicitly overridden."""
    if Path(TEST_MAPS_FILE) != paths.TEST_MAPS_FILE:
        return Path(TEST_MAPS_FILE) if Path(TEST_MAPS_FILE).is_absolute() else Path(DEM_DIR) / TEST_MAPS_FILE
    if Path(DEM_DIR).resolve() != paths.DEM_DIR:
        return Path(DEM_DIR) / "test_maps.json"
    return paths.TEST_MAPS_FILE


def test_maps():
    """The maps the model is scored on, those that exist here."""
    try:
        with open(test_maps_path(), encoding="utf-8") as f:
            names = [str(n) for n in json.load(f)]
    except (OSError, ValueError, TypeError):
        names = DEFAULT_TEST_MAPS
    have = {dem_name(p) for p in dem_files()}
    return [n for n in names if n in have]


def set_test_maps(names):
    path = test_maps_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    with open(staged, "w", encoding="utf-8") as f:
        json.dump(sorted(set(names)), f, indent=1)
    os.replace(staged, path)


def extents():
    """(left, bottom, right, top) in metres of every map here."""
    out = {}
    for r in records():
        if "transform" in r and "size" in r:
            a, (H, W) = r["transform"], r["size"]
            out[dem_name(r["working_dem"])] = (a[2], a[5] + a[4] * H, a[2] + a[0] * W, a[5])
    for p in dem_files():
        if dem_name(p) not in out:
            with rasterio.open(p) as src:
                out[dem_name(p)] = tuple(src.bounds)
    return out


def held_out_maps(names=None):
    """The test maps (or `names`) and every map touching them: never trained on, and
    kept out of the painter. Neighbouring ground looks alike, so holding out only the map
    itself would let its surroundings leak into training."""
    names = set(test_maps() if names is None else names)
    if not names:
        return names
    box = extents()
    near = []
    for n in names & box.keys():
        l, b, r, t = box[n]
        m = (r - l) / 2                                # half a map around it: its neighbours
        near.append((l - m, b - m, r + m, t + m))
    return names | {n for n, (l, b, r, t) in box.items()
                    if any(l < R and r > L and b < T and t > B for L, B, R, T in near)}


def dem_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def site_id(path):
    """Keep tiles from one parent site grouped for future training/evaluation."""
    record = map_record(dem_name(path))
    if record and (record.get("group") or record.get("site")):
        return record.get("group") or record["site"]
    with rasterio.open(path) as src:
        return src.tags().get("site_id", dem_name(path))


def companion(dem_path, kind):
    """A manifest layer, or <name>_<kind>.tif beside a legacy DEM."""
    record = map_record(dem_name(dem_path))
    if record:
        path = record_path(record, kind)
        if path and os.path.isfile(path):
            return path
    stem = os.path.splitext(dem_path)[0]
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        if os.path.exists(stem + f"_{kind}" + ext):
            return stem + f"_{kind}" + ext
    return None


def _output_component(value):
    value = os.fspath(value)
    if (not value or value.startswith(".") or "/" in value or "\\" in value
            or "\x00" in value):
        raise ValueError(f"Output names must be single, non-hidden path components: {value!r}")
    return value


def _output_path(root, relative):
    """Contain both directory and file symlinks without creating anything on reads."""
    root = Path(root).resolve()
    target = root / relative
    resolved = target.resolve()
    if not _within(resolved, root) or _within(resolved, Path(DEM_DIR).resolve()):
        raise ValueError(f"Output path must stay under its output root and outside processed data: {target}")
    return str(target)


def _output_relative_dir(name):
    name = _output_component(name)
    dem = dem_path(name)
    if not dem:
        raise ValueError(f"No DEM path for map {name!r}")
    return Path(dem).resolve().relative_to(Path(DEM_DIR).resolve()).parent


def out_dir(name, final=True):
    """Mirror a DEM's parent inside saved paintings or painting drafts."""
    return _output_path(OUT_DIR if final else DRAFT_DIR, _output_relative_dir(name))


def _legacy_output_file(name, filename, final=True):
    name, filename = _output_component(name), _output_component(filename)
    if final:
        return _output_path(LEGACY_OUT_DIR, Path(name) / filename)
    suffix = {"painting.npy": ".npy", "accepted.npy": ".accepted.npy",
              "confidence.npy": ".confidence.npy"}.get(filename, "." + filename)
    return _output_path(LEGACY_DRAFT_DIR, name + suffix)


def _mirrored_output_file(name, filename, final=True):
    """Read the previous one-tree layout, never mistaking a bucket for a site."""
    root = Path(LEGACY_MIRRORED_DIR).resolve()
    folder = root / _output_relative_dir(name)
    for reserved in (OUT_DIR, DRAFT_DIR, MAP_DIR, LEGACY_OUT_DIR, LEGACY_DRAFT_DIR):
        reserved = Path(reserved).resolve()
        if _within(reserved, root) and _within(folder.resolve(), reserved):
            return None
    if not final:
        filename = {"painting.npy": "draft.npy", "accepted.npy": "draft.accepted.npy",
                    "confidence.npy": "draft.confidence.npy"}.get(filename, "draft." + filename)
    return _output_path(root, folder / f"{_output_component(name)}_{_output_component(filename)}")


def _painting_output_file(name, filename, final=True):
    """Painting artifacts, and read compatibility for TIFFs formerly saved beside them."""
    root = OUT_DIR if final else DRAFT_DIR
    return _output_path(root, Path(out_dir(name, final)) / f"{name}_{filename}")


def output_file(name, filename, *, existing=False, final=True):
    """Resolve a painting artifact or a saved GeoTIFF in the separate maps bucket.

    Legacy read fallbacks select a whole saved/draft set. Missing sidecars never
    fall through independently, and metadata-only edits do not hide old labels.
    """
    name, filename = _output_component(name), _output_component(filename)
    export = final and filename in ("labels.tif", "map.tif")
    target = (_output_path(MAP_DIR, _output_relative_dir(name) / f"{name}_{filename}")
              if export else _painting_output_file(name, filename, final))
    if existing:
        markers = ("painting.npy", "labels.tif") if final else ("painting.npy",)
        current = [output_file(name, m, final=final) for m in markers]
        if final:
            current.append(_painting_output_file(name, "labels.tif"))
        if any(os.path.isfile(p) for p in current):
            # Before the first save into maps/, old TIFFs remain readable. Once a
            # new labels export exists, absent predictions must not revive old ones.
            if export and not os.path.isfile(output_file(name, "labels.tif")):
                previous = _painting_output_file(name, filename)
                if os.path.isfile(previous):
                    return previous
        else:
            for resolver in (_mirrored_output_file, _legacy_output_file):
                candidates = [resolver(name, m, final) for m in markers]
                if any(p and os.path.isfile(p) for p in candidates):
                    return resolver(name, filename, final)
    return target


def meta_get(name, final=None):
    """Saved metadata plus staged draft changes, or saved-only with final=True.

    Verification is explicit metadata, never inferred from a folder or a site name.
    """
    record = map_record(name) or {}
    annotations = record.get("annotations", {})
    metadata = {key: annotations.get(key, record.get(key))
                for key in ("verified", "label_status", "restricted", "provenance")
                if key in annotations or key in record}
    sources = [_legacy_output_file(name, "meta.json"), _mirrored_output_file(name, "meta.json"),
               output_file(name, "meta.json")]
    if final is not True:
        sources.append(output_file(name, "meta.json", final=False))
    for path in sources:
        if path is None:
            continue
        try:
            with open(path) as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                metadata.update(saved)
        except (OSError, ValueError):
            pass
    return deepcopy(metadata)


def meta_set(name, *, final=True, **kw):
    """Stage metadata in drafts, or explicitly commit it with the saved painting."""
    m = meta_get(name)
    m["name"] = name
    m.update(kw)
    path = output_file(name, "meta.json", final=final)
    draft = output_file(name, "meta.json", final=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = output_file(name, "meta.json.tmp", final=final)
    with open(temporary, "w") as f:
        json.dump(m, f, indent=2)
    os.replace(temporary, path)
    if final and os.path.isfile(draft):
        os.remove(draft)


def painting_files(name, final=False, *, existing=False):
    """Painting, acceptance and confidence from one draft or saved output set.

    Canonical paths are always returned for writing. With existing=True, legacy
    files remain readable until a canonical painting of the same kind exists.
    """
    return tuple(output_file(name, f, existing=existing, final=final)
                 for f in ("painting.npy", "accepted.npy", "confidence.npy"))


def save_painting(name, cells, final=False, accepted=None, confidence=None):
    """Recoverable draft files stay separate from explicit saved exports. `accepted` marks
    cells filled from a prediction instead of painted, and `confidence` how sure you were
    of each cell; saving without them clears them (every painted cell is then SURE)."""
    path, accepted_path, confidence_path = painting_files(name, final)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if confidence is not None and not ((confidence != SURE) & (cells > 0)).any():
        confidence = None                            # all sure: the default needs no file
    for p, a in ((accepted_path, accepted), (confidence_path, confidence), (path, cells)):
        if a is None or (p == accepted_path and not a.any()):
            if os.path.exists(p):
                os.remove(p)
            continue
        # Replace atomically so a restart cannot leave a half-written painting.
        temporary = _output_path(OUT_DIR if final else DRAFT_DIR, Path(p + ".tmp"))
        with open(temporary, "wb") as f:
            np.save(f, a)
        os.replace(temporary, p)


@catalog_snapshot()
def seed_paintings(names=None):
    """Explicitly initialize both buckets from supplied paintings, never on browsing.

    Any existing painting, sidecar, export or metadata is user work, including an
    empty painting. Skip that map entirely rather than overwrite or guess which
    version should initialize the other bucket.
    """
    if names is None:
        names = [dem_name(p) for p in dem_files()]
    seeded = []
    artifacts = ("painting.npy", "accepted.npy", "confidence.npy", "meta.json",
                 "labels.tif", "map.tif", "thumb.png")
    for name in dict.fromkeys(names):
        source = annotation_path(name, "painting")
        if not source or not os.path.isfile(source):
            continue
        prior = []
        for final in (False, True):
            for filename in artifacts:
                prior.extend((output_file(name, filename, final=final),
                              _painting_output_file(name, filename, final),
                              _mirrored_output_file(name, filename, final),
                              _legacy_output_file(name, filename, final)))
        if any(p and os.path.exists(p) for p in prior):
            continue
        cells = np.load(source, allow_pickle=False)
        if cells.shape != (GRID, GRID) or not np.isin(cells, (0, *CODES)).all():
            raise ValueError(f"Invalid bundled painting: {name}")
        if not cells.any():
            continue
        flags = {}
        for kind in ("accepted", "confidence"):
            path = annotation_path(name, kind)
            values = np.load(path, allow_pickle=False) if path and os.path.isfile(path) else None
            allowed = (0, 1) if kind == "accepted" else (0, SURE, MOSTLY, UNSURE)
            if values is not None and (values.shape != cells.shape or not np.isin(values, allowed).all()):
                raise ValueError(f"Invalid bundled {kind} flags: {name}")
            flags[kind] = values
        for final in (True, False):
            save_painting(name, cells, final=final, **flags)
        seeded.append(name)
    return seeded


def _painting_source(name, final=None):
    """Resolve once, retaining bundle identity even though its public final flag is True."""
    candidates = [(output_file(name, "painting.npy", existing=True, final=saved), saved, False)
                  for saved in ((False, True) if final is None else (final,))]
    # A user's raster-only save must not silently resurrect the bundle's cell labels.
    if final is not False and not os.path.isfile(output_file(name, "labels.tif", existing=True)):
        bundled_path = annotation_path(name, "painting")
        if bundled_path:
            candidates.append((bundled_path, True, True))
    for path, saved, bundled in candidates:
        if path and os.path.isfile(path):
            cells = np.load(path, allow_pickle=False)
            if cells.shape == (GRID, GRID):
                return cells.astype(np.uint8), saved, bundled
    return np.zeros((GRID, GRID), np.uint8), None, False


def painting_source(name, final=None):
    """Return (independent cells copy, final): draft False, saved/bundled True, absent None.

    Precedence is draft, user save, bundled painting. An empty user painting still
    overrides bundled labels. Browsing never copies bundles; explicit seeding can
    initialize independent saved and draft snapshots without changing the bundle.
    final=True reads saved/bundled only; final=False reads only the draft.
    """
    cells, final, _ = _painting_source(name, final)
    return cells, final


def load_painting(name):
    return painting_source(name)[0]


def has_saved_labels(name):
    """Whether saved/bundled labels contain terrain units, ignoring all drafts."""
    cells, source = painting_source(name, final=True)
    if source is not None:
        # An explicitly cleared save overrides older bundled/raster labels.
        return bool(np.isin(cells, list(CODES)).any())
    path = output_file(name, "labels.tif", existing=True)
    if not os.path.isfile(path):
        path = annotation_path(name, "labels")
    if not path or not os.path.isfile(path):
        return False
    with rasterio.open(path) as source:
        for _, block in source.block_windows(1):
            labels = source.read(1, window=block, masked=True)
            if np.isin(labels.compressed(), list(CODES)).any():
                return True
    return False


@catalog_snapshot()
def painted_maps():
    """(painted IDs, accepted IDs) for effective draft/saved/bundled paintings.

    Includes bundled maps without creating output folders. As with legacy output
    IDs, callers displaying maps should intersect these IDs with dem_files().
    Gallery must use this API rather than only globbing output metadata folders;
    annotation_path(name, 'labels') exposes the bundle's optional label raster.
    """
    candidates = {os.path.basename(p)[:-4] for p in glob.glob(f"{LEGACY_DRAFT_DIR}/*.npy")
                  if not p.endswith((".accepted.npy", ".confidence.npy"))}
    candidates |= {os.path.basename(os.path.dirname(p))
                   for p in glob.glob(f"{LEGACY_OUT_DIR}/*/painting.npy")}
    candidates |= {dem_name(p) for p in dem_files()}
    # Keep flat/uninstalled IDs discoverable without scanning evaluation archives.
    for root, suffix in ((OUT_DIR, "_painting.npy"), (DRAFT_DIR, "_painting.npy"),
                         (LEGACY_MIRRORED_DIR, "_painting.npy"), (LEGACY_MIRRORED_DIR, "_draft.npy")):
        candidates |= {p.name[:-len(suffix)] for p in Path(root).glob(f"*{suffix}")}
    painted, accepted = set(), set()
    for name in candidates:
        cells, final, bundled = _painting_source(name)
        if not cells.any():
            continue
        painted.add(name)
        flags = _beside(name, final, 1, cells, bundled=bundled)
        if flags is not None and (flags.astype(bool) & (cells > 0)).any():
            accepted.add(name)
    return painted, accepted


def _beside(name, final, which, cells, *, bundled=None):
    """Flags from exactly the selected painting source, never a lower-priority one."""
    if final is None:
        return None
    if bundled is None:
        _, _, bundled = _painting_source(name, final)
    path = (annotation_path(name, ("painting", "accepted", "confidence")[which])
            if bundled else painting_files(name, final, existing=True)[which])
    if path and os.path.isfile(path):
        a = np.load(path, allow_pickle=False)
        if a.shape == cells.shape:
            return a
    return None


def load_accepted(name, final=None):
    """Approved prediction flags, optionally from saved/bundled or draft only."""
    cells, final, bundled = _painting_source(name, final)
    a = _beside(name, final, 1, cells, bundled=bundled)
    return np.zeros((GRID, GRID), bool) if a is None else a.astype(bool) & (cells > 0)


def load_confidence(name, final=None):
    """How sure you were of each painted cell of load_painting(name) -- or of the saved
    painting, with final=True (falling back to bundled, never draft): SURE unless
    marked otherwise, 0 where unpainted. final=False reads only the draft."""
    cells, final, bundled = _painting_source(name, final)
    a = _beside(name, final, 2, cells, bundled=bundled)
    conf = np.where(a > 0, a, SURE) if a is not None else np.full(cells.shape, SURE)
    return np.where(cells > 0, conf, 0).astype(np.uint8)


# ---------------------------------------------------------------- features

def _feature_plan(dem_path):
    """Iterative dependency ordering: reject cycles before any heavy terrain builds."""
    root = Path(DEM_DIR).resolve()

    def record_at(path):
        # Map IDs are unique, so look up by ID and confirm the path, not every record.
        record = map_record(dem_name(path))
        return record if record and record_path(record) == path else {}

    target = str(Path(dem_path).resolve())
    dependencies, state, order = {}, {}, []
    pending = [(target, False)]
    while pending:
        path, leaving = pending.pop()
        if leaving:
            state[path] = 2
            order.append(path)
            continue
        if state.get(path) == 1:
            raise ValueError(f"Cyclic feature_sources dependency: {path}")
        if state.get(path) == 2:
            continue
        state[path] = 1
        sources = record_at(path).get("feature_sources", [])
        dependencies[path] = [str(_catalog_path(root, p)) for p in sources]
        pending.append((path, True))
        pending.extend((p, False) for p in reversed(dependencies[path]))
    return order, dependencies


def _feature_warp(values, source, target):
    """Strict nearest transport, unlike permissive display-companion warp().

    Untagged rasters can only be copied on exactly equal grids (including CRS).
    Otherwise both real CRSs are required; no guessed Earth or lunar projection.
    """
    source_shape = (source["height"], source["width"])
    target_shape = (target["height"], target["width"])
    if values.ndim not in (2, 3) or values.shape[-2:] != source_shape:
        raise ValueError("Feature array shape does not match its source grid")
    if (source_shape == target_shape and source["transform"] == target["transform"]
            and source["crs"] == target["crs"]):
        return values.copy()
    if source["crs"] is None or target["crs"] is None:
        raise ValueError("Feature transport requires both CRSs unless grids match exactly")
    out = np.full((*values.shape[:-2], *target_shape), np.nan, np.float32)
    reproject(values.astype(np.float32, copy=False), out,
              src_transform=source["transform"], src_crs=source["crs"],
              dst_transform=target["transform"], dst_crs=target["crs"],
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest,
              warp_mem_limit=64, num_threads=1)
    return out


def _terrain_features(z, px):
    """Six terrain channels, calculated before cutting a shared context into tiles."""
    zf, hole = fill_nan(z)
    gy, gx = np.gradient(zf, px)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    values = dict(slope=slope, rough=local_std(slope, window(ROUGH_M, px)),
                  svf=sky_view(zf, px),
                  rel_local=zf - ndi.gaussian_filter(zf, window(LOCAL_M, px)),
                  curv=ndi.gaussian_laplace(zf, window(CURV_M, px)) / px ** 2,
                  svf_local=sky_view(zf, px, max_m=SKY_LOCAL_M, n_ray=16, ds=1))
    for value in values.values():
        value[hole] = np.nan
    return values


def _source_contexts(leaves):
    """Join adjacent native leaf DEMs without resampling terrain or inventing a CRS.

    Only aligned, non-overlapping grids share a context. Recursive consumers still
    own exactly their declared footprints. Different grids keep strict feature
    transport, and disconnected terrain is not used as neighbouring context.
    """
    groups = []
    for path in leaves:
        with rasterio.open(path) as src:
            tr = src.transform
            if max(src.shape) > MAX_SIDE:
                raise ValueError(f"Feature source build changed its native grid: {path}")
            if src.crs is None or tr.b or tr.d or tr.a <= 0 or tr.e >= 0:
                continue
            tile = (path, tr, src.height, src.width)
            for crs, origin, tiles in groups:
                if (src.crs, tr.a, tr.e) != (crs, origin.a, origin.e):
                    continue
                offset = (~origin) @ (tr.c, tr.f)
                if np.allclose(offset, np.round(offset), rtol=0, atol=1e-6):
                    tiles.append(tile)
                    break
            else:
                groups.append((src.crs, tr, [tile]))
    contexts = {}
    for _, origin, tiles in groups:
        rectangles = []
        for path, tr, height, width in tiles:
            col, row = (~origin) @ (tr.c, tr.f)
            row, col = round(row), round(col)
            rectangles.append((path, row, col, row + height, col + width))
        pending = list(rectangles)
        while pending:
            group = [pending.pop()]
            for _, top, left, bottom, right in group:
                for item in pending[:]:
                    _, t, l, b, r = item
                    # Shared edges, not just isolated corner contact.
                    if ((min(bottom, b) > max(top, t) and min(right, r) >= max(left, l))
                            or (min(bottom, b) >= max(top, t) and min(right, r) > max(left, l))):
                        group.append(item)
                        pending.remove(item)
            if len(group) < 2:
                continue
            # Overlapping leaves may feed disjoint intermediate owners. Do not
            # merge their elevations; the normal ownership checks still apply.
            if any(min(a[3], b[3]) > max(a[1], b[1]) and min(a[4], b[4]) > max(a[2], b[2])
                   for i, a in enumerate(group) for b in group[i + 1:]):
                continue
            top, left = min(p[1] for p in group), min(p[2] for p in group)
            bottom, right = max(p[3] for p in group), max(p[4] for p in group)
            terrain = np.full((bottom - top, right - left), np.nan, np.float32)
            for path, t, l, b, r in group:
                terrain[t-top:b-top, l-left:r-left] = read(path)[0]
            features = _terrain_features(terrain, origin.a)
            for path, t, l, b, r in group:
                contexts[path] = {name: values[t-top:b-top, l-left:r-left]
                                  for name, values in features.items()}
    return contexts


def build(dem_path):
    """Return (X, names, hillshade, output profile, pixel size), all caller-owned.

    Without feature_sources this is the original local DEM build. With linkage,
    adjoining, aligned source DEMs share terrain context before the six BASE_FEATS
    are computed and nearest-transported. Other channels, hillshade and geometry
    remain local. Uncovered pixels and local DEM holes are NaN; overlapping source
    footprints raise even
    where source terrain is absent. Missing CRS is allowed only for exact grids.

    A dependency DAG is built once per call, with each source computed once and
    released after its last consumer. There are no persistent feature/disk caches,
    so edits to source rasters or manifests are visible on the next build.
    """
    order, dependencies = _feature_plan(dem_path)
    consumers = {path: 0 for path in order}
    for sources in dependencies.values():
        for path in sources:
            consumers[path] += 1
    contexts = _source_contexts([path for path in order[:-1] if not dependencies[path]])
    features = {}
    result = None
    for path in order:
        result = _build_local(path, context=contexts.pop(path, None))
        X, names, _, profile, _ = result
        indices = [names.index(name) for name in BASE_FEATS]
        sources = dependencies[path]
        if sources:
            terrain = np.isfinite(X[names.index("slope")])
            occupied = np.zeros(terrain.shape, bool)
            transported = np.full((len(BASE_FEATS), *terrain.shape), np.nan, np.float32)
            for source in sources:
                values, grid = features[source]
                footprint = _feature_warp(
                    np.ones((grid["height"], grid["width"]), np.float32), grid, profile) == 1
                if (occupied & footprint).any():
                    raise ValueError(f"Overlapping feature source ownership for {path}: {source}")
                occupied |= footprint
                warped = _feature_warp(values, grid, profile)
                transported[:, footprint] = warped[:, footprint]
                consumers[source] -= 1
                if consumers[source] == 0:
                    del features[source]
            transported[:, ~terrain] = np.nan
            X[indices] = transported
        if consumers[path]:
            # A linked context must be the whole native tile, not a silently
            # decimated feature grid different from the declared source raster.
            with rasterio.open(path) as src:
                if ((src.height, src.width) != X.shape[-2:]
                        or src.transform != profile["transform"] or src.crs != profile["crs"]):
                    raise ValueError(f"Feature source build changed its native grid: {path}")
            features[path] = (X[indices].copy(), profile)
    assert result is not None  # the plan always contains the requested DEM
    return result


def _build_local(dem_path, *, context=None):
    """Local display geometry with optional features from a joined terrain context."""
    z, tr, crs, px = read(dem_path, MAX_SIDE)
    ref = dict(height=z.shape[0], width=z.shape[1], transform=tr, crs=crs)
    prof: dict[str, Any] = dict(driver="GTiff", dtype="uint8", count=1, nodata=255,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256, **ref)

    zf, hole = fill_nan(z)
    f = _terrain_features(z, px) if context is None else dict(context)
    f.update(rel=zf - ndi.gaussian_filter(zf, window(REL_M, px)),
             zscene=zf - np.median(zf))
    names = list(LAYERS)

    for kind, resamp in (("psr", Resampling.nearest), ("cpr", Resampling.bilinear),
                             ("nac", Resampling.bilinear)):
        p = companion(dem_path, kind)
        if not p:
            continue
        v, t2, c2, _ = read(p)
        v = warp(v, t2, c2, ref, resamp)
        good = np.isfinite(v)
        if good.any() and kind == "cpr":       # drop physically absurd fill values
            lo, hi = np.percentile(v[good], [0.5, 99.5])
            v[(v < lo) | (v > hi)] = np.nan
        f[kind] = v
        names.append(kind)

    for v in f.values():
        v[hole] = np.nan
    X = np.stack([f[n] for n in names]).astype(np.float32)
    hs = hillshade(zf, px)
    hs[hole] = 0                         # show the missing border instead of filled terrain
    return X, names, hs, prof, px


def hillshade(z, px, az=315, alt=25):
    zs = ndi.gaussian_filter(z, 2)
    gy, gx = np.gradient(zs, px)
    sl, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(-gy, gx)
    return np.clip(np.sin(np.radians(alt)) * np.cos(sl) +
                   np.cos(np.radians(alt)) * np.sin(sl) *
                   np.cos(np.radians(az) - asp), 0, 1)


# ---------------------------------------------------------------- the model

def cell_pixels(X, cells, keep=None, per_cell=120):
    """Sample feature rows from inside every painted cell (optionally a subset)."""
    n, H, W = X.shape
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for i, j in zip(*np.nonzero(cells if keep is None else (cells * keep))):
        r0, r1 = i * H // GRID, (i + 1) * H // GRID
        c0, c1 = j * W // GRID, (j + 1) * W // GRID
        blk = X[:, r0:r1, c0:c1].reshape(n, -1)
        idx = np.flatnonzero(np.isfinite(blk[0]))     # band 0 = slope: is there terrain?
        if idx.size == 0:
            continue
        idx = rng.choice(idx, min(per_cell, idx.size), replace=False)
        xs.append(blk[:, idx].T)
        ys.append(np.full(idx.size, cells[i, j]))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def blockwise(predict_rows, X, rows=256, dtype=np.uint8, fill=255):
    """Run a per-pixel model over a raster in row blocks, so a large one does not go
    through the model in a single piece. Pixels with no terrain come back `fill`."""
    n, H, W = X.shape
    out = np.full((H, W), fill, dtype)
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        blk = X[:, r0:r1].reshape(n, -1).T
        ok = np.isfinite(blk[:, 0])           # band 0 = slope: is there terrain?
        row = np.full(blk.shape[0], fill, dtype)
        if ok.any():
            row[ok] = predict_rows(blk[ok]).astype(dtype)
        out[r0:r1] = row.reshape(r1 - r0, W)
    return out


def train(X, Xs, cells):
    """Fit the chosen model on the painted cells and predict the preview grid.
    (None, None) until two units are painted."""
    model = backend().fit(X, cells)
    if model is None:
        return None, None
    return model, backend().predict(model, Xs)


def predict(model, X):
    return backend().predict(model, X)


def per_pixel(values, shape):
    """A (GRID, GRID) cell array spread onto a raster grid, on the painter's cell edges."""
    rows = np.searchsorted([i * shape[0] // GRID for i in range(1, GRID)],
                           np.arange(shape[0]), side="right")
    cols = np.searchsorted([j * shape[1] // GRID for j in range(1, GRID)],
                           np.arange(shape[1]), side="right")
    return values[np.ix_(rows, cols)]


def cells_to_labels(cells, shape, valid=None):
    """Blow the 120x120 painting up to the raster grid, 255 where nothing was painted."""
    lab = per_pixel(cells, shape).astype(np.uint8)
    lab[lab == 0] = 255
    if valid is not None:
        if valid.shape != lab.shape:
            raise ValueError("Terrain mask must match the output label grid")
        lab[~valid] = 255
    return lab


def labels_to_cells(lab, k=4):
    """The majority unit of a unit raster in each painting cell, 0 where a cell has no
    unit -- the reverse of cells_to_labels. Samples k x k points per cell, so it works
    for rasters coarser or finer than the painting grid."""
    H, W = lab.shape
    r = ((np.arange(GRID * k) + .5) * H / (GRID * k)).astype(int)
    c = ((np.arange(GRID * k) + .5) * W / (GRID * k)).astype(int)
    blocks = lab[np.ix_(r, c)].reshape(GRID, k, GRID, k)
    counts = np.stack([(blocks == code).sum((1, 3)) for code in CODES])
    best = np.array(list(CODES), np.uint8)[counts.argmax(0)]
    return np.where(counts.max(0) > 0, best, 0).astype(np.uint8)


def smart_fill(suggested, seed=None, inside=None, unit=None, min_patch=4):
    """Painting cells chosen from a prediction on the painting grid (`suggested`, 0 where
    none). With `seed` (row, col): the connected region the model predicts as the same unit
    as that cell. With `inside` (a cell mask) and `unit`: the cells in it predicted as
    `unit`, without specks. Holes smaller than `min_patch` cells are filled either way."""
    if seed is not None:
        if suggested[seed] == 0:
            return np.zeros(suggested.shape, bool)
        parts, _ = ndi.label(suggested == suggested[seed])
        region = parts == parts[seed]
    else:
        region = inside & (suggested == unit)
        parts, _ = ndi.label(region)
        region &= np.bincount(parts.ravel())[parts] >= min_patch
    holes, _ = ndi.label(~region)
    small = (np.bincount(holes.ravel())[holes] < min_patch) & (holes > 0)
    if inside is not None:
        small &= inside
    return (region | small) & (suggested > 0)


def write_map(path, arr, prof):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr, 1)
        d.write_colormap(1, CMAP)
