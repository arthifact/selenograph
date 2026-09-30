#!/usr/bin/env python3
"""Offline A3CLR22 public tiles (numpy + rasterio + standard library).

Defaults: data/raw_data/*.zip -> data/processed_data/<region>/.
The older data/raw_data/essentials/ layout is also accepted.
Each region contains metadata/, tiles/<stable-id>/ and working_dems.json with
region-relative paths. Tiles are 1024 square, with unpadded variable-sized edges;
the full native SfS extent and all-nodata tiles are retained. Full-region aligned
layers are temporary only and removed after tile verification, before publication.
The root manifest.json owns only the public selection. Other registered site
folders may coexist; their catalogs/paths are checked but their data is not rebuilt.

Run with --help. Existing published regions are immutable. --max-seconds is a
cooperative deadline (checked between IO chunks/blocks); an unfinished region is
restarted on resume. Unix-only: directory flock excludes concurrent writers and
is released automatically on exit/crash; verification takes a read-only shared
lock. Completion refers to the requested selection, not excluded regions.
No labels are created.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

PROJECT = Path(__file__).resolve().parent.parent
TILE_SIZE = 1024
CONFIG = {"schema": 3, "collection": "public", "grid": "native_sfs_tiles", "tile_size": TILE_SIZE,
          "edge_tiles": "retain_extent", "resampling": "nearest",
          "dtype": "float32", "compression": "DEFLATE", "nodata": "NaN",
          "scale_offset": "preserve_metadata_do_not_apply", "labels": False}
PRODUCTS = {"GLDOMOS": "nac", "GLDELEV": "sfs", "GLDMASK": "image_count",
            "GLDSBCT": "solar_bins", "GLDBRES": "best_resolution",
            "GLDSIGM": "uncertainty", "GLDISGM": "uncertainty"}
REQUIRED = {"nac", "sfs", "image_count", "readme"}
OPTIONAL = {"solar_bins", "best_resolution", "uncertainty"}
MASKS = {"valid_data": "finite, unmasked NAC AND SfS; not an annotation label",
         "sfs_support": "finite, unmasked SfS AND image_count > 0; not an annotation label"}
CAVEATS = ["Full SfS extent retained, including nodata; no filling or normalization.",
           "Nearest-neighbor alignment does not create native 5 m detail: inspect each native grid.",
           "NAC may have a half-pixel offset; quality may be 15 m despite a _005 filename.",
           "GLDSIGM/GLDISGM naming differs between some files and README uncertainty descriptions; consult preserved README.",
           "NAC is a maximally-lit composite, not uniform-illumination calibrated reflectance.",
           "Raw samples retained without applying scale/offset; float64 to float32 may round radiometry.",
           "Count samples must be integral and exactly representable in float32; compression is lossless.",
           "Zero support flags are not zero-valued annotation labels; no labels or ML products exist."]
ARCHIVE_RE = re.compile(r"A3CLR22_(\d+)_([A-Za-z0-9_]+)\.zip\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkpoint(deadline=None):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Time budget reached; published regions remain reusable.")


def safe_path(path):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlink path: {path}")
    return path


@contextmanager
def dataset_lock(output, verify=False):
    """Unix advisory lock on the directory inode: no lock file or stale cleanup."""
    safe_path(output)
    if not verify:
        output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, (fcntl.LOCK_SH if verify else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Dataset in use by another process: {output}") from exc
        yield
    finally:
        os.close(fd)


def digest(path, deadline=None):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while True:
            checkpoint(deadline)
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "bytes": path.stat().st_size}


def read_json(path):
    safe_path(path)
    require(path.is_file() and path.stat().st_size <= 16 * 1024 * 1024, f"Invalid JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    # A sibling-filesystem temporary file keeps interrupted replacements out of inventories.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent.parent,
                                     prefix=".selenograph-json-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def discover(raw):
    found = {}
    archives = [*raw.glob("*.zip"), *(raw / "essentials").glob("*.zip")]
    for path in sorted(archives):
        match = ARCHIVE_RE.fullmatch(path.name)
        if match:
            safe_path(path)
            region = match[2].lower()
            require(region not in found, f"Duplicate region: {region}")
            found[region] = path
    require(found, f"No A3CLR22 archives in {raw}; download the regional ZIPs from Zenodo first")
    return found


def inventory(archive):
    """Select exact top-level regional products, never subtrees or GLDSBCTALL."""
    selected, seen = {}, set()
    match = ARCHIVE_RE.fullmatch(archive.name)
    if match is None:
        raise ValueError(f"Invalid archive name: {archive.name}")
    roi = f"A3{int(match[1]):02d}"
    product_re = re.compile(rf"{roi}_(GLDOMOS|GLDELEV|GLDMASK|GLDSBCT|GLDBRES|GLDSIGM|GLDISGM)_\d+\.tif", re.I)
    with zipfile.ZipFile(archive) as z:
        require(len(z.infolist()) <= 100000, f"Excessive ZIP inventory: {archive}")
        for info in z.infolist():
            name, mode = info.filename, info.external_attr >> 16
            parts = PurePosixPath(name).parts
            require(name == info.orig_filename and name and not name.startswith("/")
                    and "\\" not in name and ":" not in name
                    and all(p not in ("", ".", "..") for p in name.rstrip("/").split("/")),
                    f"Unsafe ZIP path: {name!r}")
            require(stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR), f"Special ZIP member: {name}")
            canonical = name.rstrip("/").casefold()
            require(canonical not in seen, f"Duplicate ZIP member: {name}")
            seen.add(canonical)
            if info.is_dir() or any(p.startswith(".") or p == "__MACOSX" for p in parts):
                continue
            if len(parts) == 2 and parts[0] == archive.stem:
                basename = parts[1]
            elif len(parts) == 1:
                basename = parts[0]
            else:
                continue
            match = product_re.fullmatch(basename)
            key = PRODUCTS[match[1].upper()] if match else None
            if re.fullmatch(r"README(?:\.(?:txt|md))?", basename, re.I):
                key = "readme"
            elif basename.lower().endswith(".csv") and info.file_size <= 1024 * 1024:
                key = f"evidence_{sum(k.startswith('evidence_') for k in selected):02d}"
            if key is None:
                continue
            require(key not in selected, f"Ambiguous root product {key}: {archive}")
            require(not info.flag_bits & 1 and 0 <= info.file_size <= 64 * 1024**3
                    and info.file_size <= max(1, info.compress_size) * 10000, f"Unsafe ZIP size/encryption: {name}")
            if key == "readme":
                require(info.file_size <= 1024 * 1024, "README too large")
            selected[key] = info
    require(REQUIRED <= selected.keys(), f"Missing required products in {archive}: {REQUIRED - selected.keys()}")
    require(sum(i.file_size for i in selected.values()) <= 128 * 1024**3 and len(selected) <= 24,
            "Selected ZIP contents exceed safety limits")
    return selected


def archive_record(archive, published, deadline=None):
    record = {"name": archive.name, **digest(archive, deadline)}
    expected = published.get(archive.name) if published is not None else None
    if published is not None:
        if expected is None or not re.fullmatch(r"md5:[0-9a-fA-F]{32}", expected):
            raise ValueError(f"Missing/invalid published MD5: {archive.name}")
        require(record["md5"] == expected[4:].lower(), f"Published MD5 mismatch: {archive.name}")
    record["published_md5"] = expected[4:].lower() if expected else None
    return record


def grid(src):
    return {"width": src.width, "height": src.height, "transform": list(src.transform)[:6],
            "crs_wkt": src.crs.to_wkt() if src.crs else None, "spacing": list(src.res)}


def validate_sfs_grid(src):
    """Reject geographic, rotated, singular, flipped or nonfinite target grids."""
    t = src.transform
    require(src.count == 1 and src.width > 0 and src.height > 0
            and src.crs is not None and src.crs.is_projected
            and all(math.isfinite(v) for v in (*t, *src.bounds, t.determinant))
            and t.a > 0 and t.e < 0 and t.b == t.d == 0 and t.determinant < 0,
            "Invalid native SfS grid: require finite north-up projected grid with positive spacing")
    return grid(src)


def native_record(src):
    nodata = src.nodata
    return {"grid": grid(src), "dtype": src.dtypes[0],
            "nodata": str(nodata) if nodata is not None and not math.isfinite(nodata) else nodata,
            "scales": list(src.scales), "offsets": list(src.offsets), "units": list(src.units),
            "tags": src.tags(), "band_tags": src.tags(1), "descriptions": list(src.descriptions),
            "mask_flags": [[f.name for f in flags] for flags in src.mask_flag_enums]}


def layer_path(tile_id, layer):
    folder = "imagery" if layer == "nac" else "elevation" if layer == "sfs" else "quality"
    stem = tile_id if layer == "sfs" else f"{tile_id}_{layer}"
    return f"tiles/{tile_id}/{folder}/{stem}.tif"


def tile_layout(region, target):
    """Canonical row-major, zero-based windows: no gaps, overlap, padding or crop."""
    require(all(type(target[k]) is int and target[k] > 0 for k in ("width", "height")), "Invalid coverage grid")
    transform = rasterio.Affine(*target["transform"])
    for row, top in enumerate(range(0, target["height"], TILE_SIZE)):
        for col, left in enumerate(range(0, target["width"], TILE_SIZE)):
            width, height = min(TILE_SIZE, target["width"] - left), min(TILE_SIZE, target["height"] - top)
            window = {"col_off": left, "row_off": top, "width": width, "height": height}
            tile_grid = {**target, "width": width, "height": height,
                         "transform": list(transform * rasterio.Affine.translation(left, top))[:6]}
            yield {"tile_id": f"zenodo-{region}-r{row:03d}-c{col:03d}", "row": row, "col": col,
                   "window": window, "grid": tile_grid}


def tile_counts(tile, outputs):
    pixels = {key: outputs[layer_path(tile["tile_id"], key)]["pixels"] for key in ("sfs", *MASKS)}
    return {"valid_count": pixels["sfs"]["valid"],
            **{f"{key}_count": pixels[key]["total"] - pixels[key]["zero"] for key in MASKS}}


def catalog_record(region, tile, layers):
    paths = {key: layer_path(tile["tile_id"], key) for key in sorted(layers)}
    return {"working_dem": paths["sfs"], "tile_id": tile["tile_id"], "group": region, "site": region,
            "collection": "public", "size": [tile["grid"]["height"], tile["grid"]["width"]],
            "transform": tile["grid"]["transform"], "layers": paths,
            **{key: tile[key] for key in ("valid_count", "valid_data_count", "sfs_support_count")}}


def evidence_path(region, key):
    return f"metadata/{region}_{key}.{'txt' if key == 'readme' else 'csv'}"


def profile(target, nodata: float | None = float("nan")):
    return {"driver": "GTiff", "width": target["width"], "height": target["height"],
            "transform": rasterio.Affine(*target["transform"]), "crs": target["crs_wkt"],
            "count": 1, "dtype": "float32", "nodata": nodata, "tiled": True,
            "blockxsize": 256, "blockysize": 256, "compress": "deflate", "predictor": 3,
            "BIGTIFF": "IF_SAFER"}


def source_with_alpha(src):
    """Expose the intrinsic source mask as alpha; GDAL warp can ignore mask bands."""
    filename = escape(os.path.abspath(src.name))
    nodata = f"<NoDataValue>{src.nodata}</NoDataValue>" if src.nodata is not None else ""
    bands = "".join(
        f'<VRTRasterBand dataType="{dtype}" band="{number}">{extra}'
        f'<SimpleSource><SourceFilename relativeToVRT="0">{filename}</SourceFilename>'
        f'<SourceBand>{band}</SourceBand></SimpleSource></VRTRasterBand>'
        for number, dtype, band, extra in ((1, "Float64", "1", nodata),
                                          (2, "Byte", "mask,1", "<ColorInterp>Alpha</ColorInterp>")))
    return (f'<VRTDataset rasterXSize="{src.width}" rasterYSize="{src.height}">'
            f'<SRS>{escape(src.crs.to_wkt())}</SRS>'
            f'<GeoTransform>{",".join(map(str, src.transform.to_gdal()))}</GeoTransform>'
            f'{bands}</VRTDataset>').encode("utf-8")


def align_raster(original, destination, target, layer, deadline=None):
    """Warp raw samples, respecting source masks and nonfinite values; never fill."""
    with rasterio.open(original) as src:
        require(src.count == 1 and src.crs is not None and src.dtypes[0] in
                ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"),
                f"Unsupported raster: {original}")
        record = native_record(src)
        with MemoryFile(source_with_alpha(src)) as memory, memory.open() as masked, \
                WarpedVRT(masked, crs=target["crs_wkt"], transform=rasterio.Affine(*target["transform"]),
                          width=target["width"], height=target["height"], dtype="float64", src_alpha=2,
                          nodata=float("nan"), resampling=Resampling.nearest, warp_mem_limit=64) as vrt, \
                rasterio.open(destination, "w", **profile(target)) as dst:
            dst.scales, dst.offsets = src.scales, src.offsets
            if src.units[0]:
                dst.set_band_unit(1, src.units[0])
            for _, window in dst.block_windows(1):
                checkpoint(deadline)
                values = vrt.read(1, window=window, masked=True)
                valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data) & (vrt.read(2, window=window) > 0)
                # An explicit alpha mask can take precedence over declared nodata in GDAL.
                if src.nodata is not None:
                    valid &= values.data != src.nodata
                with np.errstate(over="ignore", invalid="ignore"):
                    result = np.where(valid, values.data, np.nan).astype("float32")
                require(np.isfinite(result[valid]).all(), f"float32 overflow: {original}")
                if layer in ("image_count", "solar_bins"):
                    samples = values.data[valid]
                    require(np.all(samples >= 0) and np.all(samples == np.floor(samples))
                            and np.all(result[valid].astype("float64") == samples), f"Inexact/invalid counts: {original}")
                dst.write(result, 1, window=window)
    return record


def write_tile(original, destination, tile, deadline=None):
    window = Window(**tile["window"])
    with rasterio.open(original) as src, rasterio.open(destination, "w", **profile(tile["grid"])) as dst:
        dst.scales, dst.offsets = src.scales, src.offsets
        if src.units[0]:
            dst.set_band_unit(1, src.units[0])
        for _, block in dst.block_windows(1):
            checkpoint(deadline)
            row, col = window.row_off + block.row_off, window.col_off + block.col_off
            source_window = Window.from_slices(rows=(row, row + block.height), cols=(col, col + block.width))
            dst.write(src.read(1, window=source_window), 1, window=block)


def write_masks(root, tile_id, target, deadline=None):
    with rasterio.open(root / layer_path(tile_id, "nac")) as nac, \
            rasterio.open(root / layer_path(tile_id, "sfs")) as sfs, \
            rasterio.open(root / layer_path(tile_id, "image_count")) as count:
        for name in MASKS:
            with rasterio.open(root / layer_path(tile_id, name), "w", **profile(target, None)) as dst:
                for _, w in dst.block_windows(1):
                    checkpoint(deadline)
                    good = np.isfinite(sfs.read(1, window=w))
                    other = nac.read(1, window=w) if name == "valid_data" else count.read(1, window=w)
                    good &= np.isfinite(other) & (True if name == "valid_data" else other > 0)
                    dst.write(good.astype("float32"), 1, window=w)


def raster_stats(path, target, is_mask, deadline=None):
    valid = zeros = 0
    with rasterio.open(path) as src:
        require(validate_sfs_grid(src) == target and src.dtypes == ("float32",), f"Grid/type mismatch: {path}")
        require(src.compression and src.compression.value.upper() == "DEFLATE", f"Compression mismatch: {path}")
        require(src.nodata is None if is_mask else src.nodata is not None and math.isnan(src.nodata), f"Nodata mismatch: {path}")
        for _, w in src.block_windows(1):
            checkpoint(deadline)
            a = src.read(1, window=w)
            finite = np.isfinite(a)
            require(not np.isinf(a).any() and np.array_equal(src.read_masks(1, window=w) > 0, finite), f"Invalid mask: {path}")
            if is_mask:
                require(np.isin(a, [0, 1]).all(), f"Nonbinary support mask: {path}")
            valid += int(finite.sum())
            zeros += int((a == 0).sum())
        return {"total": src.width * src.height, "valid": valid,
                "nodata": src.width * src.height - valid, "zero": zeros}


def tree(root, *, allow_hardlinks=False):
    files, directories = set(), set()
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(parent) / name
            require(not path.is_symlink(), f"Symlink in output: {path}")
            require(path.is_dir() or (path.is_file() and (allow_hardlinks or path.stat().st_nlink == 1)),
                    f"Special/linked output: {path}")
            (directories if path.is_dir() else files).add(path.relative_to(root).as_posix())
    return files, directories


def registered_path(output, parent, value, must_exist=False, *, allow_hardlinks=False):
    """Check catalog paths without following links or leaving the processed root."""
    require(isinstance(value, str) and value and not any(c in value for c in ("\x00", "\\", ":")),
            f"Invalid registered path: {value!r}")
    candidate = parent / value
    require(candidate.is_relative_to(output), f"Registered path outside output: {value!r}")
    path = output
    # Inspect before normalizing '..': a symlink or hidden component must not be
    # concealed by a later parent step. Sibling-site references are permitted.
    for part in candidate.relative_to(output).parts:
        if part == "..":
            require(path != output, f"Registered path outside output: {value!r}")
            path = path.parent
        else:
            require(not part.startswith("."), f"Hidden registered path: {value!r}")
            path = path / part
            require(not path.is_symlink(), f"Symlink registered path: {path}")
    if must_exist or path.exists():
        require(path.is_file() and (allow_hardlinks or path.stat().st_nlink == 1), f"Invalid registered file: {path}")
    return path


def registered_ids(root, output):
    """Validate generic catalogs, not public provenance or raster contents."""
    require((root / "working_dems.json").is_file(), f"Unregistered output directory: {root}")
    files, _ = tree(root, allow_hardlinks=True)  # Never follow directory links.
    ids, raster_paths = set(), set()
    for name in sorted(n for n in files if PurePosixPath(n).name == "working_dems.json"):
        catalog = registered_path(output, root, name, must_exist=True)
        records = read_json(catalog)
        require(isinstance(records, list) and records, f"Invalid registered catalog: {catalog}")

        def path(value, must_exist=False, *, raster=False, parent=catalog.parent):
            result = registered_path(output, parent, value, must_exist, allow_hardlinks=raster)
            if raster:
                raster_paths.add(result)
            return result

        def layers(value, catalog=catalog):
            if isinstance(value, str):
                path(value, raster=True)
            elif isinstance(value, dict):
                if "path" in value:
                    if value["path"] is not None:
                        path(value["path"], raster=True)
                elif "status" not in value:
                    for nested in value.values():
                        layers(nested)
            else:
                require(value is None, f"Invalid registered layer: {catalog}")

        for record in records:
            require(isinstance(record, dict), f"Invalid registered record: {catalog}")
            dem = path(record.get("working_dem"), must_exist=True, raster=True)
            require(dem.stem not in ids, f"Duplicate map ID {dem.stem!r}: {catalog}")
            ids.add(dem.stem)
            if "layers" in record:
                require(isinstance(record["layers"], dict), f"Invalid registered layers: {catalog}")
                for value in record["layers"].values():
                    layers(value)
            if "annotations" in record:
                annotations = record["annotations"]
                require(isinstance(annotations, dict), f"Invalid registered annotations: {catalog}")
                for kind in ("painting", "confidence", "accepted", "labels"):
                    if annotations.get(kind) is not None:
                        path(annotations[kind], raster=kind == "labels")
            if "feature_sources" in record:
                require(isinstance(record["feature_sources"], list), f"Invalid registered feature_sources: {catalog}")
                for value in record["feature_sources"]:
                    path(value, raster=True)
    # Declared raster inputs are immutable and may share preserved-source inodes.
    # Catalogs, metadata and undeclared files retain the single-link restriction.
    for name in files:
        file_path = root / name
        require(file_path.stat().st_nlink == 1 or file_path in raster_paths, f"Special/linked output: {file_path}")
    return ids


def verify_region(root, region, archive, selected, deadline=None):
    """Validate exact tile coverage, catalog, inventory, hashes, grids and masks."""
    metadata_name = f"metadata/{region}_metadata.json"
    m = read_json(root / metadata_name)
    require(m["config"] == CONFIG and m["region"] == region
            and all(m["archive"][k] == archive[k] for k in ("name", "bytes", "sha256", "md5"))
            and m["archive"]["published_md5"] in (None, archive["md5"]), f"Region source/config mismatch: {region}")
    require(set(m["sources"]) == set(selected), f"Source selection mismatch: {region}")
    expected = {metadata_name, f"metadata/{region}_quality_report.json", "working_dems.json"}
    raster_keys = set()
    for key, info in selected.items():
        source = m["sources"][key]
        require((source["member"], source["bytes"], source["crc32"]) == (info.filename, info.file_size, info.CRC),
                f"Source inventory mismatch: {region}/{key}")
        if key == "readme" or key.startswith("evidence_"):
            expected.add(evidence_path(region, key))
        else:
            raster_keys.add(key)
    require(m["grid"] == m["sources"]["sfs"]["native"]["grid"] and m["mask_definitions"] == MASKS,
            f"SfS grid/mask definition mismatch: {region}")
    layout = list(tile_layout(region, m["grid"]))
    require(m["tile_count"] == len(layout) == len(m["tiles"]), f"Tile coverage/count mismatch: {region}")
    rasters = {}
    for tile, canonical in zip(m["tiles"], layout):
        checkpoint(deadline)
        require(all(tile[key] == value for key, value in canonical.items()), f"Tile coverage/window mismatch: {region}")
        rasters.update({layer_path(tile["tile_id"], key): (tile, key) for key in raster_keys | MASKS.keys()})
    expected.update(rasters)
    expected_dirs = {p.as_posix() for name in expected for p in PurePosixPath(name).parents if p != PurePosixPath(".")}
    files, dirs = tree(root)
    require(files == expected and dirs == expected_dirs, f"Unlisted/missing paths: {region}")
    require(set(m["outputs"]) == expected - {metadata_name}, f"Output inventory mismatch: {region}")
    for name, record in m["outputs"].items():
        path = root / name
        require(digest(path, deadline) == record["checksum"], f"Modified output: {path}")
        if name in rasters:
            tile, layer = rasters[name]
            require(record["layer"] == layer and record["grid"] == tile["grid"]
                    and record["tile_id"] == tile["tile_id"], f"Layer/grid mismatch: {path}")
            require(raster_stats(path, tile["grid"], layer in MASKS, deadline) == record["pixels"], f"Pixel counts mismatch: {path}")
            with rasterio.open(path) as src:
                native = m["sources"][layer]["native"] if layer not in MASKS else {"scales": [1.0], "offsets": [0.0], "units": [None]}
                require(all(list(getattr(src, k)) == native[k] for k in ("scales", "offsets", "units")), f"Units/scales mismatch: {path}")
    for tile in m["tiles"]:
        tile_id = tile["tile_id"]
        require(all(tile[key] == value for key, value in tile_counts(tile, m["outputs"]).items()), f"Tile valid counts mismatch: {tile_id}")
        with rasterio.open(root / layer_path(tile_id, "nac")) as nac, rasterio.open(root / layer_path(tile_id, "sfs")) as sfs, \
                rasterio.open(root / layer_path(tile_id, "image_count")) as count:
            for name in MASKS:
                with rasterio.open(root / layer_path(tile_id, name)) as mask:
                    for _, w in mask.block_windows(1):
                        checkpoint(deadline)
                        other = nac.read(1, window=w) if name == "valid_data" else count.read(1, window=w)
                        expected_mask = np.isfinite(sfs.read(1, window=w)) & np.isfinite(other)
                        if name == "sfs_support":
                            expected_mask &= other > 0
                        require(np.array_equal(mask.read(1, window=w), expected_mask), f"Support mismatch: {tile_id}/{name}")
    require(read_json(root / "working_dems.json") ==
            [catalog_record(region, tile, raster_keys | MASKS.keys()) for tile in m["tiles"]], f"Catalog path/record mismatch: {region}")
    quality = read_json(root / f"metadata/{region}_quality_report.json")
    require(quality == m["quality"] and quality["missing_optional"] == sorted(OPTIONAL - selected.keys()), f"Quality report mismatch: {region}")
    return {"archive": m["archive"], "tile_count": len(layout),
            "metadata_sha256": digest(root / metadata_name, deadline)["sha256"]}


def build_region(output, region, archive, source, selected, deadline=None, *, other_ids=()):
    """Extract with CRC checking, then atomically publish; caller holds dataset_lock."""
    scratch = output / ".staging"
    scratch.mkdir()
    try:
        originals, aligned, root = scratch / "originals", scratch / "aligned", scratch / region
        originals.mkdir()
        aligned.mkdir()
        (root / "metadata").mkdir(parents=True)
        total = sum(i.file_size for i in selected.values())
        require(shutil.disk_usage(output).free > total + 64 * 1024**2, "Insufficient extraction disk space")
        sources = {}
        with zipfile.ZipFile(archive) as z:
            for key, info in selected.items():
                destination = originals / key
                with z.open(info) as src, destination.open("xb") as dst:
                    written = 0
                    while True:
                        checkpoint(deadline)
                        block = src.read(1024 * 1024)  # Reading through EOF checks the full member CRC.
                        if not block:
                            break
                        written += len(block)
                        require(written <= info.file_size, f"Expanded ZIP member: {info.filename}")
                        dst.write(block)
                require(written == info.file_size, f"Truncated ZIP member: {info.filename}")
                sources[key] = {"member": info.filename, "crc32": info.CRC, **digest(destination, deadline)}
        with rasterio.open(originals / "sfs") as sfs:
            target = validate_sfs_grid(sfs)
        require(not any(tile["tile_id"] in other_ids for tile in tile_layout(region, target)),
                f"Duplicate map ID in region: {region}")
        raster_keys = set(selected) - {k for k in selected if k == "readme" or k.startswith("evidence_")}
        needed = target["width"] * target["height"] * 4 * (len(raster_keys) + len(MASKS)) * 2
        require(shutil.disk_usage(output).free > needed + 64 * 1024**2, "Insufficient raster disk space")
        for key in selected:
            if key in raster_keys:
                sources[key]["native"] = align_raster(originals / key, aligned / f"{key}.tif", target, key, deadline)
            else:
                shutil.copyfile(originals / key, root / evidence_path(region, key))
        tiles, outputs = [], {}
        layers = raster_keys | MASKS.keys()
        for tile in tile_layout(region, target):
            checkpoint(deadline)
            tile_id = tile["tile_id"]
            for folder in ("imagery", "elevation", "quality"):
                (root / "tiles" / tile_id / folder).mkdir(parents=True)
            for key in sorted(raster_keys):
                write_tile(aligned / f"{key}.tif", root / layer_path(tile_id, key), tile, deadline)
            write_masks(root, tile_id, tile["grid"], deadline)
            for key in sorted(layers):
                name = layer_path(tile_id, key)
                outputs[name] = {"checksum": digest(root / name, deadline), "layer": key, "tile_id": tile_id,
                                 "grid": tile["grid"], "pixels": raster_stats(root / name, tile["grid"], key in MASKS, deadline)}
            tile.update(tile_counts(tile, outputs))
            tiles.append(tile)
        quality = {"missing_optional": sorted(OPTIONAL - selected.keys()), "warnings": CAVEATS +
                   ([] if source["published_md5"] else ["No published MD5 supplied; local SHA256 is provenance, not publisher authentication."])}
        write_json(root / f"metadata/{region}_quality_report.json", quality)
        write_json(root / "working_dems.json", [catalog_record(region, tile, layers) for tile in tiles])
        for name in ["working_dems.json", f"metadata/{region}_quality_report.json",
                     *(evidence_path(region, key) for key in selected if key not in raster_keys)]:
            outputs[name] = {"checksum": digest(root / name, deadline)}
        write_json(root / f"metadata/{region}_metadata.json",
                   {"config": CONFIG, "region": region, "archive": source, "grid": target,
                    "tile_count": len(tiles), "tiles": tiles, "sources": sources,
                    "outputs": outputs, "quality": quality, "mask_definitions": MASKS})
        result = verify_region(root, region, source, selected, deadline)
        # Keep the aligned sources until all tiles pass; never publish a regional copy.
        shutil.rmtree(aligned)
        shutil.rmtree(originals)
        checkpoint(deadline)
        require(not (output / region).exists(), f"Refusing to overwrite region: {region}")
        root.rename(output / region)
        return result
    finally:
        shutil.rmtree(scratch)


def run(args):
    deadline = time.monotonic() + args.max_seconds if args.max_seconds is not None else None
    raw, output = safe_path(args.raw_data), safe_path(args.output)
    protected = {raw, PROJECT / "raw_data", PROJECT / "reference_data", raw.parent / "reference_data",
                 PROJECT / "data/raw_data", PROJECT / "data/reference_data"}
    require(all(output != p and p not in output.parents and output not in p.parents for p in protected)
            and "reference_data" not in output.parts, "Output overlaps raw/reference data")
    with dataset_lock(output, args.verify):
        return run_locked(args, raw, output, deadline)


def run_locked(args, raw, output, deadline=None):
    """Preparation/verification body; caller must hold dataset_lock throughout."""
    archives = discover(raw)
    selectors = args.region or []
    selected_regions = set()
    for selector in selectors:
        matches = [r for r, p in archives.items() if (match := ARCHIVE_RE.fullmatch(p.name)) is not None
                   and selector.lower() in (r, p.stem.lower(), match[1])]
        require(len(matches) == 1, f"Unknown/ambiguous region: {selector}")
        selected_regions.update(matches)
    if not selectors and not args.verify:
        selected_regions = set(archives)
    published, record_path = None, safe_path(raw / "zenodo-record-17954508.json")
    if record_path.exists():
        entries = read_json(record_path)["files"]
        published = {}
        for item in entries:
            require(item["key"] not in published, "Duplicate Zenodo file key")
            published[item["key"]] = item["checksum"]
    manifest_path = safe_path(output / "manifest.json")
    has_manifest = manifest_path.exists()
    if has_manifest:
        manifest = read_json(manifest_path)
        require(manifest["config"] == CONFIG and manifest["status"] in ("partial", "complete"), "Unknown output configuration")
    else:
        require(not args.verify, "No dataset manifest")

        manifest = {"config": CONFIG, "collection": "public", "status": "partial", "requested_regions": [],
                    "included_regions": [], "regions": {}, "requested_archives": {}, "tile_count": 0,
                    "completion_scope": "requested_regions", "available_regions": sorted(archives), "all_available_regions_included": False}
    previous = set(manifest["requested_regions"])
    require(manifest["collection"] == "public" and manifest["included_regions"] == sorted(manifest["regions"])
            and manifest["requested_regions"] == sorted(previous)
            and manifest["tile_count"] == sum(r["tile_count"] for r in manifest["regions"].values())
            and set(manifest["regions"]) <= previous
            and (manifest["status"] != "complete" or set(manifest["regions"]) == previous), "Inconsistent manifest")
    require(manifest["completion_scope"] == "requested_regions"
            and previous <= set(manifest["available_regions"])
            and type(manifest["all_available_regions_included"]) is bool
            and manifest["all_available_regions_included"] ==
            (manifest["status"] == "complete" and set(manifest["regions"]) == set(manifest["available_regions"])),
            "Inconsistent completion scope")
    requested = previous | selected_regions
    require(requested <= archives.keys(), "Unknown manifest regions")
    require(manifest["requested_archives"] == {r: archives[r].name for r in sorted(previous)}, "Requested archive inventory mismatch")
    present, other_ids = set(), set()
    if output.exists():
        for path in output.iterdir():
            safe_path(path)
            if path.name == "manifest.json":
                require(path.is_file() and path.stat().st_nlink == 1, "Invalid manifest path")
            elif path.name == ".staging" and has_manifest and manifest["status"] == "partial" and not args.verify:
                require(path.is_dir(), "Invalid staging directory")
                tree(path)  # Never traverse symlinks when discarding interrupted work.
            elif path.name in previous:
                require(path.is_dir() and (manifest["status"] == "partial" or path.name in manifest["regions"]),
                        f"Unlisted output: {path}")
                present.add(path.name)
            else:
                require(path.is_dir() and not path.name.startswith(".") and path.name not in requested,
                        f"Unlisted output: {path}")
                ids = registered_ids(path, output)
                require(not ids & other_ids, f"Duplicate map IDs: {sorted(ids & other_ids)}")
                other_ids.update(ids)
    require(set(manifest["regions"]) <= present, "Missing published region directory")
    if args.verify:
        require(present == set(manifest["regions"]), "Unfinalized region; resume preparation first")
    results = {}
    for region in sorted(present):
        members = inventory(archives[region])
        source = archive_record(archives[region], published, deadline)
        result = verify_region(output / region, region, source, members, deadline)
        if region in manifest["regions"]:
            require(result == manifest["regions"][region], f"Modified region metadata: {region}")
        ids = {r["tile_id"] for r in read_json(output / region / "working_dems.json")}
        require(not ids & other_ids, f"Duplicate map IDs: {sorted(ids & other_ids)}")
        results[region] = result
    if args.verify:
        require(manifest["status"] == "complete" and present == set(manifest["requested_regions"])
                and selected_regions <= present, "Dataset is partial; resume preparation")
        print(f"Verified requested selection: {len(results)} of {len(archives)} available regions, {manifest['tile_count']} public tiles (read-only).")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": CONFIG, "collection": "public", "status": "partial", "completion_scope": "requested_regions",
                "available_regions": sorted(archives), "all_available_regions_included": False,
                "requested_regions": sorted(requested), "requested_archives": {r: archives[r].name for r in sorted(requested)},
                "included_regions": sorted(results), "regions": results,
                "tile_count": sum(r["tile_count"] for r in results.values())}
    write_json(manifest_path, manifest)
    if (output / ".staging").exists():
        shutil.rmtree(output / ".staging")
    for region in sorted(requested - results.keys()):
        checkpoint(deadline)
        members = inventory(archives[region])
        source = archive_record(archives[region], published, deadline)
        results[region] = build_region(output, region, archives[region], source, members, deadline, other_ids=other_ids)
        manifest["included_regions"] = sorted(results)
        manifest["tile_count"] = sum(r["tile_count"] for r in results.values())
        write_json(manifest_path, manifest)
        print(f"Published {region}: {results[region]['tile_count']} tiles", flush=True)
    # Every included region was validated in this invocation; only now claim completeness.
    manifest["status"] = "complete"
    manifest["all_available_regions_included"] = set(results) == set(archives)
    write_json(manifest_path, manifest)
    print(f"Completed requested selection: {len(results)} of {len(archives)} available regions, {manifest['tile_count']} public tiles in {output}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, default=PROJECT / "data/raw_data")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/processed_data")
    parser.add_argument("--region", action="append", help="Region name, archive stem, or numeric ID; repeat for a subset")
    parser.add_argument("--verify", action="store_true", help="Read-only verification of all included regions and source archives")
    parser.add_argument("--max-seconds", type=float, help="Cooperative time budget; exit 2 on interruption; resume with the same command")
    args = parser.parse_args(argv)
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("--max-seconds must be finite and positive")
    try:
        with rasterio.Env(GDAL_CACHEMAX=64 * 1024**2, GDAL_PAM_ENABLED="NO"):
            return run(args)
    except (TimeoutError, KeyboardInterrupt) as exc:
        print(f"Incomplete: {exc or 'interrupted'}", file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, RasterioError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
