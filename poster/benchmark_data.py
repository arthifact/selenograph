"""Read-only, cell-level inputs for provisional geology feature ablations.

No fitting, label-dependent features, persistent caches, or output files. The return
matrix always has FEATURE_NAMES columns (including unavailable optional channels).
Targets are legacy 120 x 120 painting cells, NOT independent 5 m pixel truth.
"""
import json
import zipfile
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage as ndi

from app import core
from app.dataset_utils import (
    _json, _path, _Hashes, _timed, _Grid, _clean, _read, _metric_grid, _intersects, _warp, _edges, _cell_sum, _expand, _cell_mean, _class_counts, _snapshot, _claim,
    GRID, MIN_COVERAGE, REGIONS, GROUPS,
)
from app import dataset_utils


def _references(root, hashes):
    refs, count = dataset_utils._references(root, hashes)
    for ref in refs:
        ref["sums"] = np.zeros((GRID, GRID, len(FEATURE_NAMES)), np.float64)
        ref["counts"] = np.zeros((GRID, GRID, len(FEATURE_NAMES)), np.float64)
    return refs, count


TERRAIN = ["slope", "rough", "rel_local", "curv", "svf", "svf_local", "rel", "zscene"]
IMAGE = ["nac", "nac_contrast", "nac_texture_30m", "nac_texture_100m", "nac_gradient"]
QUALITY = ["image_count", "solar_bins", "best_resolution", "uncertainty"]
RADAR = ["radar_s1_log", "radar_cpr"]
LOLA = ["lola_slope", "lola_rough", "lola_rel_local", "lola_curv"]
FEATURE_NAMES = TERRAIN + IMAGE + QUALITY + RADAR + LOLA
def _eligible(saved, draft, accepted, ground, area, joint):
    """Sequential, mutually exclusive exclusions; support is deliberately absent."""
    remaining = np.ones(saved.shape, bool)
    excluded = {}
    for reason, good in (
        ("invalid_either_target", np.isin(saved, (1, 2, 3)) & np.isin(draft, (1, 2, 3))),
        ("accepted_prediction_either_snapshot", ~accepted),
        ("reference_terrain_below_90pct_full_cell", ground >= MIN_COVERAGE * area),
        ("public_joint_below_90pct_reference_terrain", (ground > 0) & (joint >= MIN_COVERAGE * ground)),
    ):
        excluded[reason] = int((remaining & ~good).sum())
        remaining &= good
    return remaining, excluded


def _gaussian_stats(values, metres, px):
    """Normalized convolution over finite samples, not zero-filled image statistics."""
    good = np.isfinite(values)
    sigma = core.window(metres, px)
    weight = ndi.gaussian_filter(good.astype(np.float64), sigma, mode="reflect")
    v = np.where(good, values, 0).astype(np.float64)
    mean = np.divide(ndi.gaussian_filter(v, sigma, mode="reflect"), weight,
                     out=np.full(values.shape, np.nan), where=weight > 1e-12)
    second = np.divide(ndi.gaussian_filter(v * v, sigma, mode="reflect"), weight,
                       out=np.full(values.shape, np.nan), where=weight > 1e-12)
    return mean, np.sqrt(np.maximum(second - mean * mean, 0))


def _image_features(nac, px):
    mean, std100 = _gaussian_stats(nac, 100, px)
    _, std30 = _gaussian_stats(nac, 30, px)
    # A local relative floor makes normalization scale-equivariant; constant fields
    # have zero contrast. The 1e-12 floor merely prevents division by numerical zero.
    floor = np.maximum(1e-6 * np.abs(mean), 1e-12)
    contrast = (nac - mean) / np.maximum(std100, floor)
    gy, gx = np.gradient(nac.astype(np.float64), px)
    values = (nac, contrast, std30, std100, np.hypot(gx, gy))
    return {name: _clean(value, np.isfinite(nac)) for name, value in zip(IMAGE, values)}


def _local_terrain(z, px):
    if not np.isfinite(z).any():
        return {name: np.full(z.shape, np.nan, np.float32) for name in LOLA}
    filled, holes = core.fill_nan(z)
    gy, gx = np.gradient(filled, px)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    values = (slope, core.local_std(slope, core.window(core.ROUGH_M, px)),
              filled - ndi.gaussian_filter(filled, core.window(core.LOCAL_M, px)),
              ndi.gaussian_laplace(filled, core.window(core.CURV_M, px)) / px ** 2)
    return {name: _clean(value, ~holes) for name, value in zip(LOLA, values)}


def _accumulate(ref, name, values):
    valid = ref["terrain"] & np.isfinite(values)
    index = FEATURE_NAMES.index(name)
    ref["sums"][..., index] += _cell_sum(np.where(valid, values, 0))
    ref["counts"][..., index] += _cell_sum(valid)


def _public_metadata(public_root, raw_root, regions, hashes):
    """Validate published metadata and available raw archives/members without extraction."""
    manifest = _json(public_root / "manifest.json")
    hashes.file(public_root / "manifest.json", "public/manifest.json")
    result = {}
    for region in sorted(regions):
        folder = _path(public_root, region)
        relative = f"metadata/{region}_metadata.json"
        meta = _json(_path(folder, relative))
        expected = manifest["regions"][region]["metadata_sha256"]
        hashes.file(_path(folder, relative), f"public/{region}/{relative}", expected)
        working = "working_dems.json"
        entry = meta.get("outputs", {}).get(working, {})
        hashes.file(folder / working, f"public/{region}/{working}",
                    entry.get("checksum", {}).get("sha256"))
        archive = meta.get("archive", {})
        raw = _path(raw_root, "essentials/" + archive["name"]) if archive.get("name") else None
        status = "not_available; source hashes retained from verified public metadata"
        if raw is not None and raw.is_file():
            hashes.file(raw, "raw/essentials/" + archive["name"], archive["sha256"])
            with zipfile.ZipFile(raw) as z:
                for name, source in meta.get("sources", {}).items():
                    with z.open(source["member"]) as stream:
                        actual = hashes.stream(stream)
                    if actual != source["sha256"]:
                        raise ValueError(f"SHA256 mismatch: {region} raw source {name}")
                    hashes.records[f"raw/{archive['name']}!{source['member']}"] = {
                        "sha256": actual, "verified_expected": True}
            status = "archive and listed source members verified"
        result[region] = {"metadata": meta, "raw_source_verification": status}
    return result


def _public_features(folder, record, grid, meta, hashes, timings):
    layers = record["layers"]
    dem_path = _path(folder, layers["sfs"])
    arrays = {}
    for name in ("sfs", "nac", "valid_data", "sfs_support", *QUALITY):
        relative = layers.get(name)
        if relative is None:
            if name in ("sfs", "nac", "valid_data", "sfs_support"):
                raise ValueError(f"Required public layer absent: {name}")
            arrays[name] = np.full(grid.shape, np.nan, np.float32)
            continue
        path = _path(folder, relative)
        expected = meta["outputs"][relative]["checksum"]["sha256"]
        with _timed(timings, "checksums"):
            hashes.file(path, f"public/{folder.name}/{relative}", expected)
        arrays[name], actual_grid = _read(path)
        if actual_grid != grid:
            raise ValueError(f"Processed public layers must already be aligned: {path}")
    px = _metric_grid(grid)
    if min(grid.shape) < 2 or max(grid.shape) > 1024:
        raise ValueError("Public feature context must be one tile, 2..1024 pixels per axis")
    with _timed(timings, "public_terrain"):
        if np.isfinite(arrays["sfs"]).any():
            stack, names, _, profile, _ = core.build(str(dem_path))
            profile = cast(dict[str, Any], profile)
            built_grid = _Grid((profile["height"], profile["width"]), profile["transform"], profile["crs"])
            if built_grid != grid:
                raise ValueError("core.build changed the public tile grid")
            features = {name: _clean(stack[names.index(name)], np.isfinite(arrays["sfs"]))
                        for name in TERRAIN}
            del stack
        else:
            features = {name: np.full(grid.shape, np.nan, np.float32) for name in TERRAIN}
    with _timed(timings, "public_image"):
        features.update(_image_features(arrays["nac"], px))
    with _timed(timings, "public_quality"):
        features.update({name: arrays[name] for name in QUALITY})
        joint = np.isfinite(arrays["sfs"]) & np.isfinite(arrays["nac"])
        for name in ("valid_data", "sfs_support"):
            values = arrays[name]
            if np.any(np.isfinite(values) & ~np.isin(values, (0, 1))):
                raise ValueError(f"Nonbinary public mask: {name}")
        if not np.array_equal(joint, arrays["valid_data"] == 1):
            raise ValueError("Public valid_data mask disagrees with actual NAC/SfS validity")
        support = (arrays["sfs_support"] == 1) & np.isfinite(arrays["sfs"])
    return features, joint, support


def _optional(ref, root, lola_path, hashes, timings):
    available: dict[str, dict[str, Any]] = {}
    for name, layer_name in zip(RADAR, ("radar-s1", "radar-cpr")):
        layer = ref["record"]["layers"].get(layer_name, {})
        relative = layer.get("path")
        available[name] = {"available": False, "native_pixel_size_m": layer.get("native_pixel_size_m")}
        if not relative or not _path(root, relative).is_file():
            continue
        path = _path(root, relative)
        with _timed(timings, "checksums"):
            hashes.file(path, "reference/" + relative, layer["sha256"])
        with _timed(timings, "radar"):
            values, grid = _read(path)
            if name == "radar_s1_log":
                values[values < 0] = np.nan
                values = np.log1p(values)
            warped = _warp(values, grid, ref["grid"])
            _accumulate(ref, name, warped)
        available[name].update(available=bool(np.isfinite(warped).any()), grid=grid.metadata())
    available["lola"] = {"available": lola_path is not None}
    if lola_path is not None:
        with _timed(timings, "lola"):
            target = ref["grid"]
            with rasterio.open(lola_path) as src:
                native = _Grid.read(src)
                with WarpedVRT(src, crs=target.crs, transform=target.transform,
                               height=target.shape[0], width=target.shape[1],
                               resampling=Resampling.bilinear, dtype="float32", nodata=np.nan,
                               warp_mem_limit=64) as vrt:
                    z = _clean(vrt.read(1), vrt.read_masks(1) > 0)
            for name, values in _local_terrain(z, _metric_grid(target)).items():
                _accumulate(ref, name, values)
            available["lola"].update(available=bool(np.isfinite(z).any()), native_grid=native.metadata(),
                                     native_pixel_size_m=[abs(native.transform.a), abs(native.transform.e)])
    return available


def _finish(ref, optional):
    ground = ref["ground"]
    joint_count = _cell_sum(ref["joint"] & ref["terrain"])
    support_count = _cell_sum(ref["support"] & ref["terrain"])
    support_fraction = np.divide(support_count, ground, out=np.zeros(ground.shape), where=ground > 0)
    keep, excluded = _eligible(ref["saved"], ref["draft"], ref["accepted"],
                               ground, ref["area"], joint_count)
    means = np.full(ref["sums"].shape, np.nan, np.float32)
    valid = (ground[..., None] > 0) & (ref["counts"] >= MIN_COVERAGE * ground[..., None])
    np.divide(ref["sums"], ref["counts"], out=means, where=valid)
    tile_id, site = ref["record"]["tile_id"], ref["record"]["site_id"]
    samples = np.array([f"{tile_id}:{row}:{col}" for row, col in np.argwhere(keep)], dtype=str)
    snapshots = {}
    for variant, snapshot in ref["snapshots"].items():
        confidence = snapshot.get("confidence")
        snapshots[variant] = {
            "provenance": snapshot["provenance"],
            "class_counts": _class_counts(snapshot["painting"]),
            "accepted_observed": "accepted" in snapshot,
            "accepted_cells": int(snapshot["accepted"].sum()) if "accepted" in snapshot else None,
            "confidence_observed": confidence is not None,
            "confidence_counts": {str(c): int(((confidence == c) & (snapshot["painting"] > 0)).sum())
                                  for c in range(4)} if confidence is not None else None,
        }
    paired = np.isin(ref["saved"], (1, 2, 3)) & np.isin(ref["draft"], (1, 2, 3))
    differences = ref["saved"] != ref["draft"]
    X = means[keep]
    meta = {
        "tile_id": tile_id, "site_id": site, "group": GROUPS[site],
        "public_region": REGIONS[site], "grid": ref["grid"].metadata(),
        "source_map_id": next(iter(snapshots.values()))["provenance"].get("private_map_id"),
        "reference_record": ref["record"], "snapshots": snapshots,
        "saved_variant_used": "saved" if "saved" in snapshots else "draft",
        "draft_variant_used": "draft" if "draft" in snapshots else "saved",
        "public_tiles": ref["public_tiles"], "optional": optional,
        "cells_total": GRID * GRID, "cells_kept": int(keep.sum()), "excluded_sequential": excluded,
        "saved_class_counts": _class_counts(ref["saved"], keep),
        "draft_class_counts": _class_counts(ref["draft"], keep),
        "snapshot_differences_all_cells": int(differences.sum()),
        "snapshot_differences_both_valid": int((differences & paired).sum()),
        "snapshot_differences_kept": int((differences & keep).sum()),
        "reference_valid_pixels": int(ground.sum()),
        "public_joint_pixels_on_reference": int(joint_count.sum()),
        "public_joint_fraction_on_reference": float(joint_count.sum() / ground.sum()) if ground.sum() else None,
        "public_support_fraction_on_reference": float(support_count.sum() / ground.sum()) if ground.sum() else None,
        "kept_support_below_90pct": int((keep & (support_fraction < MIN_COVERAGE)).sum()),
        "kept_reference_terrain_fraction_min": float((ground / ref["area"])[keep].min()) if keep.any() else None,
        "feature_missing_counts": {name: int((~np.isfinite(X[:, i])).sum()) for i, name in enumerate(FEATURE_NAMES)},
    }
    return X, ref["saved"][keep], ref["draft"][keep], samples, support_fraction[keep].astype(np.float32), meta


def load_data(public_root: Path, reference_root: Path, raw_root: Path, progress=None) -> dict:
    """Extract one row per eligible original cell; progress, if supplied, accepts text.

    Root-relative source paths come from catalogs, except the predeclared region
    mapping and optional raw future-experiments/LDEM_83S_10MPP_ADJ.tiff. Missing
    optional layers stay NaN. Missing historical accepted/confidence flags stay
    unknown in metadata. No support threshold is used to remove evaluation rows.
    """
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=64 * 1024 * 1024, PROJ_NETWORK="OFF"):
        return _load_data(Path(public_root).resolve(), Path(reference_root).resolve(),
                          Path(raw_root).resolve(), progress)


def _load_data(public_root, reference_root, raw_root, progress):
    start = perf_counter()
    timings = {name: 0.0 for name in ("checksums", "reference_validation", "public_terrain",
                                     "public_image", "public_quality", "reprojection_aggregation", "radar", "lola")}
    hashes = _Hashes()

    def report(message):
        if progress is not None:
            progress(message)

    report("Validating reference artifact hashes, label legends and snapshot geometry")
    with _timed(timings, "reference_validation"):
        refs, artifact_count = _references(reference_root, hashes)
    report("Validating public metadata and available raw source hashes (streaming, once per file)")
    with _timed(timings, "checksums"):
        regions = _public_metadata(public_root, raw_root,
                                   {REGIONS[r["record"]["site_id"]] for r in refs}, hashes)
        lola_path = _path(raw_root, "future-experiments/LDEM_83S_10MPP_ADJ.tiff")
        if lola_path.is_file():
            report("Hashing optional raw LOLA once; raster reads will be bounded reference windows")
            hashes.file(lola_path, "raw/future-experiments/LDEM_83S_10MPP_ADJ.tiff")
        else:
            lola_path = None
        hashes.file(Path(core.__file__), "code/app/core.py")
        hashes.file(Path(__file__), "code/poster/benchmark_data.py")
    public_grids = {}
    for region, region_info in sorted(regions.items()):
        folder = _path(public_root, region)
        targets = [r for r in refs if REGIONS[r["record"]["site_id"]] == region]
        records = _json(folder / "working_dems.json")
        ids = [r["tile_id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate public tile IDs: {region}")
        for record in sorted(records, key=lambda r: r["tile_id"]):
            if record.get("collection") != "public" or record.get("site") != region:
                raise ValueError(f"Nonpublic/inconsistent region record: {record['tile_id']}")
            with rasterio.open(_path(folder, record["layers"]["sfs"])) as src:
                grid = _Grid.read(src)
            hits = [r for r in targets if _intersects(grid, r["grid"])]
            if not hits:
                continue
            report(f"Extracting {record['tile_id']} -> {', '.join(r['record']['tile_id'] for r in hits)}")
            features, joint, support = _public_features(folder, record, grid,
                                                        region_info["metadata"], hashes, timings)
            public_grids[record["tile_id"]] = grid.metadata()
            with _timed(timings, "reprojection_aggregation"):
                for ref in hits:
                    target = ref["grid"]
                    footprint = _warp(np.ones(grid.shape, np.float32), grid, target) == 1
                    _claim(ref, footprint, record["tile_id"])
                    ref["joint"] |= _warp(joint.astype(np.float32), grid, target) == 1
                    ref["support"] |= _warp(support.astype(np.float32), grid, target) == 1
                    ref["public_tiles"].append(record["tile_id"])
                    for name, values in features.items():
                        _accumulate(ref, name, _warp(values, grid, target))
            del features, joint, support
    batches, ref_metadata = [], []
    for ref in refs:
        report(f"Aggregating cells and optional radar/LOLA: {ref['record']['tile_id']}")
        optional = _optional(ref, reference_root, lola_path, hashes, timings)
        X, saved, draft, ids, support, meta = _finish(ref, optional)
        batches.append((X, saved, draft, ids, support, np.full(len(ids), GROUPS[ref["record"]["site_id"]], dtype="U11")))
        ref_metadata.append(meta)
    def concatenate(index, shape, dtype):
        return np.concatenate([b[index] for b in batches]).astype(dtype, copy=False) if batches else np.empty(shape, dtype)
    X = concatenate(0, (0, len(FEATURE_NAMES)), np.float32)
    saved, draft = concatenate(1, (0,), np.uint8), concatenate(2, (0,), np.uint8)
    ids, support = concatenate(3, (0,), str), concatenate(4, (0,), np.float32)
    groups = concatenate(5, (0,), str)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("Duplicate original-cell sample IDs")
    timings["total"] = perf_counter() - start
    metadata = {
        "contract_version": 1, "feature_names": list(FEATURE_NAMES), "samples": len(ids),
        "roots": {"public": str(public_root), "reference": str(reference_root), "raw": str(raw_root)},
        "reference_artifacts_verified": artifact_count, "source_hashes": hashes.records,
        "public_regions": {name: {"grid": info["metadata"]["grid"],
                                   "sources": info["metadata"].get("sources", {}),
                                   "raw_source_verification": info["raw_source_verification"]}
                           for name, info in regions.items()},
        "public_tile_grids": public_grids, "references": ref_metadata, "timings_seconds": timings,
        "saved_class_counts": _class_counts(saved), "draft_class_counts": _class_counts(draft),
        "group_counts": {name: int((groups == name).sum()) for name in sorted(set(groups.tolist()))},
        "snapshot_differences_kept": int((saved != draft).sum()),
        "feature_availability": {name: bool(np.isfinite(X[:, i]).any()) for i, name in enumerate(FEATURE_NAMES)},
        "missingness": {name: {"count": int((~np.isfinite(X[:, i])).sum()),
                                "fraction": float((~np.isfinite(X[:, i])).mean()) if len(X) else None}
                        for i, name in enumerate(FEATURE_NAMES)},
        "policies": {
            "regions": REGIONS, "groups": GROUPS, "snapshot": "saved else draft; sensitivity draft else saved",
            "cell_edges": "floor(i*height/120), floor(j*width/120); 40/45 m edges on 1024x1024 5 m grids",
            "eligibility": "both targets in 1..3; no accepted=true in either snapshot; >=90% full-cell reference terrain; >=90% of reference terrain has joint public NAC/SfS",
            "missing_flags": "accepted and confidence absent = unknown, NOT certified independent/sure; confidence is not an eligibility filter",
            "support": "sfs_support positive pixels / valid reference-terrain pixels; NEVER an evaluation-row filter",
            "aggregation": "finite per-channel mean on valid reference terrain; >=90% reference-terrain coverage per channel else NaN; nearest reprojection; reject overlapping public ownership",
            "public_terrain": "core.build on each whole public tile; use only TERRAIN names, not companion predictions/private DEMs",
            "image": "finite-weight normalized Gaussian convolution, reflect edges; sigma=width/sqrt(12)/pixel_m. Textures: standard deviation at 30/100 m. Contrast=(NAC-mean100)/max(std100,1e-6*abs(mean100),1e-12). Gradient=hypot(dNAC/dx,dNAC/dy) using numpy.gradient; invalid stencils stay NaN. Center must be valid. No label fitting or global normalization.",
            "quality": "processed aligned image_count, solar_bins, best_resolution, uncertainty; no label interpretation; native grids retained in public_regions.sources",
            "radar": "log1p(S1) only where S1>=0; CPR unchanged; finite abs(value)<1e30; native resolution from reference record (~118.45 m)",
            "lola": "raw LOLA bilinear-warped to each <=1024 reference grid, then core slope/rough/rel_local/curv math and physical windows; native 10 m on audited input, NOT public tile context; no broad-window halo",
            "terrain_windows_m": {"rough": core.ROUGH_M, "rel_local": core.LOCAL_M, "curv": core.CURV_M,
                                  "rel": core.REL_M, "svf_local": core.SKY_LOCAL_M},
            "hashes": "all listed reference artifacts validated; used public rasters and available raw archives/members validated; unavailable historical source paths are provenance only; raw LOLA SHA256 computed once (no independent expected hash)",
        },
        "limitations": ["Unreviewed professor-derived snapshots; missing confidence does not establish review.",
                        "Primary-map vectors are absent; historical GBM interaction and edits prevent an untouched-final-test claim.",
                        "Cells are coarse labels, not independent 5 m truth; N1014 and MS1 share nobile-1 group.",
                        "Only three geographic groups; feature selection is development validation, not SOTA/final-test evidence.",
                        "Tile-edge/context effects match core.build; optional LOLA instead uses reference-window context.",
                        "NAC is not uniformly calibrated reflectance; Nobile-2 NAC is native 10 m; several quality grids are native 15 m.",
                        "Missing/zero SfS support is retained for paired screening; coverage and class-dependent missingness must be reported.",
                        "No PSR channel: class 3 also includes crater interiors and is not a PSR/ice target.",
                        "Reference collection remains local-use/CUI restricted."],
    }
    report(f"Extraction complete: {len(ids)} cells, {len(FEATURE_NAMES)} features; no support-based test filtering")
    return {"X": X, "feature_names": list(FEATURE_NAMES), "y_saved": saved, "y_draft": draft,
            "groups": groups, "sample_ids": ids, "support_fraction": support, "metadata": metadata}
