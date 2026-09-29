"""Local, read-only reference extraction and app-compatible pixel RF training.

Run explicitly with ``python -B -m poster.reference_training``.
Nothing is extracted or fitted at import time. Source snapshots and all derived
models/reports remain CUI/local-use restricted. The headline validation unit is an
original painting cell, not an independently labelled terrain pixel. These labels
were already used for feature selection: this is development validation, not an
untouched final test, even though each geographic fold fits a fresh model.

The private benchmark_data helpers are intentionally shared: they are the audited
hash, snapshot, lunar-grid, nearest-warp and floor-cell geometry implementation.
The benchmark's cell-mean feature matrix and NAC eligibility are NOT used here.
"""

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import rasterio

from app import assistant_model as assistant
from poster import benchmark_data as data
from app.evaluation_metrics import _software_versions, _metrics, score, _atomic_json
from app import core, import_reference_maps, paths
from poster import storage

GROUPS = data.GROUPS
CLASSES = (1, 2, 3)
RENDER_FIELDS = (
    "tile_id", "map_id", "site_id", "group", "snapshot", "grid", "px_m",
    "X", "hillshade", "reference", "cells", "eligible", "valid", "support",
    "prediction", "confidence", "cell_prediction",
)




def _terrain_features(folder, record, grid, metadata, hashes):
    """Verify terrain/masks, then select the six app features on a WHOLE tile.

    valid_data encodes joint NAC/SfS availability. Its checksum, binary values,
    grid and implication of valid SfS are checked, but its zeros do not remove
    terrain. Neither NAC nor support is required to be positive for eligibility.
    """
    arrays = {}
    for name in ("sfs", "valid_data", "sfs_support"):
        relative = record["layers"].get(name)
        if not relative:
            raise ValueError(f"Required public terrain layer absent: {name}")
        path = data._path(folder, relative)
        hashes.file(path, f"public/{folder.name}/{relative}",
                    metadata["outputs"][relative]["checksum"]["sha256"])
        arrays[name], actual = data._read(path)
        if actual != grid:
            raise ValueError(f"Processed public layers must already be aligned: {path}")
    data._metric_grid(grid)
    if min(grid.shape) < 2 or max(grid.shape) > 1024:
        raise ValueError("Public feature context must be one tile, 2..1024 pixels per axis")
    terrain = np.isfinite(arrays["sfs"])
    for name in ("valid_data", "sfs_support"):
        values = arrays[name]
        if np.any(np.isfinite(values) & ~np.isin(values, (0, 1))):
            raise ValueError(f"Nonbinary public mask: {name}")
    if np.any((arrays["valid_data"] == 1) & ~terrain):
        raise ValueError("Public valid_data mask claims invalid SfS terrain")
    support = (arrays["sfs_support"] == 1) & terrain
    if not terrain.any():
        return (np.full((len(core.BASE_FEATS), *grid.shape), np.nan, np.float32),
                np.full(grid.shape, np.nan, np.float32), support)
    stack, names, hillshade, profile, _ = core.build(str(data._path(folder, record["layers"]["sfs"])))
    profile = cast(dict[str, Any], profile)
    built = data._Grid((profile["height"], profile["width"]), profile["transform"], profile["crs"])
    if built != grid:
        raise ValueError("core.build changed the public tile grid")
    X = assistant.features(stack, names)
    X = np.stack([data._clean(channel, terrain) for channel in X])
    return X, data._clean(hillshade, terrain), support


def _map_ids(reference_root, hashes):
    manifest = data._json(reference_root / "dataset.json")
    result = {}
    for item in manifest.get("sites", []):
        path = data._path(reference_root, item["path"])
        site = data._json(path)
        hashes.file(path, "reference/" + item["path"])
        if site.get("map_id"):
            result[item["site_id"]] = site["map_id"]
    # Some older catalogs omit the sites index; use their preserved site.json.
    for path in sorted(reference_root.glob("sites/*/site.json")):
        relative = path.relative_to(reference_root).as_posix()
        path = data._path(reference_root, relative)
        site = data._json(path)
        hashes.file(path, "reference/" + relative)
        if site.get("map_id"):
            result.setdefault(site.get("site_id", path.parent.name), site["map_id"])
    if reference_root.resolve() == Path(import_reference_maps.REFERENCE_ROOT).resolve():
        for record in import_reference_maps.reference_records():
            result.setdefault(record["site_id"], record["reference_map"])
    return result


def _finish_tile(ref, map_ids, hashes):
    record, grid = ref["record"], ref["grid"]
    site, tile_id = record["site_id"], record["tile_id"]
    if site not in map_ids:
        raise ValueError(f"No map_id in preserved site.json/reference records for {site}")
    variant = "saved" if "saved" in ref["snapshots"] else "draft"
    snapshot = ref["snapshots"][variant]
    cells = snapshot["painting"]
    valid = ref["terrain"] & np.isfinite(ref["X"]).all(axis=0)
    # _eligible's two-target check deliberately receives the SAME chosen snapshot.
    # We borrow its audited coverage math, not the benchmark's paired-target policy.
    eligible, excluded = data._eligible(cells, cells, ref["accepted"], ref["ground"],
                                        ref["area"], data._cell_sum(valid))
    excluded["invalid_selected_target"] = excluded.pop("invalid_either_target")
    excluded["public_finite_terrain_below_90pct_reference_terrain"] = excluded.pop(
        "public_joint_below_90pct_reference_terrain")
    reference = data._expand(cells, grid.shape).astype(np.uint8, copy=True)
    reference[~ref["terrain"] | ~np.isin(reference, CLASSES)] = 255
    ref["X"][:, ~valid] = np.nan
    annotation = next(a for a in record["annotations"] if a["snapshot_variant"] == variant)
    confidence = snapshot.get("confidence")
    painting_hash = assistant.painting_hash(cells, confidence, ref["accepted"])
    provenance = {
        "reference_group": GROUPS[site], "public_region": data.REGIONS[site],
        "source_kind": "reference_snapshot", "geographic_group": GROUPS[site],
        "site_id": site, "map_id": map_ids[site], "tile_id": tile_id, "snapshot": variant,
        "public_tiles": ref["public_tiles"], "public_tile_ids": ref["public_tile_ids"],
        "painting_hash": painting_hash,
        "source_revision": {
            "reference_dataset_sha256": hashes.records["reference/dataset.json"]["sha256"],
            "public_manifest_sha256": hashes.records["public/manifest.json"]["sha256"],
            "snapshot_labels_sha256": annotation["sha256"],
        },
        "footprint": grid.metadata(), "grid": grid.metadata(),
        "confidence_observed": confidence is not None,
        "confidence_counts": ({str(c): int(((confidence == c) & (cells > 0)).sum())
                               for c in range(4)} if confidence is not None else None),
        "accepted_observed": {v: "accepted" in s for v, s in ref["snapshots"].items()},
        "known_accepted_cells_union": int(ref["accepted"].sum()),
        "weight_policy": "uniform 1.0; unreviewed snapshot, NOT certified sure",
    }
    ground = int(ref["terrain"].sum())
    return {
        "tile_id": tile_id, "map_id": map_ids[site], "site_id": site, "group": GROUPS[site],
        "snapshot": variant, "grid": grid.metadata(), "px_m": float(data._metric_grid(grid)),
        "X": ref["X"], "hillshade": ref["hillshade"], "reference": reference,
        "cells": cells, "eligible": eligible, "valid": valid,
        "support": ref["support"] & ref["terrain"],
        "reference_terrain_pixels": ground,
        "public_coverage_fraction": float(valid.sum() / ground) if ground else 0.0,
        "exclusions": excluded, "provenance": provenance,
        "prediction": np.full(grid.shape, 255, np.uint8),
        "confidence": np.full(grid.shape, np.nan, np.float32),
        "cell_prediction": np.full((data.GRID, data.GRID), 255, np.uint8),
    }


def load_data(public_root=None, reference_root=None, raw_root=None, progress=None):
    """Return pixel ``tiles`` and hash/provenance ``metadata`` without fitting/writes."""
    started = perf_counter()
    public_root = Path(public_root or paths.DEM_DIR).resolve()
    reference_root = Path(reference_root or paths.PROFESSOR_MAPS_DIR).resolve()
    raw_root = Path(raw_root or paths.DATA_DIR / "raw_data").resolve()
    report = progress or (lambda message: None)
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=64 * 1024 * 1024, PROJ_NETWORK="OFF"):
        hashes = data._Hashes()
        report("Validating preserved reference hashes, labels and geometry")
        refs, count = data._references(reference_root, hashes)
        map_ids = _map_ids(reference_root, hashes)
        report("Validating public metadata and available raw sources; no LOLA features")
        regions = data._public_metadata(public_root, raw_root,
                                        {data.REGIONS[r["record"]["site_id"]] for r in refs}, hashes)
        for module in (data, data.dataset_utils, core, assistant, import_reference_maps, core.backend()):
            if module.__file__ is None:
                raise RuntimeError(f"Cannot record source code provenance: {module.__name__}")
            code_path = Path(module.__file__)
            hashes.file(code_path, "code/" + module.__name__.replace(".", "/") + ".py")
        hashes.file(Path(__file__), "code/poster/reference_training.py")
        for ref in refs:
            ref["X"] = np.full((len(core.BASE_FEATS), *ref["grid"].shape), np.nan, np.float32)
            ref["hillshade"] = np.full(ref["grid"].shape, np.nan, np.float32)
            ref["public_tile_ids"] = []
            ref["feature_grids"] = {}
            # The cell-mean buffers from the shared validator are never features here.
            ref.pop("sums")
            ref.pop("counts")
        public_grids, catalog_counts = {}, {}
        for region, info in sorted(regions.items()):
            folder = data._path(public_root, region)
            records = data._json(folder / "working_dems.json")
            ids = [r["tile_id"] for r in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate public tile IDs: {region}")
            catalog_counts[region] = len(records)
            targets = [r for r in refs if data.REGIONS[r["record"]["site_id"]] == region]
            for record in sorted(records, key=lambda r: r["tile_id"]):
                if record.get("collection") != "public" or record.get("site") != region:
                    raise ValueError(f"Nonpublic/inconsistent region record: {record['tile_id']}")
                with rasterio.open(data._path(folder, record["layers"]["sfs"])) as src:
                    grid = data._Grid.read(src)
                hits = [r for r in targets if data._intersects(grid, r["grid"])]
                if not hits:
                    continue
                report(f"Building whole-public-tile terrain features: {record['tile_id']}")
                X, hillshade, support = _terrain_features(folder, record, grid, info["metadata"], hashes)
                map_name = core.dem_name(record.get("working_dem", record["layers"]["sfs"]))
                if map_name in public_grids:
                    raise ValueError(f"Duplicate public DEM map name: {map_name}")
                public_grids[map_name] = {"tile_id": record["tile_id"], "grid": grid.metadata()}
                for ref in hits:
                    target = ref["grid"]
                    ref["feature_grids"][str(data._path(folder, record["layers"]["sfs"]))] = grid
                    footprint = data._warp(np.ones(grid.shape, np.float32), grid, target) == 1
                    data._claim(ref, footprint, record["tile_id"])
                    for index, channel in enumerate(X):
                        warped = data._warp(channel, grid, target)
                        ref["X"][index, footprint] = warped[footprint]
                    warped = data._warp(hillshade, grid, target)
                    ref["hillshade"][footprint] = warped[footprint]
                    ref["support"] |= data._warp(support.astype(np.float32), grid, target) == 1
                    ref["public_tiles"].append(map_name)
                    ref["public_tile_ids"].append(record["tile_id"])
        # Use exactly the same adjoining source context as the app, separately
        # for each reference map. All source hashes/ownership were checked above.
        for ref in refs:
            grids = ref.pop("feature_grids")
            contexts = core._source_contexts(list(grids))
            for path, features in contexts.items():
                grid, target = grids[path], ref["grid"]
                footprint = data._warp(np.ones(grid.shape, np.float32), grid, target) == 1
                for index, name in enumerate(core.BASE_FEATS):
                    warped = data._warp(features[name], grid, target)
                    ref["X"][index, footprint] = warped[footprint]
        tiles = [_finish_tile(ref, map_ids, hashes) for ref in sorted(refs, key=lambda r: r["record"]["tile_id"])]
        manifest = data._json(public_root / "manifest.json")
        metadata = {
            "roots": {"public": str(public_root), "reference": str(reference_root), "raw": str(raw_root)},
            "reference_artifacts_verified": count, "source_hashes": hashes.records,
            "software_versions": _software_versions(),
            "public_tile_grids": public_grids, "public_tiles_used": len(public_grids),
            "public_catalog_tiles_by_region": catalog_counts,
            "public_manifest_tiles": sum(r.get("tile_count", 0) for r in manifest["regions"].values()),
            "public_regions": {name: {"raw_source_verification": info["raw_source_verification"],
                                      "sources": info["metadata"].get("sources", {})}
                               for name, info in regions.items()},
        }
    metadata["timings_seconds"] = {"extraction": perf_counter() - started}
    return {"tiles": tiles, "metadata": metadata}


def sample_tiles(tiles):
    """One deterministic assistant.sample call per tile, never one per snapshot."""
    samples = {}
    for tile in tiles:
        key = tile["tile_id"]
        if key in samples:
            raise ValueError(f"Duplicate reference tile: {key}")
        keep = tile["valid"] & data._expand(tile["eligible"], tile["valid"].shape)
        labels = np.where(keep, tile["reference"], 255).astype(np.uint8)
        X, y, w = assistant.sample(tile["X"], labels, keep.astype(np.float32), labels)
        if X is None:
            continue
        if y is None or w is None:
            raise ValueError("assistant.sample returned incomplete training rows")
        if not np.isin(y, CLASSES).all() or not np.isfinite(X).all():
            raise ValueError("Reference samples must have known classes and six finite features")
        if any(int((y == c).sum()) > 2000 for c in CLASSES):
            raise ValueError("assistant.sample exceeded 2000 pixels/class/tile")
        samples[key] = dict(tile["provenance"], X=X, y=y, w=w)
    return samples


def cell_majority(prediction, valid=None):
    """Original floor-edged 120x120 cells; ignore nodata; lowest code wins ties."""
    if valid is None:
        valid = np.ones(prediction.shape, bool)
    counts = np.stack([data._cell_sum(valid & (prediction == c)) for c in CLASSES])
    result = (counts.argmax(axis=0) + 1).astype(np.uint8)
    result[counts.sum(axis=0) == 0] = 255
    return result






def _majority_class(tiles):
    counts = np.zeros(3, dtype=np.int64)
    for tile in tiles:
        labels = tile["cells"][tile["eligible"]]
        counts += [int((labels == c).sum()) for c in CLASSES]
    if not counts.sum():
        raise ValueError("No unsampled eligible training cells for majority baseline")
    return int(counts.argmax() + 1)


def _component(value):
    if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe output path component: {value!r}")
    return value


def validate(tiles, samples, progress=None):
    """Fit exactly three independent leave-geographic-group-out assistants.

    Predictions/confidence attached to tiles are OOF, never the final-fit output.
    Fold models and arrays stay in memory; this workflow never installs an app model.
    """
    started = perf_counter()
    groups = sorted(set(GROUPS.values()))
    if sorted({t["group"] for t in tiles}) != groups:
        raise ValueError("Reference validation requires all three predeclared geographic groups")
    by_id = {t["tile_id"]: t for t in tiles}
    if len(by_id) != len(tiles) or not set(samples).issubset(by_id):
        raise ValueError("Duplicate tiles or samples outside the reference tile set")
    for key, sample in samples.items():
        if sample["geographic_group"] != by_id[key]["group"]:
            raise ValueError(f"Sample geography disagrees with its reference tile: {key}")
    folds = []
    aggregate, baseline_aggregate = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
    for group in groups:
        fold_started = perf_counter()
        train_tiles = [t for t in tiles if t["group"] != group]
        test_tiles = [t for t in tiles if t["group"] == group]
        train = {k: s for k, s in samples.items() if by_id[k]["group"] != group}
        train_groups = sorted({s["geographic_group"] for s in train.values()})
        if train_groups != [g for g in groups if g != group] or not any(t["eligible"].any() for t in test_tiles):
            raise ValueError(f"Insufficient eligible train/test groups for fold {group}")
        majority = _majority_class(train_tiles)
        if progress:
            progress(f"Fitting independent held-out fold: {group}")
        fit_started = perf_counter()
        bundle = assistant.fit(train)
        fit_seconds = perf_counter() - fit_started
        if bundle["model"] is None:
            raise ValueError(f"Fold {group} has no fitted model (at least two training classes required)")
        matrix, baseline_matrix = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
        for tile in test_tiles:
            pred = assistant.predict(bundle, tile["X"], list(core.BASE_FEATS))
            conf = assistant.confidence(bundle, tile["X"], list(core.BASE_FEATS))
            if pred is None or conf is None:
                raise ValueError(f"Fold {group} did not produce predictions/confidence")
            pred, conf = np.asarray(pred, np.uint8).copy(), np.asarray(conf, np.float32).copy()
            if pred.shape != tile["valid"].shape or conf.shape != pred.shape:
                raise ValueError("Backend returned the wrong prediction grid")
            if not np.isin(pred[tile["valid"]], CLASSES).all():
                raise ValueError("Backend returned missing/invalid predictions on finite terrain")
            if not np.isfinite(conf[tile["valid"]]).all():
                raise ValueError("Backend returned nonfinite confidence on finite terrain")
            pred[~tile["valid"]], conf[~tile["valid"]] = 255, np.nan
            cells = cell_majority(pred, tile["valid"])
            keep = tile["eligible"]
            metrics = score(tile["cells"][keep], cells[keep])
            baseline = score(tile["cells"][keep], np.full(int(keep.sum()), majority, np.uint8))
            matrix += metrics["confusion_matrix"]
            baseline_matrix += baseline["confusion_matrix"]
            tile.update(prediction=pred, confidence=conf, cell_prediction=cells, metrics=metrics)
        metrics, baseline = _metrics(matrix), _metrics(baseline_matrix)
        folds.append({"group": group, "train_groups": train_groups, "train_tiles": sorted(train),
                      "test_tiles": sorted(t["tile_id"] for t in test_tiles),
                      "sampled_pixels": sum(len(s["y"]) for s in train.values()),
                      "metrics": metrics, "baseline": baseline, "majority_class": majority,
                      "model_revision": bundle["revision"],
                      "fit_seconds": fit_seconds, "elapsed_seconds": perf_counter() - fold_started})
        aggregate += matrix
        baseline_aggregate += baseline_matrix
    validation = {
        "method": "leave-one-geographic-group-out; 3 fresh pixel RFs; original-cell majority vote",
        "aggregate": _metrics(aggregate),
        "group_mean_macro_f1": float(np.mean([f["metrics"]["macro_f1"] for f in folds])),
        "baseline_aggregate": _metrics(baseline_aggregate), "folds": folds,
        "timings_seconds": {"fit": sum(f["fit_seconds"] for f in folds),
                            "total": perf_counter() - started},
    }
    return validation, {}




def _summary_markdown(report):
    validation = report["validation"]
    pooled, baseline = validation["aggregate"], validation["baseline_aggregate"]
    lines = ["# Poster development results", "",
             "Local-use restrictions follow the input data. These coarse labels were already used for feature selection; this is not an untouched final test.", "",
             f"Run: {report['run_id']}",
             f"Scored painting cells: {pooled['n']}",
             f"Held-out accuracy: {pooled['accuracy']:.3%}",
             f"Held-out macro F1: {pooled['macro_f1']:.4f}",
             f"Majority baseline accuracy: {baseline['accuracy']:.3%}", "",
             "| Held-out location | Cells | Accuracy | Macro F1 |", "|---|---:|---:|---:|"]
    for fold in validation["folds"]:
        metrics = fold["metrics"]
        lines.append(f"| {fold['group']} | {metrics['n']} | {metrics['accuracy']:.3%} | {metrics['macro_f1']:.4f} |")
    lines += ["", "## Limitations", "", *[f"- {item}" for item in report.get("limitations", [])],
              "", "Reproduce with `python -B -m poster.reference_training`. The app model is never changed.", ""]
    return "\n".join(lines)


def finalize_report_exports(run_dir, report):
    """Synchronize derived reports/summary only; no fitting, model or source writes.

    The caller supplies its canonical final report (including timings).
    Neither the argument nor results.json/latest.json is changed. Missing renderer
    outputs are optional: always write summary.md, update figures/report.json only
    if figures/ exists, and rewrite a ZIP only when exports.bundle is declared.
    A declared but missing/corrupt ZIP is an error, not a silently skipped export.

    Each replacement is atomic, not a multi-file transaction. Everything is staged
    before publishing; a later publication error can safely be retried with the
    same report. No results/latest pointer should be published until this succeeds.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or run_dir.name != _component(report["run_id"]):
        raise ValueError("Report finalization requires the existing matching run directory")
    canonical = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    summary = _summary_markdown(report).encode("utf-8")
    summary_path = data._path(run_dir, "summary.md")
    poster_dir = Path(report.get("poster_dir", run_dir)).resolve()
    figure_directory = data._path(poster_dir, "figures")
    text_outputs = [(summary_path, summary)]
    if figure_directory.is_dir():
        text_outputs.append((data._path(poster_dir, "figures/report.json"), canonical))
    bundle_name = report.get("exports", {}).get("bundle")
    bundle = data._path(poster_dir, bundle_name) if bundle_name else None
    if bundle is not None and not bundle.is_file():
        raise FileNotFoundError(f"Declared report bundle is missing: {bundle}")
    staged = []
    try:
        if bundle is not None:
            temporary = bundle.with_name(f".{bundle.name}-{uuid.uuid4().hex}.tmp")
            staged.append((temporary, bundle))
            replacements = {"report.json": canonical, "summary.md": summary}
            with zipfile.ZipFile(bundle, "r") as source, zipfile.ZipFile(temporary, "x") as target:
                target.comment = source.comment
                written = set()
                for info in source.infolist():
                    if info.filename in replacements:
                        if info.filename not in written:
                            target.writestr(copy.copy(info), replacements[info.filename])
                            written.add(info.filename)
                        continue
                    # Preserve uncompressed byte contents and entry metadata; never
                    # extract members or sweep unrelated run files into the bundle.
                    with source.open(info) as incoming, target.open(copy.copy(info), "w") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                for name, payload in replacements.items():
                    if name not in written:
                        target.writestr(name, payload, compress_type=zipfile.ZIP_DEFLATED)
        for destination, payload in text_outputs:
            temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
            staged.append((temporary, destination))
            with temporary.open("xb") as stream:
                stream.write(payload)
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def run(*, public_root=None, reference_root=None, raw_root=None, output_root=None,
        progress=print, renderer=None):
    """Make poster results and figures under poster/; app and sources are read-only."""
    started = perf_counter()
    timings = {}
    if core.MODEL != "rf":
        raise ValueError("Poster reference evaluation requires SELENOGRAPH_MODEL=rf")
    directory = storage.output_path(output_root or storage.ROOT / "results")
    source_roots = [Path(p).resolve() for p in (
        public_root or paths.DEM_DIR, reference_root or paths.PROFESSOR_MAPS_DIR,
        raw_root or paths.DATA_DIR / "raw_data")]
    if any(directory.is_relative_to(root) for root in source_roots):
        raise ValueError("Poster outputs must not be inside an immutable source dataset")
    if renderer is None:
        from poster.reference_figures import export_figures
        renderer = export_figures
    with data._timed(timings, "extraction"):
        extracted = load_data(*source_roots, progress=progress)
    tiles, metadata = extracted["tiles"], extracted["metadata"]
    metadata.setdefault("software_versions", _software_versions())
    with data._timed(timings, "sampling"):
        samples = sample_tiles(tiles)
    now = dt.datetime.now(dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = directory / run_id
    with data._timed(timings, "validation"):
        validation, _ = validate(tiles, samples, progress)
    timings["training_fit_total"] = validation["timings_seconds"]["fit"]
    by_id = {t["tile_id"]: t for t in tiles}
    report = {
        "run_id": run_id, "created_at": now.isoformat(), "poster_dir": str(run_dir),
        "schema": json.loads(json.dumps(assistant.schema())),
        "feature_names": list(core.BASE_FEATS),
        "software_versions": metadata["software_versions"], "timings_seconds": timings,
        "training": {
            "tiles": len(samples), "map_groups": len({by_id[k]["map_id"] for k in samples}),
            "geography_groups": len({s["geographic_group"] for s in samples.values()}),
            "sampled_pixels": sum(len(s["y"]) for s in samples.values()),
            "samples_by_tile": {k: len(s["y"]) for k, s in samples.items()},
        },
        "validation": validation,
        "tiles": [{**{k: t[k] for k in ("tile_id", "map_id", "site_id", "group", "snapshot", "px_m", "grid",
                                       "reference_terrain_pixels", "public_coverage_fraction", "metrics", "exclusions")},
                   "eligible_cells": int(t["eligible"].sum()),
                   "class_counts": data._class_counts(t["cells"], t["eligible"]),
                   "supported_valid_pixels": int((t["support"] & t["valid"]).sum()),
                   "provenance": t["provenance"]} for t in tiles],
        "public": {k: metadata[k] for k in ("public_tiles_used", "public_manifest_tiles", "public_catalog_tiles_by_region")},
        "policies": {"outputs": "poster results and figures only; models and OOF arrays stay in memory",
                     "split": "three independent geographic folds; N1014 and MS1 are grouped together",
                     "labels": "coarse original painting cells; accepted predictions excluded across snapshots"},
        "limitations": [
            "Preserved professor-derived paintings are unreviewed; missing confidence is not certified sure.",
            "These labels were already used for feature selection; this is development validation, not an untouched final test.",
            "Painting cells are spatially correlated coarse labels, not independent 5 m geological truth.",
            "Only three geographic groups. Class 3 includes crater interiors, not a PSR or ice detector.",
            "Sources, results and figures remain local-use/CUI restricted where applicable.",
        ],
        "provenance": dict(metadata, samples={k: {n: v for n, v in sample.items() if n not in ("X", "y", "w")}
                                               for k, sample in samples.items()}),
        "exports": {},
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    with data._timed(timings, "figure_export"):
        report["exports"].update(renderer([{k: t[k] for k in RENDER_FIELDS} for t in tiles], report, run_dir / "figures"))
    timings["elapsed_seconds"] = perf_counter() - started
    finalize_report_exports(run_dir, report)
    _atomic_json(run_dir / "results.json", report)
    if progress:
        progress(f"Completed poster results: {run_dir / 'results.json'}")
    return run_dir, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Results directory inside poster/ (default: poster/results)")
    args = parser.parse_args(argv)
    run(output_root=args.output)


if __name__ == "__main__":
    main()
