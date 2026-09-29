"""Independent selected-map evaluation; never update the active model.

catalog/plan read metadata and small painting arrays only. run builds terrain and
fits fresh fold models from the supplied assistant's retained training memory.
Only painted maps from core's processed-data catalog can be selected. Verification
controls training membership; unverified paintings can also be evaluated.
"""

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import rasterio

from app import assistant_model as assistant
from app import dataset_utils as data
from app import core, paths
from app import evaluation_metrics as report_helpers

GRID = core.GRID
CLASSES = (1, 2, 3)


def canonical_group(value, aliases=None):
    value = str(value) if value is not None else ""
    return (aliases or {}).get(value, value)


def _coalesce_groups(registry, samples):
    """Union only metadata-declared location equivalences; never infer from names.

    geography_aliases may be a list of aliases of the record's group, or an
    explicit alias->group mapping. Sample geographic_group is the preferred
    canonical spelling, followed by declared record aliases. Conflicting spellings
    of linked groups are conservatively coalesced, not split into leaky folds.
    """
    parents, preference = {}, {}

    def root(value):
        value = str(value)
        parents.setdefault(value, value)
        while value != parents[value]:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def link(alias, group, priority):
        if alias is None or group is None:
            return
        alias, group = str(alias), str(group)
        parents[root(alias)] = root(group)
        preference[group] = max(preference.get(group, 0), priority)

    for record in registry.values():
        group = record["group"]
        link(group, group, 1)
        declared = record.get("geography_aliases") or []
        if isinstance(declared, dict):
            for alias, target in declared.items():
                if not isinstance(target, str):
                    raise ValueError("geography_aliases mapping values must be group names")
                link(alias, target, 2)
        else:
            if isinstance(declared, str):
                declared = [declared]
            for alias in declared:
                link(alias, group, 2)
    for key, sample in samples.items():
        group = (sample.get("geographic_group") or sample.get("reference_group") or
                 registry.get(key, {}).get("group") or sample.get("public_region") or sample.get("site_id"))
        if not group:
            continue
        priority = 3 if sample.get("geographic_group") else 1
        link(group, group, priority)
        for alias in (sample.get("public_region"), sample.get("site_id"), sample.get("reference_group"),
                      registry.get(key, {}).get("group")):
            if alias:
                link(alias, group, priority)
    components = {}
    for value in parents:
        components.setdefault(root(value), []).append(value)
    aliases = {}
    for members in components.values():
        canonical = min(members, key=lambda value: (-preference.get(value, 0), value))
        aliases.update({value: canonical for value in members})
    return {key: dict(record, group=canonical_group(record["group"], aliases))
            for key, record in registry.items()}, aliases


def _painting(name):
    # The public final flag is True for both saved and bundled paintings. Retain
    # core's resolved source identity so an imported draft is not reported as saved.
    cells, final, bundled = core._painting_source(name)
    if cells.shape != (GRID, GRID) or not np.isin(cells, (0, *CLASSES)).all():
        raise ValueError(f"Invalid painting cells: {name}")
    confidence = core.load_confidence(name)
    accepted = core.load_accepted(name)
    if confidence.shape != cells.shape or accepted.shape != cells.shape:
        raise ValueError(f"Invalid painting flags: {name}")
    sure = np.isin(cells, CLASSES) & (confidence == core.SURE) & ~accepted.astype(bool)
    snapshot = "saved" if final is True else "draft" if final is False else "none"
    if bundled:
        provenance = ((core.map_record(name) or {}).get("annotations") or {}).get("provenance")
        declared = provenance.get("snapshot") if isinstance(provenance, dict) else None
        snapshot = declared if isinstance(declared, str) and declared else "bundled"
    return cells, sure, snapshot


@core.catalog_snapshot()
def catalog():
    """Painted processed maps, including unverified paintings and current drafts.

    Planning reads metadata and painting arrays only. Scoring uses Sure manual
    cells; accepted predictions do not become their own answer keys.
    """
    result = {}
    processed_root = Path(core.DEM_DIR).resolve()
    for path in core.dem_files():
        name = core.dem_name(path)
        if name in result:
            raise ValueError(f"Duplicate processed map ID: {name}")
        relative = Path(path).resolve().relative_to(processed_root)
        record = core.map_record(name) or {}
        section = core.section(dict(record, working_dem=str(relative)))
        annotations = record.get("annotations") or {}
        if not isinstance(annotations, dict):
            raise ValueError(f"Expected an annotations dictionary for {name}")
        metadata = core.meta_get(name)
        site = record.get("site_id") or record.get("site") or core.map_site(name) or name
        group = record.get("group") or core.map_site(name) or site
        cells, sure, snapshot = _painting(name)
        verified = metadata.get("verified") is True
        if not np.isin(cells, CLASSES).any():
            continue
        if not sure.any():
            reason = "No sure manual cells remain after excluding accepted/unsure cells"
        else:
            reason = None
        result[name] = {
            "name": name, "tile_id": name, "map_id": record.get("map_id", name), "site_id": str(site),
            "group": str(group), "geography_aliases": record.get("geography_aliases", []),
            "source_kind": record.get("source_kind", "processed_map"), "section": section,
            "path": str(path), "snapshot": snapshot, "grid": record.get("grid"),
            "label_source": metadata.get("label_source") or annotations.get("label_source") or record.get("label_source") or (
                f"{snapshot} cell labels; sure cells excluding accepted predictions" if cells.any() else None),
            "painted": True, "verified": verified, "available": reason is None, "reason": reason,
            "label_status": metadata.get("label_status") or annotations.get("label_status") or
                            record.get("label_status") or "user verification flag; label independence not established",
            "provenance": metadata.get("provenance", annotations.get("provenance")),
            "feature_sources": record.get("feature_sources"),
            "restricted": bool(metadata.get("restricted") or record.get("restricted") or
                               record.get("restrictions") or annotations.get("restrictions")),
        }
    return _coalesce_groups(result, {})[0]


def _sample_group(key, sample, registry, aliases):
    for value in (sample.get("geographic_group"), sample.get("reference_group"),
                  registry.get(key, {}).get("group"), sample.get("public_region"), sample.get("site_id")):
        if value:
            return canonical_group(value, aliases)
    return canonical_group(core.map_site(key) or key, aliases)


def _training_labels(samples):
    values = []
    for sample in samples.values():
        labels = np.asarray(sample["y"])
        weights = np.asarray(sample.get("w", np.ones(len(labels))))
        values.append(labels[np.isin(labels, CLASSES) & (weights > 0)])
    return np.concatenate(values) if values else np.empty(0, np.uint8)


def _plan(bundle, names, registry, aliases):
    samples = bundle.get("samples", {})
    verified = assistant.verified_samples(samples)
    plans = []
    for name in dict.fromkeys(names):
        meta = registry.get(name, {})
        # Each turn starts from the complete verified set. Other selected maps,
        # including neighbours in the same location, remain available to train.
        train = {key: sample for key, sample in verified.items() if key != name}
        if not meta.get("available"):
            reason = meta.get("reason") or "No painted labels in the processed catalog"
        elif not len(_training_labels(train)):
            reason = "No verified training samples remain after leaving out this map"
        else:
            reason = None
        plans.append({"group": meta.get("group", name), "maps": [name],
                      "mode": "leave-one-map-out" if name in verified else "painted-holdout",
                      "train_keys": sorted(train),
                      "excluded_keys": sorted(set(samples) - set(train)), "reason": reason})
    return plans


@core.catalog_snapshot()
def plan(bundle, names):
    """Quick metadata/sample-memory fold plan; no terrain extraction or fitting."""
    registry, aliases = _coalesce_groups(catalog(), bundle.get("samples", {}))
    return _plan(bundle, names, registry, aliases)


def _cell_ids(shape):
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("A nonempty 2-D raster is required")
    # core.per_pixel owns the floor-edge convention, including duplicate edges when
    # a public edge raster is smaller than the painting grid.
    return core.per_pixel(np.arange(GRID * GRID).reshape(GRID, GRID), shape)


def cell_sum(values):
    values = np.asarray(values)
    return np.bincount(_cell_ids(values.shape).ravel(), weights=values.ravel(),
                       minlength=GRID * GRID).reshape(GRID, GRID)


def cell_majority(prediction, valid=None):
    prediction = np.asarray(prediction)
    valid = np.ones(prediction.shape, bool) if valid is None else valid
    counts = np.stack([cell_sum(valid & (prediction == c)) for c in CLASSES])
    cells = (counts.argmax(axis=0) + 1).astype(np.uint8)
    cells[counts.sum(axis=0) == 0] = 255
    return cells


def _load_map(meta):
    stack, names, shade, profile, px = core.build(meta["path"])
    profile = cast(dict[str, Any], profile)
    X = assistant.features(stack, names).astype(np.float32, copy=True)
    grid = data._Grid(X.shape[1:], profile["transform"], profile["crs"])
    # Feature-source coverage can be smaller than the target DEM footprint. Keep
    # its native terrain/labels for coverage denominators and complete label panels.
    z, transform, crs, _ = core.read(meta["path"], core.MAX_SIDE)
    if data._Grid(z.shape, transform, crs) != grid:
        raise ValueError(f"Feature grid differs from the target DEM: {meta['name']}")
    terrain = np.isfinite(z)
    valid = terrain & np.isfinite(X).all(axis=0)
    cells, manual, snapshot = _painting(meta["name"])
    reference = core.cells_to_labels(cells, grid.shape, terrain)
    ground, area = cell_sum(terrain), cell_sum(np.ones(grid.shape, bool))
    finite = cell_sum(valid)
    known = cell_sum(valid & np.isin(reference, CLASSES))
    eligible = (manual & (area > 0) & (ground >= data.MIN_COVERAGE * area) & (ground > 0)
                & (finite >= data.MIN_COVERAGE * ground) & (known >= data.MIN_COVERAGE * ground))
    X[:, ~valid] = np.nan
    return {
        "tile_id": meta["name"], "map_id": meta["map_id"], "site_id": meta["site_id"],
        "group": meta["group"], "snapshot": snapshot, "grid": grid.metadata(), "px_m": float(px),
        "X": X, "hillshade": np.asarray(shade, np.float32), "reference": reference,
        "cells": cells, "eligible": eligible, "valid": valid, "support": np.zeros(grid.shape, bool),
        "reference_terrain_pixels": int(terrain.sum()),
        "public_coverage_fraction": float(valid.sum() / terrain.sum()) if terrain.any() else 0.0,
        "exclusions": {"no_pixel_cells": int((area == 0).sum()),
                       "not_sure_manual_cells": int((~manual).sum()),
                       "insufficient_coverage": int((manual & ~eligible).sum())},
    }


def _load_tiles(names, registry, progress):
    tiles, skipped, provenance = [], [], {"maps": {}}
    for name in names:
        meta = registry[name]
        if progress:
            progress(f"Loading evaluation map: {name}")
        tile = _load_map(meta)
        provenance["maps"][name] = meta
        if not tile["eligible"].any():
            skipped.append({"name": name, "group": meta["group"], "reason": "No eligible labelled cells with sufficient finite terrain coverage"})
            continue
        tile = dict(tile, label_source=meta["label_source"], source_kind=meta["source_kind"],
                    label_status=meta["label_status"], source_label=meta["label_source"],
                    review_status=meta["label_status"], verified=meta["verified"],
                    restricted=meta.get("restricted", False), section=meta["section"])
        tiles.append(tile)
    return tiles, skipped, provenance



@core.catalog_snapshot()
def run(bundle, names, progress=None, output_root=None, model_root=None, poster_root=None):
    """Evaluate selected maps; return (report_path, report), leaving the bundle unchanged.

    Publish one dated JSON in output_root/evaluations, with no images or sidecars.
    Fold models and predictions stay in memory; model_root and poster_root are
    accepted only for call compatibility. No valid planned folds means ValueError
    before any output creation.
    """
    if core.MODEL not in ("rf", "gbm"):
        raise ValueError("Selected-map evaluation supports the RF and GBM pixel backends")
    if bundle.get("schema") != assistant.schema():
        raise ValueError("Assistant schema differs from the current feature/backend schema")
    selected = list(dict.fromkeys(names))
    registry, aliases = _coalesce_groups(catalog(), bundle.get("samples", {}))
    plans = _plan(bundle, selected, registry, aliases)
    if not any(p["reason"] is None for p in plans):
        detail = "; ".join(f"{p['group']}: {p['reason']}" for p in plans) or "No maps selected"
        raise ValueError("No evaluable folds: " + detail)
    skipped = []
    for fold in plans:
        for name in fold["maps"]:
            reason = registry.get(name, {}).get("reason") or ("Map is not in the current catalog" if name not in registry else None)
            reason = reason or fold["reason"]
            if reason:
                skipped.append({"name": name, "group": fold["group"], "reason": reason})
    wanted = [n for p in plans if p["reason"] is None for n in p["maps"] if registry.get(n, {}).get("available")]
    with rasterio.Env(GDAL_PAM_ENABLED="NO", PROJ_NETWORK="OFF"):
        tiles, extraction_skips, provenance = _load_tiles(wanted, registry, progress)
    skipped.extend(extraction_skips)
    if not tiles:
        raise ValueError("No selected maps have eligible labelled cells after terrain extraction")
    now = dt.datetime.now(dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    base = (Path(output_root or paths.OUTPUT_DIR) / "evaluations").resolve()
    report_path = base / f"evaluation-{run_id}.json"
    roots = [Path(paths.DATA_DIR).resolve(), Path(core.DEM_DIR).resolve()]
    roots += [Path(r["path"]).resolve().parent for r in registry.values() if r.get("path")]
    if any(report_path.resolve().is_relative_to(root) for root in roots):
        raise ValueError("Evaluation outputs must not be inside source datasets")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an evaluation report: {report_path}")
    samples = bundle["samples"]
    folds, scored = [], []
    aggregate, base_aggregate = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
    for planned in plans:
        tests = [t for t in tiles if t["tile_id"] in planned["maps"]]
        if planned["reason"] or not tests:
            continue
        train = {k: samples[k] for k in planned["train_keys"]}
        labels = _training_labels(train)
        counts = np.array([int((labels == c).sum()) for c in CLASSES])
        majority = int(counts.argmax() + 1)
        if progress:
            progress(f"Evaluating map: {planned['maps'][0]}")
        started = perf_counter()
        single_class = len(np.unique(labels)) == 1
        fitted = None if single_class else assistant.fit(train)
        if fitted is not None and fitted["model"] is None:
            reason = "Backend could not fit the retained training samples; no in-sample fallback"
            skipped.extend({"name": t["tile_id"], "group": t["group"], "reason": reason} for t in tests)
            continue

        matrix, baseline_matrix = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
        train_groups = sorted({_sample_group(k, s, registry, aliases) for k, s in train.items()})
        if set(planned["maps"]) & set(train):
            raise ValueError("Unsafe fold: evaluation map is present in training")
        for tile in tests:
            if single_class:
                # A one-unit training set can still predict that unit everywhere.
                pred = np.full(tile["valid"].shape, majority, np.uint8)
                confidence = np.ones(tile["valid"].shape, np.float32)
            else:
                pred = assistant.predict(fitted, tile["X"], list(core.BASE_FEATS))
                confidence = assistant.confidence(fitted, tile["X"], list(core.BASE_FEATS))
            if pred is None or confidence is None:
                raise ValueError("Fold backend returned no predictions/confidence")
            pred, confidence = np.asarray(pred, np.uint8).copy(), np.asarray(confidence, np.float32).copy()
            if pred.shape != tile["valid"].shape or confidence.shape != pred.shape:
                raise ValueError("Fold prediction grid differs from the selected map")
            if not np.isin(pred[tile["valid"]], CLASSES).all() or not np.isfinite(confidence[tile["valid"]]).all():
                raise ValueError("Fold returned missing predictions on finite terrain")
            pred[~tile["valid"]], confidence[~tile["valid"]] = 255, np.nan
            cell_pred = cell_majority(pred, tile["valid"])
            keep = tile["eligible"]
            metrics = report_helpers.score(tile["cells"][keep], cell_pred[keep])
            baseline = report_helpers.score(tile["cells"][keep], np.full(int(keep.sum()), majority, np.uint8))
            matrix += metrics["confusion_matrix"]
            baseline_matrix += baseline["confusion_matrix"]
            tile.update(prediction=pred, confidence=confidence, cell_prediction=cell_pred, metrics=metrics)
            scored.append(tile)

        folds.append({"group": planned["group"], "mode": planned["mode"],
                      "model_kind": "single-class" if single_class else core.MODEL,
                      "train_groups": train_groups, "train_tiles": sorted(train),
                      "test_tiles": sorted(t["tile_id"] for t in tests),
                      "sampled_pixels": sum(len(s["y"]) for s in train.values()),
                      "metrics": report_helpers._metrics(matrix), "baseline": report_helpers._metrics(baseline_matrix),
                      "majority_class": majority, "model_revision": fitted["revision"] if fitted else None,
                      "elapsed_seconds": perf_counter() - started})
        aggregate += matrix
        base_aggregate += baseline_matrix
    if not folds:
        raise ValueError("No fold could be fitted; active model unchanged, no latest report published")
    report = {
        "run_id": run_id, "created_at": now.isoformat(), "schema": json.loads(json.dumps(assistant.schema())),
        "feature_names": list(core.BASE_FEATS), "model_revision": bundle.get("revision"),
        "restricted": any(s.get("restricted", registry.get(k, {}).get("restricted", True))
                          for k, s in samples.items()) or any(t["restricted"] for t in scored),
        "training": {"tiles": len(samples), "map_groups": len({s.get("map_id", k) for k, s in samples.items()}),
                     "geography_groups": len({_sample_group(k, s, registry, aliases) for k, s in samples.items()}),
                     "sampled_pixels": sum(len(s["y"]) for s in samples.values()),
                     "samples_by_tile": {k: len(s["y"]) for k, s in samples.items()}},
        "validation": {"method": "independent selected-map evaluation", "aggregate": report_helpers._metrics(aggregate),
                       "map_mean_macro_f1": float(np.mean([f["metrics"]["macro_f1"] for f in folds])),
                       "baseline_aggregate": report_helpers._metrics(base_aggregate), "folds": folds},
        "tiles": [{**{k: t[k] for k in ("tile_id", "map_id", "site_id", "group", "snapshot", "grid", "px_m",
                                       "source_kind", "label_source", "label_status", "verified", "section", "reference_terrain_pixels",
                                       "public_coverage_fraction", "exclusions", "metrics")},
                   "eligible_cells": int(t["eligible"].sum()),
                   "source_label": t["label_source"], "review_status": t["label_status"]} for t in scored],
        "selected_maps": selected, "skipped": skipped, "plan": plans,
        "software_versions": report_helpers._software_versions(),
        "policies": {
            "catalog": "only core.dem_files processed maps; sections follow core.section (declared section or top-level processed-data folder); source_kind does not select evaluation behavior",
            "split": "one fold per selected map; leave only that map out of verified training samples for its turn, then restore it for the next; unverified evaluation maps do not enter training",
            "baseline": "majority of retained sampled training labels with positive weights; unweighted counts, ties lowest code; NOT the unsampled-cell reference-training baseline",
            "labels": "core painting/accepted/confidence APIs for every map, including bundled annotations; sure cells only, accepted predictions excluded; no vector-label special case",
            "verification": "evaluation does not require verification; core.meta_get verified=true controls training membership only; source label_status remains unchanged",
            "training_verification": "only currently verified map samples train each fold; missing verification and revoked verification are excluded",
            "confidence": "missing painting confidence follows core defaults, not a certification of independent truth",
            "geometry": "floor cell edges via core.per_pixel and bincount; zero-pixel cells excluded; >=90% reference terrain and >=90% finite terrain/known-label coverage",
            "features": "six ordered assistant terrain features from core.build for every map; feature_sources transport belongs to core; no dataset-specific extractor or persistent X cache",
            "support": "not an eligibility filter; unobserved support shown as false, not evidence of measured lack of support",
            "exports": "one dated JSON in output/evaluations embeds results and provenance; no images or sidecars; predictions remain in memory",
            "models": "fresh fold fits in memory only; a single remaining class predicts that class everywhere; no model artifacts, active-model writes or in-sample fallback",
        },
        "limitations": [
            "Verification controls training, not evaluation eligibility or independent label certification. Source label_status is preserved.",
            "Maps are held out individually, not by geography. Neighbouring or overlapping maps can remain in training; scores do not measure transfer to unseen locations.",
            "Pixel targets expand coarse painting cells; pixels are not independent geological truth. Class 3 is not an ice/PSR or crater-object detector.",
            "Only selected, evaluable maps contribute to pooled and map-mean metrics; unavailable or impossible selections are disclosed in skipped.",
            "Missing historical confidence/acceptance flags cannot establish label independence; core painting defaults apply uniformly to all datasets.",
            "All sources and derived artifacts remain local-use/CUI restricted where applicable; no upload or publication is authorized.",
        ],
        "provenance": provenance,
    }
    base.mkdir(parents=True, exist_ok=True)
    report_helpers._atomic_json(report_path, report)
    return report_path, report


def _read_report(path, run_id):
    """Validate a completed JSON evaluation report."""
    report = data._json(path)
    if not isinstance(report, dict) or report.get("run_id") != run_id:
        raise ValueError("Report identity differs from its filename")
    validation = report.get("validation")
    if not isinstance(validation, dict) or not isinstance(validation.get("aggregate"), dict):
        raise ValueError("Expected completed evaluation metrics")
    for key in ("aggregate", "baseline_aggregate"):
        metrics = validation.get(key, {})
        if not isinstance(metrics, dict):
            raise ValueError("Expected metric objects")
        for field in ("accuracy", "macro_f1"):
            value = metrics.get(field)
            if value is not None and (type(value) not in (int, float) or not 0 <= value <= 1):
                raise ValueError("Invalid evaluation metric")
    count = validation["aggregate"].get("n")
    if type(count) is not int or count <= 0:
        raise ValueError("Expected scored cells")
    for container, key in ((report, "selected_maps"), (report, "tiles"),
                           (report, "skipped"), (validation, "folds")):
        values = container.get(key, [])
        kind = str if key == "selected_maps" else dict
        if not isinstance(values, list) or any(not isinstance(value, kind) for value in values):
            raise ValueError(f"Invalid report {key}")
    if (not isinstance(report.get("provenance"), dict) or
            not isinstance(report["provenance"].get("maps"), dict)):
        raise ValueError("Expected embedded map provenance")
    json.dumps(report, allow_nan=False)
    return report


def latest_report(output_root=None):
    """Return (report_path, report) for the newest valid dated JSON; read-only.

    Dates come from the UTC run IDs, not file mtimes. Incomplete, corrupt or unsafe
    candidates do not hide older completed runs. Only output_root/evaluations is
    searched; browsing never creates files or directories.
    """
    root = (Path(output_root or paths.OUTPUT_DIR) / "evaluations").resolve()
    for candidate in sorted(root.glob("evaluation-*.json"), reverse=True):
        try:
            match = re.fullmatch(r"evaluation-(\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8})\.json", candidate.name)
            if match is None or candidate.is_symlink() or not candidate.is_file():
                continue
            run_id = match[1]
            dt.datetime.strptime(run_id[:23], "%Y%m%dT%H%M%S.%fZ")
            result = data._path(root, candidate.name)
            return result, _read_report(result, run_id)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue
    return None
