"""Persistent independent-check scores for verified processed-map labels.

Answer keys are saved user paintings or bundled annotations, excluding uncertain
and accepted-prediction cells. Verification is permission to score, not proof of
independent truth. Legacy labels-only TIFF exports remain supported. No original
vectors or source collections are discovered here. The Progress page uses app.evaluation for selected-map rotation.
"""
import datetime
import functools
import json
import os
import uuid
from pathlib import Path

import numpy as np
import rasterio

from app import assistant_model, core, paths
# Compatibility for older callers/tests that patched this module attribute.
# No evaluation path calls the offline importer.
from app import import_reference_maps as reference

KEY_VERSION = 3    # verified processed annotations only, never live vector keys


def _key_files(name):
    path = core.dem_path(name)
    if not path or not Path(path).is_file() or core.meta_get(name, final=True).get("verified") is not True:
        return None
    painting, accepted, confidence = core.painting_files(name, final=True, existing=True)
    labels = core.output_file(name, "labels.tif", existing=True)
    source = "your verified labels"
    if not any(Path(p).is_file() for p in (painting, labels)):
        painting, accepted, confidence, labels = (
            core.annotation_path(name, kind) for kind in ("painting", "accepted", "confidence", "labels"))
        source = "verified bundled labels"
    if any(p and Path(p).is_file() for p in (painting, labels)):
        return source, painting, accepted, confidence, labels
    return None


@core.catalog_snapshot()
def key_source(name):
    """Source of a verified saved/bundled key, never an unsaved draft or original vector."""
    files = _key_files(name)
    return files[0] if files else None


@core.catalog_snapshot()
def candidates():
    """Verified keys for existing maps in the processed catalog, in catalog order."""
    out = {}
    for path in core.dem_files():
        name = core.dem_name(path)
        source = key_source(name)
        if source:
            out[name] = source
    return out


def _cell_array(path, name):
    if not path or not Path(path).is_file():
        return None
    values = np.load(path, allow_pickle=False)
    if values.shape != (core.GRID, core.GRID):
        raise ValueError(f"Invalid answer-key cell grid for {name}: {path}")
    return values


@core.catalog_snapshot()
def answer_key(name):
    """(codes on the DEM grid, 255 where unscored; None for the legacy crater mask).

    A saved painting overrides bundled labels even when erased. Flags always come
    from the chosen source, never a draft or lower-priority bundled annotation.
    TIFF-only exports are a fallback, masked by both their nodata and DEM coverage.
    """
    files = _key_files(name)
    if files is None:
        return None
    _, painting, accepted, confidence, labels = files
    with rasterio.open(core.dem_path(name)) as src:
        terrain = src.read(1, masked=True)
        valid = ~np.ma.getmaskarray(terrain) & np.isfinite(terrain.data)
        shape, transform, crs = src.shape, src.transform, src.crs
    cells = _cell_array(painting, name)
    if cells is not None:
        if not np.isin(cells, (0, *core.CODES)).all():
            raise ValueError(f"Invalid answer-key unit code for {name}")
        codes = core.cells_to_labels(cells, shape)
    elif labels and Path(labels).is_file():
        with rasterio.open(labels) as src:
            if src.shape != shape or src.transform != transform or src.crs != crs:
                raise ValueError(f"Answer-key raster must match the DEM grid: {name}")
            values = src.read(1, masked=True)
            codes = np.where(~np.ma.getmaskarray(values) & np.isin(values.data, list(core.CODES)),
                             values.data, 255).astype(np.uint8)
    else:
        return None
    sure = np.ones((core.GRID, core.GRID), bool)
    levels = _cell_array(confidence, name)
    if levels is not None:
        sure &= levels == core.SURE
    approved = _cell_array(accepted, name)
    if approved is not None:
        sure &= ~approved.astype(bool)
    codes[~valid | ~core.per_pixel(sure, shape)] = 255
    return (codes, None) if np.isin(codes, list(core.CODES)).any() else None


@functools.lru_cache(maxsize=8)
def _features(dem_path, revision, recipe):
    X, names, *_ = core.build(dem_path)
    return X, names


@core.catalog_snapshot()
def test_inputs():
    """inputs_for() the test maps."""
    return inputs_for(core.test_maps())


@core.catalog_snapshot()
def inputs_for(names):
    """(name, features, feature names, answer key, small-crater mask or None) for each of
    `names` that has an answer key."""
    recipe = json.dumps(assistant_model.schema()["recipe"], sort_keys=True)
    out = []
    for name in names:
        got = answer_key(name)
        if got is None:
            continue
        path = core.dem_path(name)
        if not path:
            raise ValueError(f"No processed DEM for answer key: {name}")
        stat = os.stat(path)
        X, feats = _features(path, (stat.st_mtime_ns, stat.st_size), recipe)
        out.append((name, X, feats, *got))
    return out


def score(bundle, maps=None, preds=None):
    """How the bundle's model maps the test maps (or `maps`, from inputs_for), or None
    without a model or answer keys. `preds` ({name: prediction}) saves predicting again.
    `accuracy` is the share of scored pixels it gets right over all test maps, and per
    map in `maps`; `iou` is each unit's overlap with the answer keys; `baseline` is the
    accuracy of always guessing the commonest unit. `small_craters` says how much of the
    single small craters the keys map it calls shadowed floor (`found`), against how
    much of the plain highlands it calls that (`on_highlands`, the false alarms)."""
    if bundle["model"] is None:
        return None
    per_map, keys, guesses, found, alarms = {}, [], [], [], []
    for name, X, names, key, small in test_inputs() if maps is None else maps:
        pred = preds[name] if preds else assistant_model.predict(bundle, X, names)
        if pred is None:
            continue
        scored = np.isin(key, list(core.CODES)) & np.isin(pred, list(core.CODES))
        if scored.any():
            per_map[name] = round(float((key[scored] == pred[scored]).mean()), 3)
            keys.append(key[scored])
            guesses.append(pred[scored])
        if small is not None:
            found.append(pred[small & scored] == 3)
            alarms.append(pred[scored & ~small & ((key == 1) | (key == 2))] == 3)
    if not per_map:
        return None
    craters = None
    if found and sum(f.size for f in found):
        f, a = np.concatenate(found), np.concatenate(alarms)
        craters = dict(found=round(float(f.mean()), 3), pixels=int(f.size),
                       on_highlands=round(float(a.mean()), 3) if a.size else None)
    k, p = np.concatenate(keys), np.concatenate(guesses)
    iou = {}
    for code, unit in core.CODES.items():
        union = int(((k == code) | (p == code)).sum())
        iou[unit] = round(int(((k == code) & (p == code)).sum()) / union, 3) if union else None
    share = np.bincount(k, minlength=max(core.CODES) + 1)
    return dict(accuracy=round(float((k == p).mean()), 3), iou=iou,
                baseline=round(float(share.max() / share.sum()), 3), pixels=int(k.size),
                maps=per_map, small_craters=craters)


@core.catalog_snapshot()
def record(bundle):
    """Score the bundle and save one JSON in output/evaluations, or nothing."""
    result = score(bundle)
    if result is None:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    entry = dict(kind="independent_check", run_id=run_id,
                 revision=bundle["revision"], test_maps=sorted(core.test_maps()),
                 key_version=KEY_VERSION,
                 at=now.isoformat(timespec="seconds"),
                 backend=core.MODEL, recipe=assistant_model.schema()["recipe"],
                 inputs=list(core.BASE_FEATS),
                 trained_on=sorted(bundle["samples"]), **result)
    content = json.dumps(entry, indent=2, allow_nan=False) + "\n"
    path = Path(paths.OUTPUT_DIR) / "evaluations" / f"independent-check-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".tmp")
    try:
        staged.write_text(content, encoding="utf-8")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return entry


def history():
    """Logged scores comparable with now: the same test maps and answer keys, whatever
    model or inputs made them -- which is how changes to those are judged. Oldest first."""
    root = Path(paths.OUTPUT_DIR) / "evaluations"
    now = (sorted(core.test_maps()), KEY_VERSION)
    out = []
    for path in sorted(root.glob("independent-check-*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            e = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(e, dict) and e.get("kind") == "independent_check" and \
                (e.get("test_maps"), e.get("key_version")) == now:
            out.append(e)
    return out


@core.catalog_snapshot()
def folds():
    """Verified processed keys grouped by their declared site, for legacy rotation."""
    out = {}
    for name in candidates():
        out.setdefault(core.map_site(name) or name, []).append(name)
    return {site: sorted(names) for site, names in sorted(out.items())}
