"""Optional legacy poster rotation; artifacts stay inside poster/legacy."""
import datetime
import json
from pathlib import Path
import numpy as np
from app import assistant_model, core
from app.evaluate import KEY_VERSION, folds, inputs_for, score
from poster import storage

VALIDATION_LOG = "validation.jsonl"
MAPS_DIR = "validation_maps"

@core.catalog_snapshot()
def validate(bundle):
    """Legacy rotation over verified processed keys, excluding each site's geography.
    Logs and returns the per-site scores and their mean (each site counts equally), or
    None when there is nothing to train or score. Each scored map's prediction and
    confidence are kept (saved_maps) for pictures of what the model makes of ground it
    has never seen."""
    folder = storage.output_path(storage.ROOT / "legacy") / MAPS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.npz"):
        old.unlink()
    sites = {}
    for site, names in folds().items():
        held = core.held_out_maps(names)
        train = assistant_model.retained_samples(bundle["samples"], held)
        fold = assistant_model.fit(train)
        if fold["model"] is None:
            continue
        maps = inputs_for(names)
        preds = {}
        for name, X, feats, *_ in maps:
            preds[name] = assistant_model.predict(fold, X, feats)
            confidence = assistant_model.confidence(fold, X, feats)
            if confidence is None:
                raise ValueError(f"No model confidence for validation map: {name}")
            np.savez_compressed(folder / f"{name}.npz", pred=preds[name], site=site,
                                conf=confidence.astype(np.float16),
                                revision=bundle["revision"], trained_on=sorted(train))
        result = score(fold, maps, preds)
        if result:
            sites[site] = dict(result, trained_on=sorted(train))
    if not sites:
        return None

    def mean(values):
        values = [v for v in values if v is not None]
        return round(float(np.mean(values)), 3) if values else None

    craters = [r["small_craters"] for r in sites.values() if r["small_craters"]]
    entry = dict(revision=bundle["revision"], key_version=KEY_VERSION,
                 at=datetime.datetime.now().isoformat(timespec="seconds"),
                 backend=core.MODEL, inputs=list(core.BASE_FEATS),
                 trained_on=sorted(bundle["samples"]), sites=sites,
                 mean=dict(accuracy=mean(r["accuracy"] for r in sites.values()),
                           baseline=mean(r["baseline"] for r in sites.values()),
                           iou={u: mean(r["iou"][u] for r in sites.values())
                                for u in core.CODES.values()},
                           small_craters=dict(
                               found=mean(c["found"] for c in craters),
                               on_highlands=mean(c["on_highlands"] for c in craters))
                           if craters else None))
    path = storage.output_path(storage.ROOT / "legacy") / VALIDATION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def validation_history():
    """Logged validations with today's answer keys. Oldest first."""
    path = storage.output_path(storage.ROOT / "legacy") / VALIDATION_LOG
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("key_version") == KEY_VERSION:
            out.append(e)
    return out


def saved_maps():
    """The latest validation's maps: {name: dict(pred, conf, site, trained_on)}, each
    made by a model trained without that map's site."""
    folder = storage.output_path(storage.ROOT / "legacy") / MAPS_DIR
    out = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as f:
            out[path.stem] = dict(pred=f["pred"], conf=f["conf"].astype(np.float32),
                                  site=str(f["site"]), trained_on=list(f["trained_on"]))
    return out
