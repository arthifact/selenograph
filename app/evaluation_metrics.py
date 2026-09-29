"""Evaluation metrics and JSON publication; no training or figure exports."""
import json
import os
import platform
import uuid
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import rasterio

CLASSES = (1, 2, 3)


def _software_versions():
    """Installed runtime versions, without importing/fitting optional estimators."""
    versions = {"python": platform.python_version(),
                "gdal": rasterio.__gdal_version__, "proj": rasterio.__proj_version__}
    for package in ("numpy", "scipy", "rasterio", "scikit-learn", "joblib", "Pillow"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    n = int(matrix.sum())
    per_class = {}
    for i, code in enumerate(CLASSES):
        tp = int(matrix[i, i])
        support, predicted = int(matrix[i].sum()), int(matrix[:, i].sum())
        union = support + predicted - tp
        per_class[str(code)] = {
            "precision": float(tp / predicted) if predicted else 0.0,
            "recall": float(tp / support) if support else 0.0,
            "f1": float(2 * tp / (support + predicted)) if support + predicted else 0.0,
            "iou": float(tp / union) if union else 0.0, "support": support,
        }
    return {"n": n, "accuracy": float(np.trace(matrix) / n) if n else 0.0,
            "macro_f1": float(np.mean([c["f1"] for c in per_class.values()])),
            "per_class": per_class, "confusion_matrix": matrix.tolist()}


def score(truth, prediction):
    """Fixed three-class metrics; 0/255 truth is unknown, never a fourth class.

    Missing predictions for known targets are errors, not silently dropped rows.
    Absent-class and empty-set metrics are zero (fixed-label macro, zero_division=0).
    """
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction shapes differ")
    known = np.isin(truth, CLASSES)
    truth, prediction = truth[known].astype(np.int64), prediction[known]
    if not np.isin(prediction, CLASSES).all():
        raise ValueError("Missing/invalid prediction for an eligible target")
    index = (truth - 1) * 3 + prediction.astype(np.int64) - 1
    return _metrics(np.bincount(index, minlength=9).reshape(3, 3))


def _atomic_json(path, value):
    staged = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with staged.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
