"""
LightGBM: gradient-boosted trees, one prediction per pixel. The default.

Refits from current verified paintings when Update model is clicked.

Sees each pixel on its own: it knows a pixel's slope and roughness, not what its
neighbours were labelled. Spatial context reaches it only through the features that
already average over a window (roughness, relative elevation, sky-view factor).

The backend interface is documented beside core.MODEL.
"""
import numpy as np

from app import core


NAME = "LightGBM"


def fit(X, cells):
    """X: (n_feats, H, W) feature stack. cells: (GRID, GRID) painting, 0 = unpainted.
    Returns None when fewer than two units are painted -- nothing to tell apart yet."""
    Xtr, ytr = core.cell_pixels(X, cells)
    return fit_rows(Xtr, ytr)


def fit_rows(Xtr, ytr, w=None):
    """Fit the shared assistant from sampled human labels across maps, each counted by
    its weight `w` (how sure the painter was)."""
    if Xtr is None or len(np.unique(ytr)) < 2:
        return None
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.1, num_leaves=31,
                             min_child_samples=20, class_weight="balanced", verbose=-1,
                             random_state=0)
    clf.fit(Xtr, ytr, sample_weight=w)
    return clf


def predict(model, X):
    """(n_feats, H, W) -> (H, W) uint8 of unit codes, 255 where there is no terrain."""
    return core.blockwise(model.predict, X)


def confidence(model, X):
    """(n_feats, H, W) -> (H, W) float32: how sure the model is of each pixel's unit (the
    probability it gives it), NaN where there is no terrain."""
    return core.blockwise(lambda rows: model.predict_proba(rows).max(1), X,
                          dtype=np.float32, fill=np.nan)
