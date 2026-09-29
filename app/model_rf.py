"""
Random forest: bagged decision trees, one prediction per pixel.

Same shape of model as LightGBM -- per-pixel, tree-based, trained on the pixels inside
the cells you painted -- but grown independently and averaged instead of boosted in
sequence. Usually a little slower to train and a little smoother at the contacts, which
makes it a fair comparison rather than a different kind of thing.

Needs no NaN handling of its own: scikit-learn's trees have taken missing values
natively since 1.4, and the feature stack carries NaN wherever the DEM has holes.

The backend interface is documented beside core.MODEL.
"""
import numpy as np

from app import core


NAME = "Random forest"


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
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                 class_weight="balanced", n_jobs=-1, random_state=0)
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
