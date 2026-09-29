"""
U-Net: a convolutional segmentation network.  NOT IMPLEMENTED YET -- this file is the
placeholder, so switching core.MODEL to "unet" fails with a clear message instead of an
import error.

Why it is worth having, and why it is not just a swap
-----------------------------------------------------
LightGBM and the random forest label each pixel from its own feature vector. A U-Net
labels a whole tile at once, so it can learn that a crater floor is a connected region
inside a rim rather than a scatter of steep pixels. That is the thing the per-pixel
models cannot do, and it is the reason to try it.

Implementing this backend requires a spatial training pipeline:

* `fit` gets a (GRID, GRID) painting where most cells are 0 = unpainted. Those are not
  background, they are unknown, so the loss has to ignore them -- a masked cross-entropy
  over the painted cells only, not a plain one.
* Training tiles have to come from the raster with their spatial context intact, not as
  the shuffled per-pixel rows `core.cell_pixels` hands back. That sampler is the wrong
  tool here; read tiles from X directly.
* The assistant currently pools pixel rows through `fit_rows`. A U-Net needs stored
  spatial training tiles and an explicit update action that can handle longer fits.
* It will want a GPU, and `torch` plus `segmentation-models-pytorch` in requirements.txt
  (about 2 GB). Neither is installed right now.

Changing core.MODEL selects this placeholder. Implementing it also requires adapting
the assistant's training storage and sampler to preserve spatial context.
"""
from app import core


NAME = "U-Net (not implemented)"

_MSG = ("The U-Net backend is a placeholder and has nothing behind it yet. "
        "Set MODEL = \"gbm\" (or \"rf\") in core.py, or implement model_unet.py -- "
        "the file documents what it needs.")


def fit(X, cells):
    """X: (n_feats, H, W) feature stack. cells: (GRID, GRID) painting, 0 = unpainted.
    Should return a trained model, or None when fewer than two units are painted."""
    raise NotImplementedError(_MSG)


def fit_rows(Xtr, ytr, w=None):
    """A spatial training pipeline is needed before this backend can be used."""
    raise NotImplementedError(_MSG)


def predict(model, X):
    """(n_feats, H, W) -> (H, W) uint8 of unit codes, 255 where there is no terrain."""
    raise NotImplementedError(_MSG)


def confidence(model, X):
    """(n_feats, H, W) -> (H, W) float32 probability of the predicted unit."""
    raise NotImplementedError(_MSG)
