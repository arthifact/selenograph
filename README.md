# Selenograph

Paint and review lunar geological maps with a Random Forest assistant. The app
has three pages: **Paint**, **Gallery** and **Progress**.

This repository includes **237 processed tiles across 13 sites**, current
paintings on **13 verified maps**, saved GIS exports and the trained model. You can
continue that work on another computer immediately. Raw downloads, private source
collections, poster work and historical backups stay outside Git.

## Start

Install [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/).
On macOS, Git LFS is available with `brew install git-lfs`.
Then run:

```sh
git lfs install
git clone https://github.com/arthifact/selenograph.git
cd selenograph
git lfs pull
uv venv --python 3.12
uv pip install -r requirements.txt
uv run --no-project streamlit run selenograph.py
```

Open **http://localhost:8501**. The initial dataset download is about **1.4 GB**;
Git LFS downloads the terrain rasters and trained model. Original source ZIPs are
not needed to use the app. GitHub's ordinary source ZIP may contain LFS pointers;
use the clone instructions above to get the complete working dataset.

For a download without Git, use **`selenograph-working-dataset.zip`** from the
[release page](https://github.com/arthifact/selenograph/releases/latest). It contains
the actual rasters and model. Extract it, open the `selenograph` folder, and run the
three `uv` setup commands above. Use Git when syncing work between computers.

Random Forest is the default. To use the optional LightGBM backend, install
`requirements-gbm.txt`; macOS also needs `brew install libomp`.

## Paint, save and train

1. In **Paint**, choose a **Section** and **Map**, or browse **All**. Choose a
   background and paint geological units. The six terrain features are the model's
   inputs; changing the displayed background does not change training.
2. Changes autosave as drafts. **Restore saved** replaces the draft with the last
   saved painting. **Save labels + map** saves the labels and overwrites that
   tile's GIS exports.
3. Mark a map **Verified** when its labels are ready for training. **Update model**
   rebuilds the model from the latest paintings on every currently verified map
   and refreshes the prediction. Yellow button text means training inputs changed;
   green means they match the last update. Browsing and painting do not retrain.
4. Turn on both **Prediction** and **Painting** to see disagreements in yellow.
   Unpainted cells and missing predictions are not compared. **Colour strength**
   controls the highlight; **Flip** switches between the individual layers.
5. **Gallery** shows all tiles by default. Filter by verified maps, saved unverified
   paintings or section. Missing data appears in black, with edge tiles positioned
   inside the same square as full tiles.

**Progress → Evaluation** compares predictions with Sure hand-labeled cells,
excluding approved predictions. Each selected map is scored separately: if it is
in training, only that map is excluded for its turn. Other maps, including its
neighbours, remain in training. Unverified maps use the full verified training set.
The active model stays unchanged. A map cannot score if its exclusion leaves no
training samples. Update the model before evaluating new verified paintings.
Each evaluation creates one JSON report, without image exports.

## Continue on another computer

Use the same setup commands on the lab computer. Before switching computers,
commit and push your paintings, exports and model:

```sh
git add output/ models/
git commit -m "Save mapping progress"
git push
```

On the other computer, pull before starting work:

```sh
git pull --ff-only
git lfs pull
uv run --no-project streamlit run selenograph.py
```

Training status compares relative paths and file contents, so changing computers
does not make the saved model appear outdated. Finish syncing one computer before
editing the same painting on another.

## Files

| Folder | Contents |
|---|---|
| `data/processed_data/` | Native 5 m SfS terrain, imagery, quality layers and tile catalogs |
| `data/raw_data/` | Empty placeholder for optional original downloads; contents ignored |
| `output/painting_drafts/` | Current edits, confidence and verification flags |
| `output/paintings/` | Explicitly saved labels and metadata |
| `output/maps/` | Saved label and prediction GeoTIFFs for GIS |
| `output/evaluations/` | One JSON per evaluation |
| `models/rf.joblib` | Trained Random Forest and its training samples |

Each output directory mirrors the DEM's parent path relative to
`data/processed_data/`. **Save labels + map** overwrites `<tile>_labels.tif`
(your labels) and `<tile>_map.tif` (the prediction), retaining coordinates,
projection, unit colours and the NoData mask. Without a trained model, only the
label TIFF is saved. Observation rasters remain unchanged.

Expert labels were transferred from the former duplicate maps onto 13 public
tiles. Their sources and edits remain recorded in the metadata; there are no
separate reference sections in the app. Original data is linked in
[data/sources.md](data/sources.md). Verification means ready for training; it is
not independent certification or a license change. Only load model files you trust.

The terrain recipe is version 6: slope, roughness, sky view, local height,
curvature and local sky view. See [training guidance](docs/TRAINING.md) and
[dataset details](docs/DATASET.md).

## Rebuild data or check the app

Rebuilding is optional. Download the regional ZIPs into `data/raw_data/` as
explained in [data/sources.md](data/sources.md), then run:

```sh
uv run --no-project python data/prepare_dataset.py
uv run --no-project python data/prepare_dataset.py --verify
```

The preparation script creates the processed layout and reuses completed regions.
Verification checks against the original downloads. To check the installed app
without those downloads:

```sh
uv run --no-project python -B -m tests.smoke_app_layout
```

The smoke check browses the app without editing paintings, exporting files or
training. Run the test suite with:

```sh
uv run --no-project python -B -m unittest discover -s tests -q
```
