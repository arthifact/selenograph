# Dataset and saved work

The app uses one public observation catalog: **237 tiles across 13 site folders**.
All training terrain uses the native **5 m shape-from-shading DEMs**. Imagery and
quality layers are included for inspection. Missing observations are not filled;
edge tiles retain their actual extent and appear inside a full square in the app.

## Installed layout

```text
data/
  sources.md
  prepare_dataset.py
  raw_data/                            empty download placeholder in Git
  processed_data/
    manifest.json
    <site>/
      working_dems.json
      metadata/                        source hashes, grids, quality and readme
      tiles/<tile-id>/
        elevation/<tile-id>.tif
        imagery/<tile-id>_nac.tif
        quality/<tile-id>_<layer>.tif
```

The DEM's filename stem is its stable map ID. Tile row and column numbers are
zero-based. Catalog paths are relative, so the installed dataset works after
cloning or moving the project. Original downloads and historical backups are
excluded from Git. The processed TIFFs and active model use Git LFS.

## Paintings and exports

Expert labels from the former duplicate maps were transferred onto **13 public
tiles**, with their confidence, approval, verification and source metadata. The
app uses their ordinary public tile names. There are no duplicate reference
sections. Original source names remain descriptive metadata, not dependencies.

Each output bucket mirrors the DEM's exact parent path relative to
`data/processed_data/`. For example:

```text
data/processed_data/site/tiles/id/elevation/id.tif
output/painting_drafts/site/tiles/id/elevation/id_painting.npy
output/paintings/site/tiles/id/elevation/id_painting.npy
output/maps/site/tiles/id/elevation/id_labels.tif
output/maps/site/tiles/id/elevation/id_map.tif
```

Painting arrays use `<id>_painting.npy`, `<id>_confidence.npy` and
`<id>_accepted.npy`; `<id>_meta.json` records flags and source information.
Changes in Paint and Gallery are drafts. **Save labels + map** commits the current
painting and metadata, then replaces the GIS label and prediction exports.
**Restore saved** restores the saved painting to the draft. Browsing does not
create output directories or modify observations.

Only verified maps train the assistant. **Update model** reads their current
paintings and writes `models/rf.joblib`; it does not save paintings or GIS exports.
An evaluation saves one JSON under `output/evaluations/`, leaving the active model
unchanged. There is no persistent test-map selection in the app.

## Optional rebuild

Follow [data/sources.md](../data/sources.md): download the regional ZIPs into
`data/raw_data/` without extracting them, then run:

```sh
uv run --no-project python data/prepare_dataset.py
```

The script creates missing processed directories, aligns companion layers to the
native SfS grid, creates 1024-pixel tiles and writes relative catalogs. Existing
completed regions are verified and reused. Interrupted work can resume with the
same command. Preparation does not alter your paintings or train a model.

To verify against the original ZIPs:

```sh
uv run --no-project python data/prepare_dataset.py --verify
```

The older `raw_data/essentials/` input layout is also accepted. Raw preparation
uses Unix file locking (macOS/Linux). Original public files are attributed in
regional readmes and linked from the source guide. Label restrictions recorded
in painting metadata are preserved; verification is a training flag.

See [setup](../README.md#start) and [training details](TRAINING.md).
