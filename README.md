# Selenograph

Map lunar geological units with imagery, terrain views and an assistant that learns
from your paintings. The app has three pages: **Paint**, **Gallery** and **Progress**.
It works with ordinary processed datasets; private reference maps are not required.
This repository contains the source code, documentation and tests. Datasets, saved
paintings, exported maps and trained models stay local and are not included.

## Sharing and privacy

- Share only authorized code, data and derived work. Include `requirements.txt`,
  `.streamlit/config.toml`, dataset manifests, provenance and source citations.
- Recipients create their own environment. Exclude `.venv/`, `.uv-cache/`, Python
  caches, `.agents/`, `.claude/` and secrets from shared copies.
- Review `data/`, `output/` and `models/` before sharing. Private references,
  their trained models and exports still require permission; Verified is not
  publication approval. Do not load untrusted model files. These workflows upload nothing.
- For approved data sharing, copy the whole `data/processed_data/` tree as regular
  files, preserving relative paths. This materializes source hard links safely and
  keeps linked feature inputs available. Never edit source-linked rasters in place.

## Start

Install [uv](https://docs.astral.sh/uv/), then run from this folder:

```sh
# First setup on a new computer; keep an existing working .venv:
uv venv --python 3.12
uv pip install -r requirements.txt
uv run --no-project streamlit run selenograph.py
```

Open **http://localhost:8501**. Random forest is the default; optional LightGBM
requires OpenMP on macOS (`brew install libomp`).

## Paint, review and evaluate

1. In **Paint**, choose a **Section** (the literal site folder), then a **Map**.
   **All** lists every available tile without a separate location filter.
   Choose a background, paint units and set confidence.
   Changing the background does not change the training inputs.
2. Changes autosave to `output/painting_drafts/`. Only **Save labels + map** in Paint
   commits them to `output/paintings/` and replaces that map's GeoTIFF exports in
   `output/maps/`. **Update model** retrains from the latest
   paintings on all currently verified maps,
   then refreshes the current prediction. It saves only the model; evaluations run
   separately. The button text is yellow when verified training inputs have changed,
   green when they match the last update. Browsing and painting do not train.
3. **Gallery** shows all installed maps in compact cards. Filter by **Verified only**,
   **Saved, unverified** (saved painted labels, excluding drafts and verified maps),
   or **Section**; the default shows all maps. Gallery and Paint show missing areas
   in black inside a full square, keeping edge tiles aligned with their neighbours.
   **Verified** permits training; it is a user trust flag, not
   independent certification. Changes remain drafts until you Save in Paint.
4. In **Progress → Evaluation**, select painted maps, including unverified maps.
   Each is scored separately. A selected training map is left out only for its own
   turn, then restored for the next; other maps in the same section remain in training.
   Unverified painted maps use the complete verified training set. The active model
   stays unchanged. If leaving a map out leaves no training samples, that map cannot score.

Bundled inputs stay immutable. Evaluation uses current draft/saved paintings and
compares Sure hand-painted cells, excluding approved predictions. Verification controls
training membership, not whether a painted map can be evaluated. Use **Update model**
to refresh training samples before evaluating new verified paintings.

## Data and model

```text
data/processed_data/<site>/  working_dems.json, observation layers, optional labels
data/raw_data/               Original public downloads
data/reference_data/         Preserved private originals; never auto-loaded by the app
output/painting_drafts/      Working paintings and metadata; mirrored DEM parent paths
output/paintings/            Saved paintings, flags, metadata and gallery thumbnails
output/maps/                 Saved GeoTIFFs for GIS; mirrored DEM parent paths
output/evaluations/          One JSON per evaluation, including results and provenance
models/                      Active assistant
```

`output/` stores evaluations in **`evaluations/`**, paintings in **`paintings/`**
and **`painting_drafts/`**, and GeoTIFF exports in **`maps/`**. Each map and painting directory mirrors the DEM's **exact parent directory relative to `data/processed_data/`**,
with no extra map folder:

```text
data/processed_data/site/tiles/id/elevation/id.tif
output/painting_drafts/site/tiles/id/elevation/id_painting.npy
output/paintings/site/tiles/id/elevation/id_painting.npy
output/maps/site/tiles/id/elevation/id_labels.tif
output/maps/site/tiles/id/elevation/id_map.tif
```

Both buckets use the same canonical filenames: `id_painting.npy`, `id_accepted.npy`,
`id_confidence.npy` and `id_meta.json`. Saved gallery thumbnails (`id_thumb.png`)
stay in `paintings/`. **Save labels + map** overwrites `maps/` exports at the same paths:
`id_labels.tif` contains your painting and `id_map.tif` contains the model prediction.
Both retain the map's coordinates, projection, unit colours and NoData mask for GIS use.
Without a trained model, only the label TIFF is saved and any previous prediction at
the current export path is removed. Older export layouts remain readable.
Draft metadata overrides
saved metadata until Save commits the effective metadata and removes that overlay.
Reference maps follow the same rule: a DEM at
`data/processed_data/site(reference)/tiles/id.tif` uses `site(reference)/tiles/`
inside each bucket, without inserting an `elevation/` or extra ID folder.

The six supplied paintings use the same explicit `core.seed_paintings()` bootstrap
as any catalog's bundled paintings. It creates independent painting/flag arrays in
**both** buckets; it copies neither label TIFFs nor metadata. Metadata stays inherited
from the bundled manifest. Any recognized existing canonical or legacy work—even
matching files or metadata alone—makes it skip the entire map, not fill a partially
seeded pair. Observation rasters are never copied or linked into output, and browsing
creates no output directories. See the
[storage and bootstrap contract](data/README.md#per-map-output-layout).

New evaluations save only one JSON file under `output/evaluations/`.

There is no public or reference wrapper under `processed_data/`. The development workspace has
**237 public tiles in 13 site folders**, plus six tiles in `mons-mouton(reference)`,
`nobile1(reference)`, `nobile2(reference)` and `nobile1-ms1(reference)`:
**243 tiles, 17 sections**. Those imported labels are user-marked verified as requested;
`label_status` preserves their historical, unreviewed provenance—not independent certification.

The active model is stored locally at **`models/rf.joblib`**. Click **Update model**
to train from your verified paintings. The current feature recipe is version 6,
which removes artificial source-tile seams; older models need retraining. The six features
remain slope, roughness, sky view, local relative height, curvature and local sky view. NAC and other sensors
remain viewing layers, not automatic predictors. See [training guidance](docs/TRAINING.md).

## Prepare or check data

To prepare tiles from authorized local ZIPs or check an existing installation:

```sh
uv run --no-project python data/prepare_dataset.py
uv run --no-project python data/prepare_dataset.py --verify
```

The default destination is `data/processed_data/`. Verification is read-only and
leaves unrelated registered sites alone. Optional offline import, manifest examples,
annotation precedence and hard-link precautions are in [data/README.md](data/README.md).
Neither preparation nor importing trains the active model or seeds output paintings.
To bootstrap bundled paintings separately, run this explicit command from the project root:

```sh
uv run --no-project python -B -c "from app import core; print(core.seed_paintings())"
```

It prints the newly seeded map IDs. Existing work is left untouched.

## Evaluation reports

Each selected-map evaluation saves `output/evaluations/evaluation-<run_id>.json`, containing
scores, coverage, exclusions, configuration and provenance. It creates no images,
ZIPs, arrays, per-run model archives or pointer files. Progress reads the latest
valid report and provides a JSON download.

Progress contains a single Evaluation workflow; it does not configure persistent test maps.

Maintainers: `uv run --no-project python -B -m unittest discover -s tests -v`.
The installed-data smoke check is read-only: `uv run --no-project python -B -m tests.smoke_app_layout`.
