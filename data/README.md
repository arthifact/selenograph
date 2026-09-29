# Data workspace

Install datasets under **`data/processed_data/<site>/`**, one site folder per
Section. There are no public or reference wrapper directories. The app discovers
`working_dems.json` catalogs there; it does not auto-load originals from
`reference_data/` or require a private reference collection.

## Storage and safe sharing

| Folder | Responsibility |
|---|---|
| `raw_data/essentials/` | Original public regional ZIPs; preserve unchanged |
| `raw_data/future-experiments/` | Optional LOLA and Mini-RF inputs; not part of the default recipe |
| `reference_data/` | Preserved private originals and annotation snapshots; protect and back up |
| `processed_data/<site>/` | Installed observations, manifests and optional immutable bundled annotations |
| `../output/painting_drafts/` | Working paintings, flags and draft metadata overrides |
| `../output/paintings/` | Explicitly saved paintings, flags, metadata and gallery thumbnails |
| `../output/maps/` | GeoTIFF exports of painted labels and model predictions, replaced on Save |
| `../output/evaluations/evaluation-<run_id>.json` | One JSON per evaluation, with results and provenance |
| `../models/` | Active assistant |
| `../poster/` | Optional, removable poster scripts and their results/figures |

Obtain permission before sharing private data, derived models or exports. Copy the
whole approved processed tree with a **regular-file copy**, preserving paths and
metadata, so cross-site `feature_sources` remain available. This materializes hard
links as independent files; recipients do not need the original reference tree for
normal app use. Do not create new links back to the source workspace. Keep citations
and provenance, and exclude environments, caches and secrets as described in the
[project sharing guidance](../README.md#sharing-and-privacy). Nothing is uploaded.

**Hard-link safety:** the current four imported sections contain **30 raster hard
links**, each sharing its source file's inode. They avoid duplicate raster storage,
not enforce write protection. Writing through either pathname changes the same
file. Never edit, reproject, overwrite or change permissions on these rasters in
place. Source datasets remain immutable; app drafts and exports belong in the
mirrored per-map directories under `output/`, not in the processed tree. Model
updates belong in `models/`.

### Per-map output layout

`output/` contains **`evaluations/`** for JSON reports and two painting directories:
**`paintings/`** and **`painting_drafts/`**, plus **`maps/`** for saved GIS exports.
Inside each, mirror the DEM's exact parent directory relative to
`data/processed_data/`. Prefix every artifact filename with the map ID (the DEM
stem); do not add another map-ID folder. Separate roots prevent collisions with
site names, so there is no arbitrary reserved-site-name requirement. For example:

```text
data/processed_data/amundsen_rim/tiles/id/elevation/id.tif
output/painting_drafts/amundsen_rim/tiles/id/elevation/
  id_painting.npy
  id_accepted.npy
  id_confidence.npy
  id_meta.json
output/paintings/amundsen_rim/tiles/id/elevation/
  id_painting.npy
  id_accepted.npy
  id_confidence.npy
  id_meta.json
  id_thumb.png
output/maps/amundsen_rim/tiles/id/elevation/
  id_labels.tif
  id_map.tif
```

Both buckets use the same canonical painting, flag and metadata filenames. Flag
arrays are optional. Gallery thumbnails (`id_thumb.png`) stay in `paintings/`.
GeoTIFFs belong in `maps/`: `id_labels.tif` is the saved painting and `id_map.tif`
is the model prediction. Each save overwrites these filenames. Without a trained
model, Save writes only the label TIFF and removes any previous prediction at the
current export path. Both TIFFs preserve georeferencing, unit colours and NoData.
Previous export layouts remain readable until the map is saved in the new layout.
Reference maps use the same rule with their actual folder tree:
`data/processed_data/mons-mouton(reference)/tiles/id.tif` uses
`mons-mouton(reference)/tiles/` inside each bucket. No `elevation/` folder is inserted
when absent.

App changes—including strokes, undo, training corrections and verification toggles
in Paint or Gallery—write to `painting_drafts/`. Only **Save labels + map** in Paint
updates `paintings/` and `maps/`. Model updates are separate and do not commit paintings.
Draft metadata is an overlay on saved metadata: Save commits the effective metadata
and removes the draft metadata overlay. Explicit API calls retain saved-write
compatibility (`core.meta_set` defaults to `final=True`); UI staging must pass
`final=False`. `core.meta_get` defaults to effective metadata, while `final=True`
reads saved-only overrides over bundle defaults.

Observation DEMs, imagery, masks and companion rasters stay in `data/`; none are
copied or linked into output. Bundled annotation originals also remain immutable.
Directories are created on writes, never just by browsing. Canonical outputs are
tied to the DEM's relative location: when relocating a site, move the matching
subtrees in the painting, draft and map buckets to the same new relative location.

**Explicit one-time bootstrap:** `core.seed_paintings()` is a separate operation,
not automatically run by dataset preparation, the importer or runtime browsing.
From the project root:

```sh
uv run --no-project python -B -c "from app import core; print(core.seed_paintings())"
```

This generic helper resolves nonempty bundled paintings through the installed
catalog, including the six supplied paintings; it is not reference-specific.
It creates independent painting arrays and provided acceptance/confidence flags in
both buckets using the normal save rules (default-only flag sidecars may be omitted).
It copies **no label TIFFs or metadata**: verification and provenance remain inherited
from the bundled manifest. Labels-only bundles are not seeded. The command prints
the newly seeded map IDs and neither trains a model nor manufactures verification.

If **any recognized canonical or legacy artifact** exists for a map—including an
empty painting, a flag sidecar, an export or metadata—the helper skips that entire
map, even if the existing files match the bundle. It does not overwrite existing
work, compare it for equality or fill a partially seeded saved/draft pair.

Old evaluation artifacts have been cleared. Each new evaluation saves one JSON
at `output/evaluations/evaluation-<run_id>.json`, without images, arrays, model
archives or pointer files. `output/test_maps.json` remains a root-level selection
file. The active model stays in `models/`; optional poster scripts and generated
figures stay in the independently removable `poster/` directory.

## Installed layout

```text
processed_data/
  manifest.json                         public preparation inventory only
  amundsen_rim/working_dems.json
  connecting_ridge/working_dems.json
  ...                                   13 public site folders in total
  mons-mouton(reference)/working_dems.json
  nobile1(reference)/working_dems.json
  nobile2(reference)/working_dems.json
  nobile1-ms1(reference)/working_dems.json
```

The public collection has **237 tiles across 13 sites**. The four additional
sections contain six tiles: **243 tiles in 17 sections** overall. These literal
folder names appear in **Section**. Select a section to browse all its tiles, or
**All** to browse every available tile; there is no separate Location dropdown.
Geographic metadata still defines evaluation groups. `collection` is provenance,
not a wrapper or navigation page.

Public site folders contain:

```text
<site>/
  working_dems.json
  metadata/
    <site>_metadata.json                 source grids, hashes and tile windows
    <site>_quality_report.json           caveats and layer availability
    <site>_readme.txt                    original regional documentation
  tiles/
    zenodo-<site>-r000-c000/
      elevation/zenodo-<site>-r000-c000.tif
      imagery/zenodo-<site>-r000-c000_nac.tif
      quality/zenodo-<site>-r000-c000_<layer>.tif
```

Tile row/column IDs are zero-based. **The DEM filename stem is the stable map ID**,
shared by annotations and model sample memory; it must be unique across every
catalog. Keep `tile_id` equal to that stem and do not rename IDs after annotation.
Edge tiles keep their actual dimensions, without padding or discarded slivers.

## Prepare or verify public observations

From the project root, using local source ZIPs:

```sh
uv run --no-project python data/prepare_dataset.py
uv run --no-project python data/prepare_dataset.py --verify
```

The default output is **`data/processed_data/`**. Preparation is offline and supported
on macOS/Linux. It selectively extracts regional products, aligns them to the native
SfS grid with nearest-neighbor resampling, creates support flags, cuts tiles, validates
them and publishes each finished site atomically. Completed sites are validated and
reused, not overwritten. Temporary full-region rasters are removed; individual source
images are not all expanded. Hidden staging folders are ignored by the app.

The root `manifest.json` owns only the public preparation selection. **`--verify`
is read-only**: it verifies that collection and its source archives, leaving unrelated
registered sites out of public-content verification/rebuilding. It still checks their
catalog paths and map-ID uniqueness; it does not certify their scientific labels.
Keep manifests, hashes and quality metadata. Do not add arbitrary files within a
strictly verified public site; add another properly registered site instead.
Neither command generates geological labels or trains a model.

### Layer meanings

| Name | Meaning |
|---|---|
| `sfs` / `elevation/` | SfS elevation on its native projected grid |
| `nac` | Maximum-illumination NAC orthomosaic, aligned to SfS |
| `image_count` | Illuminated-image contribution count |
| `solar_bins` | Solar illumination-bin count |
| `best_resolution` | Best input-image resolution |
| `uncertainty` | Published uncertainty product; consult the source README naming caveat |
| `valid_data` | 1 where both NAC and SfS are finite and unmasked; otherwise 0 |
| `sfs_support` | 1 where SfS is valid and image contribution count is positive; otherwise 0 |

The final two masks are **not geological labels**. Public prepared layers are
compressed float32 GeoTIFFs. Observation nodata is NaN; binary masks retain valid
zeroes. Scale/offset metadata is preserved, not applied. Compression is lossless,
although converting float64 to float32 may round samples. Resampling does not make
15 m quality products genuinely 5 m observations or certify co-registration.
Finite elevation alone is not proof of illuminated-image support. Quality layers
are preserved but do not automatically become training predictors or weights.

## Generic catalog contract

A `working_dems.json` is a JSON list of records. It registers DEMs explicitly, so
arbitrary nested imagery or quality TIFFs are not mistaken for maps. All declared
file paths are relative to that catalog, must resolve inside `processed_data/`, and
must not resolve into hidden staging folders. Sibling-site links are supported.

This illustrative `data/processed_data/survey-a/working_dems.json` uses supported
fields. Replace example paths with installed files; omit optional annotations or
feature sources when they are not available. Set `verified: true` only by an
explicit user decision, not because a folder is called a reference:

```json
[
  {
    "working_dem": "tiles/survey-a-0001/elevation/survey-a-0001.tif",
    "tile_id": "survey-a-0001",
    "section": "survey-a",
    "collection": "local-survey",
    "site": "survey-a",
    "group": "location-a",
    "geography_aliases": ["location-a-revision-1"],
    "restricted": true,
    "layers": {
      "nac": "tiles/survey-a-0001/imagery/survey-a-0001_nac.tif"
    },
    "annotations": {
      "painting": "tiles/survey-a-0001/annotations/painting.npy",
      "confidence": "tiles/survey-a-0001/annotations/confidence.npy",
      "accepted": "tiles/survey-a-0001/annotations/accepted.npy",
      "labels": "tiles/survey-a-0001/annotations/labels.tif",
      "verified": true,
      "label_status": "User-marked verified; not independent certification",
      "provenance": {
        "snapshot": "saved",
        "verification": {
          "basis": "user_requested",
          "independent_certification": false
        }
      }
    },
    "feature_sources": [
      "../context-a/tiles/context-a-0001/elevation/context-a-0001.tif"
    ]
  }
]
```

- `working_dem` supplies the DEM/grid; `layers` supplies optional named viewing
  layers. Layer paths can be strings or objects with a `path` field.
- `section` defaults to the first folder below `processed_data/`; for this layout,
  omit it or match the literal site folder. `site` and `group` describe location,
  not UI sections. Public preparation also records `size`, `transform` and counts.
- `group` should identify the **actual geography** shared by overlapping products
  or revisions. `geography_aliases` may list equivalent names or map aliases to
  group names. Evaluation also coalesces geographic provenance in retained samples;
  it does not hardcode particular dataset names into the generic evaluator.
- `annotations.painting` is a 120 × 120 NPY array: 0 unknown/unpainted, 1 smooth
  highlands, 2 rough highlands, 3 `shadowed_floor`. `confidence` uses 1 Sure, 2 Mostly,
  3 Unsure (0 unpainted); `accepted` is a same-shaped boolean prediction-acceptance
  mask. `labels` is an optional label TIFF for viewing/export, not a replacement for
  usable cell paintings in the generic selected-map evaluation.
- `verified`, `label_status`, `restricted` and `provenance` are annotation metadata
  defaults (also supported at record level). Saved metadata at
  `output/paintings/<DEM parent path>/<id>_meta.json` overrides defaults; draft
  metadata at `output/painting_drafts/<DEM parent path>/<id>_meta.json` overrides
  that effective view, including `verified: false`.
  Verification never grants sharing permission or establishes independent
  scientific truth.

### Bundled labels and output precedence

Paint, Gallery and generic evaluation use the same loader. Effective paintings
resolve **user draft → user save → bundled painting**. Even an empty user painting
can override the bundle. Confidence and acceptance flags come from that same source,
not a lower-priority snapshot. Missing confidence defaults to Sure for painted cells;
missing acceptance defaults to false. These operational defaults do **not** establish
historical confidence or label independence; preserve that uncertainty in provenance.

Browsing bundled labels creates no output copies and trains nothing. The explicit
bootstrap above seeds both buckets; subsequent app changes go only to drafts until
Save in Paint commits them. Evaluation uses effective draft/saved paintings and
can score unverified maps. Verification controls which remembered map samples train
the temporary evaluation models; it is not required to compare a painting.
A bundled `verified: true` is simply the initial user-trust flag. All writes use the
[mirrored output layout](#per-map-output-layout), never the immutable bundle.

### Optional whole-input feature context

Without `feature_sources`, the six terrain features are computed on the map's own
DEM. With it, the generic loader computes those same features on each declared
**whole source DEM**, then nearest-transports them to the selected map's grid.
Display terrain, other viewing layers and output geometry stay local. This is a
generic option for linked context, not a private-dataset requirement or an extra
model feature. No duplicate raster or persistent feature cache is needed.

Keep every linked source inside the processed tree. Dependencies must be acyclic;
overlapping source ownership is rejected. Different grids need valid CRSs, and
linked source builds must preserve their native grids. Uncovered terrain remains
missing rather than being filled from an unrelated DEM. Whole-input context still
has source-tile boundaries; document the context policy when comparing results.

## Optional offline import of preserved maps

For this workspace's authorized preserved collection only, run explicitly from
the project root:

```sh
# Validate sources and planned/existing sections without writing:
uv run --no-project python -B -m app.prepare_reference_section --dry-run
# Publish missing per-site sections:
uv run --no-project python -B -m app.prepare_reference_section
```

The importer creates `mons-mouton(reference)`, `nobile1(reference)`,
`nobile2(reference)` and `nobile1-ms1(reference)` directly below `processed_data/`.
Original datasets stay in `reference_data/`. It is offline, creates hard links
rather than duplicate rasters, and **does not train or replace the active model**.
Once published, these are ordinary processed maps with bundled annotations; the
app never invokes this importer or follows provenance back to live originals.

The default import marks labels verified **at the user's explicit request**.
`label_status` is **“User-marked verified; preserved unreviewed historical snapshot;
not independent certification”**. `--no-verified` opts out for a new import; do not
use it to rewrite an existing immutable bundle. The importer does not seed output
paintings; run the separate `core.seed_paintings()` command above if needed.
Later app overrides go to drafts,
and only Save in Paint commits them.
One snapshot is selected per tile, saved if present otherwise draft; observed
acceptance flags are unioned across snapshots. Missing confidence stays recorded
as unknown, not synthesized evidence of certainty.

Reruns are idempotent: matching destinations are validated and reused; differing
nonempty destinations are refused, not repaired in place. Sources and destinations
are validated before publication, each site is staged then renamed atomically, and
an interrupted batch can resume. Hard links require source and destination on the
same filesystem; there is no large-raster-copy fallback.

N1014 and MS1 share the Nobile-1 geography despite different section folders.
Their manifests' `group`/`geography_aliases` and sample provenance keep them together:
the six imported tiles represent **three geographic folds**, not four independent sites.

## Add other datasets

The app works without these preserved maps. Install another site with a valid
`working_dems.json`, DEM and optional display layers/annotations using the same
contract. For supported regional A3CLR22 ZIPs, add archives to `raw_data/essentials/`,
keep matching publisher metadata in `raw_data/zenodo-record-17954508.json`, and rerun
preparation. Other product formats need an explicit adapter and validation policy.
New geological classes require a reviewed schema and model version, not just a new
site folder. See [training and evaluation](../docs/TRAINING.md) and
[source pages and citations](sources.md).
