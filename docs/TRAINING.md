# Painting, training and evaluating geological units

## Use any installed processed dataset

The app has only **Paint**, **Gallery** and **Progress**. Install maps under
`data/processed_data/<site>/` with a `working_dems.json`; **Section** lists literal
site folders. Choose a section and then a tile, or **All** for every available tile;
there is no separate Location dropdown. Bundled labels and user paintings use the
same generic loader. Private reference maps are optional,
not a product requirement, and originals in `data/reference_data/` are never
auto-loaded by the app. See the [catalog contract](DATASET.md#generic-catalog-contract)
for annotation, geographic grouping and optional whole-input feature-context fields.

Keep source datasets immutable. Painting drafts, saved labels, verification overrides
and predictions go only to `output/`; models go to `models/`. Archived imported rasters may share hard links with their originals, so editing
either pathname in place would alter the source. Do not edit, reproject or change their permissions.
For authorized sharing, materialize the whole processed tree as regular copies,
keeping linked feature inputs and provenance together. Private sources and derived
models/figures still require permission. These workflows upload nothing.

`output/` has **`evaluations/`** for JSON reports, **`painting_drafts/`** for working
changes, **`paintings/`** for committed paintings and **`maps/`** for GeoTIFF exports.
The painting and map directories mirror the DEM's exact parent folder
relative to `data/processed_data/`, with map-ID-prefixed filenames. For example,
`data/processed_data/amundsen_rim/tiles/id/elevation/id.tif` has its working painting at
`output/painting_drafts/amundsen_rim/tiles/id/elevation/id_painting.npy` and its saved
painting at `output/paintings/amundsen_rim/tiles/id/elevation/id_painting.npy`.
Both use `id_painting.npy`, `id_accepted.npy`, `id_confidence.npy` and `id_meta.json`.
Gallery thumbnails (`id_thumb.png`) stay with saved paintings. Each **Save labels + map**
replaces `id_labels.tif` (painted labels) and `id_map.tif` (model prediction) under
`output/maps/<DEM parent>/`. Without a model, only labels are exported and the previous
prediction at that path is removed. The GeoTIFFs retain georeferencing and NoData.
There is no extra map folder. Reference maps follow the identical rule: a DEM at
`data/processed_data/survey-a/tiles/id.tif` uses
`survey-a/tiles/` inside each bucket, without inserting `elevation/`.

Changes in Paint or Gallery—including verification—persist as drafts. Only clicking
**Save labels + map** in Paint updates saved paintings and exports. Save also commits
effective metadata (saved metadata plus draft overrides), then removes the draft
metadata overlay. **Update model** rebuilds from all currently verified paintings and refreshes the
current prediction. It saves only the model, without evaluations or painting exports.
Separate output roots remove the old arbitrary reserved-site-name requirement.
When relocating a site, move its matching painting, draft and map subtrees too.

For an unmerged installation, `core.seed_paintings()` can explicitly bootstrap
a catalog's nonempty bundled paintings. The current public paintings already exist
in output. Run the helper separately using the
[bootstrap command](DATASET.md#per-map-output-layout); dataset preparation,
importing and browsing never invoke it. It creates independent painting/flag arrays
in **both** buckets, with no label TIFF or metadata copy. Metadata, including
verification and provenance, remains inherited from the bundled manifest.
Any recognized existing canonical or legacy work—including metadata or matching
files—causes the entire map to be skipped; partially seeded pairs are not filled.
Browsing creates no directories and trains no models. Observation rasters stay in
`data/` and are never copied or linked into output.

Evaluation JSONs live under `output/evaluations/`. Storage cleanup does not
retrain the installed model or change paintings.

## Start with reviewed unit definitions

The app currently uses three classes: smooth highlands, rough highlands and
`shadowed_floor`. The last name includes some crater interiors in the historical
reference conversion; it does not prove permanent shadow or ice. Agree on unit
definitions and the source-unit mapping before extensive labeling. Unpainted and
unmapped areas are unknown, not a background class.

NAC imagery and terrain backgrounds help visual interpretation. The painter has
120 × 120 cells: on a full 1024-square tile at 5 m spacing, each cell spans about
42.7 m. Smaller edge tiles have different cell sizes. Painted boundaries are not
native 5 m polygon boundaries or new subpixel measurements.

Gallery and Paint display a full square tile with missing data in black. Cropped
edge and corner tiles retain the nominal size and placement of full neighbouring
tiles. Raster NoData holes remain in their original positions. Display padding
never changes raster coordinates, stored painting cells, or training samples;
painting ignores padding and cells without terrain.

**Verified is a user-trust flag, not independent scientific certification.** Only
mark a map verified deliberately, preserving its actual review history in
`label_status` and `provenance`. The source paintings were explicitly marked verified at the user's request.
Their original review metadata remains in the migration provenance and archive.
Transferring paintings onto public tiles does not independently validate them.

## Train deliberately

This workspace uses one public catalog: **237 native 5 m SfS tiles in 13
sections**, with paintings and verification on **13 tiles**. The active model was
rebuilt from those merged paintings. The six original painting grids and duplicate
map sections are archived outside the active catalog. Normal public tile names are
used everywhere; source-map identities remain metadata only.

- **Sure**, **Mostly** and **Unsure** paintings have different training weights.
- Accepted model suggestions remain marked as accepted predictions and have reduced
  weight. They must not become independent human answer keys.
- Strokes, undo, confidence/acceptance changes and verification toggles autosave to
  drafts. **Save labels + map** commits them and saves human labels separately from
  predictions; saving does not require a prediction. **Update model** rereads all
  currently verified paintings, including draft edits made on other maps, and trains
  a fresh model on every click. It refreshes the current prediction and saves only
  the model, without committing paintings or producing evaluation files.
- Unverified, removed and cleared maps are excluded from rebuilt samples;
  previous sample arrays are never reused by this action. A fit needs at least two
  labeled classes. Browsing, changing a background and painting alone do not train.
- Effective paintings resolve **draft → user save → bundled painting**, with flags
  from the same source. User edits and metadata override the immutable bundle.
  Missing confidence defaults to Sure and missing acceptance to false in the loader;
  neither default proves historical certainty or independence.
- **Restore saved** replaces the draft painting, confidence and approval flags with
  the saved version (or bundled painting when no user save exists). Undo recovers
  the previous draft. Restoring does not save exports or retrain the model.

Random forest is the default; LightGBM is also available. Both use the six-feature
**terrain baseline (recipe version 6)**: slope, roughness, sky view, local relative
height, curvature and local sky view. These six ordered columns are used for
training, prediction and model confidence.

By default, features come from the selected DEM. Optional manifest `feature_sources`
link whole input DEMs: adjoining, aligned source DEMs are joined before the six
features are calculated, removing artificial edges at tile joins. Features are
then nearest-transported to the selected map's grid. Display geometry remains local;
missing feature coverage remains missing. This supports consistent source context
without duplicate rasters or a dataset-specific evaluator. It does not add predictors.

All installed maps use their native **5 m public SfS inputs**. There is no
per-map alternative terrain source. Imported paintings were geographically merged
onto these public grids; areas beyond public SfS coverage have no destination.
A destination painting cell needs at least 50% coverage and a strict majority label.
Certainty and approval flags are combined conservatively, with originals preserved.

**NAC is a viewing layer, not a trained feature.** PSR, radar, separate LOLA,
quality channels, scene-relative height and hillshade are excluded from the default
training recipe even when available. Quality layers remain preserved for review;
no automatic support-based exclusion or weighting is applied.

Version 5 models need an explicit **Update model** to rebuild their samples and
predictions using the corrected terrain context. Browsing leaves the saved artifact
untouched and does not use stale predictions.

Models from the previous eight-column recipe are incompatible and are rejected,
not silently reused or deleted. Archive such models outside `models/` and explicitly
retrain from reviewed paintings. Saved map metadata records the model schema
separately from display inputs. Changing inputs or classes requires an explicitly
versioned recipe and retraining.

## Evaluate selected maps without changing the active model

In **Progress → Evaluation**, only painted maps appear. Select one or several:
each selected map has its own result. Verification controls training membership,
not evaluation eligibility. The RF and GBM pixel backends are supported.

1. **Read the current painting.** Drafts take precedence over saved/bundled labels.
   Scoring uses Sure hand-painted cells, excluding approved predictions, unknown
   labels and zero-pixel cells. At least 90% of a cell must have terrain, and at least
   90% of that terrain must have finite six-feature and known-label coverage.
2. **Rotate one map at a time.** Start each turn from all currently verified samples
   in the active model's training memory. If the selected map is in that set, remove
   only its samples for this turn, then restore them for the next map's evaluation.
   Maps in the same section, location or neighbouring footprint remain in training.
   An unverified painted map is scored using the full verified training set.
3. **Fit only in memory.** Each evaluation fits its own model without changing the
   active estimator, saved samples, paintings or verification. Use **Update model**
   first to refresh training memory with new verified paintings or edits. A remaining
   single-class training set predicts that class everywhere and is recorded as such.
4. **Handle unavailable selections.** If leaving out a map leaves no usable verified
   training samples, that map cannot be evaluated. Other selected maps can still score.
   Missing usable labels/terrain or a backend failure is reported explicitly. There
   is no fallback to scoring a model on its own training map.
5. **Score cells once.** Majority-vote pixel predictions within each eligible painting
   cell; ties choose the lowest class code. Report per-map and pooled scores, an
   equal-map mean, confusion matrices, coverage and a majority-training-class baseline.

This is leave-one-map-out evaluation, not geographic holdout: neighbouring and
overlapping maps may share terrain context. The scores measure agreement with
paintings, not generalisation to untouched locations or independently certified truth.

### Evaluation reports

Each completed run saves exactly one `output/evaluations/evaluation-<run_id>.json`
containing results, the per-map training selections, coverage, skipped-map reasons,
software versions and provenance. Models and predictions stay in memory. No images,
model archives, arrays, per-run directories or latest-pointer files are written.

**Progress** displays the latest completed report and offers a JSON download.
Browsing and planning are read-only. There is no separate Independent checks section
or persistent test-map chooser. Evaluation never changes the active training set.

## Which layers should become training inputs?

**Do not automatically train on every available layer.** Storage, viewing and
training serve different purposes. All available backgrounds are shown in the
layer selector. Changing the background changes only the display, not the feature recipe.

| Layer group | Recommended role |
|---|---|
| SfS-derived terrain features | Keep as the interpretable baseline; compare smaller feature subsets on held-out regions |
| NAC imagery | Test texture/appearance features as an additional, versioned model recipe; currently display-only |
| Image counts, solar bins, resolution, uncertainty and support masks | Use a documented quality-screening/weighting policy and stratified evaluation before considering them as predictors |
| Separate LOLA and Mini-RF | Optional controlled comparisons after the baseline; do not assume resampling provides fine-scale detail |
| Reviewed reference maps/paintings | Target labels or independent evaluation references, never predictor channels |

A minimal experiment sequence is **terrain only → terrain + NAC → optional radar**,
using identical geographically separated splits and reporting per-site/per-class
results. NAC illumination and mosaic seams can become shortcuts; quality channels
can likewise reveal acquisition coverage instead of geology. Add predictors only
when they improve independent evaluation, not merely training accuracy.

The current app checks finite terrain and label confidence; public quality maps are
preserved but **not wired into automatic training screening/weights**. Choose and
validate that policy explicitly rather than inventing universal quality thresholds.
Hillshade is a visualization, not an additional independent sensor.

## Before reporting research results

1. Group overlapping tiles, products and revisions of the same physical site together.
   Folder separation alone does not prevent leakage; check manifest/sample geography.
2. Freeze reviewed label versions and document source, reviewer, class schema, grid,
   quality policy and hashes. Distinguish user verification from independent review.
3. Keep model-selection regions separate from a final evaluation region. Do not train
   on corrections from that final region before scoring the initial predictions.
4. Report per-site/per-unit overlap, precision/recall, confusion matrices, the majority
   baseline, scored coverage and all skipped maps/folds. State which workflow produced
   the numbers; historical, generic and persistent-check results are not interchangeable.
5. Record model recipe, sampling policy, parameters, software versions and label versions.
   Interactive updates alone are not a frozen reproducible study.
6. Measure mapping effort as well as predictive quality: the goal is useful human
   assistance, not merely a higher aggregate score.

## Data and interpretation limits

Public tiles include validity/support masks and source quality products. Finite
SfS elevation alone does not prove illuminated-image support. Do not convert nodata
or a zero support flag into a geological class. Review the uncertainty product's
source documentation before quantitative use. In generic evaluation, unobserved
support is not evidence of measured lack of support and is not an eligibility filter.

SfS uses NAC/LOLA constraints; the layers are not independent accuracy references.
Resampling does not increase native detail. Terrain features use each local DEM or
its declared whole-input sources; limited surrounding context can affect boundaries
and sky-view estimates. Define a consistent context policy before claiming fine-scale
generalization, and report missing linked-feature coverage rather than filling it.

Keep original datasets and processed bundles unchanged. Back up new `output/`
and `models/` work separately, and review permissions before sharing any of it.
