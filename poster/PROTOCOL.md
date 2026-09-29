# Historical poster protocol

These method notes describe the previous research workflow. Old generated reports,
figures and model archives have been deleted. Current commands are in [README.md](README.md);
they use in-memory fold models and do not install or archive an app model.

**Frozen historical protocol**

1. Validate all 88 listed reference-artifact hashes and label-TIFF/painting equality,
   public metadata, used public terrain/mask rasters and available source archives.
2. Select **saved, otherwise draft**, exactly once per tile. Six tiles represent four
   map groups; versions are not extra training maps. The snapshots were unreviewed;
   later user-marked verification does not change that historical assessment.
3. Build the six ordered features with `core.build` on each whole **public SfS tile**,
   then nearest-transport them to the reference grids. Preserved reference DEMs supply
   geometry/validity, not assumed current public SfS. No cell-mean classifier is installed.
4. Require a known selected label, at least 90% reference terrain per original cell,
   and at least 90% coverage of that terrain by all six finite public features.
   Exclude known accepted flags from either snapshot. Missing flags mean unknown
   provenance, not independently reviewed labels. Historical snapshot weights are
   uniformly 1.0, not evidence of high label confidence.
5. Retain zero/missing SfS-support areas; do not filter tests by support or NAC.
   Report support counts and coverage. N1014's public terrain coverage was **54.3%**;
   no historical reference DEM was substituted to fill the gap.
6. Sample at most 2,000 pixels per class per tile, deterministic seed 0. Fit the app's
   200-tree RF defaults: **31,500 sampled pixels** retained. Expand coarse labels using
   floor-edged 120 × 120 cells, not as independent 5 m truth.
7. Fit three fresh models, holding out Mons Mouton, Nobile-1 and Nobile-2 in turn.
   **N1014 and all MS1 tiles stay together.** No held-out labels enter a fold's fit.
   Majority-vote valid pixel predictions within each eligible original cell (ties use
   the lowest class code); score each cell once. The historical majority baseline
   chooses its class from unsampled eligible *training* cells. This differs from the
   current generic evaluator's sampled-training-label baseline.
8. Historical runs fitted a separate final model using all retained reference samples after any configured
   holdout exclusions. Retain source region/footprint provenance for later exclusions.
   The installed starter is this all-reference fit, not one of the held-out models.

**Completed historical result:** 42,215 scored cells; pooled agreement **55.6%**,
pooled macro-F1 **0.517**, equal-geography mean macro-F1 **0.506**. The training-majority
baseline was **48.4% agreement / 0.218 macro-F1**. The report records class metrics,
confusion matrices, coverage, hashes, software versions and execution timings.
Those timings are not a controlled mapping-efficiency study.

The historical assets contain six held-out reference/prediction/cell-agreement/
confidence comparisons, a six-feature montage, footprints, scores PNG/SVG, CSV and
captions. PNGs have 300-DPI metadata and at least 2400 pixels on each axis. The app's
blue/yellow-green/red palette is unchanged; exports use a separately labeled
green/orange/navy palette. Figures and ZIPs belong under `poster/<runid>/`, separate
from flat JSON reports.
