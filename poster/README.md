# Optional poster tools

This folder owns the poster scripts, tests and generated assets. The app and its
tests never import `poster`; you can delete the whole folder without breaking
painting, training or JSON evaluations. These tools reuse app code and input data
read-only. They never install or modify an app model or write to `output/`.

Run from the project root using the existing Python environment:

```sh
# Three geographic folds over the preserved reference labels; results and figures:
SELENOGRAPH_MODEL=rf .venv/bin/python -B -m poster.reference_training

# Optional feature comparisons; choose a fresh directory for each experiment:
.venv/bin/python -B -m poster.benchmark_features --max-seconds 1800 --output poster/experiments/feature-selection

# Synthetic poster-only tests:
.venv/bin/python -B -m unittest discover -s poster/tests -t .
```

`reference_training` writes `poster/results/<run_id>/results.json`, `summary.md`,
`figures/` (PNG, SVG, tables, captions and embedded report) and `poster-assets.zip`.
Models and prediction arrays stay in memory. Results include source provenance;
no model installer or latest-pointer file is used. `--output` may select another
results directory inside `poster/`; paths that escape this folder are rejected.

`benchmark_features` writes `results.json` and `summary.md` inside the chosen
`poster/experiments/` directory. No experiment changes the app feature recipe.

The supporting scripts are `benchmark_data.py` (read-only extraction),
`reference_figures.py` (figure rendering), `pictures.py` (legacy panel rendering),
and `legacy_evaluation.py` (older rotation, with artifacts under `poster/legacy/`).
Shared raster validation and scoring remain in `app/dataset_utils.py` and
`app/evaluation_metrics.py` because they are also required by the application.

These are development results from coarse, previously used labels, not an untouched
final test or independent 5 m geological truth. Input sharing restrictions apply
to derived results and figures. See [PROTOCOL.md](PROTOCOL.md) for historical method notes.

Reference extraction uses the app's corrected terrain context (recipe version 6):
adjoining source tiles are joined before neighbourhood features are computed.
Historical version 5 scores in `PROTOCOL.md` must be regenerated for current figures.
