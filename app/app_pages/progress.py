"""Evaluate painted maps independently without changing the active model."""
import importlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app import assistant_model, paths


def file_stamp(path):
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    except OSError:
        return None


def report_stamp(output_root):
    root = Path(output_root) / "evaluations"
    candidates = root.glob("evaluation-*.json")
    return tuple((str(path.relative_to(root)), file_stamp(path)) for path in sorted(candidates))


@st.cache_data(show_spinner=False, max_entries=4)
def completed_report(output_root, stamp):
    return importlib.import_module("app.evaluation").latest_report(output_root)


def percent(value):
    return "Not reported" if value is None else f"{value:.1%}"


st.subheader("Evaluation")
st.caption("Each map is evaluated separately. Training maps are left out for their turn; the active model stays unchanged.")
bundle = assistant_model.load()
samples = bundle.get("samples") or {}
evaluation = None
latest = None
try:
    evaluation = importlib.import_module("app.evaluation")
except ModuleNotFoundError as error:
    if error.name != "app.evaluation":
        raise
    st.caption("The evaluation engine is not available yet.")

maps, folds = {}, []
if evaluation is not None:
    maps = {name: meta for name, meta in evaluation.catalog().items()
            if meta.get("painted", bool(meta.get("label_source")))}
    try:
        latest = completed_report(str(paths.OUTPUT_DIR), report_stamp(paths.OUTPUT_DIR))
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"The latest evaluation report could not be read: {error}")
previous = latest[1].get("selected_maps", []) if latest else []
st.session_state.setdefault("evaluation_maps", [n for n in previous if n in maps])
st.session_state.evaluation_maps = [n for n in st.session_state.evaluation_maps if n in maps]


def map_label(name):
    status = "painted, verified" if maps[name].get("verified") else "painted"
    return f"{name} · {maps[name].get('section', 'Local')} — {status}"


names = st.multiselect(
    "Maps to evaluate", list(maps), key="evaluation_maps",
    format_func=map_label,
    help="Compare predictions with Sure hand-painted cells.",
    width="stretch")
if evaluation is not None and names:
    folds = evaluation.plan(bundle, names)
ready = [fold for fold in folds if not fold.get("reason")]
can_evaluate = bool(evaluation is not None and samples and names and set(names).issubset(maps) and ready)
st.caption(f"{len(names)} maps selected · {len(ready)} ready")
if not maps:
    st.info("Paint a map to make it available for evaluation.")
if not samples:
    st.info("No training samples are available. Train the app model from labeled maps before evaluating.")
elif names and not ready:
    st.info("No evaluation is ready. See the evaluation plan for details.")
if names:
    with st.expander("Evaluation plan", expanded=not bool(ready)):
        st.dataframe([{"Map": ", ".join(f.get("maps", [])),
                       "Training maps": len(f.get("train_keys", [])),
                       "Left out": ", ".join(f.get("maps", [])) if f.get("mode") == "leave-one-map-out" else "None",
                       "Status": f.get("reason") or "Ready"} for f in folds],
                     hide_index=True, width="stretch")
if st.button("Evaluate", icon=":material/fact_check:", type="primary", disabled=not can_evaluate,
             width="stretch") and can_evaluate and evaluation is not None:
    try:
        with st.spinner("Evaluating selected maps..."):
            message = st.empty()
            evaluation.run(bundle, names, progress=message.text)
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Evaluation could not complete: {error}")
    else:
        completed_report.clear()
        st.rerun()

if latest is None:
    st.caption("No completed evaluation report yet.")
else:
    report_path, report = latest
    validation = report.get("validation", {})
    aggregate = validation.get("aggregate") or {}
    baseline = validation.get("baseline_aggregate") or {}
    st.subheader("Latest evaluation")
    st.caption(f"Run: {report.get('run_id', 'not reported')} · {report.get('created_at', 'not reported')}. "
               "These saved scores describe the run's selection, not unsaved chooser changes.")
    with st.container(horizontal=True):
        st.metric("Evaluation accuracy", percent(aggregate.get("accuracy")), border=True)
        st.metric("Evaluation macro F1", percent(aggregate.get("macro_f1")), border=True)
        st.metric("Majority baseline accuracy", percent(baseline.get("accuracy")), border=True)
        st.metric("Scored cells", aggregate.get("n", 0), border=True)
    results = validation.get("folds", [])
    if results:
        st.dataframe([{"Map": ", ".join(f.get("maps", f.get("test_tiles", []))),
                       "Training maps": len(f.get("train_keys", f.get("train_tiles", []))),
                       "Scored cells": (f.get("metrics") or {}).get("n"),
                       "Accuracy": (f.get("metrics") or {}).get("accuracy"),
                       "Macro F1": (f.get("metrics") or {}).get("macro_f1"),
                       "Baseline accuracy": (f.get("baseline") or {}).get("accuracy"),
                       "Baseline macro F1": (f.get("baseline") or {}).get("macro_f1")}
                      for f in results], hide_index=True, width="stretch")
        bars = [{"Map": ", ".join(f.get("maps", f.get("test_tiles", []))) or f["group"],
                 "Method": method, "Macro F1": scores["macro_f1"]}
                for f in results for method, scores in (("Evaluation model", f.get("metrics") or {}),
                                                       ("Majority baseline", f.get("baseline") or {}))
                if scores.get("macro_f1") is not None]
        if bars:
            st.bar_chart(pd.DataFrame(bars), x="Map", y="Macro F1", color="Method",
                         stack=False, width="stretch")
    with st.expander("Scores and coverage"):
        st.caption("Reference painting cells are coarse labels, not independent 5 m truth. "
                   "Class 3 includes crater interiors / floor, not measured permanent shadow or ice.")
        scopes = {"All selected maps": aggregate, "Majority baseline": baseline}
        scopes.update({", ".join(f.get("maps", f.get("test_tiles", []))) or str(f["group"]):
                       f.get("metrics") or {} for f in results})
        scope = st.selectbox("Score detail", list(scopes), key="evaluation_score_detail", width="stretch")
        metrics = scopes[scope]
        if metrics.get("per_class"):
            st.dataframe([{"Class": c, **values} for c, values in metrics["per_class"].items()],
                         hide_index=True, width="stretch")
        if metrics.get("confusion_matrix") is not None:
            st.caption("Confusion matrix: reference classes in rows, predictions in columns (1, 2, 3).")
            matrix = metrics["confusion_matrix"]
            st.dataframe([{"Reference class": str(i + 1),
                           **{f"Predicted {j + 1}": int(value) for j, value in enumerate(row)}}
                          for i, row in enumerate(matrix)], hide_index=True, width="stretch")
        st.json({"selected_maps": report.get("selected_maps", []), "skipped": report.get("skipped", []),
                 "tiles": report.get("tiles", []), "policies": report.get("policies", {}),
                 "limitations": report.get("limitations", [])})
    if report.get("skipped"):
        st.caption(f"{len(report['skipped'])} skipped entries; reasons are listed under Scores and coverage.")

    st.download_button("Download evaluation JSON", json.dumps(report, indent=2, allow_nan=False) + "\n",
                       file_name=Path(report_path).name, mime="application/json",
                       icon=":material/download:", on_click="ignore", width="stretch")
