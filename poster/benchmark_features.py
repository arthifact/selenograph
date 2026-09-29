"""Private, reproducible feature ablations; never changes the interactive assistant.

Run: python -m poster.benchmark_features --help
Models live only in memory. A new results directory receives results.json and a
human-readable summary.md. These are development results derived from restricted
references, not an independent final test or an automatically deployed model.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from app import paths
from poster import benchmark_data, storage

CLASSES = (1, 2, 3)
LOCAL = ("slope", "rough", "rel_local", "curv")
FULL = LOCAL + ("svf", "svf_local")
NAC = tuple(benchmark_data.IMAGE)
RADAR = tuple(benchmark_data.RADAR)
QUALITY = tuple(benchmark_data.QUALITY)
LOLA = tuple(benchmark_data.LOLA)


@dataclass(frozen=True)
class Arm:
    name: str
    features: tuple[str, ...]
    screen_training_support: bool = False


ARMS = (
    Arm("majority", ()),
    Arm("terrain_minimal", ("slope", "rough")),
    Arm("terrain_local", LOCAL),
    Arm("terrain_local_global_sky", LOCAL + ("svf",)),
    Arm("terrain_local_local_sky", LOCAL + ("svf_local",)),
    Arm("terrain_full", FULL),
    Arm("terrain_scene", FULL + ("rel", "zscene")),
    Arm("nac_only", NAC),
    Arm("terrain_local_nac", LOCAL + NAC),
    Arm("terrain_full_nac", FULL + NAC),
    Arm("terrain_full_radar", FULL + RADAR),
    Arm("terrain_full_nac_radar", FULL + NAC + RADAR),
    Arm("terrain_full_nac_quality", FULL + NAC + QUALITY),
    Arm("terrain_full_nac_screened", FULL + NAC, True),
    Arm("lola_local", LOLA),
    Arm("terrain_full_nac_lola", FULL + NAC + LOLA),
    Arm("all_available", tuple(benchmark_data.FEATURE_NAMES)),
)


def validate_data(data):
    X = np.asarray(data["X"])
    n = len(data["sample_ids"])
    if X.ndim != 2 or X.shape != (n, len(data["feature_names"])) or n == 0:
        raise ValueError("Expected a nonempty cell-by-feature matrix")
    if len(set(data["feature_names"])) != X.shape[1] or np.isinf(X).any():
        raise ValueError("Duplicate feature names or infinite values")
    if len(set(data["sample_ids"].tolist())) != n:
        raise ValueError("Repeated original cells would inflate the evaluation")
    for key in ("groups", "y_saved", "y_draft", "support_fraction"):
        if np.asarray(data[key]).shape != (n,):
            raise ValueError(f"Unpaired sample rows: {key}")
    if len(np.unique(data["groups"])) < 3:
        raise ValueError("At least three geographic groups are required for this protocol")
    for key in ("y_saved", "y_draft"):
        if not np.isin(data[key], CLASSES).all():
            raise ValueError(f"Unknown/nodata labels must be excluded before fitting: {key}")
    support = data["support_fraction"]
    if not np.isfinite(support).all() or np.any((support < 0) | (support > 1)):
        raise ValueError("Support fraction must be finite and in [0, 1]")


def row_digest(sample_ids):
    return hashlib.sha256("\n".join(map(str, sample_ids)).encode()).hexdigest()


def training_rows(y, groups, eligible, seed, cap):
    """One deterministic class/site-balanced draw, independent of selected features."""
    rng = np.random.default_rng(seed)
    picked = []
    for group in sorted(set(groups[eligible].tolist())):
        for code in CLASSES:
            pool = np.flatnonzero(eligible & (groups == group) & (y == code))
            if len(pool):
                picked.extend(rng.choice(pool, min(cap, len(pool)), replace=False).tolist())
    return np.asarray(sorted(picked), dtype=np.int64)


def metrics(y, pred, probability=None):
    if len(y) == 0:
        return None
    if not np.isin(y, CLASSES).all() or not np.isin(pred, CLASSES).all():
        raise ValueError("Scoring requires class IDs 1, 2, 3")
    cm = confusion_matrix(y, pred, labels=list(CLASSES))
    tp = np.diag(cm).astype(float)
    actual, guessed = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, guessed, out=np.zeros(3), where=guessed > 0)
    recall = np.divide(tp, actual, out=np.zeros(3), where=actual > 0)
    f1 = np.divide(2 * tp, actual + guessed, out=np.zeros(3), where=actual + guessed > 0)
    union = actual + guessed - tp
    iou = np.divide(tp, union, out=np.zeros(3), where=union > 0)
    result = {
        "cells": len(y), "accuracy": float(tp.sum() / len(y)),
        "macro_f1": float(f1.mean()), "mean_iou": float(iou.mean()),
        "balanced_accuracy": float(recall[actual > 0].mean()), "confusion_matrix": cm.tolist(),
        "per_class": {str(c): {"support": int(actual[i]), "precision": float(precision[i]),
                              "recall": float(recall[i]), "f1": float(f1[i]), "iou": float(iou[i])}
                      for i, c in enumerate(CLASSES)},
        "absent_target_classes": [c for i, c in enumerate(CLASSES) if actual[i] == 0],
    }
    if probability is not None:
        truth = np.eye(3)[np.asarray(y, dtype=int) - 1]
        result["brier_multiclass"] = float(np.mean(np.sum((probability - truth) ** 2, axis=1)))
    return result


def fit_fold(data, arm, snapshot, held_out, seed, cap, trees):
    y, groups = data[f"y_{snapshot}"], data["groups"]
    test = np.flatnonzero(groups == held_out)
    eligible = groups != held_out
    available_train = int(eligible.sum())
    if arm.screen_training_support:
        eligible &= data["support_fraction"] >= benchmark_data.MIN_COVERAGE
    train = (np.flatnonzero(eligible) if arm.name == "majority" else
             training_rows(y, groups, eligible, seed, cap))
    if not len(train) or len(np.unique(y[train])) < 2:
        raise ValueError(f"Insufficient training classes for {arm.name}, held out {held_out}")
    if np.intersect1d(train, test).size or held_out in set(groups[train].tolist()):
        raise AssertionError("Geographic training/test leakage")
    columns = [data["feature_names"].index(f) for f in arm.features]
    start = time.perf_counter()
    if arm.name == "majority":
        # Natural training prevalence, before balanced model-sample draws.
        counts = np.array([int((y[eligible] == code).sum()) for code in CLASSES])
        code = CLASSES[int(np.argmax(counts))]
        fitted = time.perf_counter()
        pred = np.full(len(test), code, dtype=np.uint8)
        probability = np.zeros((len(test), 3))
        probability[:, CLASSES.index(code)] = 1
    else:
        model = RandomForestClassifier(n_estimators=trees, min_samples_leaf=5,
                                       class_weight="balanced", n_jobs=1, random_state=seed)
        # Native missing-value handling: never drop low-support test cells because
        # best_resolution is NaN, and never fit an imputer on held-out observations.
        model.fit(data["X"][np.ix_(train, columns)], y[train])
        fitted = time.perf_counter()
        rows = data["X"][np.ix_(test, columns)]
        pred = model.predict(rows)
        p = cast(np.ndarray, model.predict_proba(rows))
        probability = np.zeros((len(test), 3))
        for i, code in enumerate(model.classes_):
            probability[:, CLASSES.index(int(code))] = p[:, i]
    finished = time.perf_counter()
    supported = data["support_fraction"][test] >= benchmark_data.MIN_COVERAGE
    return {
        "arm": arm.name, "snapshot": snapshot, "held_out": held_out, "seed": seed,
        "features": list(arm.features), "train_groups": sorted(set(groups[train].tolist())),
        "training_cells_available": available_train, "training_cells_after_screen": int(eligible.sum()),
        "training_cells_sampled": len(train),
        "training_class_counts": {str(c): int((y[train] == c).sum()) for c in CLASSES},
        "train_rows_sha256": row_digest(data["sample_ids"][train]),
        "test_rows_sha256": row_digest(data["sample_ids"][test]),
        "test_cells": len(test), "test_supported_cells": int(supported.sum()),
        "fit_seconds": fitted - start, "predict_seconds": finished - fitted,
        "metrics": metrics(y[test], pred, probability),
        "supported_metrics": metrics(y[test][supported], pred[supported], probability[supported]),
        "low_support_metrics": metrics(y[test][~supported], pred[~supported], probability[~supported]),
    }


def summarize(runs, snapshots, groups, seeds, arms=ARMS):
    summaries = {}
    expected = len(groups) * len(seeds)
    for snapshot in snapshots:
        by_arm = {}
        for arm in arms:
            rows = [r for r in runs if r["arm"] == arm.name and r["snapshot"] == snapshot]
            if len(rows) != expected:
                continue  # Partial experiments never enter the ranking.
            by_group = {}
            for group in groups:
                subset = [r for r in rows if r["held_out"] == group]
                by_group[group] = {
                    name: float(np.mean([r["metrics"][name] for r in subset]))
                    for name in ("macro_f1", "mean_iou", "accuracy", "balanced_accuracy", "brier_multiclass")
                }
                by_group[group]["cells"] = subset[0]["test_cells"]
                by_group[group]["seed_macro_f1_range"] = [
                    min(r["metrics"]["macro_f1"] for r in subset),
                    max(r["metrics"]["macro_f1"] for r in subset)]
            means = {name: float(np.mean([v[name] for v in by_group.values()]))
                     for name in ("macro_f1", "mean_iou", "accuracy", "balanced_accuracy", "brier_multiclass")}
            seed_means = [float(np.mean([r["metrics"]["macro_f1"] for r in rows if r["seed"] == seed]))
                          for seed in seeds]
            by_arm[arm.name] = {
                "features": list(arm.features), "feature_count": len(arm.features),
                "site_balanced": means, "by_group": by_group,
                "seed_mean_macro_f1_range": [min(seed_means), max(seed_means)],
                "mean_fit_seconds": float(np.mean([r["fit_seconds"] for r in rows])),
                "mean_predict_seconds": float(np.mean([r["predict_seconds"] for r in rows])),
            }
        baseline = by_arm.get("terrain_full")
        if baseline:
            for record in by_arm.values():
                deltas = {g: record["by_group"][g]["macro_f1"] - baseline["by_group"][g]["macro_f1"]
                          for g in groups}
                record["delta_macro_f1_vs_full"] = record["site_balanced"]["macro_f1"] - baseline["site_balanced"]["macro_f1"]
                record["delta_by_group"] = deltas
                record["groups_improved"] = sum(v > 0 for v in deltas.values())
        parents = {
            "terrain_local_global_sky": "terrain_local",
            "terrain_local_local_sky": "terrain_local",
            "terrain_full": "terrain_local",
            "terrain_scene": "terrain_full",
            "terrain_local_nac": "terrain_local",
            "terrain_full_nac": "terrain_full",
            "terrain_full_radar": "terrain_full",
            "terrain_full_nac_radar": "terrain_full_nac",
            "terrain_full_nac_quality": "terrain_full_nac",
            "terrain_full_nac_screened": "terrain_full_nac",
            "terrain_full_nac_lola": "terrain_full_nac",
        }
        for name, parent in parents.items():
            if name in by_arm and parent in by_arm:
                record, baseline = by_arm[name], by_arm[parent]
                record["incremental_comparison"] = {
                    "parent": parent,
                    "delta_macro_f1": record["site_balanced"]["macro_f1"] - baseline["site_balanced"]["macro_f1"],
                    "delta_mean_iou": record["site_balanced"]["mean_iou"] - baseline["site_balanced"]["mean_iou"],
                    "delta_by_group": {g: record["by_group"][g]["macro_f1"] - baseline["by_group"][g]["macro_f1"]
                                       for g in groups},
                }
        summaries[snapshot] = by_arm
    return summaries


def recommendation(summary, snapshots, complete):
    primary = summary.get("saved", summary.get(snapshots[0], {}))
    if not complete or "terrain_full" not in primary:
        return {"status": "incomplete_no_recommendation", "app_recipe_changed": False}
    ranked = sorted(primary, key=lambda n: (-primary[n]["site_balanced"]["macro_f1"], primary[n]["feature_count"]))
    consistent = []
    for name in ranked:
        if name == "terrain_full":
            continue
        if all(name in summary[s] and
               summary[s][name]["delta_macro_f1_vs_full"] >= 0.01 and
               min(summary[s][name]["delta_by_group"].values()) >= 0 and
               summary[s][name]["site_balanced"]["mean_iou"] >= summary[s]["terrain_full"]["site_balanced"]["mean_iou"]
               for s in snapshots):
            consistent.append(name)
    return {
        "status": "development_only_not_promoted", "app_recipe_changed": False,
        "highest_mean_macro_f1": ranked[0], "consistent_candidates": consistent,
        "consistency_rule": ">=1 percentage point mean macro-F1 gain, no group loss, no mean-IoU loss, in every requested snapshot variant",
        "next_step": "Review labels and validate finalists on new independent geography before changing the deployed recipe.",
        "warning": "Seeds and repeated snapshots are not independent sites; no confidence interval or SOTA claim is justified.",
    }


def markdown_report(report):
    lines = ["# Which inputs help geological mapping?", "",
             "**Private development benchmark — not a final test or a deployable model.**", "",
             (f"Status: **{'complete' if report['complete'] else 'incomplete'}**. "
              f"{report['sample_count']:,} original painting cells; "
              f"{len(report['groups'])} geographic groups; {len(report['config']['seeds'])} model seeds."), "",
             ("Each group is held out in full. N1014 and MS1 stay together. All arms score the same cells; "
              "quality screening affects training only. No model was saved to the app."), "",
             "## Results", "",
             ("Metrics below are equally weighted across held-out groups, after averaging seeds. "
              "Macro-F1 and mean IoU weight the three target classes equally; accuracy alone can hide rare-class failures."), ""]
    for snapshot, summary in report["summary"].items():
        lines += [f"### {snapshot.title()}-preferred labels", "",
                  "| Input recipe | Features | Macro-F1 | Mean IoU | Accuracy | Δ F1 vs full terrain | Mean fit time |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in sorted(summary.items(), key=lambda item: -item[1]["site_balanced"]["macro_f1"]):
            m = row["site_balanced"]
            delta = row.get("delta_macro_f1_vs_full")
            gain = f"{delta * 100:+.2f} pp" if delta is not None else "—"
            lines.append(f"| {name} | {row['feature_count']} | {m['macro_f1']:.3f} | {m['mean_iou']:.3f} | "
                         f"{m['accuracy']:.1%} | {gain} | {row['mean_fit_seconds']:.2f} s |")
        if summary:
            lines += ["", "#### Held-out group macro-F1", "",
                      "| Recipe | " + " | ".join(report["groups"]) + " |",
                      "|---|" + "---:|" * len(report["groups"])]
            for name, row in sorted(summary.items(), key=lambda item: -item[1]["site_balanced"]["macro_f1"]):
                lines.append("| " + name + " | " + " | ".join(f"{row['by_group'][g]['macro_f1']:.3f}" for g in report["groups"]) + " |")
        lines += ["", "#### Incremental changes", "",
                  "Compare each addition against the same recipe without it, not just full terrain.", "",
                  "| Recipe | Compared with | Δ macro-F1 | Δ mean IoU | " + " | ".join(report["groups"]) + " |",
                  "|---|---|---:|---:|" + "---:|" * len(report["groups"])]
        for name, row in summary.items():
            delta = row.get("incremental_comparison")
            if delta:
                lines.append(f"| {name} | {delta['parent']} | {delta['delta_macro_f1'] * 100:+.2f} pp | "
                             f"{delta['delta_mean_iou'] * 100:+.2f} pp | " +
                             " | ".join(f"{delta['delta_by_group'][g] * 100:+.2f} pp" for g in report["groups"]) + " |")
        lines.append("")
    if report["skipped_arms"]:
        lines += ["## Unavailable comparisons", ""]
        lines += [f"- **{name}**: {reason}" for name, reason in report["skipped_arms"].items()]
        lines.append("")
    rec = report["recommendation"]
    lines += ["## Interpretation", "", f"Decision status: **{rec['status']}**."]
    if report["complete"] and "highest_mean_macro_f1" in rec:
        lines += [f"Highest mean score: **{rec['highest_mean_macro_f1']}**.",
                  "Candidates passing the conservative consistency check: **" +
                  (", ".join(rec["consistent_candidates"]) or "none") + "**.", rec["consistency_rule"] + ".",
                  "A best mean score is not proof of a reliable gain at every site. No automatic app feature change was made."]
    lines += ["", "## Feature definitions", ""]
    for arm in report["arms"]:
        lines.append(f"- **{arm['name']}**: " + (", ".join(arm["features"]) or "natural training-majority class") +
                     ("; train only on cells with SfS support ≥90%; test set unchanged." if arm["screen_training_support"] else "."))
    lines += ["", "## Reproducibility and limits", "",
              (f"- Random forest: {report['config']['trees']} trees, minimum leaf size 5, balanced class weights, "
               f"one CPU worker; cap {report['config']['cap_per_group_class']} training cells per group/class."),
              ("- Sampling is paired between feature arms. Quality-screened sampling is the declared exception. "
               "Missing feature values use the forest's training-fold-native handling; no complete-case test filtering."),
              ("- Targets are coarse painting cells, not thousands of independent 5 m pixel labels. "
               "The model is trained and scored on cell-mean features; this is not yet the app's pixel-wise training recipe."),
              ("- Support is an observational flag, not label confidence. Low-support cells remain in the overall score; "
               "their separate metrics and per-class confusion matrices are in results.json."),
              ("- The saved/draft comparison is sensitivity to historical snapshots, not independent replication. "
               "Only eight paired eligible labels differ in the current collection; this is a very weak sensitivity check."),
              ("- Historical model interaction may favor terrain recipes resembling those previously used. "
               "Missing accepted/confidence arrays prevent certifying independence, even with geographic holdouts."),
              ("- All recipes use the same NAC/SfS-covered cohort, including LOLA-only and terrain-only. "
               "This does not evaluate deployment where NAC is unavailable."),
              ("- The existing Gaussian-Laplacian curvature implementation has a nonzero constant-height response. "
               "It contains an elevation component; it is not strictly offset-invariant physical curvature. "
               "Both public SfS and LOLA use this implementation, unchanged here."),
              ("- RF uses max_features=sqrt, so changing feature count also changes split-search capacity. "
               "These are recipe comparisons, not isolated sensor-quality measurements.")]
    lines += ["- " + text for text in report["data_metadata"].get("limitations", [])]
    lines += ["", ("Source hashes, exact sample/fold hashes, feature math, exclusions, missingness, "
                   "software versions and every fold's scores are retained in `results.json`. No individual predictions, "
                   "paintings or trained models are exported. Both files inherit the reference-data sharing restrictions."), ""]
    if report.get("stopped_reason"):
        lines += [f"Stopped: {report['stopped_reason']}", ""]
    return "\n".join(lines)


def save_report(directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in (("results.json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
                       ("summary.md", markdown_report(report))):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".benchmark-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
                stream.close()
                os.replace(temporary, directory / name)
            finally:
                temporary.unlink(missing_ok=True)


def validate_options(seeds, snapshots, cap, trees):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Provide unique nonnegative seeds below 2**32")
    if not snapshots or len(set(snapshots)) != len(snapshots) or not set(snapshots) <= {"saved", "draft"}:
        raise ValueError("Snapshots must be saved and/or draft, without duplicates")
    if cap <= 0 or trees <= 0:
        raise ValueError("Training cap and tree count must be positive")


def run_benchmark(data, *, seeds=(0, 17, 42), snapshots=("saved", "draft"), cap=1000,
                  trees=200, arms=ARMS, output=None, deadline=None, progress=None):
    validate_data(data)
    validate_options(seeds, snapshots, cap, trees)
    if not arms or len({a.name for a in arms}) != len(arms):
        raise ValueError("Provide uniquely named experiment arms")
    groups = sorted(set(data["groups"].tolist()))
    report = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False, "sample_count": len(data["sample_ids"]), "groups": groups,
        "sample_ids_sha256": row_digest(data["sample_ids"]),
        "config": {"seeds": list(seeds), "snapshots": list(snapshots), "cap_per_group_class": cap,
                   "trees": trees, "min_samples_leaf": 5, "class_weight": "balanced", "n_jobs": 1,
                                      "max_features": "sqrt",
                   "support_threshold": benchmark_data.MIN_COVERAGE,
                   "metric_class_order": list(CLASSES), "absent_class_f1_iou": 0},
        "arms": [asdict(arm) for arm in arms], "data_metadata": data["metadata"],
        "software": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                     for p in ("numpy", "scipy", "rasterio", "scikit-learn")}},
        "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": [], "skipped_arms": {}, "summary": {}, "recommendation": {},
    }
    started = time.perf_counter()

    def publish():
        report["elapsed_fitting_seconds"] = time.perf_counter() - started
        report["summary"] = summarize(report["runs"], snapshots, groups, seeds, arms=arms)
        report["recommendation"] = recommendation(report["summary"], snapshots, report["complete"])
        if output is not None:
            save_report(output, report)

    try:
        for arm in arms:
            unavailable = [f for f in arm.features if f not in data["feature_names"] or
                           not np.isfinite(data["X"][:, data["feature_names"].index(f)]).any()]
            if unavailable:
                report["skipped_arms"][arm.name] = "Unavailable features: " + ", ".join(unavailable)
                continue
            if progress:
                progress(f"Benchmarking {arm.name} ({len(arm.features)} features)")
            for snapshot in snapshots:
                for group in groups:
                    for seed in seeds:
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError("Cooperative time budget reached between fits")
                        report["runs"].append(fit_fold(data, arm, snapshot, group, seed, cap, trees))
            publish()
        report["complete"] = True
    except (TimeoutError, KeyboardInterrupt) as exc:
        report["stopped_reason"] = str(exc) or "Interrupted"
    finally:
        publish()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=storage.ROOT / "experiments/feature-selection")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--cap-per-group-class", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 42])
    parser.add_argument("--snapshots", nargs="+", choices=("saved", "draft"), default=["saved", "draft"])
    parser.add_argument("--max-seconds", type=float, default=840,
                        help="Cooperative budget including extraction; checked between model fits")
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be positive and finite")
    try:
        validate_options(args.seeds, args.snapshots, args.cap_per_group_class, args.trees)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    experiments = storage.ROOT.resolve() / "experiments"
    if output == experiments or not output.is_relative_to(experiments):
        parser.error("Choose a results subdirectory inside poster/experiments")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Choose a new/empty results directory; existing experiments are not overwritten")
    deadline = time.monotonic() + args.max_seconds
    try:
        data = benchmark_data.load_data(paths.DEM_DIR, paths.PROFESSOR_MAPS_DIR,
                                        paths.DATA_DIR / "raw_data", progress=lambda m: print(m, flush=True))
        result = run_benchmark(data, seeds=tuple(args.seeds), snapshots=tuple(args.snapshots),
                               cap=args.cap_per_group_class, trees=args.trees, output=output,
                               deadline=deadline, progress=lambda m: print(m, flush=True))
    except (ValueError, OSError) as exc:
        print(f"Benchmark failed: {exc}")
        return 1
    print(f"{'Completed' if result['complete'] else 'Incomplete'}: {len(result['runs'])} fits/scores; {output / 'summary.md'}")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
