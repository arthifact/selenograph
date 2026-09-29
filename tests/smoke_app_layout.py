"""Opt-in installed-catalog smoke: .venv/bin/python -B -m tests.smoke_app_layout.

Read-only, stdout summary only. Browse every installed Section and its Map options,
bundled annotations, the smallest tile, NAC imagery, Gallery and saved evaluation scores.
Never click action buttons, train a model, export files or simulate canvas strokes.
Synthetic callers may use main() with a disposable or empty catalog.
"""
import sys

sys.dont_write_bytecode = True

import builtins
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import rasterio
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import assistant_model as assistant
from app import canvas, core, evaluate, evaluation, paths
from app import import_reference_maps as reference


ENTRYPOINT = Path(__file__).resolve().parents[1] / "selenograph.py"
PAGES = ["Paint", "Gallery", "Progress"]


def snapshot():
    """Include empty directories as well as file contents; never create a directory."""
    roots = (paths.OUTPUT_DIR, Path(core.OUT_DIR), Path(core.MAP_DIR), Path(core.LEGACY_OUT_DIR), Path(core.DRAFT_DIR),
             Path(core.LEGACY_DRAFT_DIR), Path(core.LEGACY_MIRRORED_DIR),
             Path(assistant.MODEL_DIR), paths.POSTER_DIR,
             paths.PROJECT_ROOT / "reports", core.test_maps_path())
    result = {}
    for root in roots:
        for path in [root, *root.rglob("*")] if root.is_dir() else [root]:
            if path.is_file():
                with path.open("rb") as stream:
                    result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
            elif path.is_dir():
                result[str(path)] = "directory"
    return result


@contextmanager
def read_only():
    """Refuse training, exports and filesystem mutations, including accidental autosave.

    File guards cover direct thumbnail/log/poster writes, raster writes and deletion,
    even when configured roots are outside PROJECT_ROOT. In-memory images are fine.
    """
    roots = [Path(p).resolve() for p in (
        paths.PROJECT_ROOT, core.DEM_DIR, paths.OUTPUT_DIR, core.OUT_DIR, core.MAP_DIR, core.LEGACY_OUT_DIR, core.DRAFT_DIR,
        core.LEGACY_DRAFT_DIR, core.LEGACY_MIRRORED_DIR,
        assistant.MODEL_DIR, paths.POSTER_DIR, paths.REFERENCE_DIR,
        paths.PROFESSOR_MAPS_DIR, reference.REFERENCE_ROOT, core.test_maps_path())]
    attempts = []

    def refuse(action):
        def blocked(*args, **kwargs):
            attempts.append(action)
            raise AssertionError(f"Read-only smoke test refused {action}")
        return blocked

    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if any(resolved == root or resolved.is_relative_to(root) for root in roots):
                refuse(f"filesystem mutation: {resolved}")()

    def opening(original):
        def guarded(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                check(file)
            return original(file, mode, *args, **kwargs)
        return guarded

    def mutation(original, count):
        def guarded(*args, **kwargs):
            for path in args[:count]:
                check(path)
            for key in ("path", "src", "dst"):
                if key in kwargs:
                    check(kwargs[key])
            return original(*args, **kwargs)
        return guarded

    original_os_open = os.open
    original_raster_open = rasterio.open

    def os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            check(path)
        return original_os_open(path, flags, *args, **kwargs)

    def raster_open(fp, mode="r", *args, **kwargs):
        if mode != "r":
            refuse("raster write")()
        return original_raster_open(fp, mode, *args, **kwargs)

    with ExitStack() as guards:
        for module, names in (
            (core, ("save_painting", "seed_paintings", "write_map", "meta_set", "set_test_maps", "train")),
            (assistant, ("update", "retrain", "save", "fit", "refit")),
            (evaluate, ("record",)),
            (evaluation, ("run",)),
            (reference, ("main",)),
            (core.backend(), ("fit", "fit_rows")),
            (np, ("save", "savez", "savez_compressed")),
            (assistant.joblib, ("dump",)),
        ):
            for name in names:
                action = f"{module.__name__}.{name}"
                guards.enter_context(patch.object(module, name, side_effect=refuse(action)))
        guards.enter_context(patch.object(builtins, "open", opening(builtins.open)))
        guards.enter_context(patch.object(io, "open", opening(io.open)))
        guards.enter_context(patch.object(os, "open", os_open))
        guards.enter_context(patch.object(rasterio, "open", raster_open))
        for name in ("mkdir", "remove", "unlink", "rmdir", "rename", "replace"):
            count = 2 if name in ("rename", "replace") else 1
            guards.enter_context(patch.object(os, name, mutation(getattr(os, name), count)))
        yield attempts
        assert not attempts, attempts



def main(*, expected_sections=None, expected_verified_bundles=None):
    summary = {"mode": "read-only AppTest; no browser/JavaScript simulation",
               "passed": False, "timings": []}
    before = snapshot()
    latest_canvas = {}
    render = canvas.map_canvas

    def observe_canvas(image, **kwargs):
        latest_canvas.update(tool=kwargs["tool"], legend=kwargs["legend"],
                             terrain=hashlib.sha256(kwargs["terrain"].tobytes()).hexdigest())
        stroke = render(image, **kwargs)
        assert not stroke, "Read-only smoke test must never receive a painting stroke"
        return stroke

    def timed(label, action):
        started = perf_counter()
        result = action()
        summary["timings"].append({"action": label, "seconds": round(perf_counter() - started, 3)})
        if isinstance(result, AppTest):
            assert not result.exception, [e.message for e in result.exception]
            assert not result.error, [e.value for e in result.error]
            assert not any(w.key == "map_location" or w.label == "Location" for w in result.selectbox)
        return result

    try:
        with read_only(), patch.object(canvas, "map_canvas", side_effect=observe_canvas), \
                patch.object(st, "navigation", wraps=st.navigation) as navigation:
            records = core.records()
            by_name = {core.dem_name(r["working_dem"]): r for r in records}
            installed = core.dem_files()
            hidden = core.held_out_maps()
            visible = [p for p in installed if core.dem_name(p) not in hidden]
            locked = {name for name, record in by_name.items() if record.get("read_only")}
            bundled = {name for name, record in by_name.items() if record.get("annotations")}
            verified_bundles = {name for name in bundled
                                if by_name[name]["annotations"].get("verified")}
            bundle = assistant.load()
            samples = bundle["samples"]
            assert isinstance(samples, dict)

            def record_for(path):
                return by_name.get(core.dem_name(path), {"working_dem": path})

            sections = Counter(str(core.section(record_for(p))) for p in installed)
            summary["catalog"] = dict(records=len(records), installed=len(installed),
                                       visible=len(visible), sections=dict(sections))
            summary["model"] = dict(trained=bundle["model"] is not None, remembered_maps=len(samples))
            browsed_bundles = []
            summary["annotations"] = dict(bundled=len(bundled), verified=len(verified_bundles),
                                           browsed=browsed_bundles)
            if expected_sections is not None:
                assert len(sections) == expected_sections, dict(sections)
            if expected_verified_bundles is not None:
                assert len(verified_bundles) == expected_verified_bundles, sorted(verified_bundles)
            for name, record in by_name.items():
                dem = core.record_path(record)
                assert dem and Path(dem).is_file(), f"Catalog DEM is missing: {name}"
                core.map_site(name)

            painted, accepted = core.painted_maps()
            labels = {}
            for path in installed:
                name = core.dem_name(path)
                status = [text for yes, text in (
                    (name in painted, "painted"),
                    (core.meta_get(name).get("verified"), "verified")) if yes]
                labels[path] = f"{name} — {', '.join(status)}" if status else name

            app = AppTest.from_file(str(ENTRYPOINT), default_timeout=120)
            timed("Paint cold", app.run)
            timed("Paint warm", app.run)

            def widget(key):
                return next((w for w in app.selectbox if w.key == key), None)

            def select(key, value):
                control = widget(key)
                assert control is not None, key
                if control.value != value:
                    timed(f"{key}: {value}", lambda: control.select(value).run())

            def check_maps(expected):
                assert widget("map_collection") is None
                assert widget("map_location") is None
                assert "map_location" not in app.session_state
                assert not any(w.label in ("Collection", "Location") for w in app.selectbox)
                control = widget("selected_map")
                assert control is not None
                assert control.options == [labels[p] for p in expected], control.options
                assert control.value in expected
                name = core.dem_name(control.value)
                assert app.session_state.cells.shape == (core.GRID, core.GRID)
                assert app.session_state.locked == (name in locked)
                assert (latest_canvas["tool"] == "locked") == (name in locked)
                assert app.radio[0].disabled == (name in locked)
                if bundle["model"] is None:
                    assert app.session_state.pred is None
                return name

            section_control = widget("map_section")
            assert section_control is not None
            visible_sections = sorted({str(core.section(record_for(p))) for p in visible})
            assert section_control.options == ["All", *visible_sections]
            summary["sections"] = {}
            if visible:
                check_maps(visible)
                for section in visible_sections:
                    select("map_section", section)
                    members = [p for p in visible if str(core.section(record_for(p))) == section]
                    check_maps(members)
                    summary["sections"][section] = len(members)
                selected = app.selectbox(key="selected_map").value
                select("map_section", "All")
                check_maps(visible)
                assert app.selectbox(key="selected_map").value == selected
                for path in visible:
                    if core.dem_name(path) in bundled:
                        select("selected_map", path)
                        name = check_maps(visible)
                        np.testing.assert_array_equal(app.session_state.cells, core.load_painting(name))
                        browsed_bundles.append(name)

                sized = [p for p in visible if record_for(p).get("size")]
                if sized:
                    edge = min(sized, key=lambda p: np.prod(record_for(p)["size"]))
                    select("selected_map", edge)
                    check_maps(visible)
                    summary["edge_tile"] = record_for(edge)["size"]
                nac = next((p for p in visible if core.companion(p, "nac")), None)
                if nac:
                    select("selected_map", nac)
                    check_maps(visible)
                    control = app.segmented_control(key="background")
                    assert "NAC imagery" in control.options
                    timed("NAC background", lambda: control.set_value("nac").run())
                    assert "display only" in latest_canvas["legend"]
                    features = assistant.schema()["features"]
                    assert isinstance(features, list) and "nac" not in features
                    captions = "\n".join(c.value for c in app.caption)
                    assert "Choosing a background does not change training inputs." not in captions
                    assert "NAC imagery is a display-only background" not in captions
                summary["nac_display_checked"] = nac is not None
            else:
                assert widget("selected_map") is None

            timed("Gallery", lambda: app.switch_page("app/app_pages/gallery.py").run())
            gallery_keys = {w.key for w in app.checkbox}
            page_widget = next((w for w in app.selectbox if w.key == "gallery_page"), None)
            if page_widget:
                for page_number in range(2, len(page_widget.options) + 1):
                    app.selectbox(key="gallery_page").select(page_number).run()
                    gallery_keys.update(w.key for w in app.checkbox)
            assert gallery_keys == {f"v_{core.dem_name(path)}" for path in installed}
            labeled_bundles = set()
            for name in bundled:
                label_path = core.annotation_path(name, "labels")
                if name in painted or (label_path and Path(label_path).is_file()):
                    labeled_bundles.add(name)
            assert {f"v_{name}" for name in labeled_bundles}.issubset(gallery_keys)
            summary["gallery"] = dict(verification_controls=len(gallery_keys),
                                       bundled=len(labeled_bundles),
                                       empty=any("No installed maps yet" in m.value for m in app.info))
            timed("Progress", lambda: app.switch_page("app/app_pages/progress.py").run())
            timed("Progress warm", app.run)
            assert "Evaluation" in [h.value for h in app.subheader]
            assert "Independent checks" not in [h.value for h in app.subheader]
            if not bundle["samples"] or not app.multiselect(key="evaluation_maps").value:
                assert next(b for b in app.button if b.label == "Evaluate").disabled
            assert len(app.multiselect(key="evaluation_maps").options) == len(evaluation.catalog())
            assert [w.key for w in app.multiselect] == ["evaluation_maps"]
            assert not app.get("image")
            summary["evaluation"] = dict(
                report_available=any(m.label == "Evaluation accuracy" for m in app.metric),
                json_download=any(w.label == "Download evaluation JSON" for w in app.get("download_button")))
            timed("Return to Paint", lambda: app.switch_page("app/app_pages/paint.py").run())
            for call in navigation.call_args_list:
                assert [page.title for page in call.args[0]] == PAGES
            summary["navigation"] = PAGES
            summary["passed"] = True
    except Exception as error:
        summary["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        after = snapshot()
        changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
        summary["persistence"] = dict(entries_before=len(before), entries_after=len(after), changed=changed)
        summary["passed"] = summary["passed"] and not changed
        print(json.dumps(summary, indent=2), flush=True)
        assert not changed, changed
    return summary


if __name__ == "__main__":
    main(expected_sections=17, expected_verified_bundles=6)
