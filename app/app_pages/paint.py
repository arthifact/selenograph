"""Open with the shared assistant, paint corrections, and explicitly update its preview.

Only Save labels + map exports human labels and a separate predicted raster. Predicted
cells become labels only when you approve them (Approve prediction, or Smart fill with
the model's own unit): they stay marked as approved and teach the model at half weight.
"""
import datetime
import os


import numpy as np
import streamlit as st
from scipy import ndimage as ndi

from app import core, assistant_model, canvas, evaluate, pictures

from app.core import CMAP, CODES, GRID

VIEW = 800          # hillshade pixels per side, at least
STRIDE = 4          # predict every 4th pixel while painting; full grid only on save
MAP_H = 620         # keeps the whole map inside a laptop window without scrolling
UNDO = 30           # remembered paint steps

PAINT = {n.replace("_", " "): c for c, n in CODES.items()}
PAINT["erase"] = 0
LAYERS = {"Prediction": ":material/auto_awesome: Prediction",
          "Painting": ":material/brush: Painting",      # drawn in this order
          "Confidence": ":material/equalizer: Confidence"}  # replaces the background
TOOLS = {"paint": ":material/gesture: Paint", "smart": ":material/format_color_fill: Smart fill"}
CERTAINTY = {core.SURE: "Sure", core.MOSTLY: "Mostly", core.UNSURE: "Unsure"}
# Background layers include display-only terrain layers and NAC imagery. Each is
# drawn in grey, and the legend says what bright means.
ESSENTIAL_BACKGROUNDS = ("hillshade", "nac", "zscene")
BACKGROUNDS = {"hillshade": "Hillshade", "slope": "Slope", "rough": "Roughness",
               "rel": "Relative height", "zscene": "Elevation", "svf": "Sky view",
               "rel_local": "Local height", "curv": "Curvature", "svf_local": "Local sky view",
               "psr": "Shadow (PSR)", "cpr": "Radar (CPR)", "nac": "NAC imagery"}
LEGEND = {"slope": "Slope: bright is steep",
          "rough": "Roughness: bright is rough",
          "rel": "Relative height: bright is above its 4 km surroundings",
          "zscene": "Elevation: bright is high",
          "svf": "Sky view: dark is enclosed (crater floors), bright is open sky",
          "rel_local": "Local height: dark is below its 150 m surroundings (small craters)",
          "curv": "Curvature: bright is bowl-shaped, dark is ridge-like",
          "svf_local": "Local sky view: dark is enclosed within 300 m (small craters)",
          "psr": "Shadow: bright is permanently shadowed",
          "cpr": "Radar CPR: bright is high",
          "nac": "NAC imagery: display only; brightness is not terrain height or a PSR label"}


def background_label(layer):
    label = BACKGROUNDS[layer] if layer in BACKGROUNDS else str(layer)
    return f":blue[{label}]" if layer in assistant_model.schema()["features"] else label


def chip(code, label, bold=None):
    """A swatch in the unit's exact map colour. No emoji matches these colours."""
    rgb = "#%02x%02x%02x" % CMAP[code][:3] if code else "transparent"
    edge = "" if code else ";border:1px solid #888"
    tail = f" &mdash; <b>{bold}</b>" if bold is not None else ""
    return (f'<span style="display:inline-block;width:0.8em;height:0.8em;'
            f'background:{rgb}{edge};border-radius:2px;margin-right:0.5em;'
            f'vertical-align:-0.05em"></span>{label}{tail}')


def data_stamp(dem_path):
    """Invalidate terrain caches for the whole feature tree, not just the display DEM."""
    order, _ = core._feature_plan(dem_path)
    dependencies = {str(path) for path in core.manifest_files()}
    dependencies.update(order)
    for path in order:
        for kind in ("psr", "cpr", "nac"):
            companion = core.companion(path, kind)
            if companion:
                dependencies.add(companion)
    stamps = []
    for path in sorted(dependencies):
        try:
            stat = os.stat(path)
            stamps.append((path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino))
        except FileNotFoundError:
            stamps.append((path, None))
    return (core.FEATURE_VERSION, tuple(stamps))


@st.cache_data(show_spinner="Reading the DEM and building features (once per map)...",
               max_entries=3)
def load(dem_path, revision):
    X, names, hs, prof, px = core.build(dem_path)
    step = display_step(hs.shape)
    return X, names, hs[::step, ::step], X[:, ::STRIDE, ::STRIDE], prof, px


def display_step(shape):
    """Draw every nth pixel, keeping at least VIEW pixels a side."""
    return max(1, min(shape) // VIEW)


@st.cache_data(show_spinner=False, max_entries=3)
def paintable_cells(valid):
    """Painting cells with actual DEM coverage, using the existing cell edges."""
    ids = core.per_pixel(np.arange(GRID * GRID).reshape(GRID, GRID), valid.shape)
    return np.bincount(ids[valid], minlength=GRID * GRID).reshape(GRID, GRID) > 0


@st.cache_data(show_spinner=False, max_entries=16)
def backdrop(dem_path, revision, layer, _X, _names, _hs):
    """The chosen background as brightness 0-1 on the display grid: the hillshade, or a
    display layer stretched between its 2nd and 98th percentiles on this map."""
    if layer not in _names:
        return _hs
    step = display_step(_X.shape[1:])
    v = _X[_names.index(layer), ::step, ::step]
    good = np.isfinite(v)
    if not good.any():
        return np.zeros_like(_hs)
    lo, hi = np.percentile(v[good], [2, 98])
    if hi <= lo:                                   # e.g. a 0/1 shadow mask
        lo, hi = v[good].min(), v[good].max()
    return np.clip(np.where(good, (v - lo) / max(hi - lo, 1e-6), 0), 0, 1).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=4)
def hatching(shape):
    """Which pixels each confidence level colours: sure painting solid, mostly-sure in
    diagonal stripes, unsure in dots -- the map convention for less certain units, and
    visible at any colour strength, unlike fading."""
    i, j = np.indices(shape)
    return {core.SURE: np.ones(shape, bool),
            core.MOSTLY: (i + j) // 3 % 2 == 0,
            core.UNSURE: (i % 4 < 2) & (j % 4 < 2)}


def widen_stroke(m):
    """A dragged stroke comes back one cell wide, which is too fiddly to paint with. A
    loop fill comes back solid. Tell them apart by how much of the bounding box is
    filled, and fatten only the stroke -- so a drag paints a band and a loop still fills
    exactly what you drew."""
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return m
    bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return ndi.binary_dilation(m, iterations=1) if ys.size / bbox < 0.25 else m


def run(fn, *a):
    """Every call into the model goes through here, so a backend that is a placeholder
    (or a MODEL naming a file that does not exist) says so instead of throwing a
    traceback at someone who just wanted to paint."""
    try:
        return fn(*a)
    except (NotImplementedError, ValueError, OSError) as e:
        st.error(str(e))
        st.stop()


def flip():
    """Show only the prediction, then only the painting, and back: compare in place.
    The confidence background stays as it is."""
    alone = [layer for layer in ss.layers if layer != "Confidence"] == ["Prediction"]
    ss.layers = (["Painting"] if alone else ["Prediction"]) + \
        [layer for layer in ss.layers if layer == "Confidence"]


ss = st.session_state
ss.setdefault("layers", ["Painting"])
ss.setdefault("tool", "paint")
ss.setdefault("certainty", core.SURE)
# Test maps and the maps touching them are not listed: the model is scored there, so
# they are never painted or trained on. The Progress page chooses the test maps.
hidden = core.held_out_maps()
dems = [d for d in core.dem_files() if core.dem_name(d) not in hidden]
catalog = {core.dem_name(r["working_dem"]): r for r in core.records()}


def category(path):
    record = catalog.get(core.dem_name(path), {"working_dem": path})
    return str(core.section(record))


ss.pop("map_location", None)
with st.sidebar:
    st.subheader("1 · Map")
    sections = ["All", *sorted({category(d) for d in dems})]
    if ss.get("map_section") not in sections:
        ss.map_section = "All"
    selected = st.selectbox("Section", sections, key="map_section", width="stretch")
    if selected != "All":
        dems = [d for d in dems if category(d) == selected]

if not dems:
    st.subheader("No DEM found")
    st.markdown(
        "Install a prepared dataset under `data/processed_data/`, then reload this page. "
        "Each top-level dataset folder is a Section. Nested DEMs, display layers and "
        "bundled annotations are discovered through `working_dems.json`, not by "
        "dropping rasters into arbitrary folders.")
    if hidden:
        st.caption("Prepared maps may also be hidden because they are test maps or touch "
                   "one. Review the test-map selection on the Progress page.")
    st.stop()

try:
    model_name = core.backend().NAME
except ModuleNotFoundError:
    st.error(f'`core.MODEL` is `"{core.MODEL}"`, but there is no `model_{core.MODEL}.py` '
             f'next to `core.py`. Pick one of: '
             + ", ".join(f"`{m}`" for m in core.model_files()))
    st.stop()

model_stamp = assistant_model.artifact_stamp()
assistant_changed = ("assistant" not in ss or
                     ss.assistant.get("schema") != assistant_model.schema() or
                     ss.get("assistant_stamp") != model_stamp)
if assistant_changed:
    ss.assistant = run(assistant_model.load)
    ss.assistant_stamp = model_stamp
    ss.pop("conf_for", None)
painted, _ = core.painted_maps()



def marked(path):
    """Keep the DEM path as the value and append metadata-only status labels."""
    n = core.dem_name(path)
    labels = []
    if n in painted:
        labels.append("painted")
    if core.meta_get(n).get("verified"):
        labels.append("verified")
    return f"{n} — {', '.join(labels)}" if labels else n


with st.sidebar:
    if ss.get("selected_map") not in dems:
        ss.selected_map = ss.get("dem") if ss.get("dem") in dems else dems[0]
    dem = st.selectbox("Map", dems, format_func=marked,
                       label_visibility="collapsed", key="selected_map")
    if dem is None or dem not in dems:
        st.info("Choose an installed map from the current Section.")
        st.stop()

    if hidden:
        st.caption(f"{len(hidden)} maps are not listed: the test maps and the maps "
                   "touching them. The model is scored there and never learns from them. "
                   "Choose test maps on the Progress page.")
    name = core.dem_name(dem)
    record = catalog.get(name, {})
    metadata = core.meta_get(name)
    revision = (os.stat(dem).st_mtime_ns, os.stat(dem).st_size)
    if (ss.get("dem"), ss.get("dem_revision"), ss.get("backend")) != (dem, revision, core.MODEL):
        ss.dem, ss.cells, ss.pred, ss.hist = dem, None, None, []
        ss.dem_revision = revision
        ss.backend = core.MODEL

feature_revision = run(data_stamp, dem)
features_changed = ss.get("feature_revision") != feature_revision
X, names, hs, Xs, prof, px = load(dem, feature_revision)
shape = (prof["height"], prof["width"])
cell_width, cell_height = shape[1] * px / GRID, shape[0] * px / GRID
cell_size = (f"{cell_width:.0f}" if shape[0] == shape[1] else
             f"{cell_width:.0f} × {cell_height:.0f}")
tile_bounds = pictures.tile_bounds(name, shape, prof["transform"])
terrain_valid = np.isfinite(X[names.index("zscene") if "zscene" in names else 0])
valid_cells = paintable_cells(terrain_valid)

if ss.get("cells") is None or not {"accepted", "confidence", "test_site", "locked"} <= ss.keys():
    ss.cells = core.load_painting(name)
    ss.accepted = core.load_accepted(name)       # cells filled from a prediction
    ss.confidence = core.load_confidence(name)   # how sure you were of each cell
    ss.locked = bool(record.get("read_only"))
    ss.test_site = name in hidden                # scored here, never trained on
    ss.assistant = run(assistant_model.load)
    with st.spinner("Generating a starting map..."):
        ss.pred = run(assistant_model.predict, ss.assistant, Xs, names)
    ss.hist = []                     # (cells, accepted, confidence) before each step
elif assistant_changed or features_changed:
    ss.pred = run(assistant_model.predict, ss.assistant, Xs, names)
ss.feature_revision = feature_revision
if features_changed:
    ss.pop("conf_for", None)

assert ss.cells is not None
# Provenance may be refreshed without changing a raster's revision.
ss.locked = bool(record.get("read_only"))
ss.test_site = name in hidden
ss.accepted = ss.accepted & (ss.cells > 0)
ss.confidence = np.where(ss.cells > 0, np.where(ss.confidence > 0, ss.confidence, core.SURE),
                         0).astype(np.uint8)
# Each painted cell teaches the model by how sure you were of it; approved predictions
# (accepted cells) count at half weight until you paint over them. `hand` is what you
# painted yourself, which Smart fill never overwrites.
hand = np.where(ss.accepted, 0, ss.cells).astype(np.uint8)
sure = np.where(hand > 0, ss.confidence, 0).astype(np.uint8)
suggested = (core.labels_to_cells(ss.pred) if ss.pred is not None
             else np.zeros((GRID, GRID), np.uint8))
fill = (ss.cells == 0) & (suggested > 0)
pending = assistant_model.pending(ss.assistant, name, ss.cells, ss.confidence, ss.accepted)
editable = not ss.locked and not ss.test_site
has_painting = bool(np.any(ss.cells))
saved_cells, saved_source = core.painting_source(name, final=True)
saved_accepted = core.load_accepted(name, final=True)
saved_confidence = core.load_confidence(name, final=True)
can_restore = editable and saved_source is not None and any(
    not np.array_equal(current, saved) for current, saved in zip(
        (ss.cells, ss.accepted, ss.confidence), (saved_cells, saved_accepted, saved_confidence)))
# Clearing existing labels must remain committable without a model or metadata.
has_existing_labels = any(path and os.path.isfile(path) for path in (
    core.painting_files(name, final=True, existing=True)[0],
    core.output_file(name, "labels.tif", existing=True),
    core.annotation_path(name, "painting"), core.annotation_path(name, "labels")))
has_staged_metadata = os.path.isfile(core.output_file(name, "meta.json", final=False))
is_verified = metadata.get("verified") is True
verified_training_maps = assistant_model.verified_maps()
can_update = bool(verified_training_maps or ss.assistant["samples"])
training_changed = run(assistant_model.training_changed, ss.assistant, verified_training_maps)
can_save = editable and (has_painting or has_existing_labels or has_staged_metadata
                         or ss.assistant["model"] is not None)
smart_off = not editable or ss.pred is None
if smart_off:
    ss.tool = "paint"

with st.sidebar:
    st.subheader("2 · Paint as")
    if ss.locked:
        st.caption("This dataset marks the map read only. Painting, training updates and saving are disabled.")
    tool = st.segmented_control(
        "Tool", list(TOOLS), format_func=lambda value: TOOLS[value] if value in TOOLS else str(value),
        key="tool", required=True, label_visibility="collapsed", disabled=smart_off,
        help=("Smart fill needs an editable map and a prediction." if smart_off else
              "Click to fill a predicted region; loop to limit the fill. Existing painting is kept."))
    assert tool is not None
    unit = st.radio("Paint as", list(PAINT), label_visibility="collapsed",
                    disabled=not editable)
    certainty = st.segmented_control(
        "How sure", list(CERTAINTY),
        format_func=lambda value: CERTAINTY[value] if value in CERTAINTY else str(value), key="certainty",
        required=True, disabled=not editable,
        help="Lower confidence reduces training weight. Mostly uses stripes; Unsure uses dots.")

    with st.container(gap="xxsmall"):
        st.subheader("3 · Painted")
        st.html("".join(
            f'<div style="margin:.12em 0">{chip(c, lbl, int((ss.cells == c).sum()))}</div>'
            for lbl, c in PAINT.items() if c))
    if ss.accepted.any():
        st.caption(f"{int(ss.accepted.sum())} of these cells are approved predictions: they "
                   "teach the model at half weight. Paint over one to correct it.")
    levels = {core.CONFIDENCE[k][0]: int((sure == k).sum()) for k in core.CONFIDENCE}
    if levels["mostly"] or levels["unsure"]:
        st.caption("Training weights (defaults included): " + " · ".join(
            f"{('full weight' if n == 'sure' else n)} {v}" for n, v in levels.items()) + " cells.")
    accept = st.button(
        "Approve prediction", icon=":material/done_all:", width="stretch",
        disabled=not editable or not fill.any(),
        help=("This map is read only." if ss.locked else
              "Fill unpainted cells with the prediction. Undo is available."))
    row = st.container(horizontal=True)
    undo = row.button("Undo", icon=":material/undo:", width="stretch",
                      disabled=not editable or not ss.hist)
    clear = row.button("Clear", icon=":material/delete:", width="stretch",
                       disabled=not editable or not has_painting)
    restore = st.button(
        "Restore saved", icon=":material/restore:", width="stretch", disabled=not can_restore,
        help="Replace your draft with the saved painting. Undo is available.")

    st.subheader("4 · Update Model")
    update_help = (("New verified edits." if training_changed else "Model up to date.") +
                   " Retrain from all verified paintings." if can_update else
                   "Verify a painted map to enable training.")
    update_color = "yellow" if training_changed else "green"
    update = st.button(f":{update_color}[Update model]", key="update_model",
                       icon=":material/refresh:", width="stretch", disabled=not can_update,
                       help=update_help)
    scores = evaluate.history()
    now = next((i for i, e in enumerate(scores) if e["revision"] == ss.assistant["revision"]),
               None)
    if now is not None:
        e = scores[now]
        change = (e["accuracy"] - scores[now - 1]["accuracy"]) * 100 if now else None
        st.metric(f"Test score ({len(e['maps'])} test maps)", f"{e['accuracy']:.0%} right",
                  None if change is None else f"{change:+.0f} points since the last update",
                  border=True,
                  help="Agreement with the answer keys. See Progress for details.")

    st.subheader("5 · Finish")
    verified = st.checkbox("Verified",
                           value=bool(metadata.get("verified", False)),
                           key=f"verified_{name}", disabled=not editable,
                           help="Use this map for training. Save to commit.")
    if editable and verified != bool(metadata.get("verified", False)):
        core.meta_set(name, final=False, verified=verified)
        st.rerun()
    save = st.button("Save labels + map", icon=":material/save:",
                     width="stretch", type="primary", disabled=not can_save)
    alpha = st.slider("Colour strength", 0.0, 1.0, 0.25, 0.05)

def step(cells, accepted, confidence):
    """Replace the painting, remembering the previous one for Undo. Never on a locked
    map, even from a stale page."""
    if not editable or all(np.array_equal(before, after) for before, after in
                           zip((ss.cells, ss.accepted, ss.confidence),
                               (cells, accepted, confidence))):
        return
    ss.hist = (ss.hist + [(ss.cells.copy(), ss.accepted.copy(), ss.confidence.copy())])[-UNDO:]
    ss.cells, ss.accepted, ss.confidence = cells, accepted, confidence
    core.save_painting(name, ss.cells, accepted=ss.accepted,       # survive a restart
                       confidence=ss.confidence)


def sure_of(where):
    """The confidence with `where` painted at the chosen level."""
    assert certainty is not None
    return np.where(where, certainty, ss.confidence).astype(np.uint8)


if undo and editable and ss.hist:
    ss.cells, ss.accepted, ss.confidence = ss.hist.pop()
    core.save_painting(name, ss.cells, accepted=ss.accepted, confidence=ss.confidence)
    st.rerun()
if clear and editable and has_painting:
    step(np.zeros((GRID, GRID), np.uint8), np.zeros((GRID, GRID), bool),
         np.zeros((GRID, GRID), np.uint8))
    st.rerun()
if restore and can_restore:
    step(saved_cells, saved_accepted, saved_confidence)
    st.rerun()
if accept and editable and fill.any():
    step(np.where(fill, suggested, ss.cells).astype(np.uint8), ss.accepted | fill,
         sure_of(fill))
    st.rerun()
if update and can_update:
    with st.spinner("Training from all verified maps..."):
        message = st.empty()
        result = run(assistant_model.retrain, message.text)
    ss.assistant = result
    ss.assistant_stamp = assistant_model.artifact_stamp()
    ss.pred = run(assistant_model.predict, result, Xs, names)
    if ss.pred is not None:
        ss.layers = ["Prediction"] + [layer for layer in ss.layers if layer == "Confidence"]
    ss.pop("conf_for", None)
    st.rerun()

st.markdown(
    f'<div style="font-size:1.35rem;font-weight:600;line-height:1.3">'
    f'{name} — painting {chip(PAINT[unit], f"<b>{unit}</b>")}</div>'
    f'<div style="font-size:.85rem;opacity:.6;margin:.1em 0 .3em">'
    + (f"Smart fill: click a region to paint the model's outline of it as {unit}, or loop "
       f"an area to take only its {unit}. " if tool == "smart" else
       "Loop to fill, drag to paint a band, click for one cell. ") +
    f'Scroll to zoom, right-drag to pan, hold Space to hide the colours. '
    f"{model_name} · {px:.3g} m pixels · {cell_size} m cells"
    f'</div>', unsafe_allow_html=True)


available_backgrounds = ["hillshade", *names]

primary_backgrounds = dict.fromkeys([*ESSENTIAL_BACKGROUNDS, *assistant_model.schema()["features"]])
backgrounds = [layer for layer in primary_backgrounds if layer in available_backgrounds]
backgrounds += [layer for layer in available_backgrounds if layer not in backgrounds]
# Fall back only when a view is unavailable on this map.
selected_background = ss.get("background")
ss.background = selected_background if selected_background in backgrounds else "hillshade"
with st.container(horizontal=True, vertical_alignment="center"):
    shown = st.pills("Show", list(LAYERS), selection_mode="multi",
                     format_func=lambda value: LAYERS[value] if value in LAYERS else str(value),
                     key="layers", label_visibility="collapsed",
                     help="Model confidence: black means low or missing; white means high.")
    confident = "Confidence" in shown and ss.pred is not None
    st.button("Flip", icon=":material/swap_horiz:", on_click=flip, shortcut="F",
              help="Switch between prediction and painting.")
background_help = ("Turn Confidence off to choose a background." if confident else
                   "Brightness is scaled separately for each map.")
background = st.segmented_control(
    "Background", backgrounds, format_func=background_label,
    key="background", required=True, label_visibility="collapsed",
    width="stretch", wrap=True, disabled=confident, help=background_help)
assert background is not None
st.caption(":blue[Training inputs] · other layers are viewing-only.")

if confident:
    if ss.get("conf_for") != (name, ss.assistant["revision"], feature_revision):
        with st.spinner("Computing model confidence for every pixel..."):
            ss.conf = run(assistant_model.confidence, ss.assistant, Xs, names)
        ss.conf_for = (name, ss.assistant["revision"], feature_revision)
    base = pictures.confidence_base(ss.conf, hs.shape)
    legend = "Model score: black is low or missing confidence, white is high; not certainty of correctness"
    peek = "Model confidence, no colours"
else:
    base = backdrop(dem, feature_revision, background, X, names, hs)
    legend = LEGEND.get(background, "")
    peek = f"{BACKGROUNDS.get(background, background)}, no colours"
# Both layers are drawn into the picture the same way, so flipping between them compares
# like with like.
painting = []
if "Painting" in shown:
    for level, drawn in hatching(base.shape).items():       # less sure painting is hatched
        codes = core.cells_to_labels(np.where(ss.confidence == level, ss.cells, 0), base.shape)
        codes[~drawn] = 0
        painting.append((codes, alpha))
img = pictures.render(base, [(ss.pred if "Prediction" in shown else None, alpha), *painting])
missing = ~terrain_valid[::display_step(shape), ::display_step(shape)]
if not confident and background in names:
    missing |= ~np.isfinite(X[names.index(background), ::display_step(shape), ::display_step(shape)])
img = pictures.tile_frame(img, tile_bounds, missing=missing)
bare = pictures.tile_frame(pictures.terrain(base), tile_bounds, missing=missing)
stroke = canvas.map_canvas(img, terrain=bare, grid=GRID, height=MAP_H,
                           bounds=tile_bounds, valid_cells=valid_cells,
                           legend=legend, peek=peek,
                           tool="locked" if not editable else tool, key=f"canvas_{name}")
if stroke:
    cells = [c for c in stroke["cells"] if 0 <= c < GRID * GRID and valid_cells.flat[c]]
    stroke = dict(stroke, cells=cells) if cells else None

# A stroke arrives once; the id guards against applying it twice.
if stroke and stroke.get("id") != ss.get("last_stroke") and not editable:
    ss.last_stroke = stroke["id"]
    st.toast("This map is not editable.")
elif stroke and stroke.get("id") != ss.get("last_stroke"):
    ss.last_stroke = stroke["id"]
    hit = np.zeros(GRID * GRID, bool)
    hit[np.asarray(stroke["cells"], int)] = True
    hit = np.asarray(widen_stroke(hit.reshape(GRID, GRID)), dtype=bool)
    hit &= valid_cells
    code = PAINT[unit]
    if tool == "smart" and code:
        seed = divmod(int(stroke["cells"][0]), GRID)
        region = (core.smart_fill(suggested, seed=seed) if stroke.get("click") else
                  core.smart_fill(suggested, inside=hit, unit=code))
        region &= (hand == 0) & valid_cells      # never over hand labels or missing terrain
        if region.any():
            # Cells where you agree with the model are accepted; where you relabel its
            # region they are your correction, which Update learns from.
            step(np.where(region, code, ss.cells).astype(np.uint8),
                 (ss.accepted & ~region) | (region & (suggested == code)), sure_of(region))
            st.rerun()
        elif not stroke.get("click"):
            st.toast(f"The model predicts no unpainted {unit} in that area.")
        elif suggested[seed] == 0:
            st.toast("There is no prediction at that spot.")
        else:
            st.toast("That region is already painted by hand.")
    else:
        # Painted over, an accepted prediction becomes your own label.
        step(np.where(hit, code, ss.cells).astype(np.uint8), ss.accepted & ~hit,
             sure_of(hit) if code else np.where(hit, 0, ss.confidence).astype(np.uint8))
        st.rerun()

if save and can_save:
    labels_path = core.output_file(name, "labels.tif")
    prediction_path = core.output_file(name, "map.tif")
    accepted_path = core.output_file(name, "accepted.npy")
    valid = np.isfinite(X[0])
    lab = core.cells_to_labels(ss.cells, shape, valid=valid)
    full_prediction = None
    if ss.assistant["model"] is not None:
        with st.spinner("Predicting over the whole map..."):
            full_prediction = run(assistant_model.predict, ss.assistant, X, names)
    core.write_map(labels_path, lab, prof)
    core.save_painting(name, ss.cells, final=True, accepted=ss.accepted,
                       confidence=ss.confidence)
    if full_prediction is not None:
        core.write_map(prediction_path, full_prediction, prof)
    elif os.path.exists(prediction_path):
        os.remove(prediction_path)            # do not leave an obsolete prediction on save
    areas = {CODES[c]: round(float((lab == c).sum()) * px * px / 1e6, 3) for c in CODES}
    thumb = pictures.blacken(pictures.render(hs, [(ss.pred, 0.6), (ss.cells, 0.6)]),
                             ~terrain_valid)
    thumb.thumbnail((360, 360))
    thumb.save(core.output_file(name, "thumb.png"))
    core.meta_set(name, verified=bool(verified), dem=dem, inputs=names, model=core.MODEL,
                  model_schema=ss.assistant["schema"],
                  site_id=core.site_id(dem),
                  model_revision=ss.assistant["revision"],
                  prediction_saved=full_prediction is not None,
                  corrections_pending_update=bool(pending),
                  saved_at=datetime.datetime.now().isoformat(timespec="seconds"),
                  cells={CODES[c]: int((ss.cells == c).sum()) for c in CODES},
                  cells_accepted={CODES[c]: int(((ss.cells == c) & ss.accepted).sum())
                                  for c in CODES},
                  cells_by_confidence={core.CONFIDENCE[k][0]: int((sure == k).sum())
                                       for k in core.CONFIDENCE},
                  painted_area_km2=areas,
                  painted_frac=round(float((lab != 255).sum()) / max(1, int(valid.sum())), 4))
    st.success(("Saved and marked **verified**" if verified else "Saved") +
               f" — `{labels_path}` (your painting"
               + (f", with accepted predictions marked in `{os.path.basename(accepted_path)}`)"
                  if ss.accepted.any() else ")") +
               (f" and `{prediction_path}` (the prediction)" if full_prediction is not None else
                ". A prediction can be saved after training the first model."))
