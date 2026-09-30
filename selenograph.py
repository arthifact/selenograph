"""
Selenograph — a lunar geologic unit mapper.

  uv run --no-project streamlit run selenograph.py

Selenography is the mapping of the Moon's surface. Paint and review installed
processed datasets, browse bundled and saved annotations, and follow held-out
map evaluation. Source datasets remain unchanged.
"""
import streamlit as st
from pathlib import Path
from app import core, paths

st.set_page_config(page_title="Selenograph", layout="wide")
with core.catalog_snapshot():
    required = [paths.MODEL_DIR / f"{core.MODEL}.joblib", *map(Path, core.dem_files())]
    for path in required:
        if (path.is_file() and path.stat().st_size < 1024
                and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")):
            st.error("The dataset or model has not finished downloading. Run `git lfs pull` "
                     "in the project folder, then reload the app.")
            st.stop()
    st.navigation([
        st.Page("app/app_pages/paint.py", title="Paint", icon=":material/brush:", default=True),
        st.Page("app/app_pages/gallery.py", title="Gallery", icon=":material/photo_library:"),
        st.Page("app/app_pages/progress.py", title="Progress", icon=":material/trending_up:"),
    ], position="top").run()
