"""
Selenograph — a lunar geologic unit mapper.

  uv run --no-project streamlit run selenograph.py

Selenography is the mapping of the Moon's surface. Paint and review installed
processed datasets, browse bundled and saved annotations, and follow held-out
map evaluation and independent checks. Source datasets remain unchanged.
"""
import streamlit as st
from app import core

st.set_page_config(page_title="Selenograph", layout="wide")
with core.catalog_snapshot():
    st.navigation([
        st.Page("app/app_pages/paint.py", title="Paint", icon=":material/brush:", default=True),
        st.Page("app/app_pages/gallery.py", title="Gallery", icon=":material/photo_library:"),
        st.Page("app/app_pages/progress.py", title="Progress", icon=":material/trending_up:"),
    ], position="top").run()
