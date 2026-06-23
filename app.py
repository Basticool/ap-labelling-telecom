"""Norm AP Labeling UI — entry point.

Set APP_MODE below:
  "simple"     — single labeler, no login, all norms accessible directly.
  "multi_user" — login required; admin allocates norms to users via job system.
"""
import streamlit as st
from app.startup import run_startup

# ── Configuration ──────────────────────────────────────────────────────────────
APP_MODE = "multi_user"

st.set_page_config(
    page_title="Norm AP Labeling",
    layout="wide",
    initial_sidebar_state="expanded",
)

run_startup(APP_MODE)

# ── Page routing ───────────────────────────────────────────────────────────────
from app.pages import admin, export, iaa, labeling, login  # noqa: E402

if APP_MODE == "multi_user" and not st.session_state.get("username"):
    pg = st.navigation(
        [st.Page(login.render, title="Login", url_path="login", default=True)]
    )
else:
    if APP_MODE == "multi_user":
        is_admin = st.session_state.get("username") == "admin"
        pages = [
            st.Page(labeling.render, title="Label", url_path="label", default=True),
        ]
        if is_admin:
            pages += [
                st.Page(admin.render, title="Admin", url_path="admin"),
                st.Page(iaa.render, title="IAA", url_path="iaa"),
                st.Page(export.render, title="Export", url_path="export"),
            ]
    else:
        pages = [
            st.Page(labeling.render, title="Label", url_path="label", default=True),
            st.Page(iaa.render, title="IAA", url_path="iaa"),
            st.Page(export.render, title="Export", url_path="export"),
        ]
    pg = st.navigation(pages)

pg.run()
