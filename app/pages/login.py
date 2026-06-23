"""Username-only login page for multi-user mode.

Flow
----
Known user with jobs      → log in immediately.
Known user with no jobs   → shown unclaimed bundles to claim; can also skip.
Unknown user              → shown unclaimed bundles; must claim one to register.
"""
from __future__ import annotations

import streamlit as st

from app.config import JOBS_DIR, USERS_FILE
from app.modules.job_manager import (
    claim_bundle,
    get_job_units,
    get_unclaimed_bundles,
    get_user_jobs,
)
from app.modules.storage import append_jsonl, now_iso, read_jsonl


def _load_users() -> list[str]:
    return [r["username"] for r in read_jsonl(USERS_FILE)]


def _ensure_admin() -> None:
    if "admin" not in _load_users():
        append_jsonl(USERS_FILE, {"username": "admin", "created_at": now_iso()})


def _register_user(username: str) -> None:
    append_jsonl(USERS_FILE, {"username": username, "created_at": now_iso()})


def _has_pending_work(username: str) -> bool:
    for job in get_user_jobs(username, JOBS_DIR):
        if any(u.get("unit_status") != "completed" for u in get_job_units(job["job_id"], JOBS_DIR)):
            return True
    return False


def _render_bundle_picker(username: str, is_new_user: bool) -> None:
    """Render the bundle selection screen."""
    ss = st.session_state
    chunk_conv_map: dict = ss.get("chunk_conv_map", {})
    bundles = get_unclaimed_bundles(JOBS_DIR)

    if is_new_user:
        st.title("Norm AP Labeling — Register")
        st.markdown(
            f"Username **{username}** is not registered yet. "
            "Claim a bundle below to create your account."
        )
    else:
        st.title("Norm AP Labeling — Claim a bundle")
        st.markdown(
            f"Welcome back, **{username}**! You have no norms assigned yet. "
            "Claim a bundle to get started, or skip if an admin will assign your norms."
        )

    if not bundles:
        st.warning("No bundles are available right now. Ask an admin to create one.")
        if not is_new_user:
            if st.button("Continue without claiming", key="skip_claim"):
                ss["username"] = username
                del ss["_pending_claim"]
                st.rerun()
        if st.button("← Back", key="back_no_bundles"):
            del ss["_pending_claim"]
            st.rerun()
        return

    bundle_options = {
        b["bundle_id"]: (
            f"{b.get('name', b['bundle_id'])}  —  "
            f"{len(b.get('chunk_ids', []))} chunk(s), "
            f"{b.get('n_convs', '?')} conversation(s)"
        )
        for b in bundles
    }
    selected_bid = st.radio(
        "Available bundles",
        list(bundle_options.keys()),
        format_func=lambda bid: bundle_options[bid],
    )

    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        label = "Claim & Register" if is_new_user else "Claim bundle"
        if st.button(label, type="primary", key="claim_btn"):
            try:
                if is_new_user:
                    _register_user(username)
                claim_bundle(selected_bid, username, chunk_conv_map, JOBS_DIR)
                ss["username"] = username
                del ss["_pending_claim"]
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with col2:
        if not is_new_user and st.button("Skip", key="skip_claim"):
            ss["username"] = username
            del ss["_pending_claim"]
            st.rerun()
    with col3:
        if st.button("← Back", key="back_btn"):
            del ss["_pending_claim"]
            st.rerun()


def render() -> None:
    _ensure_admin()
    ss = st.session_state

    # ── Bundle-claiming step ───────────────────────────────────────────────────
    if "_pending_claim" in ss:
        username, is_new_user = ss["_pending_claim"]
        _render_bundle_picker(username, is_new_user)
        return

    # ── Normal login step ──────────────────────────────────────────────────────
    st.title("Norm AP Labeling — Login")
    username = st.text_input("Username").strip()
    if st.button("Login", type="primary"):
        if not username:
            st.error("Enter a username.")
            return
        users = _load_users()
        if username not in users:
            ss["_pending_claim"] = (username, True)
        elif _has_pending_work(username):
            ss["username"] = username
        else:
            ss["_pending_claim"] = (username, False)
        st.rerun()
