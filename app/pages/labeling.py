"""Main labeling interface — prediction-review mode.

Layout
------
Sidebar : chunk selector with per-chunk progress badges.
Main    : one bordered container per message, showing full content on the left
          and prop checkboxes on the right → Save & Next button.

For each observation-type AP the model's prediction is shown as the
pre-selected radio value.  The annotator confirms it or flips it.
Auto-labeled APs (tool_call / tool_result / structural) are shown as
locked checkboxes only when True.
"""
from __future__ import annotations

import json

import streamlit as st

from app.config import JOBS_DIR, LABELS_DIR
from app.modules.job_manager import (
    cleanup_empty_and_completed_jobs,
    get_completed_conv_ids_job,
    get_completed_conv_ids_simple,
    get_job_conv_ids_filter_for_chunk,
    get_job_units,
    get_user_jobs,
    save_simple_label,
    save_unit_labels,
)
from app.modules.storage import now_iso, read_jsonl


# ── Display helpers ────────────────────────────────────────────────────────────

def _role_badge(role: str) -> str:
    return {"assistant": "🤖 assistant", "user": "👤 user", "tool": "🔧 tool"}.get(
        role, role
    )


def _short_prop(prop_id: str) -> str:
    for prefix in ("agent_called_", "agent_", "auth_tool_", "user_", "order_"):
        if prop_id.startswith(prefix):
            return prop_id[len(prefix):]
    return prop_id


def _render_message_content(msg: dict) -> None:
    role = msg.get("role", "")
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

    if role == "assistant" and tool_calls:
        for tc in tool_calls:
            name = tc.get("name", "?")
            args = tc.get("arguments") or {}
            st.code(
                f"{name}(\n"
                + ",\n".join(f"  {k} = {json.dumps(v)}" for k, v in args.items())
                + "\n)",
                language="python",
            )
        if content:
            st.write(content)
    elif role == "tool":
        try:
            parsed = json.loads(content)
            st.json(parsed, expanded=False)
        except (json.JSONDecodeError, TypeError):
            st.text(content)
    else:
        st.write(content or "*(empty)*")


# ── State helpers ──────────────────────────────────────────────────────────────

def _chk_key(chunk_id: str, conv_id: str, msg_idx: int, prop_id: str) -> str:
    return f"chk_{chunk_id}_{conv_id}_{msg_idx}_{prop_id}"


def _get_completed_ids(chunk_id: str, app_mode: str, ss: dict) -> set[str]:
    if app_mode == "multi_user":
        jobs = get_user_jobs(ss.get("username", ""), JOBS_DIR)
        for job in jobs:
            if chunk_id in job.get("chunk_ids", []):
                return get_completed_conv_ids_job(job["job_id"], chunk_id, JOBS_DIR)
        return set()
    return get_completed_conv_ids_simple(LABELS_DIR, chunk_id)


def _save_labels(
    chunk_id: str, conv_id: str, turns: list[dict], app_mode: str, ss: dict
) -> None:
    if app_mode == "multi_user":
        jobs = get_user_jobs(ss.get("username", ""), JOBS_DIR)
        for job in jobs:
            if chunk_id in job.get("chunk_ids", []):
                save_unit_labels(
                    job["job_id"], conv_id, chunk_id, turns,
                    ss.get("username", ""), JOBS_DIR,
                )
                return
    else:
        save_simple_label(LABELS_DIR, chunk_id, conv_id, turns)


def _get_saved_turns(chunk_id: str, conv_id: str, app_mode: str, ss: dict) -> list[dict]:
    if app_mode == "multi_user":
        for job in get_user_jobs(ss.get("username", ""), JOBS_DIR):
            if chunk_id in job.get("chunk_ids", []):
                for unit in get_job_units(job["job_id"], JOBS_DIR):
                    if unit["conv_id"] == conv_id and unit["chunk_id"] == chunk_id:
                        return unit.get("turns", [])
        return []
    for rec in read_jsonl(LABELS_DIR / f"{chunk_id}.jsonl"):
        if rec.get("conv_id") == conv_id:
            return rec.get("turns", [])
    return []


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    ss = st.session_state
    if ss.pop("scroll_to_top", False):
        st.components.v1.html(
            "<script>window.parent.document.querySelector("
            "'section[data-testid=\"stMain\"]').scrollTo(0, 0);</script>",
            height=0,
        )

    chunks: list[dict] = ss["chunks"]
    chunk_conv_map: dict = ss["chunk_conv_map"]
    propositions: dict = ss["propositions"]
    auto_prop_ids: set = ss["auto_prop_ids"]
    obs_prop_ids: set = ss["obs_prop_ids"]
    merged_conv_preds: dict = ss["merged_conv_preds"]
    app_mode: str = ss.get("app_mode", "simple")

    if not chunks:
        st.warning("No conversations found in the dataset.")
        return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Norm AP Labeler")

        available_chunks = chunks

        if app_mode == "multi_user":
            st.caption(f"Logged in as **{ss.get('username')}**")
            if st.button("Logout", key="logout_btn"):
                del ss["username"]
                st.rerun()
            st.divider()

            jobs = get_user_jobs(ss.get("username", ""), JOBS_DIR)
            assigned_chunk_ids: set[str] = set()
            for job in jobs:
                assigned_chunk_ids.update(job.get("chunk_ids", []))
            available_chunks = [c for c in chunks if c["chunk_id"] in assigned_chunk_ids]
            if not available_chunks:
                st.info("No chunks assigned to you yet. Ask an admin.")
                return

        available_chunk_ids = [c["chunk_id"] for c in available_chunks]

        def _chunk_label(cid: str) -> str:
            chunk = next((c for c in available_chunks if c["chunk_id"] == cid), None)
            label = chunk["label"] if chunk else cid
            convs = chunk_conv_map.get(cid, [])
            n_total = len(convs)
            completed = _get_completed_ids(cid, app_mode, ss)
            n_done = len(completed)
            icon = "✓" if n_done >= n_total else ("◑" if n_done > 0 else "○")
            return f"{icon} {label}  ({n_done}/{n_total})"

        _current = ss.get("_active_chunk")
        if _current not in available_chunk_ids:
            _current = available_chunk_ids[0]
        chunk_idx = available_chunk_ids.index(_current)

        selected_chunk_id = st.radio(
            "Chunks:",
            available_chunk_ids,
            index=chunk_idx,
            format_func=_chunk_label,
        )

        if selected_chunk_id != ss.get("_active_chunk"):
            ss["_active_chunk"] = selected_chunk_id

    chunk_id: str = selected_chunk_id  # type: ignore[assignment]
    convs = chunk_conv_map.get(chunk_id, [])

    # For overlap jobs, restrict to the specific conversations assigned to this user
    if app_mode == "multi_user":
        _conv_filter = get_job_conv_ids_filter_for_chunk(ss.get("username", ""), chunk_id, JOBS_DIR)
        if _conv_filter is not None:
            _filter_set = set(_conv_filter)
            convs = [c for c in convs if c.get("convID", "") in _filter_set]

    n_total = len(convs)

    # ── Chunk header ───────────────────────────────────────────────────────────
    chunk_meta = next((c for c in chunks if c["chunk_id"] == chunk_id), {})
    st.subheader(f"Chunk: `{chunk_id}`")

    # ── Find next pending conversation ─────────────────────────────────────────
    completed_ids = _get_completed_ids(chunk_id, app_mode, ss)
    pending = [
        (i, c) for i, c in enumerate(convs)
        if c.get("convID", "") not in completed_ids
    ]

    n_done = n_total - len(pending)
    st.progress(n_done / n_total if n_total else 1.0, text=f"{n_done}/{n_total} conversations labeled")

    _view_key = f"_view_conv_idx_{chunk_id}"
    view_idx = ss.get(_view_key)

    if not pending and view_idx is None:
        st.success(f"All {n_total} conversations for **{chunk_id}** are labeled.")
        return

    if view_idx is not None:
        conv_pos = max(0, min(view_idx, n_total - 1))
        conv = convs[conv_pos]
        is_reviewing = conv.get("convID", "") in completed_ids
    else:
        conv_pos, conv = pending[0]
        is_reviewing = False

    conv_id = conv.get("convID", "")
    messages = conv.get("conversation", [])

    # ── Conversation header ────────────────────────────────────────────────────
    review_badge = "  *(reviewing — already labeled)*" if is_reviewing else ""
    st.markdown(
        f"**Conversation {conv_pos + 1} of {n_total}**{review_badge}"
        f"&nbsp;|&nbsp; `{conv_id}`"
    )
    source = conv.get("source", "")
    if source:
        st.caption(f"Source: {source}")

    # ── Build display message list (skip system) ───────────────────────────────
    conv_preds: dict[int, dict[str, bool]] = merged_conv_preds.get(conv_id, {})
    display_msgs = [(i, m) for i, m in enumerate(messages) if m.get("role") != "system"]

    if not display_msgs:
        st.warning("No displayable messages in this conversation.")
        if st.button("Skip conversation"):
            _save_labels(chunk_id, conv_id, [], app_mode, ss)
            st.rerun()
        return

    # ── Pre-fill saved labels when reviewing a completed conversation ──────────
    if is_reviewing:
        saved_turns = _get_saved_turns(chunk_id, conv_id, app_mode, ss)
        for saved_turn in saved_turns:
            orig_i = saved_turn.get("msg_idx")
            if orig_i is None:
                continue
            for prop_id, val in saved_turn.get("ap_labels", {}).items():
                if prop_id in obs_prop_ids:
                    key = _chk_key(chunk_id, conv_id, orig_i, prop_id)
                    if key not in ss:
                        ss[key] = "yes" if val in (True, "yes") else "no"

    # ── Column header legend ───────────────────────────────────────────────────
    st.caption(
        "Left: full message content. "
        "Right: observation APs pre-filled with deepseek's prediction — confirm or override. "
        "🔒 props are auto-labeled ground truth (not editable)."
    )

    # ── Per-message labeling rows ──────────────────────────────────────────────
    for i, (orig_i, msg) in enumerate(display_msgs):
        role = msg.get("role", "")
        turn_preds: dict[str, bool] = conv_preds.get(orig_i, {})

        with st.container(border=True):
            col_content, col_checks = st.columns([3, 2])

            with col_content:
                st.caption(f"Turn {orig_i} · {_role_badge(role)}")
                _render_message_content(msg)

            with col_checks:
                turn_auto_props = sorted(p for p in turn_preds if p in auto_prop_ids and turn_preds[p])
                turn_manual_props = sorted(
                    p for p in turn_preds if p in obs_prop_ids and turn_preds[p] is not None
                )

                # Auto-labeled props — grey locked display, only shown when True
                if turn_auto_props:
                    st.caption("🔒 auto-labeled (ground truth):")
                    for prop_id in turn_auto_props:
                        st.checkbox(
                            _short_prop(prop_id),
                            value=True,
                            disabled=True,
                            help=prop_id,
                            key=_chk_key(chunk_id, conv_id, orig_i, prop_id),
                        )

                # Observation props — editable, pre-filled from deepseek
                if turn_manual_props:
                    if turn_auto_props:
                        st.divider()
                    st.caption("Review deepseek predictions:")
                    for prop_id in turn_manual_props:
                        model_val = turn_preds[prop_id]
                        badge = "🟢 deepseek: yes" if model_val is True else "🔴 deepseek: no"

                        chk_key = _chk_key(chunk_id, conv_id, orig_i, prop_id)
                        if chk_key not in ss:
                            ss[chk_key] = "yes" if model_val else "no"

                        st.radio(
                            f"{_short_prop(prop_id)}  *{badge}*",
                            options=["no", "yes"],
                            horizontal=True,
                            help=prop_id + " — " + propositions.get(prop_id, {}).get("description", ""),
                            key=chk_key,
                        )

    # ── Action buttons ─────────────────────────────────────────────────────────
    st.divider()
    col_prev, col_save, col_skip, _ = st.columns([1, 1, 1, 5])
    with col_prev:
        if st.button("← Prev", key=f"prev_{chunk_id}_{conv_id}", disabled=conv_pos == 0):
            ss[_view_key] = conv_pos - 1
            ss["scroll_to_top"] = True
            st.rerun()

    with col_save:
        if st.button("Save & Next ▶", type="primary", key=f"save_{chunk_id}_{conv_id}"):
            turns = []
            for i, (orig_i, msg) in enumerate(display_msgs):
                turn_preds_save = conv_preds.get(orig_i, {})
                ap_labels: dict[str, bool] = {}
                for prop_id, val in turn_preds_save.items():
                    if prop_id in auto_prop_ids:
                        ap_labels[prop_id] = bool(val)
                    elif prop_id in obs_prop_ids and val is not None:
                        raw = ss.get(_chk_key(chunk_id, conv_id, orig_i, prop_id), "no")
                        ap_labels[prop_id] = raw in (True, "yes")
                deepseek_labels = {
                    p: bool(v)
                    for p, v in turn_preds_save.items()
                    if p in obs_prop_ids and v is not None
                }
                turns.append({
                    "msg_idx": orig_i,
                    "role": msg.get("role", ""),
                    "ap_labels": ap_labels,
                    "deepseek_labels": deepseek_labels,
                })
            _save_labels(chunk_id, conv_id, turns, app_mode, ss)
            if app_mode == "multi_user":
                cleanup_empty_and_completed_jobs(JOBS_DIR)
            ss.pop(_view_key, None)
            ss["scroll_to_top"] = True
            st.rerun()

    with col_skip:
        if st.button("Skip ▷", key=f"skip_{chunk_id}_{conv_id}"):
            _save_labels(chunk_id, conv_id, [], app_mode, ss)
            ss.pop(_view_key, None)
            st.rerun()
