"""Admin panel — multi-user mode only.

Tabs
----
Users       : add / remove labeler accounts.
Allocate    : assign norms to users (creates a job per assignment).
Jobs        : view all jobs, inspect progress, delete jobs.
Bundles     : create/assign bundles, including overlap bundles for IAA study.
"""
from __future__ import annotations

import streamlit as st

from app.config import JOBS_DIR, USERS_FILE
from app.modules.job_manager import (
    claim_bundle,
    create_bundle,
    create_job,
    delete_bundle,
    delete_job,
    get_all_bundles,
    get_all_jobs,
    get_completed_conv_ids_job,
    get_job_units,
    update_job_status,
)
from app.modules.storage import append_jsonl, clear_cache, now_iso, read_jsonl, write_jsonl

# ── Overlap bundle plan ────────────────────────────────────────────────────────
# (name, norm_id, n_take, original_labeler)
_OVERLAP_PLAN = [
    ("overlap-01  N0-cancel",            "N0-cancel",            12, "Louise"),
    ("overlap-02  N0-exchange",           "N0-exchange",          12, "Louise"),
    ("overlap-03  N0-return",             "N0-return",            12, "Louise"),
    ("overlap-04  N0-modify-order-items", "N0-modify-order-items",12, "Louise"),
    ("overlap-05  N1-cancel",             "N1-cancel",            12, "Louise"),
    ("overlap-06  N3-return",             "N3-return",            12, "leif"),
    ("overlap-07  N2-cancel",             "N2-cancel",            12, "Bastien"),
    ("overlap-08  glm_cancel",            "glm_cancel",           12, "Bastien"),
    ("overlap-09  glm_cancel+return",     "glm_cancel+return",    12, "leif"),
    ("overlap-10  glm_mod_user_addr",     "glm_mod_user_addr",    12, "Anuj"),
]

_OVERLAP_EXCLUDE = {"Louise", "batch1", "batch5", "admin"}

_SAME_PERSON: list[tuple[str, frozenset[str]]] = [
    ("Bastien", frozenset({"Bastien", "bastien", "Basti"})),
]
_CANONICAL  = {a: c for c, aliases in _SAME_PERSON for a in aliases}
_ALIAS_GROUP = {a: aliases for c, aliases in _SAME_PERSON for a in aliases}


def _conv_labeler_from_jobs() -> dict[str, str]:
    """Build {conv_id: labeled_by} from all completed job units."""
    result: dict[str, str] = {}
    for job in get_all_jobs(JOBS_DIR):
        for unit in get_job_units(job["job_id"], JOBS_DIR):
            cid = unit.get("conv_id", "")
            if unit.get("unit_status") == "completed" and unit.get("labeled_by") and cid:
                result[cid] = unit["labeled_by"]
    return result


def _compute_overlap_eligible(
    conv_ids: list[str],
    original_labeler: str,
    all_users: list[str],
    conv_labeler: dict[str, str],
) -> list[str]:
    conv_set = set(conv_ids)
    orig_aliases = _ALIAS_GROUP.get(original_labeler, frozenset({original_labeler}))
    excluded = _OVERLAP_EXCLUDE | orig_aliases
    seen: set[str] = set()
    eligible: list[str] = []
    for user in sorted(all_users):
        if user in excluded:
            continue
        person_aliases = _ALIAS_GROUP.get(user, frozenset({user}))
        if any(conv_labeler.get(c) in person_aliases for c in conv_set):
            continue
        canon = _CANONICAL.get(user, user)
        if canon not in seen:
            seen.add(canon)
            eligible.append(canon)
    return eligible


# ── User helpers ───────────────────────────────────────────────────────────────

def _load_users() -> list[str]:
    return [r["username"] for r in read_jsonl(USERS_FILE)]


def _add_user(username: str) -> str | None:
    users = _load_users()
    if username in users:
        return f"User **{username}** already exists."
    append_jsonl(USERS_FILE, {"username": username, "created_at": now_iso()})
    return None


def _remove_user(username: str) -> str | None:
    if username == "admin":
        return "Cannot remove admin."
    jobs = [j for j in get_all_jobs(JOBS_DIR) if j["username"] == username and j["status"] != "completed"]
    if jobs:
        return f"User **{username}** has {len(jobs)} active job(s). Delete them first."
    records = read_jsonl(USERS_FILE)
    write_jsonl(USERS_FILE, [r for r in records if r["username"] != username])
    return None


# ── Render ─────────────────────────────────────────────────────────────────────

def render() -> None:
    if st.session_state.get("username") != "admin":
        st.error("Admin access only.")
        return

    st.title("Admin")
    tab_users, tab_alloc, tab_jobs, tab_bundles = st.tabs(["Users", "Allocate Chunks", "Jobs", "Bundles"])

    # ── Users tab ─────────────────────────────────────────────────────────────
    with tab_users:
        st.subheader("Manage users")
        users = _load_users()
        st.write(f"**{len(users)}** registered users: {', '.join(users)}")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            new_user = st.text_input("New username").strip()
            if st.button("Add user", key="add_user"):
                if not new_user:
                    st.error("Enter a username.")
                elif err := _add_user(new_user):
                    st.error(err)
                else:
                    st.success(f"Added **{new_user}**.")
                    st.rerun()

        with col2:
            del_user = st.selectbox(
                "Remove user",
                [u for u in users if u != "admin"],
                key="del_user_sel",
            )
            if del_user and st.button("Remove user", key="rem_user"):
                if err := _remove_user(del_user):
                    st.error(err)
                else:
                    st.success(f"Removed **{del_user}**.")
                    st.rerun()

    # ── Allocate tab ──────────────────────────────────────────────────────────
    with tab_alloc:
        st.subheader("Assign chunks to a user")
        chunks: list[dict] = st.session_state.get("chunks", [])
        chunk_conv_map: dict = st.session_state.get("chunk_conv_map", {})
        non_admin_users = [u for u in _load_users() if u != "admin"]

        if not non_admin_users:
            st.info("No labeler accounts yet. Create users in the Users tab.")
        else:
            target_user = st.selectbox("Assign to user", non_admin_users, key="alloc_user")

            existing_jobs = get_all_jobs(JOBS_DIR)
            already_assigned: set[str] = set()
            for j in existing_jobs:
                if j["username"] == target_user:
                    already_assigned.update(j.get("chunk_ids", []))

            available_chunk_ids = [c["chunk_id"] for c in chunks if c["chunk_id"] not in already_assigned]
            to_assign = st.multiselect(
                "Chunks to assign",
                available_chunk_ids,
                format_func=lambda cid: next((c["label"] for c in chunks if c["chunk_id"] == cid), cid),
                key="alloc_chunks",
            )

            if to_assign:
                n_convs = sum(len(chunk_conv_map.get(c, [])) for c in to_assign)
                st.caption(f"This will create a job with {n_convs} conversation(s) total.")
            if st.button("Create job", disabled=not to_assign, key="create_job"):
                job_id = create_job(target_user, to_assign, chunk_conv_map, JOBS_DIR)
                st.success(f"Job `{job_id}` created for **{target_user}**: {', '.join(to_assign)}")
                st.rerun()

            if already_assigned:
                st.caption(f"Already assigned to **{target_user}**: {', '.join(sorted(already_assigned))}")

    # ── Jobs tab ──────────────────────────────────────────────────────────────
    with tab_jobs:
        st.subheader("All jobs")
        all_jobs = get_all_jobs(JOBS_DIR)
        if not all_jobs:
            st.info("No jobs yet.")
        else:
            for job in all_jobs:
                job_id = job["job_id"]
                units = get_job_units(job_id, JOBS_DIR)
                total_u = len(units)
                done_u = sum(1 for u in units if u["unit_status"] == "completed")
                update_job_status(job_id, JOBS_DIR)

                with st.expander(
                    f"`{job_id}` — {job['username']} | "
                    f"{done_u}/{total_u} units | {job.get('status', '?')}",
                    expanded=False,
                ):
                    st.write(f"**Chunks:** {', '.join(job.get('chunk_ids', []))}")
                    st.write(f"**Created:** {job.get('created_at', '?')}")

                    for chunk_id in job.get("chunk_ids", []):
                        chunk_units = [u for u in units if u["chunk_id"] == chunk_id]
                        n_done = sum(1 for u in chunk_units if u["unit_status"] == "completed")
                        st.write(f"  • `{chunk_id}`: {n_done}/{len(chunk_units)}")

                    if st.button(f"Delete job {job_id}", key=f"del_{job_id}"):
                        delete_job(job_id, JOBS_DIR)
                        st.warning(f"Deleted job `{job_id}`.")
                        st.rerun()

    # ── Bundles tab ───────────────────────────────────────────────────────────
    with tab_bundles:
        st.subheader("Bundles")
        st.caption(
            "Bundles are pools of norms that any user can claim on login. "
            "Once claimed, the bundle is locked to that user and a job is created automatically."
        )

        if st.button("↺ Refresh from GitHub", key="refresh_bundles"):
            clear_cache()
            st.rerun()

        chunks: list[dict] = st.session_state.get("chunks", [])
        chunk_conv_map: dict = st.session_state.get("chunk_conv_map", {})

        with st.expander("Create new bundle", expanded=False):
            bundle_name = st.text_input("Bundle name (optional label for the admin)", key="bundle_name")
            bundle_chunks = st.multiselect(
                "Chunks to include",
                [c["chunk_id"] for c in chunks],
                format_func=lambda cid: next((c["label"] for c in chunks if c["chunk_id"] == cid), cid),
                key="bundle_chunks",
            )
            if bundle_chunks:
                n_convs = sum(len(chunk_conv_map.get(c, [])) for c in bundle_chunks)
                st.caption(f"{len(bundle_chunks)} chunk(s) · {n_convs} conversation(s)")
            if st.button("Create bundle", disabled=not bundle_chunks, key="create_bundle"):
                bid = create_bundle(
                    bundle_name or f"Bundle {now_iso()[:10]}",
                    bundle_chunks,
                    chunk_conv_map,
                    JOBS_DIR,
                )
                st.success(f"Bundle `{bid}` created.")
                st.rerun()

        with st.expander("Create overlap bundles (inter-annotator agreement study)", expanded=True):
            st.caption(
                "Creates overlap bundles sampling from already human-labeled norms. "
                "Each bundle shows eligible labelers (those who haven't labeled "
                "those conversations before)."
            )
            existing_bundle_names = {b["name"] for b in get_all_bundles(JOBS_DIR)}
            pending = [p for p in _OVERLAP_PLAN if p[0] not in existing_bundle_names]

            norm_convs: dict = st.session_state.get("norm_convs", {})
            if not pending:
                st.success("All overlap bundles already exist. Assign them below.")
            else:
                st.write(f"**{len(pending)}** bundle(s) not yet created:")
                for name, norm_id, n_take, orig in pending:
                    available = len(chunk_conv_map.get(norm_id, norm_convs.get(norm_id, [])))
                    st.caption(f"  • {name}  ({norm_id}, {min(n_take, available)} convs, original: {orig})")

                if st.button("Create all missing overlap bundles", key="create_overlap", type="primary"):
                    all_users = _load_users()
                    conv_labeler = _conv_labeler_from_jobs()
                    created = 0
                    for name, norm_id, n_take, original_labeler in pending:
                        convs_for_chunk = chunk_conv_map.get(norm_id, norm_convs.get(norm_id, []))
                        conv_ids = sorted(
                            c.get("convID", "") for c in convs_for_chunk
                        )[:n_take]
                        eligible = _compute_overlap_eligible(
                            conv_ids, original_labeler, all_users, conv_labeler
                        )
                        create_bundle(
                            name,
                            [norm_id],
                            chunk_conv_map,
                            JOBS_DIR,
                            conv_ids_filter={norm_id: conv_ids},
                            eligible_labelers=eligible,
                            original_labeler=original_labeler,
                        )
                        created += 1
                    st.success(f"Created {created} overlap bundle(s).")
                    st.rerun()

        st.divider()

        # Purge legacy bundles (old norm_ids schema, no chunk_ids)
        from app.modules.storage import read_jsonl as _raw_read, write_jsonl as _raw_write
        _bundles_path = JOBS_DIR / "bundles.jsonl"
        _all_raw = _raw_read(_bundles_path)
        _legacy = [b for b in _all_raw if not b.get("chunk_ids")]
        if _legacy:
            st.warning(f"{len(_legacy)} legacy bundle(s) from an older dataset are hidden.")
            if st.button("Purge legacy bundles", key="purge_legacy"):
                _raw_write(_bundles_path, [b for b in _all_raw if b.get("chunk_ids")])
                st.success("Legacy bundles removed.")
                st.rerun()

        all_bundles = get_all_bundles(JOBS_DIR)
        if not all_bundles:
            st.info("No bundles yet.")
        else:
            for bundle in all_bundles:
                bid = bundle["bundle_id"]
                name = bundle.get("name", bid)
                n_convs = bundle.get("n_convs", "?")
                claimed_by = bundle.get("claimed_by")
                status = f"claimed by **{claimed_by}**" if claimed_by else "unclaimed"
                with st.expander(f"`{name}` — {status} · {n_convs} conversations", expanded=False):
                    st.write(f"**ID:** `{bid}`")
                    st.write(f"**Chunks:** {', '.join(bundle.get('chunk_ids', []))}")
                    st.write(f"**Created:** {bundle.get('created_at', '?')[:10]}")
                    if claimed_by:
                        st.write(f"**Claimed by:** {claimed_by} at {bundle.get('claimed_at', '?')[:10]}")
                        st.write(f"**Job:** `{bundle.get('job_id', '?')}`")
                    else:
                        orig = bundle.get("original_labeler")
                        eligible = bundle.get("eligible_labelers")
                        if orig:
                            st.write(f"**Original labeler:** {orig}")
                        if eligible:
                            st.write(f"**Eligible labelers:** {', '.join(eligible)}")

                        non_admin_users = [u for u in _load_users() if u != "admin"]
                        assignable = eligible if eligible else non_admin_users
                        if assignable:
                            col_sel, col_btn, col_del = st.columns([2, 2, 1])
                            with col_sel:
                                sel_user = st.selectbox(
                                    "Assign to",
                                    assignable,
                                    key=f"assign_sel_{bid}",
                                    label_visibility="collapsed",
                                )
                            with col_btn:
                                if st.button(f"Assign to {sel_user}", key=f"assign_{bid}", type="primary"):
                                    try:
                                        claim_bundle(bid, sel_user, chunk_conv_map, JOBS_DIR)
                                        st.success(f"Bundle `{name}` assigned to **{sel_user}**.")
                                        st.rerun()
                                    except ValueError as exc:
                                        st.error(str(exc))
                            with col_del:
                                if st.button("Delete", key=f"del_bundle_{bid}"):
                                    delete_bundle(bid, JOBS_DIR)
                                    st.warning(f"Deleted bundle `{bid}`.")
                                    st.rerun()
                        else:
                            if st.button(f"Delete bundle {bid}", key=f"del_bundle_{bid}"):
                                delete_bundle(bid, JOBS_DIR)
                                st.warning(f"Deleted bundle `{bid}`.")
                                st.rerun()
