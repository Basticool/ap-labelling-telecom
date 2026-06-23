"""Export labeled data in the final_submission/data/annotations format.

Produces three JSON files (downloadable individually or as a zip):

  conversations.json      {convID: conv_dict} — only labeled conversations
  ap_annotations.json     [{convID, msgIdx, labelerID, labels, metadata}]
  labelers.json           {labelerID: labeler_dict}

AP labels:
  - Observation APs  → attributed to the human annotator (UUID derived from username)
  - Auto APs         → attributed to the auto labeler (fixed UUID)

The labels field contains only the names of APs that are True at that turn,
matching the final_submission annotation format.
"""
from __future__ import annotations

import io
import json
import uuid
import zipfile
from collections import defaultdict

import pandas as pd
import streamlit as st

from app.config import JOBS_DIR, LABELS_DIR
from app.modules.job_manager import get_all_jobs, get_job_units
from app.modules.storage import read_jsonl

# Fixed UUID for the deterministic auto-labeler (matches final_submission/labelers.json)
_AUTO_LABELER_ID = "122ca4fc-e786-5446-a1ad-7302961a02be"
_AUTO_LABELER_ENTRY = {
    "labelerID": _AUTO_LABELER_ID,
    "name": "auto",
    "type": "auto",
    "metadata": {
        "script": "auto_labeler.py",
        "ap_kinds_handled": ["tool_call", "tool_result", "structural"],
        "deterministic": True,
    },
}

# Namespace for deriving stable labelerIDs from usernames
_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def _human_labeler_id(username: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, f"norm-labeling/{username}"))


def _collect_units(app_mode: str) -> list[dict]:
    if app_mode == "simple":
        units = []
        chunks: list[dict] = st.session_state.get("chunks", [])
        for chunk in chunks:
            for rec in read_jsonl(LABELS_DIR / f"{chunk['chunk_id']}.jsonl"):
                if rec.get("unit_status") == "completed":
                    units.append(rec)
        return units
    units = []
    for job in get_all_jobs(JOBS_DIR):
        for unit in get_job_units(job["job_id"], JOBS_DIR):
            if unit.get("unit_status") == "completed":
                units.append(unit)
    return units


def _build_ap_annotations(units: list[dict]) -> list[dict]:
    """Flatten units into per-(convID, msgIdx, labelerID) annotation records.

    Auto-labeled APs and human-reviewed APs are emitted as separate records
    with their respective labelerIDs.  The labels field contains only the
    names of APs that are True, matching the final_submission schema.
    """
    # Accumulate per (conv_id, msg_idx, labeler_id) → set of true AP names
    # Use two passes: one for auto, one for human.
    # Key: (conv_id, msg_idx, labeler_id) → sorted list of true AP names
    buckets: dict[tuple[str, int, str], set[str]] = defaultdict(set)

    for unit in units:
        conv_id = unit.get("conv_id", "")
        labeled_by = unit.get("labeled_by", "unknown")
        labeled_at = unit.get("labeled_at", "")
        human_lid = _human_labeler_id(labeled_by) if labeled_by != "auto" else _AUTO_LABELER_ID

        for turn in unit.get("turns", []):
            msg_idx: int = turn.get("msg_idx", -1)
            if msg_idx < 0:
                continue
            auto_props = set(turn.get("auto_labeled_props", []))
            ap_labels: dict = turn.get("ap_labels", {})

            for prop_id, val in ap_labels.items():
                is_true = val in (True, "yes")
                if prop_id in auto_props:
                    if is_true:
                        buckets[(conv_id, msg_idx, _AUTO_LABELER_ID)].add(prop_id)
                else:
                    if is_true:
                        buckets[(conv_id, msg_idx, human_lid)].add(prop_id)
                    else:
                        # Ensure the bucket exists even for turns with no true obs APs
                        buckets.setdefault((conv_id, msg_idx, human_lid), set())

    records = []
    for (conv_id, msg_idx, labeler_id), true_aps in sorted(buckets.items()):
        records.append({
            "convID": conv_id,
            "msgIdx": msg_idx,
            "labelerID": labeler_id,
            "labels": sorted(true_aps),
            "metadata": {},
        })
    return records


def _build_conversations_json(units: list[dict], all_conversations: dict) -> dict:
    labeled_conv_ids = {u.get("conv_id", "") for u in units}
    return {
        cid: conv
        for cid, conv in all_conversations.items()
        if cid in labeled_conv_ids
    }


def _build_labelers_json(units: list[dict]) -> dict:
    labelers: dict[str, dict] = {_AUTO_LABELER_ID: _AUTO_LABELER_ENTRY}
    for unit in units:
        lb = unit.get("labeled_by", "")
        if lb and lb != "auto":
            lid = _human_labeler_id(lb)
            if lid not in labelers:
                labelers[lid] = {
                    "labelerID": lid,
                    "name": lb,
                    "type": "human",
                    "metadata": {},
                }
    return labelers


def render() -> None:
    app_mode = st.session_state.get("app_mode", "simple")
    st.title("Export labels")

    units = _collect_units(app_mode)

    if not units:
        st.info("No completed labels to export yet.")
        return

    # Summary table
    by_chunk: dict[str, int] = defaultdict(int)
    for r in units:
        by_chunk[r.get("chunk_id", r.get("norm_id", "?"))] += 1
    st.write(f"**{len(units)}** completed units across **{len(by_chunk)}** chunk(s).")
    st.dataframe(
        pd.DataFrame(
            [{"chunk_id": k, "labeled_conversations": v} for k, v in sorted(by_chunk.items())]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    st.subheader("Download")

    all_conversations: dict = st.session_state.get("conversations", {})
    ap_annotations = _build_ap_annotations(units)
    conversations_out = _build_conversations_json(units, all_conversations)
    labelers_out = _build_labelers_json(units)

    st.write(
        f"**{len(ap_annotations)}** AP annotation records · "
        f"**{len(conversations_out)}** conversations · "
        f"**{len(labelers_out)}** labelers"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "ap_annotations.json",
            data=json.dumps(ap_annotations, ensure_ascii=False, indent=2).encode(),
            file_name="ap_annotations.json",
            mime="application/json",
        )
    with col2:
        st.download_button(
            "conversations.json",
            data=json.dumps(conversations_out, ensure_ascii=False, indent=2).encode(),
            file_name="conversations.json",
            mime="application/json",
        )
    with col3:
        st.download_button(
            "labelers.json",
            data=json.dumps(labelers_out, ensure_ascii=False, indent=2).encode(),
            file_name="labelers.json",
            mime="application/json",
        )
    with col4:
        # All three files bundled as a zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ap_annotations.json", json.dumps(ap_annotations, ensure_ascii=False, indent=2))
            zf.writestr("conversations.json", json.dumps(conversations_out, ensure_ascii=False, indent=2))
            zf.writestr("labelers.json", json.dumps(labelers_out, ensure_ascii=False, indent=2))
        st.download_button(
            "Download all (.zip)",
            data=buf.getvalue(),
            file_name="norm_ap_labels.zip",
            mime="application/zip",
            type="primary",
        )
