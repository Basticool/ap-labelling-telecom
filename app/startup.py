"""Load all data sources once per session and cache in st.session_state."""
from __future__ import annotations

import re
import streamlit as st


def _build_chunks(
    norm_convs: dict[str, list[dict]],
    norms_with_obs: set[str],
    conversations: dict,
) -> list[dict]:
    """
    If all norms with obs APs cover the same set of conversations (flat mode),
    generate sliding-window chunks of 30 traces with 5-trace overlap (step=25).
    Otherwise one chunk per norm (chunk_id == norm_id).
    """
    labeling_norms = {k: v for k, v in norm_convs.items() if k in norms_with_obs}
    conv_id_sets = [frozenset(c["convID"] for c in convs) for convs in labeling_norms.values() if convs]

    if len(set(conv_id_sets)) <= 1:
        # Flat: every norm covers the same traces — use sliding-window chunks
        all_conv_ids = sorted(conversations.keys())
        size, step = 30, 25
        chunks = []
        for i, start in enumerate(range(0, len(all_conv_ids), step)):
            ids = all_conv_ids[start:start + size]
            if not ids:
                break
            chunks.append({
                "chunk_id": f"chunk-{i:02d}",
                "label": f"Chunk {i:02d}  (traces {start + 1}–{start + len(ids)})",
                "conv_ids": ids,
            })
        return chunks

    # Norm-partitioned: each norm that needs obs labeling becomes one chunk
    return [
        {
            "chunk_id": norm_id,
            "label": norm_id,
            "conv_ids": [c["convID"] for c in convs],
        }
        for norm_id, convs in sorted(labeling_norms.items())
    ]


def run_startup(app_mode: str) -> None:
    if st.session_state.get("_startup_done"):
        return

    from app.config import (
        DEFAULT_CONVERSATIONS_PATH,
        DEFAULT_NORMS_PATH,
        DEFAULT_PREDICTIONS_PATH,
        DEFAULT_PROPS_PATH,
        JOBS_DIR,
        LABELS_DIR,
    )
    from app.modules.auto_labeler import build_auto_label_sensors, compute_auto_labels
    from app.modules.data_loader import (
        load_conversations,
        load_norms,
        load_predictions,
        load_propositions,
    )
    from app.modules.job_manager import cleanup_empty_and_completed_jobs
    from app.modules.storage import ensure_dir, now_iso, read_jsonl, write_jsonl

    ensure_dir(LABELS_DIR)
    ensure_dir(JOBS_DIR)
    cleanup_empty_and_completed_jobs(JOBS_DIR)

    with st.spinner("Loading data…"):
        conversations = load_conversations(DEFAULT_CONVERSATIONS_PATH)
        predictions = load_predictions(DEFAULT_PREDICTIONS_PATH)
        norms = load_norms(DEFAULT_NORMS_PATH)
        propositions = load_propositions(DEFAULT_PROPS_PATH)

    # Group conversations by normID based on which predictions exist.
    norm_conv_ids: dict[str, set[str]] = {}
    for norm_id, conv_map in predictions.items():
        norm_conv_ids[norm_id] = set(conv_map.keys())

    norm_convs: dict[str, list[dict]] = {
        norm_id: [
            conversations[cid]
            for cid in sorted(cid_set)
            if cid in conversations
        ]
        for norm_id, cid_set in norm_conv_ids.items()
    }

    # Classify props
    tool_call_props: dict[str, str] = {
        prop_id: defn["metadata"]["tool_name"]
        for prop_id, defn in propositions.items()
        if defn.get("metadata", {}).get("ap_kind") == "tool_call"
        and defn.get("metadata", {}).get("tool_name")
    }
    auto_prop_kinds = {"tool_call", "tool_result", "structural"}
    auto_prop_ids: set[str] = {
        prop_id
        for prop_id, defn in propositions.items()
        if defn.get("metadata", {}).get("ap_kind") in auto_prop_kinds
    }
    obs_prop_ids: set[str] = {
        prop_id
        for prop_id, defn in propositions.items()
        if defn.get("metadata", {}).get("ap_kind") == "observation"
    }

    known_props = set(propositions.keys())

    def _aps_for_norm(norm_id: str) -> list[str]:
        ap_set: set[str] = set()
        visited: set[str] = set()
        to_visit = [norm_id]
        while to_visit:
            nid = to_visit.pop()
            if nid in visited or nid not in norms:
                continue
            visited.add(nid)
            defn = norms[nid]
            for field in ("precondition", "obligation"):
                tokens = set(re.findall(r'\b[a-z][a-z0-9_]*\b', defn.get(field, "")))
                ap_set |= tokens & known_props
            if defn.get("reparative"):
                to_visit.append(defn["reparative"])
        return sorted(ap_set)

    norm_props: dict[str, list[str]] = {
        norm_id: _aps_for_norm(norm_id)
        for norm_id in norm_conv_ids
        if norm_id in norms
    }

    # Norms that have at least one observation prop → need human labeling
    norms_with_obs: set[str] = {
        norm_id
        for norm_id, props_list in norm_props.items()
        if any(p in obs_prop_ids for p in props_list)
    }

    # Merge predictions across all norms: {conv_id: {msg_idx: {ap: bool}}}
    # This lets the labeling UI show all APs per turn regardless of norm grouping.
    merged_conv_preds: dict[str, dict[int, dict[str, bool]]] = {}
    for norm_id, conv_map in predictions.items():
        for conv_id, msg_map in conv_map.items():
            for msg_idx, labels in msg_map.items():
                turn = merged_conv_preds.setdefault(conv_id, {}).setdefault(int(msg_idx), {})
                turn.update(labels)

    # Build chunks (flat sliding-window or one-per-norm)
    chunks = _build_chunks(norm_convs, norms_with_obs, conversations)
    chunk_conv_map: dict[str, list[dict]] = {
        chunk["chunk_id"]: [conversations[cid] for cid in chunk["conv_ids"] if cid in conversations]
        for chunk in chunks
    }

    sensors = build_auto_label_sensors(propositions)

    # Pre-compute auto-labels: {norm_id: {conv_id: [{prop_id: bool} per message]}}
    with st.spinner("Pre-computing auto-labels…"):
        norm_auto_labels: dict[str, dict[str, list[dict[str, bool]]]] = {}
        for norm_id, conv_list in norm_convs.items():
            props_for_norm = norm_props.get(norm_id, [])
            norm_sensors = {p: sensors[p] for p in props_for_norm if p in sensors}
            conv_labels: dict[str, list[dict[str, bool]]] = {}
            for conv in conv_list:
                conv_id = conv.get("convID", "")
                messages = conv.get("conversation", [])
                conv_labels[conv_id] = compute_auto_labels(messages, norm_sensors)
            norm_auto_labels[norm_id] = conv_labels

    # Auto-save norms where every prop is auto-labeled (no human review needed)
    fully_auto_norms = [
        norm_id for norm_id in norm_convs
        if norm_id in norm_props and norm_id not in norms_with_obs
    ]
    if fully_auto_norms:
        with st.spinner("Auto-saving pre-labeled norms…"):
            for norm_id in fully_auto_norms:
                props_for_norm = norm_props.get(norm_id, [])
                auto_props = [p for p in props_for_norm if p in auto_prop_ids]
                labels_path = LABELS_DIR / f"{norm_id}.jsonl"

                existing = read_jsonl(labels_path)
                completed_conv_ids = {
                    rec["conv_id"] for rec in existing
                    if rec.get("unit_status") == "completed"
                }

                new_records = []
                for conv in norm_convs[norm_id]:
                    conv_id = conv.get("convID", "")
                    if conv_id in completed_conv_ids:
                        continue
                    messages = conv.get("conversation", [])
                    auto_labels_for_conv = norm_auto_labels.get(norm_id, {}).get(conv_id, [])
                    turns = []
                    for orig_i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            continue
                        msg_auto = (
                            auto_labels_for_conv[orig_i]
                            if orig_i < len(auto_labels_for_conv)
                            else {}
                        )
                        turns.append({
                            "msg_idx": orig_i,
                            "role": msg.get("role", ""),
                            "ap_labels": {p: bool(msg_auto.get(p, False)) for p in auto_props},
                            "auto_labeled_props": auto_props,
                        })
                    new_records.append({
                        "conv_id": conv_id,
                        "chunk_id": norm_id,
                        "unit_status": "completed",
                        "labeled_by": "auto",
                        "labeled_at": now_iso(),
                        "turns": turns,
                    })

                if new_records:
                    write_jsonl(labels_path, existing + new_records)

    st.session_state.update({
        "conversations": conversations,
        "norms": norms,
        "propositions": propositions,
        "norm_convs": norm_convs,
        "norm_props": norm_props,
        "predictions": predictions,
        "merged_conv_preds": merged_conv_preds,
        "chunks": chunks,
        "chunk_conv_map": chunk_conv_map,
        "tool_call_props": tool_call_props,
        "auto_prop_ids": auto_prop_ids,
        "obs_prop_ids": obs_prop_ids,
        "norms_with_obs": norms_with_obs,
        "norm_auto_labels": norm_auto_labels,
        "app_mode": app_mode,
        "_startup_done": True,
    })
