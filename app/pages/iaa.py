"""Inter-Annotator Agreement page.

Compares saved human labels against deepseek predictions stored alongside
each turn record.  Shows per-AP and per-norm agreement rate + Cohen's kappa.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import streamlit as st

from app.config import LABELS_DIR
from app.modules.storage import read_jsonl


def _cohen_kappa(agreed: int, n: int, n_human_yes: int, n_deepseek_yes: int) -> float:
    if n == 0:
        return float("nan")
    p_o = agreed / n
    p_A = n_human_yes / n
    p_B = n_deepseek_yes / n
    p_e = p_A * p_B + (1 - p_A) * (1 - p_B)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def _load_pairs(
    labels_dir: Path,
    obs_prop_ids: set[str],
) -> tuple[
    dict[str, list[tuple[bool, bool]]],   # ap  → [(human, deepseek)]
    dict[str, dict[str, list[tuple[bool, bool]]]],  # norm → ap → pairs
]:
    per_ap: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    per_norm_ap: dict[str, dict[str, list[tuple[bool, bool]]]] = defaultdict(lambda: defaultdict(list))

    if not labels_dir.exists():
        return dict(per_ap), {}

    for label_file in sorted(labels_dir.glob("*.jsonl")):
        norm_id = label_file.stem
        for rec in read_jsonl(label_file):
            if rec.get("unit_status") != "completed":
                continue
            for turn in rec.get("turns", []):
                human_labels: dict = turn.get("ap_labels", {})
                deepseek_labels: dict = turn.get("deepseek_labels", {})
                if not deepseek_labels:
                    continue
                for ap, human_val in human_labels.items():
                    if ap not in deepseek_labels:
                        continue
                    if obs_prop_ids and ap not in obs_prop_ids:
                        continue
                    h = bool(human_val)
                    d = bool(deepseek_labels[ap])
                    per_ap[ap].append((h, d))
                    per_norm_ap[norm_id][ap].append((h, d))

    return dict(per_ap), {k: dict(v) for k, v in per_norm_ap.items()}


def _ap_table(per_ap: dict[str, list[tuple[bool, bool]]]) -> list[dict]:
    rows = []
    for ap in sorted(per_ap):
        pairs = per_ap[ap]
        n = len(pairs)
        agreed = sum(h == d for h, d in pairs)
        n_human_yes = sum(h for h, d in pairs)
        n_deepseek_yes = sum(d for h, d in pairs)
        kappa = _cohen_kappa(agreed, n, n_human_yes, n_deepseek_yes)
        rows.append({
            "AP": ap,
            "Agreement": f"{agreed/n:.1%}" if n else "—",
            "Kappa": f"{kappa:.3f}" if n else "—",
            "Human yes%": f"{n_human_yes/n:.1%}" if n else "—",
            "Deepseek yes%": f"{n_deepseek_yes/n:.1%}" if n else "—",
            "N turns": n,
        })
    return rows


def render() -> None:
    st.title("Inter-Annotator Agreement (Human vs Deepseek)")
    st.caption(
        "Only observational APs are compared — auto-labeled APs are deterministic "
        "ground truth and are excluded."
    )

    obs_prop_ids: set[str] = st.session_state.get("obs_prop_ids", set())
    per_ap, per_norm_ap = _load_pairs(LABELS_DIR, obs_prop_ids)

    if not per_ap:
        st.info("No labeled turns with deepseek comparison found yet. Save some labels first.")
        return

    # ── Overall stats ──────────────────────────────────────────────────────────
    all_pairs = [(h, d) for pairs in per_ap.values() for h, d in pairs]
    n_total = len(all_pairs)
    agreed_total = sum(h == d for h, d in all_pairs)
    n_human_yes = sum(h for h, d in all_pairs)
    n_deepseek_yes = sum(d for h, d in all_pairs)
    overall_kappa = _cohen_kappa(agreed_total, n_total, n_human_yes, n_deepseek_yes)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Agreement", f"{agreed_total/n_total:.1%}" if n_total else "—")
    col2.metric("Overall Kappa", f"{overall_kappa:.3f}" if n_total else "—")
    col3.metric("Total turn-AP comparisons", n_total)
    col4.metric("Norms labeled", len(per_norm_ap))

    st.divider()

    # ── Per-AP table ───────────────────────────────────────────────────────────
    st.subheader("Per-AP Agreement")
    rows = _ap_table(per_ap)
    try:
        import pandas as pd
        df = pd.DataFrame(rows).set_index("AP")
        st.dataframe(df, use_container_width=True)
    except ImportError:
        for row in rows:
            st.write(row)

    # ── Per-norm breakdown ─────────────────────────────────────────────────────
    if per_norm_ap:
        st.subheader("Per-Norm Breakdown")
        norm_rows = []
        for norm_id in sorted(per_norm_ap):
            ap_data = per_norm_ap[norm_id]
            flat = [(h, d) for pairs in ap_data.values() for h, d in pairs]
            n = len(flat)
            agreed = sum(h == d for h, d in flat)
            nh = sum(h for h, d in flat)
            nd = sum(d for h, d in flat)
            kappa = _cohen_kappa(agreed, n, nh, nd)
            norm_rows.append({
                "Norm": norm_id,
                "Agreement": f"{agreed/n:.1%}" if n else "—",
                "Kappa": f"{kappa:.3f}" if n else "—",
                "N": n,
            })
        try:
            import pandas as pd
            df_norm = pd.DataFrame(norm_rows).set_index("Norm")
            st.dataframe(df_norm, use_container_width=True)
        except ImportError:
            for row in norm_rows:
                st.write(row)

        with st.expander("Per-AP breakdown by norm"):
            selected_norm = st.selectbox("Norm", sorted(per_norm_ap.keys()))
            if selected_norm:
                rows_norm = _ap_table(per_norm_ap[selected_norm])
                try:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows_norm).set_index("AP"), use_container_width=True)
                except ImportError:
                    for row in rows_norm:
                        st.write(row)
