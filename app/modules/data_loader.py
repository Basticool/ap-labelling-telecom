from __future__ import annotations

import json
from pathlib import Path

from app.config import DEEPSEEK_LABELER_ID


def load_conversations(path: str | Path) -> dict[str, dict]:
    """Load conversations.json → {convID: conv_dict}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_predictions(
    path: str | Path,
) -> dict[str, dict[str, dict[int, dict[str, bool]]]]:
    """Load ap_predictions.json → {normID: {convID: {msgIdx: {ap: bool}}}}.

    If the file contains predictions from multiple labelerIDs, the Deepseek
    labeler (DEEPSEEK_LABELER_ID) is used.  If only one labelerID is present,
    that one is used regardless of its value.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    unique_ids: set[str] = {r["labelerID"] for r in records}
    if len(unique_ids) == 1:
        chosen_id = next(iter(unique_ids))
    else:
        chosen_id = DEEPSEEK_LABELER_ID

    index: dict[str, dict[str, dict[int, dict[str, bool]]]] = {}
    for rec in records:
        if rec["labelerID"] != chosen_id:
            continue
        norm_id = rec["normID"]
        conv_id = rec["convID"]
        msg_idx = rec["msgIdx"]
        labels: dict[str, bool] = rec["labels"]
        index.setdefault(norm_id, {}).setdefault(conv_id, {})[msg_idx] = labels

    return index


def load_norms(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_propositions(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
