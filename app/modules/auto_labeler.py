"""Auto-labeler for deterministic atomic propositions.

Handles three ap_kinds without requiring an LLM:

  tool_call   — True when an assistant turn contains a tool call whose name
                matches metadata.tool_name (exact match).

  tool_result — True when a tool-result turn resolves a specific source tool
                and the result payload is a valid (non-error) value.
                Needs cross-message context (ID → tool-name lookup).

  structural  — True based on the structural shape of a single message
                (multiple tool calls, text+tool-call mix, invalid cancel reason).

All other ap_kinds (observation) require human labeling and are skipped.

Public API (unchanged from the original):
    sensors = build_auto_label_sensors(propositions)
    labels  = compute_auto_labels(messages, sensors)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

# Allowed cancel reasons per policy
_VALID_CANCEL_REASONS: frozenset[str] = frozenset(
    {"no longer needed", "ordered by mistake"}
)

# Context key for the pre-built {tool_call_id: tool_name} map
_CTX_ID_MAP = "tool_call_id_to_name"


# ── Labeler callables ─────────────────────────────────────────────────────────

def _make_tool_call_labeler(tool_name: str) -> Callable[[dict, dict], bool]:
    """True when the assistant turn contains a call whose name == tool_name."""
    def labeler(message: dict, _ctx: dict) -> bool:
        if message.get("role") != "assistant":
            return False
        return any(
            tc.get("name") == tool_name
            for tc in (message.get("tool_calls") or [])
            if isinstance(tc, dict)
        )
    return labeler


def _make_tool_result_labeler(source_tools: list[str]) -> Callable[[dict, dict], bool]:
    """True when a tool-result turn resolves a source_tool call with a valid payload.

    Grounding:
      - role == "tool"
      - the call id resolves to one of source_tools (via context map or positional fallback)
      - error flag is False
      - content is a non-empty, non-error string (looks like a user_id)
    """
    source_set = frozenset(source_tools)

    def _is_valid_user_id(content: str) -> bool:
        if not content or not content.strip():
            return False
        text = content.strip()
        lower = text.lower()
        if lower.startswith("error") or "not found" in lower:
            return False
        if text.startswith("{"):
            try:
                payload = json.loads(text)
                return bool(payload.get("user_id"))
            except (json.JSONDecodeError, AttributeError):
                return False
        return bool(re.fullmatch(r"[A-Za-z0-9_\-]+", text))

    def labeler(message: dict, ctx: dict) -> bool:
        if message.get("role") != "tool":
            return False
        if message.get("error"):
            return False
        # Primary: use the id field on the message
        call_id = message.get("id", "")
        # Positional fallback for tau-bench conversations that omit id on tool turns
        if not call_id:
            msg_idx = ctx.get("_current_msg_idx", -1)
            call_id = ctx.get("_pos_id_map", {}).get(msg_idx, "")
        tool_name = ctx.get(_CTX_ID_MAP, {}).get(call_id)
        if tool_name not in source_set:
            return False
        return _is_valid_user_id(message.get("content") or "")

    return labeler


def _labeler_multiple_tool_calls(message: dict, _ctx: dict) -> bool:
    """True when an assistant turn issues two or more tool calls."""
    if message.get("role") != "assistant":
        return False
    tcs = message.get("tool_calls") or []
    return len([tc for tc in tcs if isinstance(tc, dict)]) >= 2


def _labeler_text_with_tool_call(message: dict, _ctx: dict) -> bool:
    """True when an assistant turn has  and at least one tool call."""
    if message.get("role") != "assistant":
        return False
    tcs = message.get("tool_calls") or []
    if not any(isinstance(tc, dict) for tc in tcs):
        return False
    content = (message.get("content") or "").strip()
    return len(content) > 0


def _labeler_cancel_invalid_reason(message: dict, _ctx: dict) -> bool:
    """True when a cancel_pending_order call uses a reason not in the allowed set."""
    if message.get("role") != "assistant":
        return False
    for tc in (message.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        if tc.get("name") != "cancel_pending_order":
            continue
        args = tc.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        reason = args.get("reason", "") if isinstance(args, dict) else ""
        if reason not in _VALID_CANCEL_REASONS:
            return True
    return False


# ── Dispatch table for named structural APs ──────────────────────────────────

_STRUCTURAL_LABELERS: dict[str, Callable[[dict, dict], bool]] = {
    "agent_turn_has_multiple_tool_calls": _labeler_multiple_tool_calls,
    "agent_turn_has_text_with_tool_call": _labeler_text_with_tool_call,
    "agent_called_cancel_with_invalid_reason": _labeler_cancel_invalid_reason,
}


# ── Public API ────────────────────────────────────────────────────────────────

def build_auto_label_sensors(
    propositions: dict[str, Any],
) -> dict[str, Callable[[dict, dict], bool]]:
    """Return one labeler callable per auto-groundable proposition.

    Handles ap_kinds: tool_call, tool_result, structural.
    Skips ap_kinds: observation (requires human labeling).
    """
    labelers: dict[str, Callable[[dict, dict], bool]] = {}
    for prop_id, defn in propositions.items():
        meta = defn.get("metadata", {})
        kind = meta.get("ap_kind", "")

        if kind == "tool_call":
            tool_name = meta.get("tool_name") or prop_id
            labelers[prop_id] = _make_tool_call_labeler(tool_name)

        elif kind == "tool_result":
            source_tools = meta.get("source_tools", [])
            if source_tools:
                labelers[prop_id] = _make_tool_result_labeler(source_tools)

        elif kind == "structural":
            fn = _STRUCTURAL_LABELERS.get(prop_id)
            if fn is not None:
                labelers[prop_id] = fn

    return labelers


def compute_auto_labels(
    messages: list[dict],
    labelers: dict[str, Callable[[dict, dict], bool]],
) -> list[dict[str, bool]]:
    """Return one {prop_id: bool} dict per message for all auto-groundable props.

    Builds cross-message context (tool-call ID → name map) once, then applies
    each labeler to every message.  Also builds a positional ID map for
    conversations (e.g. tau-bench) where tool-result turns lack an 'id' field.
    """
    if not labelers:
        return [{} for _ in messages]

    # Build {call_id: tool_name} from all assistant messages
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                if isinstance(tc, dict) and tc.get("id"):
                    id_to_name[tc["id"]] = tc.get("name", "")

    # Positional fallback: map each tool-turn index to its call_id when the
    # tool message itself has no 'id'.  Matches tool responses to the preceding
    # assistant tool_calls in FIFO order.
    pos_id_map: dict[int, str] = {}
    pending: list[str] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                if isinstance(tc, dict) and tc.get("id"):
                    pending.append(tc["id"])
        elif msg.get("role") == "tool" and not msg.get("id"):
            if pending:
                pos_id_map[i] = pending.pop(0)

    ctx = {_CTX_ID_MAP: id_to_name, "_pos_id_map": pos_id_map}

    result = []
    for i, msg in enumerate(messages):
        ctx["_current_msg_idx"] = i
        result.append({prop_id: fn(msg, ctx) for prop_id, fn in labelers.items()})
    return result
