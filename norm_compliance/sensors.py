from __future__ import annotations

import asyncio
import re
from typing import Any

from norm_compliance.models import APDefinition, FewShotExample, Turn
from norm_compliance.sm.base import SM


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_turn(turn: Turn) -> str:
    return f"  {turn.role}: {turn.content}"


def _render_turns(turns: list[Turn]) -> str:
    if not turns:
        return "  (none)"
    return "\n".join(_render_turn(t) for t in turns)


def _render_example(i: int, example: FewShotExample) -> str:
    """Render one few-shot example as a natural conversation block.

    Format::

        Conversation:
          agent: Sure, may I have your name?
          customer: Crystal Minh
        Current turn:
          action: Account has been pulled up for Crystal Minh.
        Classification: true
    """
    label_str = "true" if example.label is True else ("false" if example.label is False else "null")
    lines = ["Conversation:"]
    for t in example.context:
        lines.append(f"  {t.role}: {t.content}")
    lines.append("Current turn:")
    lines.append(f"  {example.target.role}: {example.target.content}")
    lines.append(f"Classification: {label_str}")
    return "\n".join(lines)


def _render_examples(examples: list[FewShotExample]) -> str:
    if not examples:
        return "(no examples provided)"
    return "\n\n".join(_render_example(i + 1, ex) for i, ex in enumerate(examples))


# ---------------------------------------------------------------------------
# ApLlmSensor
# ---------------------------------------------------------------------------

class ApLlmSensor(SM):
    """Formats an AP evaluation prompt from a stream of conversation turns.

    Maintains conversation history as state. At each step the current turn
    becomes the evaluation target; all preceding turns are the context.

    .. note::
        This deviates from the spec description of "stateless". History is
        required to provide meaningful context for the LLM evaluator.

    Parameters
    ----------
    prompt_template:
        A Python ``.format()``-style string with the following slots:

        - ``{name}`` — AP name
        - ``{description}`` — AP description
        - ``{additional_context}`` — domain background (empty string if absent)
        - ``{few_shot_examples}`` — pre-rendered labeled examples
        - ``{context}`` — formatted preceding turns
        - ``{target}`` — formatted current turn

    ap_definition:
        The :class:`~norm_compliance.models.APDefinition` this sensor evaluates.
    """

    def __init__(
        self,
        prompt_template: str,
        ap_definition: APDefinition,
        **kwargs: Any,
    ) -> None:
        self._prompt_template = prompt_template
        self._ap_definition = ap_definition
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> list[Turn]:  # type: ignore[override]
        return []

    def input_validation(self, inp: Any) -> bool:
        return isinstance(inp, Turn)

    async def get_next_values(
        self, state: list[Turn], inp: Turn
    ) -> tuple[list[Turn], str]:
        ap = self._ap_definition
        prompt = self._prompt_template.format(
            name=ap.name,
            description=ap.description,
            additional_context=ap.additional_context or "",
            few_shot_examples=_render_examples(ap.few_shot_examples),
            context=_render_turns(state),
            target=_render_turn(inp),
        )
        next_state = state + [inp]
        return next_state, prompt


# ---------------------------------------------------------------------------
# ApSensorUnion
# ---------------------------------------------------------------------------

class ApSensorUnion(SM):
    """Runs a list of :class:`ApLlmSensor` machines on the same input turn.

    Each sensor maintains its own conversation history. Outputs are collected
    into an ordered list of prompt strings — one per sensor — in construction
    order. Sensor calls are independent and executed concurrently.

    Parameters
    ----------
    sensors:
        Ordered list of :class:`ApLlmSensor` instances, one per AP.
    """

    def __init__(self, sensors: list[ApLlmSensor], **kwargs: Any) -> None:
        self._sensors = sensors
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return tuple(s.default_start_state for s in self._sensors)

    def input_validation(self, inp: Any) -> bool:
        return isinstance(inp, Turn)

    async def get_next_values(
        self, state: tuple, inp: Turn
    ) -> tuple[tuple, list[str]]:
        results = await asyncio.gather(
            *[s.get_next_values(s_i, inp) for s, s_i in zip(self._sensors, state)]
        )
        next_states = tuple(ns for ns, _ in results)
        outputs = [o for _, o in results]
        return next_states, outputs


# ---------------------------------------------------------------------------
# ApRegexSensor
# ---------------------------------------------------------------------------

class ApRegexSensor(SM):
    """Deterministic sensor that grounds a single AP by matching a regex.

    Stateless — only inspects the current turn; no conversation history needed.
    Outputs ``bool`` directly; no LLM call required.

    Parameters
    ----------
    ap_name:
        Name of the atomic proposition this sensor grounds (e.g.
        ``"find_user_id_by_email"``). Exposed via the :attr:`ap_name`
        property so :class:`ApRegexSensorUnion` can key the output dict.
    pattern:
        Regex string. For ``field="tool_name"`` matched via ``re.fullmatch``
        against each tool call name (e.g. ``"cancel_pending_order"``). For
        ``field="content"`` matched via ``re.search`` against ``Turn.content``.
    field:
        ``"tool_name"`` (default) — inspect ``Turn.metadata["tool_calls"]``.
        ``"content"`` — inspect ``Turn.content``.
    """

    def __init__(
        self, ap_name: str, pattern: str, field: str = "tool_name", **kwargs: Any
    ) -> None:
        self._ap_name = ap_name
        self._re = re.compile(pattern)
        self._field = field
        super().__init__(**kwargs)

    @property
    def ap_name(self) -> str:
        """The AP name this sensor is responsible for grounding."""
        return self._ap_name

    @property
    def default_start_state(self) -> None:  # type: ignore[override]
        return None

    def input_validation(self, inp: Any) -> bool:
        return isinstance(inp, Turn)

    async def get_next_values(self, state: None, inp: Turn) -> tuple[None, bool]:
        if self._field == "content":
            value = inp.content or ""
            return state, bool(self._re.search(value))
        elif self._field == "tool_name":
            meta = inp.metadata or {}
            # List format: metadata["tool_calls"] = [{name, arguments, result}, ...]
            tcs = meta.get("tool_calls")
            if isinstance(tcs, list):
                return state, any(
                    bool(self._re.search(tc.get("name") or ""))
                    for tc in tcs
                    if isinstance(tc, dict)
                )
            # Legacy scalar format: metadata["tool_call"] = {name, arguments}
            tc = meta.get("tool_call", {})
            value = tc.get("name", "") if isinstance(tc, dict) else ""
            return state, bool(self._re.search(value))
        return state, False


# ---------------------------------------------------------------------------
# ApRegexSensorUnion
# ---------------------------------------------------------------------------

class ApRegexSensorUnion(SM):
    """Runs a list of :class:`ApRegexSensor` machines on the same input turn.

    Each sensor is stateless and evaluated independently. Outputs are
    collected into a ``dict[str, bool]`` keyed by each sensor's
    :attr:`~ApRegexSensor.ap_name`, in construction order.

    Parameters
    ----------
    sensors:
        Ordered list of :class:`ApRegexSensor` instances, one per AP.
    """

    def __init__(self, sensors: list[ApRegexSensor], **kwargs: Any) -> None:
        self._sensors = sensors
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return tuple(s.default_start_state for s in self._sensors)

    def input_validation(self, inp: Any) -> bool:
        return isinstance(inp, Turn)

    async def get_next_values(
        self, state: tuple, inp: Turn
    ) -> tuple[tuple, dict[str, bool]]:
        results = await asyncio.gather(
            *[s.get_next_values(s_i, inp) for s, s_i in zip(self._sensors, state)]
        )
        next_states = tuple(ns for ns, _ in results)
        outputs = {s.ap_name: out for s, (_, out) in zip(self._sensors, results)}
        return next_states, outputs
