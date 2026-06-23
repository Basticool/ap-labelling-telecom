from __future__ import annotations

import asyncio
from typing import Any, Callable

from .base import SM, UNDEF


class Cascade(SM):
    """Connects two machines in series: output of m1 becomes input to m2.

    State: (state_of_m1, state_of_m2)
    Input domain: input domain of m1.
    Output domain: output domain of m2.
    """

    def __init__(self, m1: SM, m2: SM, **kwargs: Any) -> None:
        self._m1 = m1
        self._m2 = m2
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (self._m1.default_start_state, self._m2.default_start_state)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        s1, s2 = state
        ns1, o1 = await self._m1.get_next_values(s1, inp)
        ns2, o2 = await self._m2.get_next_values(s2, o1)
        return (ns1, ns2), o2


class Parallel(SM):
    """Runs two machines on the same input; outputs are paired.

    State: (state_of_m1, state_of_m2)
    Input domain: common input domain of m1 and m2.
    Output domain: (output_of_m1, output_of_m2)
    """

    def __init__(self, m1: SM, m2: SM, **kwargs: Any) -> None:
        self._m1 = m1
        self._m2 = m2
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (self._m1.default_start_state, self._m2.default_start_state)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, tuple]:
        s1, s2 = state
        (ns1, o1), (ns2, o2) = await asyncio.gather(
            self._m1.get_next_values(s1, inp),
            self._m2.get_next_values(s2, inp),
        )
        return (ns1, ns2), (o1, o2)


def _split(v: Any) -> tuple[Any, Any]:
    """Route a paired input or propagate UNDEF to both branches."""
    if v is UNDEF:
        return UNDEF, UNDEF
    return v


class Parallel2(SM):
    """Like Parallel but input is a pair routed to m1 and m2 respectively.

    State: (state_of_m1, state_of_m2)
    Input domain: (input_for_m1, input_for_m2) or UNDEF.
    Output domain: (output_of_m1, output_of_m2)
    """

    def __init__(self, m1: SM, m2: SM, **kwargs: Any) -> None:
        self._m1 = m1
        self._m2 = m2
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (self._m1.default_start_state, self._m2.default_start_state)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, tuple]:
        s1, s2 = state
        i1, i2 = _split(inp)
        (ns1, o1), (ns2, o2) = await asyncio.gather(
            self._m1.get_next_values(s1, i1),
            self._m2.get_next_values(s2, i2),
        )
        return (ns1, ns2), (o1, o2)


class ParallelAdd(SM):
    """Like Parallel but produces the scalar sum of both machines' outputs.

    State: (state_of_m1, state_of_m2)
    Input domain: common input domain of m1 and m2.
    Output domain: number.
    """

    def __init__(self, m1: SM, m2: SM, **kwargs: Any) -> None:
        self._m1 = m1
        self._m2 = m2
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (self._m1.default_start_state, self._m2.default_start_state)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        s1, s2 = state
        (ns1, o1), (ns2, o2) = await asyncio.gather(
            self._m1.get_next_values(s1, inp),
            self._m2.get_next_values(s2, inp),
        )
        return (ns1, ns2), o1 + o2


class Switch(SM):
    """Routes input to exactly one machine per step based on a condition.

    The selected machine updates its state; the other is preserved unchanged.

    State: (state_of_m1, state_of_m2)
    Input domain: any value accepted by condition and both constituent machines.
    Output domain: output of whichever machine was selected.
    """

    def __init__(self, m1: SM, m2: SM, condition: Callable[[Any], bool], **kwargs: Any) -> None:
        self._m1 = m1
        self._m2 = m2
        self._condition = condition
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (self._m1.default_start_state, self._m2.default_start_state)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        s1, s2 = state
        if self._condition(inp):
            ns1, o = await self._m1.get_next_values(s1, inp)
            return (ns1, s2), o
        else:
            ns2, o = await self._m2.get_next_values(s2, inp)
            return (s1, ns2), o


class Mux(SM):
    """Both machines update every step; condition selects which output to emit.

    Unlike Switch, both machines always advance regardless of the condition.

    State: (state_of_m1, state_of_m2)
    Input domain: any value accepted by condition and both constituent machines.
    Output domain: output domain of m1 or m2 (must be compatible).
    """

    def __init__(self, m1: SM, m2: SM, condition: Callable[[Any], bool], **kwargs: Any) -> None:
        self._m1 = m1
        self._m2 = m2
        self._condition = condition
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (self._m1.default_start_state, self._m2.default_start_state)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        s1, s2 = state
        (ns1, o1), (ns2, o2) = await asyncio.gather(
            self._m1.get_next_values(s1, inp),
            self._m2.get_next_values(s2, inp),
        )
        output = o1 if self._condition(inp) else o2
        return (ns1, ns2), output


# ---------------------------------------------------------------------------
# If — internal state tags
# ---------------------------------------------------------------------------

class _IfSelection:
    START = "START"
    RUNNING_M1 = "RUNNING_M1"
    RUNNING_M2 = "RUNNING_M2"


class If(SM):
    """Evaluates condition once on first input; permanently commits to m1 or m2.

    The decision is irrevocable — only the chosen machine runs from step 1
    onward.

    State: (selection, inner_state) where selection ∈ {START, RUNNING_M1, RUNNING_M2}
    Input domain: any value accepted by condition and both constituent machines.
    Output domain: output domain of m1 or m2 (must be compatible).
    """

    def __init__(self, m1: SM, m2: SM, condition: Callable[[Any], bool], **kwargs: Any) -> None:
        self._m1 = m1
        self._m2 = m2
        self._condition = condition
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (_IfSelection.START, None)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        selection, inner_state = state

        if selection == _IfSelection.START:
            if self._condition(inp):
                selection = _IfSelection.RUNNING_M1
                inner_state = self._m1.default_start_state
            else:
                selection = _IfSelection.RUNNING_M2
                inner_state = self._m2.default_start_state

        if selection == _IfSelection.RUNNING_M1:
            new_inner, o = await self._m1.get_next_values(inner_state, inp)
        else:
            new_inner, o = await self._m2.get_next_values(inner_state, inp)

        return (selection, new_inner), o
