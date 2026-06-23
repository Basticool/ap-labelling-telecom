from __future__ import annotations

from typing import Any

from .base import SM, PolicyType


class Accumulator(SM):
    """Maintains and outputs the running sum of all inputs received.

    State: number — the accumulated total.
    Input domain: numbers.
    Output domain: numbers.
    """

    def __init__(self, initial: float = 0, **kwargs: Any) -> None:
        self._initial = initial
        super().__init__(**kwargs)
        self._state = initial

    @property
    def default_start_state(self) -> float:  # type: ignore[override]
        return self._initial

    async def get_next_values(self, state: float, inp: float) -> tuple[float, float]:
        next_state = state + inp
        return next_state, next_state


class Memory(SM):
    """Records every input in an ordered log; output mirrors input.

    State: list — ordered sequence of all inputs received so far.
    Input domain: any.
    Output domain: same as input (output mirrors input exactly).

    Warning
    -------
    State grows unboundedly. Use deliberately in long-running machines.
    """

    @property
    def default_start_state(self) -> list[Any]:  # type: ignore[override]
        return []

    async def get_next_values(self, state: list[Any], inp: Any) -> tuple[list[Any], Any]:
        next_state = state + [inp]
        return next_state, inp
