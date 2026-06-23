from __future__ import annotations

from typing import Any, Callable

from .base import SM


class Repeat(SM):
    """Executes a constituent TSM n times in sequence, restarting on each completion.

    If n is None, repeats indefinitely (never done).

    State: (counter: int, sm_state)
    Input domain: input domain of sm.
    Output domain: output domain of sm.
    """

    def __init__(self, sm: SM, n: int | None = None, **kwargs: Any) -> None:
        self._sm = sm
        self._n = n
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (0, self._sm.default_start_state)

    def done(self, state: tuple) -> bool:
        counter, _ = state
        return self._n is not None and counter == self._n

    def _advance_if_done(self, counter: int, sm_state: Any) -> tuple[int, Any]:
        while self._sm.done(sm_state) and not self.done((counter, sm_state)):
            counter += 1
            sm_state = self._sm.default_start_state
        return counter, sm_state

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        counter, sm_state = state
        sm_state, o = await self._sm.get_next_values(sm_state, inp)
        counter, sm_state = self._advance_if_done(counter, sm_state)
        return (counter, sm_state), o


class Sequence(SM):
    """Executes a list of TSMs one after another; terminates when the last is done.

    State: (index: int, current_sm_state)
    Input domain: must be accepted by all machines in sm_list.
    Output domain: output domain of whichever machine is currently active.
    """

    def __init__(self, sm_list: list[SM], **kwargs: Any) -> None:
        if not sm_list:
            raise ValueError("Sequence requires at least one machine")
        self._sm_list = sm_list
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (0, self._sm_list[0].default_start_state)

    def done(self, state: tuple) -> bool:
        index, sm_state = state
        return self._sm_list[index].done(sm_state)

    def _advance_if_done(self, index: int, sm_state: Any) -> tuple[int, Any]:
        while self._sm_list[index].done(sm_state) and index + 1 < len(self._sm_list):
            index += 1
            sm_state = self._sm_list[index].default_start_state
        return index, sm_state

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        index, sm_state = state
        sm_state, o = await self._sm_list[index].get_next_values(sm_state, inp)
        index, sm_state = self._advance_if_done(index, sm_state)
        return (index, sm_state), o


class RepeatUntil(SM):
    """Repeatedly runs a TSM to completion; terminates when condition is true at completion.

    The condition is evaluated only when the constituent machine finishes,
    not on every step.

    State: (cond_true: bool, sm_state)
    Input domain: input domain of sm.
    Output domain: output domain of sm.
    """

    def __init__(self, sm: SM, condition: Callable[[Any], bool], **kwargs: Any) -> None:
        self._sm = sm
        self._condition = condition
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (False, self._sm.default_start_state)

    def done(self, state: tuple) -> bool:
        cond_true, sm_state = state
        return self._sm.done(sm_state) and cond_true

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        cond_true, sm_state = state
        sm_state, o = await self._sm.get_next_values(sm_state, inp)
        cond_true = self._condition(inp)
        if self._sm.done(sm_state) and not cond_true:
            sm_state = self._sm.default_start_state
        return (cond_true, sm_state), o


class Until(SM):
    """Runs a TSM, terminating early if a condition becomes true on any step.

    Terminates when the condition holds OR the constituent machine finishes —
    whichever comes first. The condition is evaluated on every step.

    State: (cond_true: bool, sm_state)
    Input domain: input domain of sm.
    Output domain: output domain of sm.
    """

    def __init__(self, sm: SM, condition: Callable[[Any], bool], **kwargs: Any) -> None:
        self._sm = sm
        self._condition = condition
        super().__init__(**kwargs)

    @property
    def default_start_state(self) -> tuple:  # type: ignore[override]
        return (False, self._sm.default_start_state)

    def done(self, state: tuple) -> bool:
        cond_true, sm_state = state
        return cond_true or self._sm.done(sm_state)

    async def get_next_values(self, state: tuple, inp: Any) -> tuple[tuple, Any]:
        cond_true, sm_state = state
        sm_state, o = await self._sm.get_next_values(sm_state, inp)
        cond_true = self._condition(inp)
        return (cond_true, sm_state), o
