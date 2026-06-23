from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from loguru import logger


# ---------------------------------------------------------------------------
# Private sentinel for detecting missing arguments
# ---------------------------------------------------------------------------

class _Missing:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _Missing()


# ---------------------------------------------------------------------------
# UNDEF sentinel
# ---------------------------------------------------------------------------

class _Undef:
    """Singleton sentinel representing an absent or undefined signal.

    Any machine that receives UNDEF as input must return UNDEF as output.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNDEF"

    def __bool__(self) -> bool:
        return False


UNDEF = _Undef()


# ---------------------------------------------------------------------------
# SM base class
# ---------------------------------------------------------------------------

PolicyType = Literal["warn", "raise"]


class SM(ABC):
    """Abstract base class for all state machines.

    At each time step, given current state and input, the machine produces an
    output and transitions to the next state — fully determined by
    ``get_next_values``.

    Parameters
    ----------
    invalid_input_policy:
        Behaviour when ``input_validation`` returns ``False``.
        ``"warn"`` (default): log a warning and return ``UNDEF``.
        ``"raise"``: log an error then raise ``ValueError``.
    done_policy:
        Behaviour when ``step`` is called on a machine whose ``done`` is
        ``True``.
        ``"warn"`` (default): log a warning, return ``UNDEF``, leave state
        unchanged.
        ``"raise"``: log an error then raise ``RuntimeError``.
    """

    default_start_state: Any = None

    def __init__(
        self,
        *,
        invalid_input_policy: PolicyType = "warn",
        done_policy: PolicyType = "warn",
    ) -> None:
        self._invalid_input_policy: PolicyType = invalid_input_policy
        self._done_policy: PolicyType = done_policy
        self._state: Any = self.default_start_state

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_next_values(self, state: Any, inp: Any) -> tuple[Any, Any]:
        """Core transition function.

        Parameters
        ----------
        state:
            Current machine state.
        inp:
            Current input value.

        Returns
        -------
        (next_state, output)
            Must not mutate ``state`` in place.
        """

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def input_validation(self, inp: Any) -> bool:
        """Return ``True`` if ``inp`` is valid for this machine.

        Default implementation always returns ``True``. Subclasses may
        override to enforce domain constraints.
        """
        return True

    def done(self, state: Any) -> bool:
        """Return ``True`` if the machine has reached a terminal state.

        Default implementation always returns ``False``. TSM subclasses
        override this.
        """
        return False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, start_state: Any = _MISSING) -> None:
        """Set the current state.

        Parameters
        ----------
        start_state:
            If provided, use this as the initial state; otherwise use
            ``default_start_state``.
        """
        self._state = (
            self.default_start_state if isinstance(start_state, _Missing) else start_state
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    async def step(self, inp: Any) -> Any:
        """Advance the machine by one step.

        Applies ``input_validation`` and ``done`` checks according to the
        configured policies, then delegates to ``get_next_values``.

        Parameters
        ----------
        inp:
            Input for this time step.

        Returns
        -------
        output
            The machine's output for this step, or ``UNDEF`` if a policy
            violation was handled with ``"warn"``.
        """
        if not self.input_validation(inp):
            msg = f"{type(self).__name__}: invalid input {inp!r}"
            if self._invalid_input_policy == "raise":
                logger.error(msg)
                raise ValueError(msg)
            logger.warning(msg)
            return UNDEF

        if self.done(self._state):
            msg = f"{type(self).__name__}: step() called on a done machine"
            if self._done_policy == "raise":
                logger.error(msg)
                raise RuntimeError(msg)
            logger.warning(msg)
            return UNDEF

        next_state, output = await self.get_next_values(self._state, inp)
        self._state = next_state
        return output

    # ------------------------------------------------------------------
    # Transduce
    # ------------------------------------------------------------------

    async def transduce(self, input_sequence: list[Any]) -> list[Any]:
        """Run the machine over a sequence of inputs.

        Calls ``initialize``, then ``step`` for each element. Stops early
        if ``done`` returns ``True`` before an element is processed.

        Parameters
        ----------
        input_sequence:
            Ordered list of inputs.

        Returns
        -------
        list
            Ordered list of outputs, one per consumed input.
        """
        self.initialize()
        outputs: list[Any] = []
        for inp in input_sequence:
            if self.done(self._state):
                break
            output = await self.step(inp)
            outputs.append(output)
        return outputs
