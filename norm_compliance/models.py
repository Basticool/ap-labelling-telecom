from __future__ import annotations

import statistics
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

class LlmConfig(BaseModel):
    """Configuration for a litellm completion call.

    Loadable from JSON:  LlmConfig.model_validate_json(s)
    Loadable from YAML:  LlmConfig(**yaml.safe_load(f))
    """

    model: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    seed: int | None = None
    top_p: float | None = None
    api_key: str | None = None
    api_base: str | None = None
    timeout: float | None = None


# ---------------------------------------------------------------------------
# Conversation primitives
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """A single turn in a conversation.

    Parameters
    ----------
    role:
        Speaker role (e.g. ``"agent"``, ``"customer"``, ``"action"``).
    content:
        Text content of the turn.
    metadata:
        Optional structured data for tool calls, action parameters, etc.
    """

    role: str
    content: str
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Atomic proposition definitions
# ---------------------------------------------------------------------------

class FewShotExample(BaseModel):
    """A labeled example for AP evaluation in Option-B format.

    The LLM sees ``context`` + ``target`` and is expected to produce
    the ``label`` answer.

    Parameters
    ----------
    context:
        Preceding turns shown to the LLM as background.
    target:
        The specific turn being evaluated.
    label:
        Whether the AP holds at ``target``. ``None`` indicates uncertainty.
    """

    context: list[Turn]
    target: Turn
    label: bool | None


class APDefinition(BaseModel):
    """Complete definition of a single atomic proposition.

    Parameters
    ----------
    name:
        Short identifier (e.g. ``"agent_apologized"``).
    description:
        Natural language description of what the AP captures.
    additional_context:
        Optional domain-specific background that aids evaluation.
    few_shot_examples:
        Labeled examples to guide LLM evaluation.
    """

    name: str
    description: str
    additional_context: str | None = None
    few_shot_examples: list[FewShotExample] = []


# ---------------------------------------------------------------------------
# DFA definition
# ---------------------------------------------------------------------------

class DfaDefinition(BaseModel):
    """Complete specification of a DFA for LTLf norm monitoring.

    Parameters
    ----------
    states:
        Finite set of DFA state identifiers.
    ap:
        Set of atomic proposition names the DFA tracks.
    initial_state:
        The starting state ``q0 ∈ states``.
    transitions:
        Nested mapping ``state -> (guard_str -> next_state)``.
        ``guard_str`` is a sympy-style PL expression string, e.g.
        ``"~a & b"`` or ``"true"``.  Exactly one guard per source state
        should evaluate to True for any given AP assignment (MONA produces
        complete, deterministic automata).
    accepting_states:
        Subset of ``states`` representing compliant (non-violating) states.
    alphabet:
        Optional list of AP assignments (frozensets of AP names that hold)
        that constrain the DFA.  Non-``None`` iff the DFA was produced via
        :func:`~norm_compliance.utils.prune_dfa` or ``ltlf_to_dfa`` with
        an alphabet argument.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: set[str]
    ap: set[str]
    initial_state: str
    transitions: dict[str, dict[str, str]]
    accepting_states: set[str]
    alphabet: list[frozenset[str]] | None = None


# ---------------------------------------------------------------------------
# Norm definitions and monitor state
# ---------------------------------------------------------------------------

class NormDefinition(BaseModel):
    """A single I/O Logic norm parsed from norms.json.

    Parameters
    ----------
    precondition:
        LTLf formula for the norm's trigger condition. Use ``"true"`` for
        norms that are always active.
    obligation:
        LTLf formula expressing what the agent must do once the precondition
        is satisfied.
    obligation_type:
        Temporal obligation category that determines DFA acceptance structure:

        - ``"maintenance"`` — must be satisfied at some future point; once
          satisfied the norm is deactivated (``F(p)`` pattern, accepting state
          has a self-loop).
        - ``"persistence"`` — must hold throughout; violated once broken and
          the failure state is irrecoverable (``G(p)`` pattern, sink failure
          state disconnected from accepting states).
        - ``"punctual"`` — must be satisfied in the very next step
          (``X(p)`` pattern).

        ``None`` leaves the type unspecified (backward-compatible default).
    reparative:
        Optional name of another norm that becomes active if this obligation
        is violated.
    metadata:
        Optional free-form dict for non-monitoring information such as the
        natural-language description of the norm, extraction provenance,
        policy section references, etc.
    """

    precondition: str
    obligation: str
    obligation_type: Literal["maintenance", "persistence", "punctual"] | None = None
    reparative: str | None = None
    metadata: dict[str, Any] | None = None


class NormMonitorState(BaseModel):
    """State of a :class:`~norm_compliance.norm_monitor.NormMonitor` at one timestep.

    Never mutate in place — use ``model_copy(update={...})`` to produce the
    next state.

    Parameters
    ----------
    preconditions_pool:
        Norms still waiting for their precondition to be satisfied.
        Maps norm name → current precondition DFA state string.
    preconditions_sat:
        Norm names whose precondition has been satisfied (obligation may or
        may not yet be active).
    obligations_active:
        Norms whose obligation is currently being monitored.
        Maps norm name → current obligation DFA state string.
    obligations_violated:
        Norms whose obligation was violated (terminal).
        Maps norm name → obligation DFA state string at violation.
    obligations_satisfied:
        Norms whose obligation was satisfied (terminal).
        Maps norm name → obligation DFA state string at satisfaction.
    """

    preconditions_pool:    dict[str, str] = {}
    preconditions_sat:     set[str] = set()
    obligations_active:    dict[str, str] = {}
    obligations_violated:  dict[str, str] = {}
    obligations_satisfied: dict[str, str] = {}


class NormMonitorOutput(BaseModel):
    """Output emitted by :class:`~norm_compliance.norm_monitor.NormMonitor` each step.

    Parameters
    ----------
    newly_active:
        Norms whose obligation became active this step (precondition just
        satisfied at the previous step).
    newly_satisfied:
        Norms whose obligation transitioned to satisfied this step.
    newly_violated:
        Norms whose obligation transitioned to violated this step.
    """

    newly_active:            set[str] = set()
    newly_satisfied:         set[str] = set()
    newly_violated:          set[str] = set()
    newly_preconditions_sat: set[str] = set()


# ---------------------------------------------------------------------------
# NormViolationGuide state
# ---------------------------------------------------------------------------

GuidePhase = Literal["init", "inactive", "active", "satisfied", "violated"]


class NormViolationGuideState(BaseModel):
    """State of a :class:`~norm_compliance.norm_guide.NormViolationGuide`.

    Parameters
    ----------
    phase:
        ``"init"``      — before any step (startState only; never in step output).
        ``"inactive"``  — precondition DFA being stepped; precondition not yet satisfied.
        ``"active"``    — obligation DFA being stepped; outcome not yet conclusive.
        ``"satisfied"`` — obligation conclusively satisfied (terminal).
        ``"violated"``  — obligation conclusively violated (terminal).
    precondition_dfa_state:
        Current precondition DFA state string. Non-None iff phase ∈ {"init", "inactive"}.
    obligation_dfa_state:
        Current obligation DFA state (str or _SinkState sentinel).
        Non-None iff phase ∈ {"active", "satisfied", "violated"}.
    """

    phase: GuidePhase
    precondition_dfa_state: str | None = None
    obligation_dfa_state: Any = None  # str | _SinkState | None


# ---------------------------------------------------------------------------
# Query statistics
# ---------------------------------------------------------------------------

class QueryStats(BaseModel):
    """Accumulated LLM query statistics maintained as SM state.

    Derived latency statistics (mean, median, p90, p99, max) are computed
    on demand from ``latencies`` and are not stored.
    """

    query_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    latencies: list[float] = []

    # ------------------------------------------------------------------
    # Derived latency statistics
    # ------------------------------------------------------------------

    @property
    def mean_latency(self) -> float | None:
        return statistics.mean(self.latencies) if self.latencies else None

    @property
    def median_latency(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    @property
    def p90_latency(self) -> float | None:
        return self._percentile(90)

    @property
    def p99_latency(self) -> float | None:
        return self._percentile(99)

    @property
    def max_latency(self) -> float | None:
        return max(self.latencies) if self.latencies else None

    def _percentile(self, p: float) -> float | None:
        if not self.latencies:
            return None
        sorted_lats = sorted(self.latencies)
        k = (len(sorted_lats) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(sorted_lats) - 1)
        return sorted_lats[lo] + (sorted_lats[hi] - sorted_lats[lo]) * (k - lo)
