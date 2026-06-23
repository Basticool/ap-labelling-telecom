from __future__ import annotations

import re


def extract_props_from_formula(formula: str, known_props: set[str]) -> set[str]:
    """Find all known proposition names referenced in an LTLf formula."""
    found = set()
    for prop in known_props:
        if re.search(r"\b" + re.escape(prop) + r"\b", formula):
            found.add(prop)
    return found


def get_norm_props(
    norm: dict,
    known_props: set[str],
    all_norms: dict | None = None,
) -> set[str]:
    """Return all propositions referenced in a norm's precondition, obligation, and reparative.

    If ``reparative`` is a norm_id reference (not a formula), the referenced
    norm's own propositions are also included.
    """
    props: set[str] = set()
    for field in ("precondition", "obligation"):
        formula = (norm.get(field) or "").strip()
        if formula and formula.lower() != "true":
            props.update(extract_props_from_formula(formula, known_props))

    reparative = (norm.get("reparative") or "").strip()
    if reparative:
        # Try formula-style extraction first
        formula_props = extract_props_from_formula(reparative, known_props)
        if formula_props:
            props.update(formula_props)
        elif all_norms and reparative in all_norms:
            # reparative is a norm_id reference — recurse (one level only)
            props.update(
                get_norm_props(all_norms[reparative], known_props, all_norms=None)
            )

    return props
