"""
NL Query Parser
===============
Parses a natural language statistical question into structured intent.
Lightweight keyword-based with clear fallback messaging.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedQuery:
    intent: str            # 'compare' | 'correlate' | 'predict' | 'describe' | 'unknown'
    outcome_hint: Optional[str]
    group_hint: Optional[str]
    subject_hint: Optional[str]
    design_hint: Optional[str]  # 'repeated' | 'independent' | None
    ambiguous: bool
    clarification_needed: Optional[str]
    raw_query: str


# ── Intent pattern lists ─────────────────────────────────────────────────────

COMPARE_PATTERNS = [
    r'\bdiff(?:er|erence)?\b', r'\bcompar\b', r'\bbetween\b',
    r'\bgroup\b', r'\beffect of\b', r'\bchange\b', r'\bimprove\b',
    r'\bgreater\b', r'\bhigher\b', r'\blower\b', r'\bacross\b',
    r'\bvs\b', r'\bversus\b', r'\bdid .+ improve\b', r'\bdid .+ change\b',
    r'\bdo .+ differ\b', r'\bare .+ different\b',
]

REPEATED_PATTERNS = [
    r'\bsession\b', r'\btime.?point\b', r'\bbefore.?after\b',
    r'\bpre.?post\b', r'\brepeat\b', r'\bwithin.?subject\b',
    r'\bpaired\b', r'\blongitudinal\b', r'\bover time\b',
    r'\bwithin\b', r'\bsame .+ across\b', r'\bacross days\b',
    r'\bacross sessions\b', r'\bacross weeks\b',
]

CORRELATE_PATTERNS = [
    r'\bcorrelat\b', r'\bassociat\b', r'\brelated?\b',
    r'\brelationship\b', r'\blink\b', r'\bbetween .+ and\b',
    r'\bis there .+ between\b', r'\bconnection\b',
]

PREDICT_PATTERNS = [
    r'\bpredict\b', r'\bregress\b', r'\bforecast\b',
    r'\bexplain(?:s)?\b', r'\bwhat .+ predicts\b',
]


def parse_query(query: str) -> ParsedQuery:
    """Parse a natural language statistical query into structured intent."""
    q_lower = query.lower()

    is_compare   = any(re.search(p, q_lower) for p in COMPARE_PATTERNS)
    is_repeated  = any(re.search(p, q_lower) for p in REPEATED_PATTERNS)
    is_correlate = any(re.search(p, q_lower) for p in CORRELATE_PATTERNS)
    is_predict   = any(re.search(p, q_lower) for p in PREDICT_PATTERNS)

    # Priority: predict > correlate > compare
    if is_predict:
        intent = "predict"
    elif is_correlate:
        intent = "correlate"
    elif is_compare:
        intent = "compare"
    else:
        intent = "unknown"

    design_hint = "repeated" if is_repeated else None

    # ── Extract column hints from query ──────────────────────────────────────
    outcome_hint = None
    group_hint   = None
    subject_hint = None

    # "compare SCORE by/across/between GROUP"
    by_match = re.search(
        r'(?:compare|test|examine|look at)\s+(\w+)\s+(?:by|across|between|among)\s+(\w+)',
        q_lower
    )
    # "effect of TREATMENT on OUTCOME"
    on_match = re.search(r'effect\s+of\s+(\w+)\s+on\s+(\w+)', q_lower)
    # "relationship between X and Y"
    between_match = re.search(r'(?:between|of)\s+(\w+)\s+and\s+(\w+)', q_lower)
    # "subject/participant COL"
    subject_match = re.search(r'(?:subject|participant|patient)\s+(\w+)', q_lower)

    if by_match:
        outcome_hint = by_match.group(1)
        group_hint   = by_match.group(2)
    elif on_match:
        group_hint   = on_match.group(1)
        outcome_hint = on_match.group(2)
    elif between_match and is_correlate:
        outcome_hint = between_match.group(1)
        group_hint   = between_match.group(2)

    if subject_match:
        subject_hint = subject_match.group(1)

    # ── Detect ambiguity ──────────────────────────────────────────────────────
    ambiguous = False
    clarification_needed = None

    CLEARLY_INDEPENDENT = [
        r'\bgroup\b', r'\bdrug\b', r'\barm\b', r'\bcondition\b',
        r'\bbetween\b', r'\bacross\b', r'\bby \w',
    ]
    is_clearly_independent = any(re.search(p, q_lower) for p in CLEARLY_INDEPENDENT)

    if intent == "unknown":
        ambiguous = True
        clarification_needed = (
            "I'm not sure what kind of analysis you're looking for. "
            "Could you clarify: are you comparing groups, looking for a "
            "correlation, or trying to predict an outcome?"
        )
    elif is_compare and not is_repeated and not is_clearly_independent:
        word_count = len(q_lower.split())
        has_temporal = re.search(r'\bimprove\b|\bchange\b|\bbefore\b|\bafter\b|\bpre\b|\bpost\b', q_lower)
        if word_count <= 4 or has_temporal:
            ambiguous = True
            clarification_needed = (
                "Are the groups independent (different people in each group) "
                "or repeated measures (the same people measured multiple times)?"
            )

    return ParsedQuery(
        intent=intent,
        outcome_hint=outcome_hint,
        group_hint=group_hint,
        subject_hint=subject_hint,
        design_hint=design_hint,
        ambiguous=ambiguous,
        clarification_needed=clarification_needed,
        raw_query=query
    )


def resolve_ambiguity_interactively(parsed: ParsedQuery) -> ParsedQuery:
    """Ask the user to resolve ambiguity before proceeding."""
    if not parsed.ambiguous:
        return parsed
    print(f"\n  ⚠  Clarification needed:")
    print(f"  {parsed.clarification_needed}")
    answer = input("  Your answer: ").strip().lower()
    if any(w in answer for w in ["repeat", "same", "within", "paired", "longitudinal"]):
        parsed.design_hint = "repeated"
    elif any(w in answer for w in ["independent", "different", "separate", "between"]):
        parsed.design_hint = "independent"
    parsed.ambiguous = False
    return parsed
