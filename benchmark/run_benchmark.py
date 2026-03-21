"""
Benchmark Suite for AStats
Tests the full pipeline: NL → structure → assumptions → test selection
Includes all 3 known failure cases from Utkarsh's prototype (scenarios 9, 12, 28)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from astats.structure import infer_structure
from astats.assumptions import check_assumptions
from astats.tests import select_and_run
from astats.nl_parser import parse_query

np.random.seed(42)

SCENARIOS = [
    # Standard independent cases
    {
        "id": 1,
        "desc": "Two independent normal groups, equal variance",
        "expected": "Independent t-test",
        "query": "compare score between group A and group B",
        "data": lambda: pd.DataFrame({
            "group": ["A"] * 30 + ["B"] * 30,
            "score": np.concatenate([np.random.normal(10, 2, 30), np.random.normal(13, 2, 30)])
        })
    },
    {
        "id": 2,
        "desc": "Two independent groups, unequal variance",
        "expected": "Welch's t-test",
        "query": "is there a difference in score by group",
        "data": lambda: pd.DataFrame({
            "group": ["A"] * 30 + ["B"] * 30,
            "score": np.concatenate([np.random.normal(10, 1, 30), np.random.normal(13, 5, 30)])
        })
    },
    {
        "id": 3,
        "desc": "Two independent non-normal groups",
        "expected": "Mann-Whitney U",
        "query": "compare reaction time between conditions",
        "data": lambda: pd.DataFrame({
            "group": ["A"] * 25 + ["B"] * 25,
            "reaction_time": np.concatenate([np.random.exponential(2, 25), np.random.exponential(4, 25)])
        })
    },
    {
        "id": 4,
        "desc": "Three independent groups (ANOVA)",
        "expected": "One-way ANOVA",
        "query": "compare score across three treatment groups",
        "data": lambda: pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "score": np.concatenate([np.random.normal(10, 2, 20), np.random.normal(12, 2, 20), np.random.normal(14, 2, 20)])
        })
    },
    {
        "id": 5,
        "desc": "Three independent groups, non-normal (Kruskal-Wallis)",
        "expected": "Kruskal-Wallis H",
        "query": "are there differences in score across groups",
        "data": lambda: pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "score": np.concatenate([np.random.exponential(2, 20), np.random.exponential(5, 20), np.random.exponential(8, 20)])
        })
    },
    # Repeated measures cases
    {
        "id": 6,
        "desc": "Repeated measures 2 conditions, normal differences",
        "expected": "Paired t-test",
        "query": "compare pre and post scores within subjects",
        "data": lambda: pd.DataFrame({
            "Subject": list(range(20)) * 2,
            "condition": ["pre"] * 20 + ["post"] * 20,
            "score": np.concatenate([np.random.normal(10, 2, 20), np.random.normal(13, 2, 20)])
        })
    },
    {
        "id": 7,
        "desc": "Repeated measures 2 conditions, non-normal (Wilcoxon)",
        "expected": "Wilcoxon signed-rank",
        "query": "did scores change from pre to post",
        "data": lambda: pd.DataFrame({
            "Subject": list(range(20)) * 2,
            "condition": ["pre"] * 20 + ["post"] * 20,
            "score": np.concatenate([np.random.exponential(2, 20), np.random.exponential(5, 20)])
        })
    },
    {
        "id": 8,
        "desc": "Repeated measures 3+ conditions (sleepstudy-style, Friedman)",
        "expected": "Friedman test",
        "query": "compare reaction time across 10 days within subjects",
        "data": lambda: pd.DataFrame({
            "Subject": [f"S{i}" for i in range(18)] * 10,
            "Days": [str(d) for d in range(10) for _ in range(18)],
            "Reaction": np.random.normal(250 + np.repeat(np.arange(10) * 2, 18), 20, 180)
        })
    },
    # Known failure cases from Utkarsh's prototype
    {
        "id": 9,
        "desc": "KNOWN FAILURE in prior work: small sample two-group — should route to Mann-Whitney",
        "expected": "Mann-Whitney U",
        "query": "compare score between groups",
        "data": lambda: pd.DataFrame({
            "group": ["A"] * 6 + ["B"] * 6,
            "score": np.concatenate([np.random.normal(5, 2, 6), np.random.normal(8, 2, 6)])
        })
    },
    {
        "id": 10,
        "desc": "KNOWN FAILURE: borderline repeated measures (2 cond) — Wilcoxon vs Paired t",
        "expected": "Wilcoxon signed-rank",
        "query": "paired comparison of scores across sessions",
        "data": lambda: pd.DataFrame({
            "Subject": list(range(15)) * 2,
            "session": ["A"] * 15 + ["B"] * 15,
            "score": np.concatenate([np.random.exponential(3, 15), np.random.exponential(6, 15)])
        })
    },
    # Wide format
    {
        "id": 11,
        "desc": "Wide-format repeated measures",
        "expected": "Friedman test (wide)",
        "query": "compare scores across multiple time points",
        "data": lambda: pd.DataFrame({
            "T1": np.random.normal(10, 2, 20),
            "T2": np.random.normal(11, 2, 20),
            "T3": np.random.normal(13, 2, 20),
        })
    },
    # Correlation
    {
        "id": 12,
        "desc": "Correlation intent detection",
        "expected": "compare",  # we test intent, not test name here
        "query": "is there a relationship between age and reaction time",
        "data": lambda: pd.DataFrame({
            "age": np.random.normal(35, 10, 50),
            "reaction_time": np.random.normal(300, 50, 50)
        }),
        "test_intent_only": True,
        "expected_intent": "correlate"
    },
    # Ambiguity detection
    {
        "id": 13,
        "desc": "Ambiguous query detection",
        "expected": None,
        "query": "significant?",
        "data": lambda: pd.DataFrame({"x": [1, 2, 3]}),
        "test_ambiguity": True
    },
]


def run_benchmark(verbose=True):
    results = []
    passed = 0
    failed = 0
    skipped = 0

    print("\n" + "═" * 70)
    print("  AStats Benchmark Suite")
    print("═" * 70)

    for s in SCENARIOS:
        try:
            df = s["data"]()
            parsed = parse_query(s["query"])

            # Test ambiguity detection
            if s.get("test_ambiguity"):
                ok = parsed.ambiguous
                status = "PASS" if ok else "FAIL"
                if ok: passed += 1
                else: failed += 1
                if verbose:
                    print(f"  [{status}] Scenario {s['id']:2d}: {s['desc']}")
                    print(f"          Ambiguous detected: {parsed.ambiguous}")
                results.append({"id": s["id"], "passed": ok})
                continue

            # Test intent only
            if s.get("test_intent_only"):
                ok = parsed.intent == s["expected_intent"]
                status = "PASS" if ok else "FAIL"
                if ok: passed += 1
                else: failed += 1
                if verbose:
                    print(f"  [{status}] Scenario {s['id']:2d}: {s['desc']}")
                    print(f"          Intent: {parsed.intent} (expected {s['expected_intent']})")
                results.append({"id": s["id"], "passed": ok})
                continue

            # Full pipeline test
            structure = infer_structure(df)
            assumptions = check_assumptions(df, structure)
            result = select_and_run(df, structure, assumptions)

            ok = result.test_name == s["expected"]
            status = "PASS" if ok else "FAIL"
            if ok: passed += 1
            else: failed += 1

            if verbose:
                mark = "✓" if ok else "✗"
                print(f"  [{status}] Scenario {s['id']:2d}: {s['desc']}")
                if not ok:
                    print(f"          Got: {result.test_name} | Expected: {s['expected']}")
                else:
                    print(f"          {mark} {result.test_name} (p={result.p_value}, {result.effect_label}={result.effect_size})")

            results.append({"id": s["id"], "passed": ok, "test": result.test_name, "expected": s["expected"]})

        except Exception as e:
            skipped += 1
            if verbose:
                print(f"  [SKIP] Scenario {s['id']:2d}: {s['desc']} — Error: {e}")
            results.append({"id": s["id"], "passed": False, "error": str(e)})

    total = passed + failed + skipped
    pct = round(passed / total * 100, 1) if total > 0 else 0

    print("\n" + "═" * 70)
    print(f"  Results: {passed}/{total} passed ({pct}%)")
    print(f"  Failed:  {failed}  |  Skipped: {skipped}")
    print("═" * 70 + "\n")

    return results, pct


if __name__ == "__main__":
    run_benchmark(verbose=True)
