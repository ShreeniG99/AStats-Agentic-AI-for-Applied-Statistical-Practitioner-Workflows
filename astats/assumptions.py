"""
Assumption Checking Module
Checks normality, variance homogeneity, sample size, independence.
Returns structured results that drive test selection.
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional
from .structure import DataStructure


@dataclass
class AssumptionResults:
    normality_ok: bool
    equal_variance_ok: bool
    sample_adequate: bool
    normality_detail: dict   # {group: (stat, p, result)}
    variance_detail: dict
    sample_size: int
    warnings: list


def check_assumptions(df: pd.DataFrame, structure: DataStructure) -> AssumptionResults:
    warnings = []
    normality_detail = {}
    variance_detail = {}

    outcome = structure.outcome_col
    group = structure.group_col

    if outcome is None:
        return AssumptionResults(
            normality_ok=False, equal_variance_ok=True,
            sample_adequate=False, normality_detail={},
            variance_detail={}, sample_size=len(df), warnings=["No outcome column identified"]
        )

    # --- Normality ---
    if structure.design == "repeated_measures" and group is not None:
        # Check normality of within-subject differences
        groups = df[group].unique()
        if len(groups) == 2:
            g1, g2 = groups
            vals1 = df[df[group] == g1][outcome].values
            vals2 = df[df[group] == g2][outcome].values
            min_len = min(len(vals1), len(vals2))
            diffs = vals1[:min_len] - vals2[:min_len]
            if len(diffs) >= 3:
                stat, p = stats.shapiro(diffs)
                normal = p > 0.05
                normality_detail["within_subject_diffs"] = {"stat": round(stat, 4), "p": round(p, 4), "normal": normal}
                normality_ok = normal
            else:
                normality_ok = False
                warnings.append("Too few observations to test normality of differences")
        else:
            # Multiple repeated conditions - check each
            normality_ok = True
            for g in groups:
                vals = df[df[group] == g][outcome].values
                if len(vals) >= 3:
                    stat, p = stats.shapiro(vals)
                    normal = p > 0.05
                    normality_detail[str(g)] = {"stat": round(stat, 4), "p": round(p, 4), "normal": normal}
                    if not normal:
                        normality_ok = False
                else:
                    normality_ok = False
    else:
        # Independent groups
        normality_ok = True
        if group is not None:
            for g in df[group].unique():
                vals = df[df[group] == g][outcome].values
                if len(vals) >= 3:
                    stat, p = stats.shapiro(vals)
                    normal = p > 0.05
                    normality_detail[str(g)] = {"stat": round(stat, 4), "p": round(p, 4), "normal": normal}
                    if not normal:
                        normality_ok = False
                else:
                    warnings.append(f"Group '{g}' has fewer than 3 observations — normality test unreliable")
                    normality_ok = False
        else:
            vals = df[outcome].values
            if len(vals) >= 3:
                stat, p = stats.shapiro(vals)
                normality_ok = p > 0.05
                normality_detail["all"] = {"stat": round(stat, 4), "p": round(p, 4), "normal": normality_ok}

    # --- Equal Variance (Levene's) ---
    equal_variance_ok = True
    if group is not None and structure.design == "independent":
        groups_data = [df[df[group] == g][outcome].dropna().values for g in df[group].unique()]
        groups_data = [g for g in groups_data if len(g) >= 2]
        if len(groups_data) >= 2:
            stat, p = stats.levene(*groups_data)
            equal_variance_ok = p > 0.05
            variance_detail = {"levene_stat": round(stat, 4), "p": round(p, 4), "equal": equal_variance_ok}
            if not equal_variance_ok:
                warnings.append("Unequal variances detected (Levene's p < 0.05) — Welch correction will be applied")

    # --- Sample Size ---
    n = len(df)
    sample_adequate = n >= 10
    if n < 10:
        warnings.append(f"Small sample (n={n}) — results should be interpreted with caution")
        # For very small samples, force non-parametric regardless of normality
        normality_ok = False
    if n < 30 and n >= 10:
        warnings.append(f"n={n} — parametric tests may be less reliable; non-parametric alternative noted")

    return AssumptionResults(
        normality_ok=normality_ok,
        equal_variance_ok=equal_variance_ok,
        sample_adequate=sample_adequate,
        normality_detail=normality_detail,
        variance_detail=variance_detail,
        sample_size=n,
        warnings=warnings
    )
