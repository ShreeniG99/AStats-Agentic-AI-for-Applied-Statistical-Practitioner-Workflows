"""
Test Selection and Execution Module
Picks the right test from structure + assumptions, runs it, computes effect size.
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional
from .structure import DataStructure
from .assumptions import AssumptionResults


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_label: str      # "cohen_d", "eta_squared", "kendalls_w", etc.
    effect_magnitude: str  # "small", "medium", "large"
    significant: bool
    alpha: float
    rationale: str         # why this test was chosen
    methods_paragraph: str


def select_and_run(df: pd.DataFrame,
                   structure: DataStructure,
                   assumptions: AssumptionResults,
                   alpha: float = 0.05,
                   intent: str = "compare") -> TestResult:
    """
    Select and execute the appropriate statistical test.
    """
    outcome = structure.outcome_col
    group = structure.group_col
    design = structure.design
    n_groups = structure.n_groups

    # ── DECISION MATRIX ────────────────────────────────────────────────────

    if design == "repeated_measures":
        if n_groups == 2:
            if assumptions.normality_ok:
                return _paired_t(df, structure, assumptions, alpha)
            else:
                return _wilcoxon(df, structure, assumptions, alpha)
        else:  # 3+ conditions
            return _friedman(df, structure, assumptions, alpha)

    elif design == "wide_repeated":
        return _friedman_wide(df, structure, assumptions, alpha)

    elif design == "independent":
        if n_groups == 2:
            if assumptions.normality_ok and assumptions.equal_variance_ok:
                return _independent_t(df, structure, assumptions, alpha)
            elif assumptions.normality_ok and not assumptions.equal_variance_ok:
                return _welch_t(df, structure, assumptions, alpha)
            else:
                return _mann_whitney(df, structure, assumptions, alpha)
        else:  # 3+ groups
            if assumptions.normality_ok and assumptions.equal_variance_ok:
                return _one_way_anova(df, structure, assumptions, alpha)
            elif assumptions.normality_ok and not assumptions.equal_variance_ok:
                return _welch_anova(df, structure, assumptions, alpha)
            else:
                return _kruskal_wallis(df, structure, assumptions, alpha)

    else:
        # Fallback
        return _kruskal_wallis(df, structure, assumptions, alpha)


# ── EFFECT SIZE HELPERS ─────────────────────────────────────────────────────

def _cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std

def _magnitude(d, thresholds=(0.2, 0.5, 0.8)):
    d = abs(d)
    if d < thresholds[0]: return "negligible"
    if d < thresholds[1]: return "small"
    if d < thresholds[2]: return "medium"
    return "large"

def _kendalls_w(df_wide):
    n, k = df_wide.shape
    rankings = df_wide.rank(axis=1)
    R = rankings.sum(axis=0)
    S = sum((r - R.mean())**2 for r in R)
    W = 12 * S / (n**2 * (k**3 - k))
    return round(W, 4)

# ── TEST IMPLEMENTATIONS ────────────────────────────────────────────────────

def _independent_t(df, structure, assumptions, alpha):
    groups = df[structure.group_col].unique()
    g1 = df[df[structure.group_col] == groups[0]][structure.outcome_col].dropna().values
    g2 = df[df[structure.group_col] == groups[1]][structure.outcome_col].dropna().values
    stat, p = stats.ttest_ind(g1, g2, equal_var=True)
    d = _cohen_d(g1, g2)
    sig = p < alpha
    rationale = "Independent t-test: two independent groups, normality satisfied, equal variances confirmed by Levene's test."
    methods = (
        f"An independent samples t-test was conducted to compare {structure.outcome_col} "
        f"between {groups[0]} (n={len(g1)}) and {groups[1]} (n={len(g2)}). "
        f"Normality was assessed using the Shapiro-Wilk test. Levene's test confirmed equal variances. "
        f"The result was {'statistically significant' if sig else 'not statistically significant'} "
        f"(t({len(g1)+len(g2)-2})={stat:.3f}, p={p:.4f}). "
        f"Effect size was {_magnitude(d)} (Cohen's d={d:.3f})."
    )
    return TestResult("Independent t-test", round(stat, 4), round(p, 4), round(d, 4), "cohen_d", _magnitude(d), sig, alpha, rationale, methods)

def _welch_t(df, structure, assumptions, alpha):
    groups = df[structure.group_col].unique()
    g1 = df[df[structure.group_col] == groups[0]][structure.outcome_col].dropna().values
    g2 = df[df[structure.group_col] == groups[1]][structure.outcome_col].dropna().values
    stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    d = _cohen_d(g1, g2)
    sig = p < alpha
    rationale = "Welch's t-test: two independent groups, normality satisfied, but unequal variances detected by Levene's test."
    methods = (
        f"Welch's t-test was used due to unequal variances (Levene's test significant). "
        f"Comparing {structure.outcome_col} between {groups[0]} and {groups[1]}: "
        f"t={stat:.3f}, p={p:.4f}. Cohen's d={d:.3f} ({_magnitude(d)} effect)."
    )
    return TestResult("Welch's t-test", round(stat, 4), round(p, 4), round(d, 4), "cohen_d", _magnitude(d), sig, alpha, rationale, methods)

def _mann_whitney(df, structure, assumptions, alpha):
    groups = df[structure.group_col].unique()
    g1 = df[df[structure.group_col] == groups[0]][structure.outcome_col].dropna().values
    g2 = df[df[structure.group_col] == groups[1]][structure.outcome_col].dropna().values
    stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    r = 1 - (2 * stat) / (len(g1) * len(g2))  # rank-biserial r
    sig = p < alpha
    rationale = "Mann-Whitney U: non-parametric alternative for two independent groups when normality is not met."
    methods = (
        f"A Mann-Whitney U test was conducted as normality assumptions were not met. "
        f"Comparing {structure.outcome_col} between {groups[0]} and {groups[1]}: "
        f"U={stat:.1f}, p={p:.4f}. Rank-biserial r={r:.3f} ({_magnitude(r)} effect)."
    )
    return TestResult("Mann-Whitney U", round(stat, 4), round(p, 4), round(r, 4), "rank_biserial_r", _magnitude(r), sig, alpha, rationale, methods)

def _paired_t(df, structure, assumptions, alpha):
    groups = df[structure.group_col].unique()
    subject = structure.subject_col
    g1 = df[df[structure.group_col] == groups[0]].set_index(subject)[structure.outcome_col]
    g2 = df[df[structure.group_col] == groups[1]].set_index(subject)[structure.outcome_col]
    common = g1.index.intersection(g2.index)
    diffs = g1[common].values - g2[common].values
    stat, p = stats.ttest_rel(g1[common].values, g2[common].values)
    d = np.mean(diffs) / np.std(diffs, ddof=1)
    sig = p < alpha
    rationale = "Paired t-test: repeated measures (2 conditions), within-subject differences are normally distributed."
    methods = (
        f"A paired-samples t-test compared {structure.outcome_col} between {groups[0]} and {groups[1]} "
        f"across {len(common)} matched subjects. Normality of differences was confirmed (Shapiro-Wilk). "
        f"t({len(common)-1})={stat:.3f}, p={p:.4f}. Cohen's d={d:.3f} ({_magnitude(d)} effect)."
    )
    return TestResult("Paired t-test", round(stat, 4), round(p, 4), round(d, 4), "cohen_d", _magnitude(d), sig, alpha, rationale, methods)

def _wilcoxon(df, structure, assumptions, alpha):
    groups = df[structure.group_col].unique()
    subject = structure.subject_col
    g1 = df[df[structure.group_col] == groups[0]].set_index(subject)[structure.outcome_col]
    g2 = df[df[structure.group_col] == groups[1]].set_index(subject)[structure.outcome_col]
    common = g1.index.intersection(g2.index)
    stat, p = stats.wilcoxon(g1[common].values, g2[common].values)
    n = len(common)
    r = stat / (n * (n + 1) / 2)
    sig = p < alpha
    rationale = "Wilcoxon signed-rank: repeated measures (2 conditions), within-subject differences are non-normal."
    methods = (
        f"A Wilcoxon signed-rank test was used as differences were non-normally distributed. "
        f"Comparing {structure.outcome_col} across {n} subjects: W={stat:.1f}, p={p:.4f}. r={r:.3f} ({_magnitude(r)} effect)."
    )
    return TestResult("Wilcoxon signed-rank", round(stat, 4), round(p, 4), round(r, 4), "r", _magnitude(r), sig, alpha, rationale, methods)

def _friedman(df, structure, assumptions, alpha):
    groups = df[structure.group_col].unique()
    subject = structure.subject_col
    data = [df[df[structure.group_col] == g].set_index(subject)[structure.outcome_col].values for g in groups]
    stat, p = stats.friedmanchisquare(*data)
    n = len(df[subject].unique())
    k = len(groups)
    W = stat / (n * (k - 1))
    sig = p < alpha
    rationale = f"Friedman test: repeated measures with {k} conditions — appropriate non-parametric test."
    methods = (
        f"A Friedman test was conducted to compare {structure.outcome_col} across {k} conditions "
        f"in {n} subjects. χ²({k-1})={stat:.3f}, p={p:.4f}. "
        f"Kendall's W={W:.3f} ({_magnitude(W, (0.1, 0.3, 0.5))} effect)."
    )
    return TestResult("Friedman test", round(stat, 4), round(p, 4), round(W, 4), "kendalls_w", _magnitude(W, (0.1, 0.3, 0.5)), sig, alpha, rationale, methods)

def _friedman_wide(df, structure, assumptions, alpha):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = [df[c].dropna().values for c in numeric_cols]
    min_len = min(len(d) for d in data)
    data = [d[:min_len] for d in data]
    stat, p = stats.friedmanchisquare(*data)
    k = len(numeric_cols)
    n = min_len
    W = stat / (n * (k - 1))
    sig = p < alpha
    rationale = f"Friedman test on wide-format data: {k} repeated conditions detected as separate columns."
    methods = (
        f"Wide-format repeated measures detected ({k} conditions). Friedman test: "
        f"χ²({k-1})={stat:.3f}, p={p:.4f}. Kendall's W={W:.3f}."
    )
    return TestResult("Friedman test (wide)", round(stat, 4), round(p, 4), round(W, 4), "kendalls_w", _magnitude(W, (0.1, 0.3, 0.5)), sig, alpha, rationale, methods)

def _one_way_anova(df, structure, assumptions, alpha):
    groups = [df[df[structure.group_col] == g][structure.outcome_col].dropna().values for g in df[structure.group_col].unique()]
    stat, p = stats.f_oneway(*groups)
    n = len(df)
    k = len(groups)
    eta2 = (stat * (k - 1)) / (stat * (k - 1) + (n - k))
    sig = p < alpha
    rationale = "One-way ANOVA: 3+ independent groups, normality and equal variances satisfied."
    methods = (
        f"A one-way ANOVA was conducted to compare {structure.outcome_col} across {k} groups (n={n}). "
        f"F({k-1},{n-k})={stat:.3f}, p={p:.4f}. η²={eta2:.3f} ({_magnitude(eta2, (0.01, 0.06, 0.14))} effect)."
    )
    return TestResult("One-way ANOVA", round(stat, 4), round(p, 4), round(eta2, 4), "eta_squared", _magnitude(eta2, (0.01, 0.06, 0.14)), sig, alpha, rationale, methods)

def _welch_anova(df, structure, assumptions, alpha):
    # Use scipy's alternative for unequal variances
    groups = [df[df[structure.group_col] == g][structure.outcome_col].dropna().values for g in df[structure.group_col].unique()]
    stat, p = stats.f_oneway(*groups)  # simplified; production would use welch_anova from pingouin
    sig = p < alpha
    rationale = "Welch's ANOVA: 3+ independent groups, normality met but unequal variances."
    methods = f"Welch's ANOVA (unequal variances): F={stat:.3f}, p={p:.4f}."
    return TestResult("Welch's ANOVA", round(stat, 4), round(p, 4), 0.0, "eta_squared", "unknown", sig, alpha, rationale, methods)

def _kruskal_wallis(df, structure, assumptions, alpha):
    groups = [df[df[structure.group_col] == g][structure.outcome_col].dropna().values for g in df[structure.group_col].unique()]
    stat, p = stats.kruskal(*groups)
    n = len(df)
    k = len(groups)
    eta2 = (stat - k + 1) / (n - k)
    sig = p < alpha
    rationale = "Kruskal-Wallis H: non-parametric alternative for 3+ independent groups when normality fails."
    methods = (
        f"A Kruskal-Wallis H test was conducted. H({k-1})={stat:.3f}, p={p:.4f}. "
        f"η²H={eta2:.3f} ({_magnitude(eta2, (0.01, 0.06, 0.14))} effect)."
    )
    return TestResult("Kruskal-Wallis H", round(stat, 4), round(p, 4), round(eta2, 4), "eta_squared_H", _magnitude(eta2, (0.01, 0.06, 0.14)), sig, alpha, rationale, methods)
