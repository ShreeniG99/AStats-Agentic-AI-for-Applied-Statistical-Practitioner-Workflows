"""
Structure Inference Module
Detects data design from a pandas DataFrame before any test is selected.
Handles: independent, repeated-measures, nested, wide-format repeated.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataStructure:
    design: str              # 'independent' | 'repeated_measures' | 'nested' | 'wide_repeated'
    n_groups: int
    n_subjects: Optional[int]
    subject_col: Optional[str]
    group_col: Optional[str]
    outcome_col: Optional[str]
    confidence: float
    notes: list


SUBJECT_KEYWORDS = ["subject", "participant", "patient", "person", "pid"]

DEMOGRAPHIC_KEYWORDS = [
    "age", "year", "id", "pid", "sex", "gender", "code",
    "week", "month", "day", "hour", "session", "trial",
    "block", "visit", "baseline", "date", "time", "duration",
    "number", "count", "index", "record", "log", "dna", "rna", "hbv", "alt", "alat", "bmi", "weight", "height", "dose"
]


def _looks_like_subject_col(col_name: str, series: pd.Series) -> bool:
    """Returns True if column looks like a subject/participant ID."""
    name_lower = col_name.lower()
    has_subject_keyword = any(kw in name_lower for kw in SUBJECT_KEYWORDS)
    if not has_subject_keyword:
        return False
    uniqueness_ratio = series.nunique() / len(series)
    return uniqueness_ratio > 0.05


def infer_structure(df: pd.DataFrame, query_hint: dict = None) -> DataStructure:
    notes = []
    subject_col = None
    group_col = None
    outcome_col = None
    confidence = 0.7

    # Step 1: query hints
    if query_hint:
        subject_col = query_hint.get("subject_col")
        group_col   = query_hint.get("group_col")
        outcome_col = query_hint.get("outcome_col")

    # Step 2: detect subject column
    if not subject_col:
        for col in df.columns:
            if _looks_like_subject_col(col, df[col]):
                subject_col = col
                notes.append(f"Detected subject column: '{col}'")
                break

    # Step 3: detect group column
    if not group_col:
        for col in df.columns:
            if col == subject_col:
                continue
            is_categorical = df[col].dtype == object
            is_low_cardinality = (df[col].dtype != object) and (df[col].nunique() <= 10)
            if (is_categorical or is_low_cardinality) and df[col].nunique() >= 2:
                group_col = col
                notes.append(f"Detected group column: '{col}' ({df[col].nunique()} groups)")
                break

    # Step 4: detect outcome column
    if not outcome_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for remove in [subject_col, group_col]:
            if remove in numeric_cols:
                numeric_cols.remove(remove)
        clean_cols = [
            c for c in numeric_cols
            if not any(kw in c.lower() for kw in DEMOGRAPHIC_KEYWORDS)
        ]
        candidate_cols = clean_cols if clean_cols else numeric_cols
        if candidate_cols:
            outcome_col = max(candidate_cols, key=lambda c: df[c].std())
            notes.append(f"Detected outcome column: '{outcome_col}'")

    # Step 5: determine design
    n_groups = df[group_col].nunique() if group_col else 1

    if subject_col:
        obs_per_subject = df.groupby(subject_col).size()
        min_obs = obs_per_subject.min()
        max_obs = obs_per_subject.max()
        mean_obs = obs_per_subject.mean()

        if min_obs <= 1:
            # Each subject appears only once — just an ID column, not repeated measures
            design = "independent"
            n_subjects = None
            subject_col = None
            notes.append(
                "Subject column found but each subject appears only once "
                "→ independent groups (subject column is just a patient ID)"
            )
            confidence = 0.9

        elif min_obs == max_obs:
            design = "repeated_measures"
            n_subjects = df[subject_col].nunique()
            confidence = 0.95
            notes.append(
                f"Each of {n_subjects} subjects has {min_obs} observations "
                f"→ repeated measures"
            )

        elif obs_per_subject.std() < mean_obs * 0.15:
            design = "repeated_measures"
            n_subjects = df[subject_col].nunique()
            confidence = 0.80
            notes.append(
                f"Subjects have {min_obs}–{max_obs} observations each "
                f"→ likely repeated measures"
            )

        else:
            design = "nested"
            n_subjects = df[subject_col].nunique()
            confidence = 0.70
            notes.append(
                f"Unequal observations per subject ({min_obs}–{max_obs}) "
                f"→ nested/hierarchical structure"
            )
    else:
        design = "independent"
        n_subjects = None
        notes.append("No subject column found → treating as independent groups")

    # Step 6: wide-format repeated measures
    # Trigger when 3+ clean numeric columns exist, even if a group column was found.
    # This handles datasets like EEG where each column = a brain region/timepoint.
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    clean_numeric = [
        c for c in numeric_cols_all
        if not any(kw in c.lower() for kw in DEMOGRAPHIC_KEYWORDS)
        and c != subject_col and c != group_col
    ]
    # Only switch to wide_repeated if there are multiple clean numeric columns
    # AND no single outcome column was clearly identified as "the" outcome
    # (i.e., the outcome is ambiguous because all numeric cols look like measurements)
    if len(clean_numeric) >= 3 and design == "independent":
        # Check: are the clean columns all similarly named (e.g. all electrode sites)?
        # Simple heuristic: if there is no group col OR the clean cols outnumber groups
        if group_col is None or len(clean_numeric) >= 3:
            design = "wide_repeated"
            n_groups = len(clean_numeric)
            outcome_col = None  # outcome is all the measurement columns together
            notes.append(
                f"Multiple numeric outcome columns detected ({', '.join(clean_numeric[:3])}{'...' if len(clean_numeric)>3 else ''}) "
                f"→ wide-format repeated measures ({n_groups} conditions)"
            )

    return DataStructure(
        design=design,
        n_groups=n_groups,
        n_subjects=n_subjects,
        subject_col=subject_col,
        group_col=group_col,
        outcome_col=outcome_col,
        confidence=confidence,
        notes=notes
    )
