"""
AStats Test Dataset Generator
==============================
Generates realistic, clinically/neuroscience-grounded datasets that
demonstrate every analysis path in the AStats pipeline.

Each dataset is based on a real study design pattern used in published research.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

np.random.seed(2026)  # reproducible
os.makedirs("test_datasets", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 1: Drug Trial — Two Independent Groups (t-test path)
# Based on: antidepressant efficacy RCT design
# Groups: placebo vs drug. Outcome: Hamilton Depression Rating Scale (HDRS)
# Expected: Independent t-test (both groups normal, equal variance)
# ═══════════════════════════════════════════════════════════════════════════

n_placebo = 45
n_drug = 48

placebo_hdrs = np.random.normal(loc=18.2, scale=4.1, size=n_placebo).clip(0, 52)
drug_hdrs    = np.random.normal(loc=13.7, scale=4.3, size=n_drug).clip(0, 52)

df1 = pd.DataFrame({
    "patient_id": [f"P{i:03d}" for i in range(1, n_placebo + n_drug + 1)],
    "treatment":  ["Placebo"] * n_placebo + ["Drug"] * n_drug,
    "age":        list(np.random.randint(22, 65, n_placebo)) + list(np.random.randint(24, 67, n_drug)),
    "sex":        list(np.random.choice(["M", "F"], n_placebo, p=[0.42, 0.58])) +
                  list(np.random.choice(["M", "F"], n_drug, p=[0.44, 0.56])),
    "HDRS_score": list(np.round(placebo_hdrs, 1)) + list(np.round(drug_hdrs, 1)),
    "baseline_HDRS": list(np.round(np.random.normal(22, 3.5, n_placebo), 1)) +
                     list(np.round(np.random.normal(21.8, 3.7, n_drug), 1)),
    "weeks_in_trial": list(np.random.choice([8, 10, 12], n_placebo, p=[0.2, 0.5, 0.3])) +
                      list(np.random.choice([8, 10, 12], n_drug, p=[0.15, 0.55, 0.3]))
})

df1.to_csv("test_datasets/01_antidepressant_trial.csv", index=False)
print(f"✓ Dataset 1: Antidepressant trial — {len(df1)} rows")
print(f"  Placebo HDRS mean: {placebo_hdrs.mean():.2f} (SD={placebo_hdrs.std():.2f})")
print(f"  Drug HDRS mean:    {drug_hdrs.mean():.2f} (SD={drug_hdrs.std():.2f})")
print(f"  Expected path: Independent t-test")
print()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 2: Sleep Study — Repeated Measures (Friedman / Paired path)
# Based on: classic Belenky et al. (2003) sleep deprivation study design
# 20 subjects measured across 9 days of sleep restriction
# Outcome: Psychomotor Vigilance Task reaction time (ms)
# Expected: Friedman test (repeated measures, 9 conditions)
# ═══════════════════════════════════════════════════════════════════════════

n_subjects = 20
days = list(range(0, 9))  # Day 0 (baseline) through Day 8

subject_baselines = np.random.normal(250, 30, n_subjects)  # individual baseline RT
subject_sensitivity = np.random.uniform(0.5, 2.5, n_subjects)  # how much they degrade

records = []
for s in range(n_subjects):
    for d in days:
        degradation = subject_sensitivity[s] * d * 4.2  # RT worsens with days
        noise = np.random.normal(0, 12)
        rt = subject_baselines[s] + degradation + noise
        records.append({
            "Subject": f"S{s+1:02d}",
            "Day": d,
            "ReactionTime_ms": round(max(150, rt), 2),
            "Lapses": max(0, int(np.random.poisson(d * 0.8))),
            "KSS_sleepiness": min(9, max(1, int(1 + d * 0.6 + np.random.normal(0, 0.8)))),
            "SleepHours": round(max(3, min(5, 5 - d * 0.12 + np.random.normal(0, 0.3))), 1)
        })

df2 = pd.DataFrame(records)
df2.to_csv("test_datasets/02_sleep_deprivation_longitudinal.csv", index=False)
print(f"✓ Dataset 2: Sleep deprivation study — {len(df2)} rows ({n_subjects} subjects × {len(days)} days)")
print(f"  RT Day 0 mean: {df2[df2.Day==0].ReactionTime_ms.mean():.1f} ms")
print(f"  RT Day 8 mean: {df2[df2.Day==8].ReactionTime_ms.mean():.1f} ms")
print(f"  Expected path: Friedman test (repeated measures, 9 conditions)")
print()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 3: Cognitive Rehabilitation — Pre/Post (Paired t-test path)
# Stroke patients measured before and after 8-week cognitive rehab program
# Outcome: Montreal Cognitive Assessment (MoCA) score
# Expected: Paired t-test (same patients, 2 timepoints, normal differences)
# ═══════════════════════════════════════════════════════════════════════════

n_patients = 35
patient_ids = [f"PT{i:03d}" for i in range(1, n_patients + 1)]
pre_moca  = np.random.normal(21.5, 3.2, n_patients).clip(10, 30)
improvement = np.random.normal(2.8, 1.6, n_patients)  # real rehab improvement
post_moca = (pre_moca + improvement).clip(10, 30)

df3_pre  = pd.DataFrame({"patient_id": patient_ids, "timepoint": "Pre",
                          "MoCA_score": np.round(pre_moca, 1),
                          "age": np.random.randint(52, 79, n_patients),
                          "months_post_stroke": np.random.randint(2, 18, n_patients),
                          "rehab_sessions_attended": [np.nan] * n_patients})
df3_post = pd.DataFrame({"patient_id": patient_ids, "timepoint": "Post",
                          "MoCA_score": np.round(post_moca, 1),
                          "age": df3_pre["age"].values,
                          "months_post_stroke": df3_pre["months_post_stroke"].values,
                          "rehab_sessions_attended": np.random.randint(12, 25, n_patients)})
df3 = pd.concat([df3_pre, df3_post], ignore_index=True)
df3.to_csv("test_datasets/03_cognitive_rehab_pre_post.csv", index=False)
print(f"✓ Dataset 3: Cognitive rehabilitation — {len(df3)} rows ({n_patients} patients × 2 timepoints)")
print(f"  Pre MoCA mean:  {pre_moca.mean():.2f}")
print(f"  Post MoCA mean: {post_moca.mean():.2f}")
print(f"  Expected path: Paired t-test")
print()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 4: Multi-Drug Comparison — 3+ Groups (One-way ANOVA path)
# Comparing 4 treatment arms in a pain management trial
# Outcome: Visual Analogue Scale (VAS) pain score (0–100)
# Expected: One-way ANOVA (4 normal groups, equal variances)
# ═══════════════════════════════════════════════════════════════════════════

group_means = {"Placebo": 62.0, "Ibuprofen_400mg": 44.5,
               "Ibuprofen_800mg": 38.2, "Celecoxib_200mg": 40.8}
n_per_group = 30

pain_records = []
for drug, mean_pain in group_means.items():
    for i in range(n_per_group):
        pain_records.append({
            "patient_id": f"{drug[:3].upper()}{i+1:03d}",
            "treatment": drug,
            "VAS_pain": round(np.clip(np.random.normal(mean_pain, 12), 0, 100), 1),
            "age": np.random.randint(25, 70),
            "sex": np.random.choice(["M", "F"]),
            "baseline_VAS": round(np.clip(np.random.normal(72, 8), 40, 100), 1),
            "hours_post_dose": np.random.choice([2, 4, 6])
        })

df4 = pd.DataFrame(pain_records)
df4.to_csv("test_datasets/04_pain_management_4groups.csv", index=False)
print(f"✓ Dataset 4: Pain management 4-arm trial — {len(df4)} rows")
for drug, grp in df4.groupby("treatment"):
    print(f"  {drug}: VAS = {grp.VAS_pain.mean():.1f} ± {grp.VAS_pain.std():.1f}")
print(f"  Expected path: One-way ANOVA")
print()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 5: Reaction Time by Condition — Non-normal (Kruskal-Wallis path)
# Three experimental conditions with skewed RT distributions (realistic!)
# RT data is almost always right-skewed in cognitive science
# Expected: Kruskal-Wallis (non-normal, 3 groups)
# ═══════════════════════════════════════════════════════════════════════════

n_per_cond = 40
conditions  = {
    "Control":      {"loc": 280, "scale": 30},
    "Single_task":  {"loc": 320, "scale": 50},
    "Dual_task":    {"loc": 420, "scale": 90},
}

rt_records = []
for cond, params in conditions.items():
    rts = stats.lognorm.rvs(s=0.35, scale=params["loc"], size=n_per_cond)
    for i, rt in enumerate(rts):
        rt_records.append({
            "participant_id": f"{cond[:3].upper()}{i+1:03d}",
            "condition": cond,
            "RT_ms": round(rt, 1),
            "accuracy_pct": round(np.clip(np.random.normal(92 - (i % 5), 6), 60, 100), 1),
            "trial_block": np.random.randint(1, 5)
        })

df5 = pd.DataFrame(rt_records)
df5.to_csv("test_datasets/05_cognitive_load_RT.csv", index=False)
print(f"✓ Dataset 5: Cognitive load reaction times — {len(df5)} rows (skewed, non-normal)")
for cond, grp in df5.groupby("condition"):
    sk = stats.skew(grp.RT_ms)
    print(f"  {cond}: RT = {grp.RT_ms.mean():.0f} ms (skew={sk:.2f})")
print(f"  Expected path: Kruskal-Wallis H")
print()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 6: Neurofeedback EEG — Wide Format Repeated Measures
# Alpha power measured at 5 electrode sites across the scalp
# Each row is one subject; each column is a brain region
# Expected: Friedman test (wide format repeated measures)
# ═══════════════════════════════════════════════════════════════════════════

n_subjects = 28
regions = ["Frontal_F3", "Frontal_F4", "Central_Cz", "Parietal_Pz", "Occipital_Oz"]

eeg_data = {}
eeg_data["subject_id"] = [f"EEG{i:03d}" for i in range(1, n_subjects + 1)]
eeg_data["group"] = np.random.choice(["Control", "ADHD"], n_subjects, p=[0.5, 0.5])
eeg_data["age"] = np.random.randint(18, 45, n_subjects)

# Alpha power typically highest occipital, lowest frontal
region_means = {"Frontal_F3": 8.2, "Frontal_F4": 8.5, "Central_Cz": 10.1,
                "Parietal_Pz": 11.8, "Occipital_Oz": 15.3}

for reg, mean_power in region_means.items():
    eeg_data[reg] = np.round(np.random.normal(mean_power, 2.5, n_subjects).clip(2, 30), 3)

df6 = pd.DataFrame(eeg_data)
df6.to_csv("test_datasets/06_eeg_alpha_power_wide.csv", index=False)
print(f"✓ Dataset 6: EEG alpha power (wide format) — {len(df6)} subjects × {len(regions)} regions")
for reg in regions:
    print(f"  {reg}: {df6[reg].mean():.2f} µV²")
print(f"  Expected path: Friedman test (wide format)")
print()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 7: Biomarker Study — Correlation / Regression
# Serum CRP (inflammation marker) vs. cognitive decline score
# Designed for correlation analysis, with realistic noise and outliers
# Expected: intent='correlate'
# ═══════════════════════════════════════════════════════════════════════════

n = 80
age = np.random.randint(60, 85, n)
crp = np.random.gamma(shape=2, scale=1.5, size=n)  # right-skewed, realistic
mmse = 28 - 0.08 * age - 0.9 * crp + np.random.normal(0, 2, n)
mmse = np.clip(np.round(mmse, 1), 0, 30)

# Add 3 realistic outliers (severe disease)
crp[5] = 18.4
mmse[5] = 14.0
crp[23] = 22.1
mmse[23] = 11.0
crp[61] = 15.8
mmse[61] = 16.0

df7 = pd.DataFrame({
    "patient_id": [f"BIO{i:03d}" for i in range(1, n + 1)],
    "age": age,
    "sex": np.random.choice(["M", "F"], n),
    "CRP_mg_L": np.round(crp, 2),
    "MMSE_score": mmse,
    "years_education": np.random.normal(12, 3, n).clip(6, 22).astype(int),
    "BMI": np.round(np.random.normal(27.5, 4.5, n).clip(18, 45), 1),
    "diagnosis": np.random.choice(["Healthy", "MCI", "Early_AD"], n, p=[0.4, 0.35, 0.25])
})

df7.to_csv("test_datasets/07_crp_cognitive_decline.csv", index=False)
print(f"✓ Dataset 7: CRP vs cognitive decline — {n} patients")
r, p = stats.pearsonr(crp, mmse)
print(f"  CRP–MMSE correlation: r={r:.3f}, p={p:.4f}")
print(f"  Expected path: Correlation / Regression analysis")
print()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET 8: Adverse Drug Reaction — Real FAERS-style structure
# This one mirrors Shreenidhi's own FAERS project
# Patients grouped by drug, outcome is adverse event severity score
# Non-normal (realistic clinical data), 3 drugs
# Expected: Kruskal-Wallis + SHAP-style exploration
# ═══════════════════════════════════════════════════════════════════════════

drugs = {
    "Tenofovir":    {"n": 55, "adr_mean": 2.1, "adr_scale": 1.8},
    "Lamivudine":   {"n": 48, "adr_mean": 1.6, "adr_scale": 1.2},
    "Entecavir":    {"n": 52, "adr_mean": 1.1, "adr_scale": 0.9},
}

adr_records = []
pid = 1
for drug, params in drugs.items():
    n_d = params["n"]
    adr_scores = stats.lognorm.rvs(s=0.7, scale=params["adr_mean"], size=n_d).clip(0, 10)
    for i in range(n_d):
        adr_records.append({
            "patient_id": f"HBV{pid:04d}",
            "drug": drug,
            "ADR_severity_score": round(adr_scores[i], 2),
            "age": np.random.randint(28, 72),
            "sex": np.random.choice(["M", "F"], p=[0.58, 0.42]),
            "HBV_DNA_log": round(np.random.normal(5.2, 1.8), 2),
            "ALT_UxN": round(np.random.gamma(3, 0.8), 2),
            "treatment_months": np.random.choice([6, 12, 18, 24, 36],
                                                  p=[0.1, 0.25, 0.3, 0.25, 0.1]),
            "renal_function": np.random.choice(["Normal", "Mildly_reduced", "Moderately_reduced"],
                                                p=[0.65, 0.25, 0.10])
        })
        pid += 1

df8 = pd.DataFrame(adr_records)
df8.to_csv("test_datasets/08_hepatitis_B_ADR.csv", index=False)
print(f"✓ Dataset 8: Hepatitis B ADR severity — {len(df8)} patients (mirrors your FAERS project)")
for drug, grp in df8.groupby("drug"):
    print(f"  {drug}: ADR score = {grp.ADR_severity_score.mean():.2f} ± {grp.ADR_severity_score.std():.2f}")
print(f"  Expected path: Kruskal-Wallis H (non-normal, 3 drugs)")
print()


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("═" * 60)
print("  All 8 datasets created in ./test_datasets/")
print("═" * 60)
print("""
  DATASET INDEX
  ─────────────────────────────────────────────────────────
  01_antidepressant_trial.csv          → Independent t-test
  02_sleep_deprivation_longitudinal.csv → Friedman test
  03_cognitive_rehab_pre_post.csv       → Paired t-test
  04_pain_management_4groups.csv        → One-way ANOVA
  05_cognitive_load_RT.csv              → Kruskal-Wallis H
  06_eeg_alpha_power_wide.csv           → Friedman (wide fmt)
  07_crp_cognitive_decline.csv          → Correlation/Regression
  08_hepatitis_B_ADR.csv                → Kruskal-Wallis H
  ─────────────────────────────────────────────────────────

  TRY THESE COMMANDS IN ASTATS:

  python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv \\
      "compare HDRS score between treatment groups" --auto

  python -m astats.cli analyze test_datasets/02_sleep_deprivation_longitudinal.csv \\
      "compare ReactionTime across days within subjects" --auto

  python -m astats.cli analyze test_datasets/03_cognitive_rehab_pre_post.csv \\
      "did MoCA scores improve from Pre to Post" --auto

  python -m astats.cli analyze test_datasets/08_hepatitis_B_ADR.csv \\
      "compare ADR severity scores across drugs" --auto
""")
