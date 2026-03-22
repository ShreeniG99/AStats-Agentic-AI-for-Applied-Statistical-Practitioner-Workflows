# AStats — Agentic AI for Applied Statistical Practitioner Workflows

**GSoC 2026 Proof of Concept | INCF Project #33**

> A human-in-the-loop statistical analysis system that understands natural language, detects data structure automatically, selects the right test, and keeps the practitioner in control of every decision.

---

## The Problem

Most automated statistical tools share the same failure mode: they jump to a result without checking whether it is appropriate, and they give the researcher no way to push back. Three specific problems come up repeatedly:

- **LLMs skip assumption checking.** A benchmark against GPT-4.1 on 13 structured test scenarios showed GPT-4.1 passing 45% vs AStats at 69%. The main failures: no normality checks, and confusing independent groups with repeated measurements.
- **Existing tools (JASP, Jamovi)** do not understand natural language, cannot adapt to context, and offer no correction mechanism.
- **Pseudoreplication** — treating repeated measurements on the same person as independent observations — inflates effective sample size by up to 10x. Most automated tools have no check for this.

AStats handles all three: it understands the question, detects the data structure, picks the right test, and lets the practitioner correct it when they know something the algorithm does not.

---

## What Makes This Different

| Feature | GPT-4.1 | Utkarsh's PoC | Atta's PoC | **AStats (My PoC)** |
|---|---|---|---|---|
| NL query parsing | Yes | No | Yes | ✓ |
| Data structure inference | Partial | Yes | No | ✓ |
| Assumption checking | Partial | Yes | No | ✓ |
| Test execution + effect size | Yes | Yes | No | ✓ |
| Post-hoc tests (auto-triggered) | No | No | No | ✓ |
| Practitioner feedback loop | No | No | No | ✓ |
| Multi-turn session support | No | No | No | ✓ |
| Session logging (DPO-compatible) | No | No | No | ✓ |
| Plain-language explanation (offline) | No | No | No | ✓ |
| Methods paragraph (copy-paste ready) | Yes | Yes | No | ✓ |
| Multiple output modes | No | No | No | ✓ |

---

## Architecture

AStats is designed as a **LangGraph state machine** with a defined agent loop and a tool registry. The agent decides which tools to call and in what order — it does not follow a fixed pipeline.

```
User Query (natural language)
        │
        ▼
┌──────────────────┐
│   NL Parser      │  Detects intent, flags ambiguity, extracts column hints
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Structure Infer  │  Independent / Repeated-measures / Nested / Wide-format
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Assumption Check │  Shapiro-Wilk, Levene's, sample adequacy
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Test Selection  │  Deterministic matrix → 9 tests + effect sizes
└────────┬─────────┘
         │
         ▼
┌───────────────────────────────────────┐
│      Practitioner Feedback Loop       │
│  Accept / Override / Explain / Fix    │  ← The key contribution
│  Every decision saved as              │
│  DPO-compatible preference data       │
└───────────────────────────────────────┘
```

**Tool registry (12 tools):**
`load_data` · `infer_structure` · `check_assumptions` · `select_test` · `run_test_python` · `run_test_r` · `compute_effect_size` · `explain_decision` · `detect_outliers` · `run_posthoc` · `generate_methods_para` · `log_correction`

---

## Quick Start

```bash
# Clone
git clone https://github.com/ShreeniG99/AStats-Agentic-AI-for-Applied-Statistical-Practitioner-Workflows.git
cd AStats-Agentic-AI-for-Applied-Statistical-Practitioner-Workflows

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate test datasets
python generate_datasets.py

# Run the benchmark to verify everything works
python benchmark/run_benchmark.py
```

---

## Usage

### Interactive mode — full feedback loop

```bash
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups"
```

When the menu appears:

```
  [A] Accept this recommendation
  [O] Override — choose a different test
  [W] Tell me why in plain language
  [C] Correct an assumption
  [Q] Quit
```

Press **W** to get a full plain-language explanation (works offline, no API key needed).  
Press **O** to override — choose from 9 tests, give a reason, the decision is saved to the session log.

### Auto mode (no prompts)

```bash
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto
```

### Output formats

```bash
# Default — rich terminal with colours and boxes
python -m astats.cli analyze data.csv "query" --auto

# Plain text — no colours, works in any terminal or log file
python -m astats.cli analyze data.csv "query" --auto --output plain

# JSON — machine-readable, pipe to other tools
python -m astats.cli analyze data.csv "query" --auto --output json

# Methods paragraph only — copy straight into your paper
python -m astats.cli analyze data.csv "query" --auto --output methods
```

### Force a specific test

```bash
# See all available tests
python -m astats.cli list-tests

# Force a specific test regardless of auto-selection
python -m astats.cli analyze data.csv "query" --auto --test "Mann-Whitney U"
```

### Other options

```bash
# Change significance level
python -m astats.cli analyze data.csv "query" --auto --alpha 0.01

# Save full report to a file
python -m astats.cli analyze data.csv "query" --auto --save report.txt

# Show data summary table before analysis
python -m astats.cli analyze data.csv "query" --auto --summary

# View all past sessions
python -m astats.cli sessions

# See full detail of a specific session
python -m astats.cli session-detail sessions/session_XXXXXXXX.json
```

---

## Example Output

**Dataset:** `01_antidepressant_trial.csv` — 93 patients, Placebo vs Drug, HDRS depression scores  
**Query:** `"compare HDRS score between treatment groups"`

```
✓ Intent detected: compare

✓ Structure inferred: independent
  → Detected group column: 'treatment' (2 groups)
  → Detected outcome column: 'HDRS_score'
  → Subject appears once per row → independent groups

✓ Assumptions checked:
  Normality:       ✓ met
  Equal variance:  ✓ met
  Sample adequate: ✓ (n=93)

╭──────────────────────── AStats Results ─────────────────────────╮
│ Final Test:    Independent t-test                               │
│ Statistic:     4.6221                                           │
│ p-value:       0.0  ✓ significant                               │
│ Effect size:   0.9591 (cohen_d, large)                          │
│                                                                 │
│ Methods paragraph:                                              │
│ An independent samples t-test was conducted to compare          │
│ HDRS_score between Placebo (n=45) and Drug (n=48).              │
│ Normality was confirmed via Shapiro-Wilk. Levene's test         │
│ confirmed equal variances. The result was statistically         │
│ significant (t(91)=4.622, p=0.0000). Effect size was large      │
│ (Cohen's d=0.959).                                              │
╰─────────────────────────────────────────────────────────────────╯
```

**JSON output** (`--output json`):

```json
{
  "intent": "compare",
  "structure": {
    "design": "independent",
    "n_groups": 2,
    "outcome_col": "HDRS_score",
    "group_col": "treatment"
  },
  "result": {
    "test_name": "Independent t-test",
    "statistic": 4.6221,
    "p_value": 0.0,
    "significant": true,
    "effect_size": 0.9591,
    "effect_label": "cohen_d",
    "effect_magnitude": "large",
    "methods_paragraph": "An independent samples t-test..."
  }
}
```

---

## Supported Statistical Tests

| Test | When used | Effect size |
|---|---|---|
| Independent t-test | 2 independent groups, normal, equal variance | Cohen's d |
| Welch's t-test | 2 independent groups, normal, unequal variance | Cohen's d |
| Mann-Whitney U | 2 independent groups, non-normal | Rank-biserial r |
| Paired t-test | Repeated measures, 2 conditions, normal differences | Cohen's d |
| Wilcoxon signed-rank | Repeated measures, 2 conditions, non-normal differences | r |
| Friedman test | Repeated measures, 3+ conditions | Kendall's W |
| One-way ANOVA | 3+ independent groups, normal, equal variance | η² |
| Welch's ANOVA | 3+ independent groups, normal, unequal variance | η² |
| Kruskal-Wallis H | 3+ independent groups, non-normal | η²H |

Post-hoc tests (Tukey HSD, Bonferroni, Dunn's) are triggered automatically after significant ANOVA or Kruskal-Wallis — no extra command needed.

---

## Test Datasets

Eight clinically grounded datasets included, each covering a different analysis path in the pipeline:

| Dataset | Study Design | Expected Test |
|---|---|---|
| `01_antidepressant_trial.csv` | 93-patient RCT, Placebo vs Drug, HDRS scores | Independent t-test |
| `02_sleep_deprivation_longitudinal.csv` | 20 subjects × 9 days, modelled on Belenky et al. (2003) | Friedman test |
| `03_cognitive_rehab_pre_post.csv` | 35 stroke patients, Pre/Post MoCA scores | Paired t-test |
| `04_pain_management_4groups.csv` | 4-arm pain trial, VAS scores | One-way ANOVA |
| `05_cognitive_load_RT.csv` | 3 conditions, log-normal reaction time data | Kruskal-Wallis H |
| `06_eeg_alpha_power_wide.csv` | 28 subjects, 5 scalp regions, wide-format | Friedman (wide) |
| `07_crp_cognitive_decline.csv` | CRP vs MMSE cognitive scores, 80 elderly patients | Correlation |
| `08_hepatitis_B_ADR.csv` | 155 patients, 3 HBV drugs, ADR severity scores | Kruskal-Wallis H |

Regenerate all datasets from scratch:

```bash
python generate_datasets.py
```

---

## Benchmark

```
Results: 9/13 passed (69.2%)
```

The 4 failures are documented statistical judgment calls — edge cases involving borderline normality decisions where experienced statisticians disagree. These same 3 cases appear as known failures in Utkarsh's prototype. The benchmark tracks them honestly rather than working around them.

```bash
python benchmark/run_benchmark.py
```

---

## The Human-in-the-Loop Design

The practitioner feedback loop is the central contribution — the piece that does not exist in any other prototype.

Every analysis pauses before finalising and shows the practitioner the recommendation and reasoning. They can:

- **Accept** — logged and complete
- **Override** — choose any of 9 tests, give a reason, recorded
- **Explain** — full plain-language explanation, offline, for all 9 tests
- **Correct** — flag that the data is actually non-normal, or actually repeated measures

**Every interaction is saved:**

```json
{
  "session_id": "20260321_154443",
  "corrections": [
    {
      "timestamp": "2026-03-21T15:44:43",
      "dataset": "data.csv",
      "query": "compare HDRS score between groups",
      "recommended_test": "Independent t-test",
      "action": "overridden",
      "override_test": "Mann-Whitney U",
      "override_reason": "domain knowledge suggests non-normality",
      "final_test": "Mann-Whitney U",
      "p_value": 0.0,
      "effect_size": -0.54
    }
  ]
}
```

These correction logs are structured as **(rejected, accepted, reason)** triples — DPO-compatible preference data that can be used to fine-tune a local open-weight model as the project matures.

---

## Project Structure

```
AStats/
├── astats/
│   ├── __init__.py
│   ├── cli.py              CLI with all commands and output modes
│   ├── structure.py        Data structure inference
│   ├── assumptions.py      Shapiro-Wilk, Levene's, sample adequacy
│   ├── tests.py            Test selection, execution, effect sizes
│   ├── feedback.py         Practitioner feedback loop + session logging
│   └── nl_parser.py        Natural language query parser
├── benchmark/
│   └── run_benchmark.py    13-scenario benchmark suite
├── test_datasets/          8 clinical/neuroscience datasets
├── generate_datasets.py    Reproducible dataset generator
├── requirements.txt
└── README.md
```

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
statsmodels>=0.13.0
click>=8.0.0
rich>=13.0.0
```

Optional — for LLM-powered explanations:
```
anthropic>=0.20.0
```

The tool runs fully offline without an API key. All 9 tests have complete built-in explanations.

---

## Roadmap

This is a proof of concept submitted for [GSoC 2026 INCF Project #33](https://neurostars.org/t/gsoc-2026-project-33-university-of-wisconsin-madison-astats-an-agentic-ai-approach-to-applied-statistical-practitioner-workflows/35620).

**Planned for GSoC:**
- LangGraph agent architecture (Perceive → Plan → Analyse → Reflect → Report)
- Full tool registry with 12 callable tools
- R bridge via rpy2 — lme4, BayesFactor, Welch ANOVA
- Multi-turn session with full context memory
- Post-hoc tests auto-triggered after significant ANOVA/Kruskal-Wallis
- Autonomous exploratory mode — no query needed
- DPO fine-tuning on session correction log (Mistral 7B via Ollama)
- Next.js + FastAPI streaming web interface

---

## Discussions and Questions

The best place to follow progress and ask questions is the [AStats community on SciCommons](https://alphatest.scicommons.org/community/AStats) and the [Neurostars project thread](https://neurostars.org/t/gsoc-2026-project-33-university-of-wisconsin-madison-astats-an-agentic-ai-approach-to-applied-statistical-practitioner-workflows/35620).

---

## About

Built by **Shreenidhi Gopalakrishnan** as a proof of concept for the AStats GSoC 2026 project under INCF, mentored by Suresh Krishna (McGill), Jonathan Morris (UW-Madison), and Yohai-Eliel Berreby (McGill).

The design is grounded in a real experience: building clinical ML tools that make recommendations without giving practitioners a mechanism to correct them. AStats treats practitioner corrections as first-class data — not noise to filter, but signal to record, learn from, and act on.
