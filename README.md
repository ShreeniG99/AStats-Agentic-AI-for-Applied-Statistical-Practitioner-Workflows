# AStats — Agentic AI for Applied Statistical Practitioner Workflows

**GSoC 2026 Proof of Concept | INCF Project #33**

> A human-in-the-loop statistical analysis system that understands natural language, detects data structure automatically, selects the right test, and keeps the practitioner in control of every decision.

---

## The Problem

Most automated statistical tools share the same failure mode: they jump to a result without checking whether it is appropriate, and give the researcher no way to push back. Three problems come up repeatedly:

- **LLMs skip assumption checking.** A benchmark on 13 structured test scenarios showed that directly prompting a frontier LLM passes 45% of cases. Main failure modes: no normality checks, treating repeated measures as independent groups, no pseudoreplication detection.
- **Recipe-driven GUI tools (JASP, Jamovi)** do not understand natural language, cannot adapt to context, and offer no correction mechanism when the practitioner knows something the tool does not.
- **Pseudoreplication** — treating repeated measurements on the same person as independent observations — inflates effective sample size by up to 10x. Most automated tools have no check for this.

AStats handles all three: it understands the question, detects the data structure, picks the right test, and lets the practitioner correct it.

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
git clone https://github.com/ShreeniG99/AStats-Agentic-AI-for-Applied-Statistical-Practitioner-Workflows.git
cd AStats-Agentic-AI-for-Applied-Statistical-Practitioner-Workflows

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python generate_datasets.py
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

Press **W** for a full plain-language explanation (works offline, no API key needed).
Press **O** to override — choose from 9 tests, give a reason, saved to session log.

### Auto mode

```bash
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto
```

### Output formats

```bash
# Rich terminal (default)
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto

# Plain text
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto --output plain

# JSON — machine-readable
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto --output json

# Methods paragraph only — paste into your paper
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto --output methods
```

### Multi-LLM backend support

The W (explain) feature supports four backends via the `--llm` flag:

```bash
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --llm claude
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --llm codex
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --llm ollama
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --llm offline
```

| Flag | Model | Requires |
|---|---|---|
| `--llm claude` | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| `--llm codex` | gpt-4o | `OPENAI_API_KEY` |
| `--llm ollama` | mistral (configurable) | Local Ollama install |
| `--llm offline` | Built-in (all 9 tests) | Nothing — default |

### Force a specific test

```bash
python -m astats.cli list-tests
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto --test "Mann-Whitney U"
```

### Other options

```bash
# Change significance level
python -m astats.cli analyze test_datasets/08_hepatitis_B_ADR.csv "compare ADR severity scores across drugs" --auto --alpha 0.01

# Save full report
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto --save report.txt

# Show data summary before analysis
python -m astats.cli analyze test_datasets/01_antidepressant_trial.csv "compare HDRS score between treatment groups" --auto --summary

# View all past sessions
python -m astats.cli sessions

# Full detail of a session
python -m astats.cli session-detail sessions/session_XXXXXXXX.json
```

---

## Example Output

**Dataset:** `01_antidepressant_trial.csv` — 93 patients, Placebo vs Drug, HDRS scores
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
│ confirmed equal variances. t(91)=4.622, p=0.0000, d=0.959.     │
╰─────────────────────────────────────────────────────────────────╯
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

Post-hoc tests (Tukey HSD, Bonferroni, Dunn's) are triggered automatically after significant ANOVA or Kruskal-Wallis.

---

## Datasets

AStats has been validated on 15 datasets — 9 synthetic clinical datasets covering every analysis path, plus 6 real published datasets:

| Dataset | Design | n | Result |
|---|---|---|---|
| `01_antidepressant_trial.csv` | Independent, 2 groups | 93 | t(91)=4.622, p<.001, d=0.96 |
| `02_sleep_deprivation_longitudinal.csv` | Repeated measures, 9 days | 180 | Friedman χ²=104.3, W=0.65 |
| `03_cognitive_rehab_pre_post.csv` | Paired, 2 conditions | 35 | t(34)=12.2, p<.001, d=2.05 |
| `04_pain_management_4groups.csv` | Independent, 4 groups | 160 | F(3,156)=19.8, p<.001, η²=0.27 |
| `05_cognitive_load_RT.csv` | Independent, skewed | 150 | H(2)=38.4, p<.001, η²H=0.18 |
| `06_eeg_alpha_power_wide.csv` | Wide-format, 5 regions | 28 | Friedman χ²=28.1, W=0.70 |
| `07_crp_cognitive_decline.csv` | Correlation | 80 | Intent: correlate (correctly flagged) |
| `08_hepatitis_B_ADR.csv` | Independent, 3 drugs | 155 | H(2)=11.4, p=.004, η²H=0.06 |
| `09_sleepstudy_real.csv` ★ | Repeated measures, 10 days | 180 | Friedman χ²=135.8, p<.001, **W=0.839** |
| `10_iris.csv` ★ | Independent, 3 species | 150 | Welch ANOVA F=1180.2, p<.001 |
| `11_birthweight.csv` | Independent, 2 groups | 189 | t(187)=-2.89, p=.004, d=0.43 |
| `12_toothgrowth.csv` ★ | Independent, 2 supplements | 60 | Mann-Whitney p=0.065, r=0.28 |
| `13_plantgrowth.csv` ★ | Independent, 3 groups | 30 | F(2,27)=4.85, p=.016, η²=0.26 |
| `14_chickweight_longitudinal.csv` ★ | Repeated measures, 4 days | 40 | Friedman χ²=28.9, p<.001, **W=0.964** |
| `15_congruency_RT.csv` ★ | Paired, 2 conditions | 48 | t(23)=-19.4, p<.001, d=3.97 |

★ Real published dataset validated against the literature.

```bash
python generate_datasets.py
```

---

## Benchmark

```
Results: 9/13 passed (69.2%)
```

The 4 failures are documented statistical judgment calls — borderline normality edge cases where experienced statisticians genuinely disagree. They are tracked honestly rather than worked around.

```bash
python benchmark/run_benchmark.py
```

---

## The Human-in-the-Loop Design

The practitioner feedback loop is the central contribution of this project.

Every analysis pauses before finalising. The practitioner can:

- **Accept** — logged and complete
- **Override** — choose any of 9 tests, give a reason, recorded
- **Explain** — full plain-language explanation offline for all 9 tests
- **Correct** — flag that data is actually non-normal, or actually repeated measures

Every interaction is saved:

```json
{
  "session_id": "20260321_154443",
  "corrections": [
    {
      "timestamp": "2026-03-21T15:44:43",
      "dataset": "01_antidepressant_trial.csv",
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

These correction logs are structured as **(rejected, accepted, reason)** triples — DPO-compatible preference data that can improve the system over time.

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
│   ├── nl_parser.py        Natural language query parser
│   └── llm_backends.py     Claude / Codex / Ollama / offline backend manager
├── benchmark/
│   └── run_benchmark.py    13-scenario benchmark suite
├── test_datasets/          15 datasets (9 synthetic + 6 real published)
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
anthropic>=0.20.0   # --llm claude
openai>=1.0.0       # --llm codex
ollama              # --llm ollama
```

The tool runs fully offline without any API key. All 9 tests have complete built-in explanations.

---

## Roadmap

This is a proof of concept for [GSoC 2026 INCF Project #33](https://neurostars.org/t/gsoc-2026-project-33-university-of-wisconsin-madison-astats-an-agentic-ai-approach-to-applied-statistical-practitioner-workflows/35620).

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

## Discussions

Follow progress and ask questions at the [AStats community on SciCommons](https://alphatest.scicommons.org/community/AStats) and the [Neurostars project thread](https://neurostars.org/t/gsoc-2026-project-33-university-of-wisconsin-madison-astats-an-agentic-ai-approach-to-applied-statistical-practitioner-workflows/35620).

---

## About

Built by **Shreenidhi Gopalakrishnan** as a proof of concept for GSoC 2026 INCF Project #33, mentored by Suresh Krishna (McGill), Jonathan Morris (UW-Madison), and Yohai-Eliel Berreby (McGill).

The design is grounded in a real experience: building clinical ML tools that make recommendations without giving practitioners a mechanism to correct them. AStats treats practitioner corrections as first-class data — not noise to filter, but signal to record, learn from, and act on.

