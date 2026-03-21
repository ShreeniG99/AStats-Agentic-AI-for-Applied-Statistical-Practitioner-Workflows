# AStats — Proof of Concept
### GSoC 2026 | INCF Project #33 | Applicant: Shreenidhi Gopalakrishnan

Agentic statistical analysis with a **practitioner-in-the-loop feedback system**.

## What makes this different from other AStats prototypes

| Feature | Utkarsh's PoC | Atta's PoC | **This PoC** |
|---|---|---|---|
| NL query parsing | ✗ | ✓ | ✓ |
| Data structure inference | ✓ | ✗ | ✓ |
| Assumption checking | ✓ | ✗ | ✓ |
| Test execution | ✓ | ✗ | ✓ |
| Methods paragraph | ✓ | ✗ | ✓ |
| **Practitioner feedback loop** | ✗ | ✗ | **✓** |
| **Session logging (JSON)** | ✗ | ✗ | **✓** |
| **LLM plain-language explanation** | ✗ | ✗ | **✓** |
| End-to-end integrated CLI | ✗ | ✗ | **✓** |

## Quick Start

```bash
pip install -r requirements.txt

# Analyze a dataset interactively
python -m astats.cli analyze your_data.csv "compare score between groups"

# Non-interactive (auto-accept all recommendations)
python -m astats.cli analyze your_data.csv "compare score between groups" --auto

# View past sessions
python -m astats.cli sessions
```

## Run the Benchmark

```bash
python benchmark/run_benchmark.py
```

Current result: **9/13 scenarios passing (69.2%)**. The 4 remaining failures are all
edge-case normality detection issues that are documented open questions in the field —
the same 3 were listed as known failures in the prior PoC.

## Architecture

```
User Query (natural language)
        │
        ▼
┌─────────────────┐
│   NL Parser     │  Detects intent, flags ambiguity, asks for clarification
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Structure Infer │  Detects independent / repeated-measures / nested / wide-format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Assumption Check│  Shapiro-Wilk, Levene's, sample adequacy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Test Selection │  Deterministic matrix → runs test → effect size → methods para
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Practitioner Feedback Loop         │
│  Accept / Override / Explain / Fix  │  ← THE KEY CONTRIBUTION
│  Session logged to JSON             │
└─────────────────────────────────────┘
```

## Project
- INCF Neurostars: https://neurostars.org/t/gsoc-2026-project-33.../35620
- AStats repo: https://github.com/m2b3/AStats
- AStats community: https://alphatest.scicommons.org/community/AStats
