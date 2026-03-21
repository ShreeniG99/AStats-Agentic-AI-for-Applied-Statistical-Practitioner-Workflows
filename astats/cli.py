#!/usr/bin/env python3
"""
AStats CLI — Agentic Statistical Analysis
==========================================
Usage examples:

  # Basic (interactive feedback loop)
  python -m astats.cli analyze data.csv "compare score by group"

  # Auto mode (no prompts, just results)
  python -m astats.cli analyze data.csv "compare score by group" --auto

  # JSON output (machine-readable, good for scripts)
  python -m astats.cli analyze data.csv "compare score by group" --auto --output json

  # Force a specific test (override auto-selection)
  python -m astats.cli analyze data.csv "compare score by group" --auto --test "Mann-Whitney U"

  # Change significance level
  python -m astats.cli analyze data.csv "compare score by group" --auto --alpha 0.01

  # Save results to a file
  python -m astats.cli analyze data.csv "compare score by group" --auto --save results.txt

  # Show only the methods paragraph (for paper writing)
  python -m astats.cli analyze data.csv "compare score by group" --auto --output methods

  # Show data summary before analysis
  python -m astats.cli analyze data.csv "compare score by group" --auto --summary

  # List saved sessions
  python -m astats.cli sessions

  # Show full detail of one session
  python -m astats.cli session-detail sessions/session_XXXXXXXX.json
"""

import click
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import print as rprint

from .nl_parser import parse_query, resolve_ambiguity_interactively
from .structure import infer_structure
from .assumptions import check_assumptions
from .tests import select_and_run, TestResult
from .feedback import SessionManager

console = Console()

AVAILABLE_TESTS = [
    "Independent t-test",
    "Welch's t-test",
    "Mann-Whitney U",
    "Paired t-test",
    "Wilcoxon signed-rank",
    "Friedman test",
    "One-way ANOVA",
    "Welch's ANOVA",
    "Kruskal-Wallis H",
]


@click.group()
def cli():
    """AStats: Agentic statistical analysis with a practitioner in the loop."""
    pass


@cli.command()
@click.argument("data_path")
@click.argument("query")
@click.option("--alpha",   default=0.05,   help="Significance level (default: 0.05)")
@click.option("--auto",    is_flag=True,   help="Skip interactive confirmations")
@click.option("--output",  default="rich", type=click.Choice(["rich", "json", "plain", "methods"]),
              help="Output format: rich (default), json, plain, or methods-only")
@click.option("--test",    default=None,   help="Force a specific test (overrides auto-selection)")
@click.option("--save",    default=None,   help="Save output to this file path")
@click.option("--summary", is_flag=True,   help="Show a data summary table before analysis")
@click.option("--session-dir", default="sessions", help="Directory to save session logs")
def analyze(data_path, query, alpha, auto, output, test, save, summary, session_dir):
    """
    Analyze a dataset with a natural language query.

    \b
    Examples:
      python -m astats.cli analyze data.csv "compare score by group"
      python -m astats.cli analyze data.csv "compare score by group" --auto --output json
      python -m astats.cli analyze data.csv "compare score by group" --test "Mann-Whitney U"
      python -m astats.cli analyze data.csv "compare score by group" --alpha 0.01 --save out.txt
    """

    # Redirect output if saving to file
    save_lines = []

    def out(text=""):
        if save:
            save_lines.append(text)
        if output in ("rich",):
            console.print(text)
        elif output in ("plain", "methods", "json"):
            pass  # handled separately at end

    # ── 1. Load data ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        sys.exit(1)

    if output == "rich":
        console.print(f"\n[bold blue]AStats[/bold blue] — Agentic Statistical Analysis")
        console.print(f"Dataset: [green]{data_path}[/green]  ({len(df)} rows × {len(df.columns)} cols)")
        console.print(f"Query:   [yellow]{query}[/yellow]\n")

    # ── 2. Optional data summary ──────────────────────────────────────────────
    if summary and output == "rich":
        _print_data_summary(df)

    # ── 3. Parse query ────────────────────────────────────────────────────────
    parsed = parse_query(query)

    if output == "rich":
        color = "cyan" if parsed.intent != "unknown" else "red"
        console.print(f"✓ Intent detected: [{color}]{parsed.intent}[/{color}]")
        if parsed.design_hint:
            console.print(f"  Design hint: {parsed.design_hint}")

    if parsed.ambiguous and not auto:
        parsed = resolve_ambiguity_interactively(parsed)

    # ── 4. Infer structure ────────────────────────────────────────────────────
    query_hint = _build_query_hint(parsed, df)
    structure = infer_structure(df, query_hint or None)

    if output == "rich":
        console.print(f"\n✓ Structure inferred: [cyan]{structure.design}[/cyan]")
        for note in structure.notes:
            console.print(f"  → {note}")

    # ── 5. Check assumptions ──────────────────────────────────────────────────
    assumptions = check_assumptions(df, structure)

    if output == "rich":
        console.print(f"\n✓ Assumptions checked:")
        console.print(f"  Normality:       {'[green]✓ met[/green]' if assumptions.normality_ok else '[red]✗ not met[/red]'}")
        console.print(f"  Equal variance:  {'[green]✓ met[/green]' if assumptions.equal_variance_ok else '[yellow]✗ not met[/yellow]'}")
        console.print(f"  Sample adequate: {'[green]✓[/green]' if assumptions.sample_adequate else '[red]✗[/red]'} (n={assumptions.sample_size})")
        if assumptions.warnings:
            for w in assumptions.warnings:
                console.print(f"  [yellow]⚠ {w}[/yellow]")

    # ── 6. Run test ───────────────────────────────────────────────────────────
    if test:
        # User forced a specific test — run it directly
        result = _run_forced_test(test, df, structure, assumptions, alpha)
        if result is None:
            console.print(f"[red]Unknown test: '{test}'. Available tests:[/red]")
            for t in AVAILABLE_TESTS:
                console.print(f"  • {t}")
            sys.exit(1)
        if output == "rich":
            console.print(f"\n[yellow]⚡ Forced test override: {test}[/yellow]")
    else:
        result = select_and_run(df, structure, assumptions, alpha=alpha, intent=parsed.intent)

    # ── 7. Practitioner feedback loop ─────────────────────────────────────────
    session = SessionManager(session_dir=session_dir, auto=auto)

    if output == "rich" and not auto:
        feedback = session.present_and_confirm(result, structure, assumptions, df)
        if feedback["action"] == "quit":
            console.print("\n[yellow]Analysis cancelled.[/yellow]")
            return
        if feedback["action"] == "overridden":
            console.print(f"\n[yellow]Override: {feedback['override_test']}[/yellow]")
            console.print(f"  Reason: {feedback['override_reason']}")
            result = _run_forced_test(feedback["override_test"], df, structure, assumptions, alpha) or result
    else:
        feedback = {"action": "accepted", "override_test": None, "override_reason": None}

    final_test = feedback.get("override_test") or result.test_name
    session.log_correction(data_path, query, result, feedback, final_test)

    # ── 8. Output ─────────────────────────────────────────────────────────────
    if output == "rich":
        _print_rich_results(result, assumptions)
        console.print(f"\n[dim]{session.get_summary()}[/dim]")

    elif output == "plain":
        _print_plain_results(result, structure, assumptions, query, data_path)

    elif output == "methods":
        print(result.methods_paragraph)

    elif output == "json":
        _print_json_results(result, structure, assumptions, parsed, query, data_path, alpha)

    # ── 9. Save to file ───────────────────────────────────────────────────────
    if save:
        _save_to_file(save, result, structure, assumptions, query, data_path)
        console.print(f"\n[green]Results saved to {save}[/green]")


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def _build_query_hint(parsed, df):
    """Match parsed column hints to actual column names in the dataframe."""
    hint = {}
    for hint_key, hint_val in [
        ("outcome_col", parsed.outcome_hint),
        ("group_col",   parsed.group_hint),
        ("subject_col", parsed.subject_hint),
    ]:
        if hint_val:
            for col in df.columns:
                if col.lower() == hint_val.lower():
                    hint[hint_key] = col
                    break
    return hint if hint else None


def _print_data_summary(df):
    """Print a quick summary table of the dataset."""
    console.print("\n[bold]Data Summary:[/bold]")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Unique")
    table.add_column("Missing")
    table.add_column("Sample values")

    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = str(df[col].nunique())
        missing = str(df[col].isna().sum())
        if df[col].dtype == object:
            sample = str(df[col].dropna().unique()[:3].tolist())
        else:
            sample = f"min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
        table.add_row(col, dtype, nunique, missing, sample)
    console.print(table)
    console.print()


def _print_rich_results(result, assumptions):
    """Print the main results panel with Rich formatting."""
    sig_str = "[green]✓ significant[/green]" if result.significant else "[red]✗ not significant[/red]"

    console.print(Panel(
        f"[bold]Final Test:[/bold]    {result.test_name}\n"
        f"[bold]Statistic:[/bold]     {result.statistic}\n"
        f"[bold]p-value:[/bold]       {result.p_value}  {sig_str}\n"
        f"[bold]Effect size:[/bold]   {result.effect_size} ({result.effect_label}, [bold]{result.effect_magnitude}[/bold])\n"
        f"[bold]Alpha:[/bold]         {result.alpha}\n\n"
        f"[dim]Why this test:[/dim]\n{result.rationale}\n\n"
        f"[bold]Methods paragraph:[/bold]\n[italic]{result.methods_paragraph}[/italic]",
        title="[bold blue]AStats Results[/bold blue]",
        border_style="blue"
    ))


def _print_plain_results(result, structure, assumptions, query, data_path):
    """Plain text output — no colours, no boxes."""
    lines = [
        "=" * 60,
        "AStats Results",
        "=" * 60,
        f"Dataset:     {data_path}",
        f"Query:       {query}",
        f"Design:      {structure.design}",
        f"Outcome:     {structure.outcome_col}",
        f"Groups:      {structure.n_groups}",
        "",
        f"Test:        {result.test_name}",
        f"Statistic:   {result.statistic}",
        f"p-value:     {result.p_value}",
        f"Significant: {'Yes' if result.significant else 'No'} (alpha={result.alpha})",
        f"Effect size: {result.effect_size} ({result.effect_label}, {result.effect_magnitude})",
        "",
        "Rationale:",
        result.rationale,
        "",
        "Methods paragraph:",
        result.methods_paragraph,
        "=" * 60,
    ]
    print("\n".join(lines))


def _print_json_results(result, structure, assumptions, parsed, query, data_path, alpha):
    """JSON output — machine-readable, good for downstream scripts."""
    output_dict = {
        "meta": {
            "dataset": data_path,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "alpha": alpha,
        },
        "intent": parsed.intent,
        "structure": {
            "design": structure.design,
            "n_groups": structure.n_groups,
            "n_subjects": structure.n_subjects,
            "outcome_col": structure.outcome_col,
            "group_col": structure.group_col,
        },
        "assumptions": {
            "normality_ok": assumptions.normality_ok,
            "equal_variance_ok": assumptions.equal_variance_ok,
            "sample_adequate": assumptions.sample_adequate,
            "sample_size": assumptions.sample_size,
            "warnings": assumptions.warnings,
        },
        "result": {
            "test_name": result.test_name,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "significant": result.significant,
            "effect_size": result.effect_size,
            "effect_label": result.effect_label,
            "effect_magnitude": result.effect_magnitude,
            "rationale": result.rationale,
            "methods_paragraph": result.methods_paragraph,
        }
    }
    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")
    print(json.dumps(output_dict, indent=2, default=_convert))


def _save_to_file(filepath, result, structure, assumptions, query, data_path):
    """Save a plain-text report to a file."""
    lines = [
        "AStats Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Dataset:       {data_path}",
        f"Query:         {query}",
        f"Design:        {structure.design}",
        f"Outcome col:   {structure.outcome_col}",
        f"Group col:     {structure.group_col}",
        f"N groups:      {structure.n_groups}",
        "",
        "ASSUMPTIONS",
        f"  Normality:      {'Met' if assumptions.normality_ok else 'Not met'}",
        f"  Equal variance: {'Met' if assumptions.equal_variance_ok else 'Not met'}",
        f"  Sample size:    n={assumptions.sample_size}",
    ]
    if assumptions.warnings:
        lines.append("  Warnings:")
        for w in assumptions.warnings:
            lines.append(f"    - {w}")
    lines += [
        "",
        "RESULTS",
        f"  Test:        {result.test_name}",
        f"  Statistic:   {result.statistic}",
        f"  p-value:     {result.p_value}",
        f"  Significant: {'Yes' if result.significant else 'No'} (alpha={result.alpha})",
        f"  Effect size: {result.effect_size} ({result.effect_label}, {result.effect_magnitude})",
        "",
        "RATIONALE",
        f"  {result.rationale}",
        "",
        "METHODS PARAGRAPH",
        result.methods_paragraph,
        "=" * 60,
    ]
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def _run_forced_test(test_name, df, structure, assumptions, alpha):
    """Run a user-specified test directly, bypassing auto-selection."""
    import numpy as np
    from scipy import stats as scipy_stats
    from .tests import (
        _independent_t, _welch_t, _mann_whitney,
        _paired_t, _wilcoxon, _friedman, _friedman_wide,
        _one_way_anova, _welch_anova, _kruskal_wallis, TestResult
    )

    dispatch = {
        "independent t-test":  _independent_t,
        "welch's t-test":      _welch_t,
        "mann-whitney u":      _mann_whitney,
        "paired t-test":       _paired_t,
        "wilcoxon signed-rank":_wilcoxon,
        "friedman test":       _friedman,
        "friedman test (wide)":_friedman_wide,
        "one-way anova":       _one_way_anova,
        "welch's anova":       _welch_anova,
        "kruskal-wallis h":    _kruskal_wallis,
    }

    key = test_name.lower().strip()
    fn = dispatch.get(key)
    if fn is None:
        return None
    try:
        return fn(df, structure, assumptions, alpha)
    except Exception as e:
        console.print(f"[red]Error running {test_name}: {e}[/red]")
        return None


# ── SESSION COMMANDS ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--session-dir", default="sessions")
def sessions(session_dir):
    """List all saved analysis sessions."""
    session_path = Path(session_dir)
    if not session_path.exists() or not list(session_path.glob("session_*.json")):
        console.print("[yellow]No sessions found.[/yellow]")
        return

    files = sorted(session_path.glob("session_*.json"))

    table = Table(title="Saved Sessions", show_lines=True)
    table.add_column("Session ID",  style="cyan")
    table.add_column("Analyses",    justify="center")
    table.add_column("Overrides",   justify="center")
    table.add_column("Last dataset")
    table.add_column("File")

    for f in files[-15:]:
        with open(f) as fh:
            data = json.load(fh)
        corrections = data.get("corrections", [])
        n = len(corrections)
        overrides = sum(1 for c in corrections if c.get("action") == "overridden")
        last_ds = corrections[-1]["dataset"] if corrections else "—"
        table.add_row(data["session_id"], str(n), str(overrides), last_ds, f.name)

    console.print(table)


@cli.command("session-detail")
@click.argument("session_file")
def session_detail(session_file):
    """Show full detail of a saved session."""
    try:
        with open(session_file) as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Could not read session: {e}[/red]")
        return

    console.print(f"\n[bold blue]Session:[/bold blue] {data['session_id']}")
    for i, c in enumerate(data.get("corrections", []), 1):
        console.print(f"\n[bold]Analysis {i}[/bold]")
        console.print(f"  Dataset:    {c['dataset']}")
        console.print(f"  Query:      {c['query']}")
        console.print(f"  Recommended:{c['recommended_test']}")
        console.print(f"  Action:     {c['action']}")
        if c.get("override_test"):
            console.print(f"  Override:   {c['override_test']} — {c['override_reason']}")
        console.print(f"  Final test: {c['final_test']}")
        console.print(f"  p-value:    {c['p_value']}")
        console.print(f"  Effect:     {c['effect_size']}")


@cli.command("list-tests")
def list_tests():
    """Show all available statistical tests you can force with --test."""
    console.print("\n[bold]Available tests for --test flag:[/bold]\n")
    for t in AVAILABLE_TESTS:
        console.print(f"  python -m astats.cli analyze data.csv \"query\" --test \"{t}\"")
    console.print()


if __name__ == "__main__":
    cli()
