"""
Practitioner Feedback Loop — AStats's unique contribution
=========================================================
This is the piece that no existing prototype has.

When AStats recommends a test, the practitioner can:
  A) Accept the recommendation → pipeline continues
  B) Override the test → AStats runs the override, logs the correction
  C) Ask "why?" → LLM explains the decision in plain language
  D) Flag an assumption as wrong → re-runs with corrected assumption

All corrections are stored in a session file (JSON), creating:
  - An audit trail of decisions
  - A basis for future fine-tuning
  - Reproducible analysis scripts

The user is ALWAYS in the loop. They can turn off interruptions with --auto.
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


@dataclass
class FeedbackEntry:
    timestamp: str
    dataset: str
    query: str
    recommended_test: str
    action: str            # 'accepted' | 'overridden' | 'assumption_corrected'
    override_test: Optional[str]
    override_reason: Optional[str]
    final_test: str
    p_value: float
    effect_size: float


class SessionManager:
    """
    Manages the practitioner correction loop for a single AStats session.
    Saves corrections to JSON for reproducibility and future fine-tuning.
    """

    def __init__(self, session_dir: str = "sessions", auto: bool = False):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.auto = auto  # if True, skip interactive prompts
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.session_dir / f"session_{self.session_id}.json"
        self.corrections = []

    def present_and_confirm(self, result, structure, assumptions, df) -> dict:
        """
        Show the recommended test to the practitioner and get their response.
        Returns a dict with 'action', 'override_test', 'override_reason'.
        """
        if self.auto:
            return {"action": "accepted", "override_test": None, "override_reason": None}

        print("\n" + "═" * 60)
        print(f"  AStats Recommendation")
        print("═" * 60)
        print(f"  Test:        {result.test_name}")
        print(f"  Statistic:   {result.statistic}")
        print(f"  p-value:     {result.p_value}")
        print(f"  Effect size: {result.effect_size} ({result.effect_label}, {result.effect_magnitude})")
        print(f"  Significant: {'Yes ✓' if result.significant else 'No ✗'} (α={result.alpha})")
        print(f"\n  Why this test:")
        print(f"  {result.rationale}")
        print("\n  Warnings:")
        for w in assumptions.warnings:
            print(f"    ⚠ {w}")
        print("═" * 60)

        print("\n  What would you like to do?")
        print("  [A] Accept this recommendation")
        print("  [O] Override — choose a different test")
        print("  [W] Tell me why in plain language (LLM explanation)")
        print("  [C] Correct an assumption")
        print("  [Q] Quit")

        choice = input("\n  Your choice [A/O/W/C/Q]: ").strip().upper()

        if choice == "A" or choice == "":
            return {"action": "accepted", "override_test": None, "override_reason": None}

        elif choice == "W":
            self._explain_with_llm(result, structure, assumptions)
            # After explanation, ask again
            return self.present_and_confirm(result, structure, assumptions, df)

        elif choice == "O":
            print("\n  Available tests:")
            test_options = [
                "Independent t-test", "Welch's t-test", "Mann-Whitney U",
                "Paired t-test", "Wilcoxon signed-rank", "Friedman test",
                "One-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis H"
            ]
            for i, t in enumerate(test_options, 1):
                print(f"  {i}. {t}")
            choice_num = input("\n  Enter test number: ").strip()
            override_reason = input("  Brief reason for override: ").strip()
            try:
                override_test = test_options[int(choice_num) - 1]
            except (ValueError, IndexError):
                print("  Invalid choice. Accepting original recommendation.")
                return {"action": "accepted", "override_test": None, "override_reason": None}
            return {"action": "overridden", "override_test": override_test, "override_reason": override_reason}

        elif choice == "C":
            print("\n  Which assumption would you like to correct?")
            print("  [1] Data is actually normally distributed")
            print("  [2] Data is actually NOT normally distributed")
            print("  [3] Groups have equal variance")
            print("  [4] This is actually a repeated measures design")
            print("  [5] This is actually an independent groups design")
            correction = input("  Choice [1-5]: ").strip()
            return {"action": "assumption_corrected", "override_test": None,
                    "override_reason": f"practitioner_correction_{correction}"}

        else:
            return {"action": "quit", "override_test": None, "override_reason": None}

    # ── Rich rule-based explanations per test ───────────────────────────────
    EXPLANATIONS = {
        "Independent t-test": (
            "You have two separate groups of people, and you want to know if their "
            "average scores are genuinely different or just different by chance. "
            "The independent t-test is the standard tool for this — it compares the "
            "means of both groups and tells you how likely that difference is to be real. "
            "We checked that your data is roughly bell-shaped (normality) and that both "
            "groups have similar spread (equal variance) — both passed, so the t-test is "
            "appropriate. If we had skipped those checks and the data was heavily skewed, "
            "the p-value could be misleading and you might incorrectly claim a significant "
            "difference."
        ),
        "Welch's t-test": (
            "Like the standard t-test, this compares two independent groups. "
            "The difference is that Levene's test showed your two groups have "
            "unequal variance — one group's scores are more spread out than the other. "
            "Welch's version corrects for this automatically by adjusting the degrees "
            "of freedom. Using a standard t-test here would give an overconfident "
            "p-value, potentially making a weak result look significant."
        ),
        "Mann-Whitney U": (
            "Your data failed the normality check, meaning the scores are not "
            "bell-shaped — they're likely skewed or have outliers. In that case, "
            "comparing means (as the t-test does) is not meaningful. Mann-Whitney U "
            "instead compares the ranks of scores between the two groups, which is "
            "robust to any shape of distribution. The trade-off is slightly less "
            "statistical power if the data actually were normal — but getting the "
            "right answer matters more than squeezing out extra power."
        ),
        "Paired t-test": (
            "The same people were measured twice — before and after something, or "
            "under two conditions. Because the two measurements come from the same "
            "person, they are correlated, and treating them as independent would waste "
            "that information and reduce your ability to detect a real effect. "
            "The paired t-test works on the difference score for each person, "
            "effectively cancelling out individual variation. We checked that those "
            "difference scores are normally distributed and they are, so this test "
            "is valid. Using an independent t-test here would be statistically wrong "
            "and would likely give a larger p-value, missing a real effect."
        ),
        "Wilcoxon signed-rank": (
            "This is the non-parametric version of the paired t-test. You have "
            "the same people measured twice, but the within-subject difference scores "
            "are not normally distributed. Wilcoxon ranks the absolute differences "
            "and checks whether the positive and negative differences are balanced. "
            "It is robust to skewed distributions and outliers. Using a paired t-test "
            "here when normality fails could give an unreliable p-value."
        ),
        "Friedman test": (
            "You have the same people measured across three or more conditions or "
            "time points. This is the non-parametric equivalent of a repeated-measures "
            "ANOVA. It ranks each person's scores across conditions and tests whether "
            "those rankings are consistent or vary systematically. Kendall's W tells "
            "you the effect size — how consistently the rankings agree across people. "
            "This test is appropriate when normality cannot be assumed across all "
            "conditions, which is common with small samples."
        ),
        "Friedman test (wide)": (
            "Your data has multiple measurement columns — each column represents a "
            "different condition, time point, or brain region. The Friedman test "
            "treats each row as one subject measured across all those columns. "
            "It checks whether the measurements differ systematically across columns "
            "rather than varying randomly. This is the right approach when the same "
            "entity is measured in multiple ways side by side."
        ),
        "One-way ANOVA": (
            "You have three or more independent groups and want to know if any of "
            "them differ from the others. ANOVA tests whether the variation between "
            "groups is larger than the variation within groups. We verified normality "
            "in each group and equal variances across groups — both passed. "
            "If you used multiple t-tests instead (one for each pair), you would "
            "inflate the false positive rate: with 4 groups and 6 comparisons, "
            "you'd have a 26% chance of a false positive at alpha=0.05. ANOVA "
            "controls this properly. The eta-squared value tells you what proportion "
            "of total variance is explained by group membership."
        ),
        "Welch's ANOVA": (
            "Similar to one-way ANOVA for three or more groups, but Levene's test "
            "showed unequal variances across groups. Welch's ANOVA adjusts for this, "
            "similar to how Welch's t-test adjusts for two groups. Using standard "
            "ANOVA with unequal variances inflates the Type I error rate."
        ),
        "Kruskal-Wallis H": (
            "You have three or more independent groups but the normality assumption "
            "failed. Kruskal-Wallis is the non-parametric version of one-way ANOVA — "
            "it ranks all observations together and tests whether those ranks are "
            "distributed equally across groups. It makes no assumptions about the "
            "shape of the distribution. The eta-squared H effect size tells you how "
            "much of the rank variance is explained by group membership. If you had "
            "used ANOVA on non-normal data, the F-statistic could be unreliable and "
            "give you a misleading p-value."
        ),
    }

    def _explain_with_llm(self, result, structure, assumptions):
        """
        Explain the statistical decision in plain language.
        Tries Anthropic API first; falls back to rich rule-based explanations.
        """
        llm_used = False

        try:
            import anthropic
            import os
            # Only try API if key is available
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("No API key")

            client = anthropic.Anthropic(api_key=api_key)
            prompt = (
                f"A researcher is using a statistical analysis tool. The tool recommended "
                f"the {result.test_name} for their data. Here is the context:\n\n"
                f"- Study design: {structure.design}\n"
                f"- Number of groups/conditions: {structure.n_groups}\n"
                f"- Sample size: {assumptions.sample_size}\n"
                f"- Normality assumption met: {assumptions.normality_ok}\n"
                f"- Equal variance assumption met: {assumptions.equal_variance_ok}\n"
                f"- Result: {result.test_name}, statistic={result.statistic}, "
                f"p={result.p_value}, effect={result.effect_size} ({result.effect_magnitude})\n\n"
                f"Explain in 4-5 plain sentences why this test was chosen, what it is "
                f"actually measuring, and what would go wrong if the researcher chose "
                f"an inappropriate test. Write for a non-statistician researcher."
            )
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}]
            )
            explanation = response.content[0].text
            llm_used = True

        except Exception:
            # Rich rule-based fallback — no API key needed
            explanation = self.EXPLANATIONS.get(
                result.test_name,
                result.rationale + "\n\n  No detailed explanation available for this test."
            )

        # Add context about assumptions
        assumption_notes = []
        if not assumptions.normality_ok:
            assumption_notes.append(
                "Your data failed the normality check, which is why a "
                "non-parametric test was selected."
            )
        if not assumptions.equal_variance_ok:
            assumption_notes.append(
                "Your groups have unequal variance, which is why a "
                "Welch-corrected version was used."
            )
        if assumptions.sample_size < 30:
            assumption_notes.append(
                f"Your sample size is small (n={assumptions.sample_size}), "
                "so results should be interpreted cautiously."
            )

        source = "LLM explanation" if llm_used else "Built-in explanation"
        print(f"\n  ┌─ 📖 Why {result.test_name}? ({source}) " + "─" * 20)
        print()
        # Word-wrap the explanation at ~65 chars
        words = explanation.split()
        line = "  │  "
        for word in words:
            if len(line) + len(word) > 70:
                print(line)
                line = "  │  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print()
        for note in assumption_notes:
            print(f"  │  ⚠ {note}")
        if assumption_notes:
            print()
        print("  └" + "─" * 50)
        print()

    def log_correction(self, dataset: str, query: str, result,
                       feedback: dict, final_test: str):
        """Save this interaction to the session file."""
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            dataset=dataset,
            query=query,
            recommended_test=result.test_name,
            action=feedback["action"],
            override_test=feedback.get("override_test"),
            override_reason=feedback.get("override_reason"),
            final_test=final_test,
            p_value=result.p_value,
            effect_size=result.effect_size
        )
        self.corrections.append(asdict(entry))
        self._save()
        return entry

    def _save(self):
        with open(self.session_file, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "corrections": self.corrections
            }, f, indent=2)

    def get_summary(self) -> str:
        n = len(self.corrections)
        overridden = sum(1 for c in self.corrections if c["action"] == "overridden")
        return (
            f"Session {self.session_id}: {n} analyses, "
            f"{overridden} practitioner overrides. "
            f"Session saved to {self.session_file}"
        )
