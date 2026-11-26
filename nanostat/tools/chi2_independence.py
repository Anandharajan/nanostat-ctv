"""Pearson chi-squared test of independence on a 2x2 table (no continuity correction)."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, two_by_two


class Chi2Independence(Tool):
    name = "chi2_independence"
    estimands = ("association",)
    designs = (Design.independent_two_group,)
    outcomes = (OutcomeType.binary,)
    two_sided_only = True
    required_checks = (Check.expected_counts,)
    r_reference = "chi2_independence"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        t = two_by_two(df, ctx)
        r = stats.chi2_contingency(t, correction=False)
        p1, p0 = t[0, 0] / t[0].sum(), t[1, 0] / t[1].sum()
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(p1 - p0),
            effect_label="risk difference (comparison - reference)",
            n=int(t.sum()),
            extra={"table": t.tolist(), "min_expected": float(r.expected_freq.min())},
        )
