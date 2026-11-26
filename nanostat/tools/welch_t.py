"""Welch two-sample t-test (unequal variances)."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, ci_tuple, two_groups


class WelchT(Tool):
    name = "welch_t"
    estimands = ("mean_difference",)
    designs = (Design.independent_two_group,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.normality, Check.outliers)
    r_reference = "welch_t"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        a, b = two_groups(df, ctx)
        r = stats.ttest_ind(a, b, equal_var=False, alternative=ctx.alternative.value)
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(a.mean() - b.mean()),
            effect_label="mean difference (comparison - reference)",
            n=len(a) + len(b),
            ci=ci_tuple(r.confidence_interval(1 - ctx.alpha)),
            extra={"df": float(r.df)},
        )
