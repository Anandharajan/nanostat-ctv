"""Yuen's 20% trimmed-mean Welch-type t-test."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, two_groups


class YuenTrimmedT(Tool):
    name = "yuen_trimmed_t"
    estimands = ("trimmed_mean_difference", "location_shift")
    designs = (Design.independent_two_group,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.outliers,)
    r_reference = None
    trim = 0.2

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        a, b = two_groups(df, ctx)
        r = stats.ttest_ind(a, b, equal_var=False, trim=self.trim, alternative=ctx.alternative.value)
        eff = float(stats.trim_mean(a, self.trim) - stats.trim_mean(b, self.trim))
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=eff,
            effect_label="20% trimmed mean difference (comparison - reference)",
            n=len(a) + len(b),
            extra={"df": float(r.df), "trim": self.trim},
        )
