"""Student two-sample t-test (pooled variance)."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, ci_tuple, two_groups


class StudentT(Tool):
    name = "student_t"
    estimands = ("mean_difference",)
    designs = (Design.independent_two_group,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.normality, Check.equal_variance, Check.outliers)
    r_reference = "student_t"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        a, b = two_groups(df, ctx)
        r = stats.ttest_ind(a, b, equal_var=True, alternative=ctx.alternative.value)
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(a.mean() - b.mean()),
            effect_label="mean difference (comparison - reference)",
            n=len(a) + len(b),
            ci=ci_tuple(r.confidence_interval(1 - ctx.alpha)),
            extra={"df": float(r.df)},
        )
