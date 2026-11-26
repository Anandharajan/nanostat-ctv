"""Exact binomial test of a single proportion against p0."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolError, ToolOutput, binary01, ci_tuple, column


class BinomialTest(Tool):
    name = "binomial_test"
    estimands = ("proportion",)
    designs = (Design.one_sample,)
    outcomes = (OutcomeType.binary,)
    required_checks = (Check.sample_size,)
    r_reference = "binomial_test"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        if ctx.p0 is None or not 0 < ctx.p0 < 1:
            raise ToolError("binomial_test requires p0 in (0, 1)")
        y = binary01(df[column(df, ctx, "y")])
        k, n = int(y.sum()), len(y)
        r = stats.binomtest(k, n, ctx.p0, alternative=ctx.alternative.value)
        return ToolOutput(
            statistic=float(k),
            p_value=float(r.pvalue),
            effect=float(k / n - ctx.p0),
            effect_label="observed proportion - p0",
            n=n,
            ci=ci_tuple(r.proportion_ci(1 - ctx.alpha)),
            extra={"k": k, "p0": ctx.p0},
        )
