"""Paired t-test on y2 - y1."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, ci_tuple, paired_arrays


class PairedT(Tool):
    name = "paired_t"
    estimands = ("mean_difference_paired",)
    designs = (Design.paired,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.pairing, Check.normality, Check.outliers)
    r_reference = "paired_t"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        y2, y1 = paired_arrays(df, ctx)
        r = stats.ttest_rel(y2, y1, alternative=ctx.alternative.value)
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float((y2 - y1).mean()),
            effect_label="mean paired difference (comparison - reference)",
            n=len(y2),
            ci=ci_tuple(r.confidence_interval(1 - ctx.alpha)),
            extra={"df": float(r.df)},
        )
