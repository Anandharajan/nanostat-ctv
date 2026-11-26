"""Pearson product-moment correlation test."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, ci_tuple, xy


class Pearson(Tool):
    name = "pearson"
    estimands = ("correlation",)
    designs = (Design.correlation,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.normality, Check.outliers)
    r_reference = "pearson"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        x, y = xy(df, ctx)
        r = stats.pearsonr(x, y, alternative=ctx.alternative.value)
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(r.statistic),
            effect_label="Pearson r",
            n=len(x),
            ci=ci_tuple(r.confidence_interval(1 - ctx.alpha)),
        )
