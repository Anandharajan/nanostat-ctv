"""Spearman rank correlation test (t approximation)."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, xy


class Spearman(Tool):
    name = "spearman"
    estimands = ("rank_correlation", "correlation")
    designs = (Design.correlation,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.sample_size,)
    r_reference = "spearman"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        x, y = xy(df, ctx)
        r = stats.spearmanr(x, y, alternative=ctx.alternative.value)
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(r.statistic),
            effect_label="Spearman rho",
            n=len(x),
        )
