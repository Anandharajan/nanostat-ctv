"""Wilcoxon signed-rank test on paired differences y2 - y1."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import (
    RunContext,
    Tool,
    ToolError,
    ToolOutput,
    paired_arrays,
    pseudo_median,
)


class WilcoxonSignedRank(Tool):
    name = "wilcoxon_signed_rank"
    estimands = ("location_shift", "mean_difference_paired")
    designs = (Design.paired,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.pairing,)
    r_reference = "wilcoxon_signed_rank"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        y2, y1 = paired_arrays(df, ctx)
        d = y2 - y1
        if (d != 0).sum() < 3:
            raise ToolError("fewer than 3 non-zero differences")
        r = stats.wilcoxon(d, alternative=ctx.alternative.value, zero_method="wilcox", method="auto")
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=pseudo_median(d),
            effect_label="pseudo-median of paired differences",
            n=len(d),
        )
