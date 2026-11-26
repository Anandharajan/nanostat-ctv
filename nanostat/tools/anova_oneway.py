"""One-way ANOVA F-test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, k_groups


class AnovaOneway(Tool):
    name = "anova_oneway"
    estimands = ("group_means_equal",)
    designs = (Design.k_group,)
    outcomes = (OutcomeType.continuous,)
    two_sided_only = True
    required_checks = (Check.normality, Check.equal_variance, Check.outliers)
    r_reference = "anova_oneway"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        gs = k_groups(df, ctx)
        r = stats.f_oneway(*gs)
        allv = np.concatenate(gs)
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in gs)
        ss_t = float(((allv - allv.mean()) ** 2).sum())
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(ss_b / ss_t) if ss_t > 0 else 0.0,
            effect_label="eta squared",
            n=len(allv),
            extra={"df1": len(gs) - 1, "df2": len(allv) - len(gs)},
        )
