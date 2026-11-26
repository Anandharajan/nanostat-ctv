"""Kruskal-Wallis H test."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, k_groups


class KruskalWallis(Tool):
    name = "kruskal_wallis"
    estimands = ("group_distributions_equal",)
    designs = (Design.k_group,)
    outcomes = (OutcomeType.continuous,)
    two_sided_only = True
    required_checks = (Check.sample_size,)
    r_reference = "kruskal_wallis"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        gs = k_groups(df, ctx)
        r = stats.kruskal(*gs)
        n = sum(len(g) for g in gs)
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(r.statistic / ((n**2 - 1) / (n + 1))),
            effect_label="epsilon squared",
            n=n,
            extra={"df": len(gs) - 1},
        )
