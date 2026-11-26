"""Mann-Whitney U (Wilcoxon rank-sum) test."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, hodges_lehmann_two, two_groups


class MannWhitney(Tool):
    name = "mann_whitney"
    estimands = ("location_shift",)
    designs = (Design.independent_two_group,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.sample_size,)
    r_reference = "mann_whitney"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        a, b = two_groups(df, ctx)
        r = stats.mannwhitneyu(a, b, alternative=ctx.alternative.value, method="auto")
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=hodges_lehmann_two(a, b),
            effect_label="Hodges-Lehmann shift (comparison - reference)",
            n=len(a) + len(b),
        )
