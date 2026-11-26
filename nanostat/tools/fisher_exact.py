"""Fisher exact test on a 2x2 table."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from nanostat.schemas import Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, two_by_two


class FisherExact(Tool):
    name = "fisher_exact"
    estimands = ("association", "odds_ratio")
    designs = (Design.independent_two_group,)
    outcomes = (OutcomeType.binary,)
    required_checks = ()
    r_reference = "fisher_exact"

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        t = two_by_two(df, ctx)
        r = stats.fisher_exact(t, alternative=ctx.alternative.value)
        p1, p0 = t[0, 0] / t[0].sum(), t[1, 0] / t[1].sum()
        return ToolOutput(
            statistic=float(r.statistic),
            p_value=float(r.pvalue),
            effect=float(p1 - p0),
            effect_label="risk difference (comparison - reference)",
            n=int(t.sum()),
            extra={"table": t.tolist(), "odds_ratio": float(r.statistic)},
        )
