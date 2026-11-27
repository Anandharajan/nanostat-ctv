"""Permutation tests: difference in means (two groups), mean paired difference (sign flips) and
Pearson r (permuting y). Exact enumeration when the permutation space is small, otherwise a
seeded Monte Carlo with p = (1 + #extreme) / (B + 1)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools.base import RunContext, Tool, ToolOutput, paired_arrays, two_groups, xy

EXACT_LIMIT = 20_000
B = 9_999
SEED = 20261127
TOL = 1e-12


def _p(null: np.ndarray, obs: float, alt: str, exact: bool) -> float:
    if alt == "greater":
        k = int((null >= obs - TOL).sum())
    elif alt == "less":
        k = int((null <= obs + TOL).sum())
    else:
        k = int((np.abs(null) >= abs(obs) - TOL).sum())
    if exact:
        return k / len(null)
    return (k + 1) / (len(null) + 1)


class PermutationTest(Tool):
    name = "permutation_test"
    estimands = ("mean_difference", "mean_difference_paired", "correlation")
    designs = (Design.independent_two_group, Design.paired, Design.correlation)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        alt = ctx.alternative.value
        rng = np.random.default_rng(SEED)
        design = ctx.design
        if design == Design.correlation or (design is None and "x" in ctx.columns):
            x, y = xy(df, ctx)
            xc = x - x.mean()
            yc = y - y.mean()
            den = math.sqrt((xc**2).sum() * (yc**2).sum())
            obs = float((xc * yc).sum() / den)
            n = len(x)
            if n <= 12 and math.factorial(n) <= EXACT_LIMIT:
                perms = np.array(list(itertools.permutations(range(n))))
                exact = True
            else:
                perms = np.argsort(rng.random((B, n)), axis=1)
                exact = False
            null = (yc[perms] * xc).sum(axis=1) / den
            return ToolOutput(
                obs,
                _p(null, obs, alt, exact),
                obs,
                "Pearson r (permutation p)",
                n,
                extra={"exact": exact, "B": len(null)},
            )
        if design == Design.paired or (
            design is None and ("pair_id" in ctx.columns or {"y1", "y2"} <= set(ctx.columns))
        ):
            y2, y1 = paired_arrays(df, ctx)
            d = y2 - y1
            n = len(d)
            obs = float(d.mean())
            if 2**n <= EXACT_LIMIT:
                signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
                exact = True
            else:
                signs = rng.choice([1.0, -1.0], size=(B, n))
                exact = False
            null = (signs * d).mean(axis=1)
            return ToolOutput(
                obs,
                _p(null, obs, alt, exact),
                obs,
                "mean paired difference (sign-flip p)",
                n,
                extra={"exact": exact, "B": len(null)},
            )
        a, b = two_groups(df, ctx)
        pooled = np.r_[a, b]
        na, n = len(a), len(a) + len(b)
        obs = float(a.mean() - b.mean())
        total = pooled.sum()
        if math.comb(n, na) <= EXACT_LIMIT:
            idx = np.array(list(itertools.combinations(range(n), na)))
            exact = True
        else:
            idx = np.argsort(rng.random((B, n)), axis=1)[:, :na]
            exact = False
        sa = pooled[idx].sum(axis=1)
        null = sa / na - (total - sa) / (n - na)
        return ToolOutput(
            obs,
            _p(null, obs, alt, exact),
            obs,
            "mean difference (permutation p)",
            n,
            extra={"exact": exact, "B": len(null)},
        )
