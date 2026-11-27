"""OLS / WLS coefficient tests and linear-restriction (F / Wald) tests.

ols_coef : t test of one coefficient (classical, HC0-HC3, or cluster-robust CR1 SE). With a 0/1
           outcome this is the linear probability model, standard in econometrics.
ols_wald : joint-zero or equality restriction on several coefficients; F(q, df) with the same SEs.
Reference distribution: t / F with n-k df (G-1 for clustered SEs) unless spec.dist = "z".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools._models import build_design, coef_test, output, ref_df, sandwich, spec_of
from nanostat.tools.base import RunContext, Tool, ToolError, ToolOutput


def fit_ols(df: pd.DataFrame, ctx: RunContext) -> tuple[np.ndarray, np.ndarray, list[str], int, float, str]:
    spec = spec_of(ctx)
    d = build_design(df, ctx)
    X, y = d.X, d.y
    if d.weights is not None:  # WLS: transform by sqrt(w)
        sw = np.sqrt(d.weights)
        X, y = X * sw[:, None], y * sw
    n, k = X.shape
    if n <= k:
        raise ToolError("not enough rows for the number of regressors")
    bread = np.linalg.inv(X.T @ X)
    b = bread @ X.T @ y
    e = y - X @ b
    if spec.se == "classical":
        cov = bread * float(e @ e) / (n - k)
    else:
        cov = sandwich(X, e, bread, spec.se, d.cluster)
    df_ = ref_df(spec.se, n, k, d.cluster)
    return b, cov, d.names, n, df_, spec.se


class OLSCoef(Tool):
    name = "ols_coef"
    estimands = ("coefficient", "linear_restriction")
    designs = (Design.regression,)
    outcomes = (OutcomeType.continuous, OutcomeType.binary)
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        b, cov, names, n, df_, se = fit_ols(df, ctx)
        stat, p, eff, extra = coef_test(
            b, cov, names, spec.exposure, spec.restriction, ctx.alternative.value, spec.dist or "t", df_
        )
        return output(stat, p, eff, f"OLS coefficient ({se} SE)", n, {**extra, "se_type": se})


class OLSWald(Tool):
    name = "ols_wald"
    estimands = ("linear_restriction",)
    designs = (Design.regression,)
    outcomes = (OutcomeType.continuous, OutcomeType.binary)
    two_sided_only = True
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        restriction = spec.restriction if spec.restriction != "each" else "joint_zero"
        b, cov, names, n, df_, se = fit_ols(df, ctx)
        stat, p, eff, extra = coef_test(b, cov, names, spec.exposure, restriction, "two-sided", spec.dist or "t", df_)
        return output(stat, p, eff, f"Wald/F test of {restriction} ({se} SE)", n, {**extra, "se_type": se})
