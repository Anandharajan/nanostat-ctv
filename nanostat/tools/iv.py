"""Two-stage least squares (2SLS) coefficient test.

Exposure(s) = endogenous regressors; `instruments` = excluded instruments; covariates/categorical =
included exogenous controls. b = (X'Pz X)^-1 X'Pz y with Pz the projection on Z = [instruments, exog].
Residuals use the ORIGINAL X. SEs: classical s^2 (Xh'Xh)^-1 with s^2 = e'e/(n-k), HC0-HC3, or CR1
cluster. First-stage F (joint significance of excluded instruments, same SE type) is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools._models import build_design, coef_test, output, ref_df, sandwich, spec_of
from nanostat.tools.base import RunContext, Tool, ToolError, ToolOutput


class IV2SLS(Tool):
    name = "iv_2sls"
    estimands = ("coefficient",)
    designs = (Design.regression,)
    outcomes = (OutcomeType.continuous, OutcomeType.binary)
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        if not spec.instruments:
            raise ToolError("iv_2sls needs at least one excluded instrument (model.instruments)")
        d = build_design(df, ctx, extra=spec.instruments)
        Zinst = d.frame[spec.instruments].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if np.isnan(Zinst).any():
            raise ToolError("instruments must be numeric")
        exog_idx = [i for i, nm in enumerate(d.names) if nm not in spec.exposure]
        X, y = d.X, d.y
        Z = np.column_stack([Zinst, X[:, exog_idx]])
        if d.weights is not None:
            sw = np.sqrt(d.weights)
            X, y, Z = X * sw[:, None], y * sw, Z * sw[:, None]
        n, k = X.shape
        if Z.shape[1] < k:
            raise ToolError("under-identified: fewer instruments than endogenous regressors")
        if np.linalg.matrix_rank(Z) < Z.shape[1]:
            raise ToolError("instrument matrix is rank deficient")
        Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
        bread = np.linalg.inv(Xh.T @ Xh)
        b = bread @ Xh.T @ y
        e = y - X @ b
        if spec.se == "classical":
            cov = bread * float(e @ e) / (n - k)
        else:
            cov = sandwich(Xh, e, bread, spec.se, d.cluster)
        df_ = ref_df(spec.se, n, k, d.cluster)
        stat, p, eff, extra = coef_test(
            b, cov, d.names, spec.exposure, spec.restriction, ctx.alternative.value, spec.dist or "z", df_
        )
        # first stage for the first endogenous regressor: F on excluded instruments
        j = d.names.index(spec.exposure[0])
        bread1 = np.linalg.inv(Z.T @ Z)
        g = bread1 @ Z.T @ X[:, j]
        u = X[:, j] - Z @ g
        cov1 = (
            bread1 * float(u @ u) / (n - Z.shape[1])
            if spec.se == "classical"
            else sandwich(Z, u, bread1, spec.se, d.cluster)
        )
        q = Zinst.shape[1]
        gi = g[:q]
        f1 = float(gi @ np.linalg.solve(cov1[:q, :q], gi)) / q
        return output(
            stat, p, eff, f"2SLS coefficient ({spec.se} SE)", n, {**extra, "se_type": spec.se, "first_stage_F": f1}
        )
