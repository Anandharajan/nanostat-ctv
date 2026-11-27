"""Linear mixed model (random intercept) and Tobit regression.

mixed_lm : statsmodels MixedLM, REML, random intercept per `model.group`; Wald z test of the fixed
           effect (R lmerTest reports Satterthwaite t; small-group results can differ, noted).
tobit    : left-censored (default limit 0) normal regression by maximum likelihood (BFGS on
           beta and log sigma); SE from the inverse of the numerically differentiated Hessian.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.regression.mixed_linear_model import MixedLM

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools._models import build_design, coef_test, output, spec_of
from nanostat.tools.base import RunContext, Tool, ToolError, ToolOutput


class MixedLinear(Tool):
    name = "mixed_lm"
    estimands = ("coefficient",)
    designs = (Design.regression,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        if not spec.group:
            raise ToolError("mixed_lm needs model.group (random-intercept grouping)")
        d = build_design(df, ctx, extra=[spec.group])
        groups = d.frame[spec.group].astype(str).to_numpy()
        if len(np.unique(groups)) < 3:
            raise ToolError("need at least 3 groups for a random intercept")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = MixedLM(d.y, d.X, groups=groups).fit(reml=True)
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"mixed model failed: {e}") from e
        k = d.X.shape[1]
        b = np.asarray(r.fe_params)
        cov = np.asarray(r.cov_params())[:k, :k]
        stat, p, eff, extra = coef_test(
            b, cov, d.names, spec.exposure, spec.restriction, ctx.alternative.value, spec.dist or "z", None
        )
        return output(
            stat,
            p,
            eff,
            "fixed effect (REML, random intercept)",
            len(d.y),
            {**extra, "group_var": float(np.asarray(r.cov_re).ravel()[0]), "resid_var": float(r.scale)},
        )


def tobit_fit(y: np.ndarray, X: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    cens = y <= limit
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    s0 = float(np.std(y - X @ ols)) or 1.0

    def nll(theta: np.ndarray) -> float:
        b, ls = theta[:-1], theta[-1]
        s = np.exp(ls)
        mu = X @ b
        ll_c = stats.norm.logcdf((limit - mu[cens]) / s)
        ll_u = stats.norm.logpdf((y[~cens] - mu[~cens]) / s) - ls
        return -float(ll_c.sum() + ll_u.sum())

    res = optimize.minimize(nll, np.r_[ols, np.log(s0)], method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    th = res.x
    # numerical Hessian (central differences)
    k = len(th)
    H = np.zeros((k, k))
    h = 1e-4 * np.maximum(1.0, np.abs(th))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            fpp = nll(th + ei + ej)
            fpm = nll(th + ei - ej)
            fmp = nll(th - ei + ej)
            fmm = nll(th - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    return th, np.linalg.inv(H)


class Tobit(Tool):
    name = "tobit"
    estimands = ("coefficient",)
    designs = (Design.regression,)
    outcomes = (OutcomeType.continuous,)
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        d = build_design(df, ctx)
        limit = 0.0 if spec.censor_left is None else float(spec.censor_left)
        if (d.y < limit - 1e-12).any():
            raise ToolError("outcome below the censoring limit")
        if not (d.y <= limit).any():
            raise ToolError("no censored observations; use OLS")
        th, cov = tobit_fit(d.y, d.X, limit)
        k = d.X.shape[1]
        stat, p, eff, extra = coef_test(
            th[:k], cov[:k, :k], d.names, spec.exposure, spec.restriction, ctx.alternative.value, spec.dist or "z", None
        )
        return output(
            stat,
            p,
            eff,
            "Tobit coefficient (latent scale)",
            len(d.y),
            {**extra, "sigma": float(np.exp(th[-1])), "censored": int((d.y <= limit).sum())},
        )
