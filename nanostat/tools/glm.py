"""Logistic and probit regression: Wald z test of a coefficient (or joint Wald chi2).

Fitted with statsmodels (maximum likelihood). SEs: classical (inverse information), HC0-HC3
(HC2/HC3 mapped to HC1 for GLMs and recorded), or cluster-robust. z reference distribution by
default, matching R glm/summary and Stata logit/probit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools._models import build_design, coef_test, output, spec_of
from nanostat.tools.base import RunContext, Tool, ToolError, ToolOutput


def _fit(model_cls: type, df: pd.DataFrame, ctx: RunContext) -> tuple[np.ndarray, np.ndarray, list[str], int, str]:
    spec = spec_of(ctx)
    d = build_design(df, ctx)
    if not set(np.unique(d.y)) <= {0.0, 1.0}:
        raise ToolError("outcome must be coded 0/1")
    if d.weights is not None:
        raise ToolError("weights are not supported for logistic/probit in this version")
    m = model_cls(d.y, d.X)
    se = spec.se
    kw: dict = {}
    if se == "cluster":
        if d.cluster is None:
            raise ToolError("se='cluster' needs a cluster column")
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(d.cluster)[0]}}
    elif se in ("HC0", "HC1", "HC2", "HC3"):
        kw = {"cov_type": "HC0" if se == "HC0" else "HC1"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = m.fit(disp=0, maxiter=200, **kw)
        except Exception as e:  # noqa: BLE001 - perfect separation, singular Hessian, ...
            raise ToolError(f"model did not converge: {e}") from e
    if not r.mle_retvals.get("converged", True):
        raise ToolError("model did not converge")
    return np.asarray(r.params), np.asarray(r.cov_params()), d.names, len(d.y), se


class _Binary(Tool):
    designs = (Design.regression,)
    outcomes = (OutcomeType.binary,)
    estimands = ("coefficient", "linear_restriction")
    required_checks = (Check.sample_size,)
    model_cls: type = sm.Logit
    label = ""

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        b, cov, names, n, se = _fit(self.model_cls, df, ctx)
        stat, p, eff, extra = coef_test(
            b, cov, names, spec.exposure, spec.restriction, ctx.alternative.value, spec.dist or "z", None
        )
        return output(stat, p, eff, f"{self.label} coefficient ({se} SE)", n, {**extra, "se_type": se})


class Logistic(_Binary):
    name = "logistic"
    model_cls = sm.Logit
    label = "log-odds"


class Probit(_Binary):
    name = "probit"
    model_cls = sm.Probit
    label = "probit"
