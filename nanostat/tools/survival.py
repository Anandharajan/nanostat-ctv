"""Survival tests: log-rank (k groups) and Cox proportional hazards (Wald test of a coefficient).

log_rank : statsmodels `survdiff` (chi2 on k-1 df), two-sided.
cox_ph   : statsmodels PHReg; ties "efron" by default (R coxph default) or "breslow"; Wald z test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff

from nanostat.schemas import Check, Design, OutcomeType
from nanostat.tools._models import coef_test, output, spec_of
from nanostat.tools.base import RunContext, Tool, ToolError, ToolOutput


def _time_event(frame: pd.DataFrame, spec) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    t = pd.to_numeric(frame[spec.time], errors="coerce").to_numpy(float)
    ev = pd.to_numeric(frame[spec.event], errors="coerce").to_numpy(float)
    if np.isnan(t).any() or np.isnan(ev).any() or not set(np.unique(ev)) <= {0.0, 1.0}:
        raise ToolError("time must be numeric and event coded 0/1")
    if (t < 0).any():
        raise ToolError("negative survival times")
    if ev.sum() == 0:
        raise ToolError("no events")
    return t, ev


class LogRank(Tool):
    name = "log_rank"
    estimands = ("survival_curves_equal",)
    designs = (Design.survival,)
    outcomes = (OutcomeType.time_to_event,)
    two_sided_only = True
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        grp = spec.group or (spec.exposure[0] if spec.exposure else None)
        if not (spec.time and spec.event and grp):
            raise ToolError("log_rank needs model.time, model.event and model.group")
        frame = df[[spec.time, spec.event, grp]].dropna()
        t, ev = _time_event(frame, spec)
        g = frame[grp].astype(str).to_numpy()
        if len(np.unique(g)) < 2:
            raise ToolError("log_rank needs at least 2 groups")
        chisq, p = survdiff(t, ev, g)
        return output(
            float(chisq),
            float(p),
            None,
            "log-rank chi-squared",
            len(t),  # type: ignore[arg-type]
            {"groups": int(len(np.unique(g)))},
        )


class CoxPH(Tool):
    name = "cox_ph"
    estimands = ("hazard_ratio", "linear_restriction")
    designs = (Design.survival,)
    outcomes = (OutcomeType.time_to_event,)
    required_checks = (Check.sample_size,)
    r_reference = None

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:
        spec = spec_of(ctx)
        if not (spec.time and spec.event and spec.exposure):
            raise ToolError("cox_ph needs model.time, model.event and model.exposure")
        cols = [
            c
            for c in dict.fromkeys(
                [spec.time, spec.event, *spec.exposure, *spec.covariates, *spec.categorical, spec.cluster]
            )
            if c
        ]
        frame = df[cols].dropna()
        t, ev = _time_event(frame, spec)
        parts = [frame[[*spec.exposure, *spec.covariates]].apply(pd.to_numeric, errors="coerce")]
        for c in spec.categorical:
            parts.append(pd.get_dummies(frame[c].astype(str), prefix=c, drop_first=True, dtype=float))
        X = pd.concat(parts, axis=1)
        if X.isna().any().any():
            raise ToolError("Cox covariates must be numeric")
        names = list(X.columns)
        kw: dict = {}
        if spec.se == "cluster" or spec.cluster:
            if not spec.cluster:
                raise ToolError("se='cluster' needs model.cluster")
            kw["groups"] = pd.factorize(frame[spec.cluster].astype(str))[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = PHReg(t, X.to_numpy(float), status=ev, ties=spec.ties, **kw).fit()
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"Cox model failed: {e}") from e
        b, cov = np.asarray(r.params), np.asarray(r.cov_params())
        stat, p, eff, extra = coef_test(
            b, cov, names, spec.exposure, spec.restriction, ctx.alternative.value, "z", None
        )
        return output(
            stat,
            p,
            eff,
            f"log hazard ratio ({spec.ties} ties)",
            len(t),
            {**extra, "hazard_ratio": float(np.exp(eff)) if np.isfinite(eff) else None},
        )
