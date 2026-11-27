"""Shared machinery for model-based tools: design matrices from a typed ModelSpec, sandwich
covariances and Wald tests. Pure numpy/pandas; conventions documented inline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nanostat.tools.base import RunContext, ToolError, ToolOutput


@dataclass
class Design_:
    y: np.ndarray
    X: np.ndarray
    names: list[str]
    cluster: np.ndarray | None
    weights: np.ndarray | None
    frame: pd.DataFrame  # analysed rows (complete cases)


def spec_of(ctx: RunContext) -> Any:
    if ctx.model is None:
        raise ToolError("this method needs a model specification (plan.model)")
    return ctx.model


def _numeric(s: pd.Series, name: str) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce")
    if v.isna().any():
        raise ToolError(f"column {name!r} is not numeric")
    return v.to_numpy(float)


def build_design(
    df: pd.DataFrame, ctx: RunContext, y_col: str | None = None, extra: list[str] | None = None
) -> Design_:
    spec = spec_of(ctx)
    y_name = y_col or spec.outcome
    if not y_name:
        raise ToolError("model needs an outcome")
    if not spec.exposure:
        raise ToolError("model needs at least one exposure (coefficient to test)")
    used = [
        c
        for c in dict.fromkeys(
            [y_name, *spec.exposure, *spec.covariates, *spec.categorical, spec.cluster, spec.weights, *(extra or [])]
        )
        if c
    ]
    missing = [c for c in used if c not in df.columns]
    if missing:
        raise ToolError(f"unknown columns {missing}")
    frame = df[used].dropna()
    if len(frame) < 5:
        raise ToolError("fewer than 5 complete rows")
    parts = [pd.DataFrame({c: _numeric(frame[c], c) for c in [*spec.exposure, *spec.covariates]}, index=frame.index)]
    for c in spec.categorical:
        d = pd.get_dummies(frame[c].astype(str), prefix=c, drop_first=True, dtype=float)
        parts.append(d)
    X = pd.concat(parts, axis=1)
    if spec.intercept:
        X.insert(0, "const", 1.0)
    keep = X.columns[(X.std(ddof=0) > 0) | (X.columns == "const")]
    dropped = [c for c in X.columns if c not in keep and c in spec.exposure]
    if dropped:
        raise ToolError(f"exposure {dropped} has no variation in the analysed rows")
    X = X[keep]
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ToolError("design matrix is rank deficient (collinear regressors)")
    y = _numeric(frame[y_name], y_name)
    cl = frame[spec.cluster].astype(str).to_numpy() if spec.cluster else None
    w = _numeric(frame[spec.weights], spec.weights) if spec.weights else None
    if w is not None and (w <= 0).any():
        raise ToolError("weights must be positive")
    return Design_(y, X.to_numpy(float), list(X.columns), cl, w, frame)


def sandwich(
    X: np.ndarray,
    resid: np.ndarray,
    bread: np.ndarray,
    se: str,
    cluster: np.ndarray | None,
    hat: np.ndarray | None = None,
) -> np.ndarray:
    """Sandwich covariance for linear models. bread = (X'X)^-1 (or its weighted/IV analogue).

    HC0: X'diag(e^2)X; HC1: * n/(n-k); HC2: e^2/(1-h); HC3: e^2/(1-h)^2;
    cluster (CR1, Stata/statsmodels default): G/(G-1) * (n-1)/(n-k) * sum_g (X_g'e_g)(X_g'e_g)'."""
    n, k = X.shape
    if se == "cluster":
        if cluster is None:
            raise ToolError("se='cluster' needs a cluster column")
        groups = pd.factorize(cluster)[0]
        G = int(groups.max()) + 1
        if G < 2:
            raise ToolError("need at least 2 clusters")
        S = np.zeros((G, k))
        np.add.at(S, groups, X * resid[:, None])
        meat = S.T @ S
        c = G / (G - 1) * (n - 1) / (n - k)
        return c * bread @ meat @ bread
    if se in ("HC2", "HC3"):
        if hat is None:
            hat = np.einsum("ij,jk,ik->i", X, bread, X)
        u2 = resid**2 / (1 - hat) ** (1 if se == "HC2" else 2)
    else:
        u2 = resid**2
    meat = (X * u2[:, None]).T @ X
    cov = bread @ meat @ bread
    if se == "HC1":
        cov *= n / (n - k)
    return cov


def ref_df(se: str, n: int, k: int, cluster: np.ndarray | None) -> float:
    """Degrees of freedom for t-based inference: G-1 for clustered SEs, n-k otherwise."""
    if se == "cluster" and cluster is not None:
        return float(len(np.unique(cluster)) - 1)
    return float(n - k)


def coef_test(
    b: np.ndarray,
    cov: np.ndarray,
    names: list[str],
    exposure: list[str],
    restriction: str,
    alternative: str,
    dist: str,
    df: float | None,
) -> tuple[float, float, float, dict[str, Any]]:
    """Single coefficient -> t/z test (one-sided allowed); several -> Wald/F test.
    Returns (statistic, p, effect, extra)."""
    idx = [names.index(e) for e in exposure if e in names]
    if len(idx) != len(exposure):
        raise ToolError(f"exposure not in model: {[e for e in exposure if e not in names]}")
    if len(idx) == 1 and restriction == "each":
        j = idx[0]
        se_j = float(np.sqrt(cov[j, j]))
        if not np.isfinite(se_j) or se_j <= 0:
            raise ToolError("non-positive standard error")
        tstat = float(b[j] / se_j)
        dist_ = stats.t(df) if (dist == "t" and df) else stats.norm()
        if alternative == "greater":
            p = float(dist_.sf(tstat))
        elif alternative == "less":
            p = float(dist_.cdf(tstat))
        else:
            p = float(2 * dist_.sf(abs(tstat)))
        return tstat, p, float(b[j]), {"se": se_j, "df": df if dist == "t" else None, "dist": dist}
    # linear restriction R b = 0
    if restriction == "equal":
        if len(idx) != 2:
            raise ToolError("restriction 'equal' needs exactly two exposures")
        R = np.zeros((1, len(b)))
        R[0, idx[0]], R[0, idx[1]] = 1.0, -1.0
    else:
        R = np.zeros((len(idx), len(b)))
        for r, j in enumerate(idx):
            R[r, j] = 1.0
    if alternative != "two-sided":
        raise ToolError("joint restrictions are two-sided only")
    rb = R @ b
    W = float(rb @ np.linalg.solve(R @ cov @ R.T, rb))
    q = R.shape[0]
    if dist == "t" and df:
        F = W / q
        return (
            F,
            float(stats.f.sf(F, q, df)),
            float(rb[0]) if q == 1 else float("nan"),
            {"q": q, "df2": df, "dist": "F"},
        )
    return W, float(stats.chi2.sf(W, q)), float(rb[0]) if q == 1 else float("nan"), {"q": q, "dist": "chi2"}


def output(stat: float, p: float, effect: float | None, label: str, n: int, extra: dict[str, Any]) -> ToolOutput:
    if not np.isfinite(p):
        raise ToolError("non-finite p-value")
    return ToolOutput(
        statistic=float(stat), p_value=float(min(max(p, 0.0), 1.0)), effect=effect, effect_label=label, n=n, extra=extra
    )
