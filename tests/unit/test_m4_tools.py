"""M4 tools against independent references (SPEC §6.1, §13): each tool is checked on >= 5 seeded
datasets against a second implementation (statsmodels where the tool is numpy, and hand-written
numpy where the tool is statsmodels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize, stats

from nanostat.schemas import Alternative, ModelSpec
from nanostat.tools import REGISTRY, ToolError
from nanostat.tools.base import RunContext

SEEDS = [1, 2, 3, 4, 5]


def ctx(spec: ModelSpec, alt: str = "two-sided") -> RunContext:
    return RunContext(columns={}, alternative=Alternative(alt), model=spec)


def data(seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 15, n)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "c": rng.normal(size=n),
            "z": rng.normal(size=n),
            "f": rng.choice(["a", "b", "c"], n),
            "g": g.astype(str),
            "w": rng.uniform(0.5, 2, n),
        }
    )
    df["y"] = 0.5 + 0.25 * df.x - 0.2 * df.c + rng.normal(size=n) + rng.normal(size=15)[g] * 0.7
    df["yb"] = (rng.random(n) < 1 / (1 + np.exp(-(-0.3 + 0.6 * df.x + 0.3 * df.c)))).astype(int)
    df["d"] = 0.6 * df.z + 0.5 * rng.normal(size=n)
    df["yiv"] = 0.4 * df.d + 0.3 * df.c + rng.normal(size=n)
    lam = np.exp(0.5 * df.x)
    t_ev = rng.exponential(1 / lam)
    t_c = rng.exponential(1.5, n)
    df["t"] = np.round(np.minimum(t_ev, t_c), 2) + 0.01  # rounding creates ties
    df["e"] = (t_ev <= t_c).astype(int)
    df["grp"] = rng.integers(0, 3, n).astype(str)
    df["yt"] = np.maximum(0.0, df.y - 0.5)
    return df


def X_of(df, cols, cat=None):
    X = df[cols].copy()
    if cat:
        X = pd.concat([X, pd.get_dummies(df[cat].astype(str), prefix=cat, drop_first=True, dtype=float)], axis=1)
    return sm.add_constant(X, has_constant="add")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("se", ["classical", "HC0", "HC1", "HC2", "HC3", "cluster"])
def test_ols_coef_vs_statsmodels(seed, se):
    df = data(seed)
    spec = ModelSpec(
        outcome="y",
        exposure=["x"],
        covariates=["c"],
        categorical=["f"],
        se=se,
        cluster="g" if se == "cluster" else None,
    )
    out = REGISTRY["ols_coef"].run(df, ctx(spec))
    X = X_of(df, ["x", "c"], "f")
    m = sm.OLS(df.y.to_numpy(), X)
    r = (
        m.fit()
        if se == "classical"
        else (
            m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.g)[0]})
            if se == "cluster"
            else m.fit(cov_type=se)
        )
    )
    se_x = float(r.bse["x"])
    df_ = (df.g.nunique() - 1) if se == "cluster" else (len(df) - X.shape[1])
    p_ref = 2 * stats.t.sf(abs(r.params["x"] / se_x), df_)
    assert out.effect == pytest.approx(r.params["x"], rel=1e-9)
    assert out.extra["se"] == pytest.approx(se_x, rel=1e-8)
    assert out.p_value == pytest.approx(p_ref, rel=1e-7)


@pytest.mark.parametrize("seed", SEEDS)
def test_wls_and_wald_vs_statsmodels(seed):
    df = data(seed)
    out = REGISTRY["ols_coef"].run(df, ctx(ModelSpec(outcome="y", exposure=["x"], covariates=["c"], weights="w")))
    r = sm.WLS(df.y, X_of(df, ["x", "c"]), weights=df.w).fit()
    assert out.p_value == pytest.approx(r.pvalues["x"], rel=1e-7)
    out = REGISTRY["ols_wald"].run(df, ctx(ModelSpec(outcome="y", exposure=["x", "c"], restriction="joint_zero")))
    r = sm.OLS(df.y, X_of(df, ["x", "c"])).fit()
    ft = r.f_test("x = 0, c = 0")
    assert out.statistic == pytest.approx(float(np.squeeze(ft.fvalue)), rel=1e-8)
    assert out.p_value == pytest.approx(float(ft.pvalue), rel=1e-7)
    out = REGISTRY["ols_wald"].run(df, ctx(ModelSpec(outcome="y", exposure=["x", "c"], restriction="equal")))
    ft = r.f_test("x - c = 0")
    assert out.p_value == pytest.approx(float(ft.pvalue), rel=1e-7)


def _newton_glm(y, X, kind):
    """Independent Newton-Raphson for logit/probit; returns beta, classical cov."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ b
        if kind == "logit":
            mu = 1 / (1 + np.exp(-eta))
            g = X.T @ (y - mu)
            H = (X * (mu * (1 - mu))[:, None]).T @ X
        else:
            Phi, phi = stats.norm.cdf(eta), stats.norm.pdf(eta)
            q = 2 * y - 1
            lam = q * stats.norm.pdf(q * eta) / stats.norm.cdf(q * eta)
            g = X.T @ lam
            wts = lam * (lam + eta)  # -d2l/deta2 for probit (q^2 = 1)
            H = (X * wts[:, None]).T @ X
            del Phi, phi
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return b, np.linalg.inv(H)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("tool,kind", [("logistic", "logit"), ("probit", "probit")])
def test_binary_glm_vs_newton(seed, tool, kind):
    df = data(seed)
    out = REGISTRY[tool].run(df, ctx(ModelSpec(outcome="yb", exposure=["x"], covariates=["c"])))
    X = X_of(df, ["x", "c"]).to_numpy(float)
    b, cov = _newton_glm(df.yb.to_numpy(float), X, kind)
    z = b[1] / np.sqrt(cov[1, 1])
    assert out.effect == pytest.approx(b[1], rel=1e-6)
    assert out.p_value == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_iv_2sls_vs_statsmodels(seed):
    from statsmodels.sandbox.regression.gmm import IV2SLS

    df = data(seed)
    out = REGISTRY["iv_2sls"].run(
        df, ctx(ModelSpec(outcome="yiv", exposure=["d"], covariates=["c"], instruments=["z"], dist="t"))
    )
    X = X_of(df, ["d", "c"])
    Z = X_of(df, ["z", "c"])
    r = IV2SLS(df.yiv.to_numpy(), X.to_numpy(), Z.to_numpy()).fit()
    j = list(X.columns).index("d")
    assert out.effect == pytest.approx(r.params[j], rel=1e-8)
    assert out.extra["se"] == pytest.approx(r.bse[j], rel=1e-8)
    assert out.extra["first_stage_F"] > 10


def _logrank(t, e, g):
    groups = np.unique(g)
    times = np.unique(t[e == 1])
    obs = np.zeros(len(groups))
    E = np.zeros(len(groups))
    V = np.zeros((len(groups), len(groups)))
    for tt in times:
        at = t >= tt
        n = at.sum()
        d = ((t == tt) & (e == 1)).sum()
        nk = np.array([(at & (g == k)).sum() for k in groups])
        dk = np.array([((t == tt) & (e == 1) & (g == k)).sum() for k in groups])
        obs += dk
        E += d * nk / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(nk / n) - np.outer(nk, nk) / n**2)
    u = (obs - E)[:-1]
    chi = float(u @ np.linalg.solve(V[:-1, :-1], u))
    return chi, float(stats.chi2.sf(chi, len(groups) - 1))


@pytest.mark.parametrize("seed", SEEDS)
def test_log_rank_vs_own(seed):
    df = data(seed)
    out = REGISTRY["log_rank"].run(df, ctx(ModelSpec(time="t", event="e", group="grp")))
    chi, p = _logrank(df.t.to_numpy(), df.e.to_numpy(), df.grp.to_numpy())
    assert out.statistic == pytest.approx(chi, rel=1e-7) and out.p_value == pytest.approx(p, rel=1e-6)


def _cox_breslow(t, e, X):
    order = np.argsort(t)
    t, e, X = t[order], e[order], X[order]

    def nll(b):
        eta = X @ b
        ll = 0.0
        for tt in np.unique(t[e == 1]):
            d = (t == tt) & (e == 1)
            risk = t >= tt
            ll += eta[d].sum() - d.sum() * np.log(np.exp(eta[risk]).sum())
        return -ll

    res = optimize.minimize(nll, np.zeros(X.shape[1]), method="BFGS", options={"gtol": 1e-9})
    b = res.x
    k = len(b)
    H = np.zeros((k, k))
    h = 1e-5
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (nll(b + ei + ej) - nll(b + ei - ej) - nll(b - ei + ej) + nll(b - ei - ej)) / (4 * h * h)
    return b, np.linalg.inv(H)


@pytest.mark.parametrize("seed", SEEDS)
def test_cox_breslow_vs_own_partial_likelihood(seed):
    df = data(seed, n=150)
    out = REGISTRY["cox_ph"].run(
        df, ctx(ModelSpec(time="t", event="e", exposure=["x"], covariates=["c"], ties="breslow"))
    )
    b, cov = _cox_breslow(df.t.to_numpy(), df.e.to_numpy(), df[["x", "c"]].to_numpy(float))
    assert out.effect == pytest.approx(b[0], abs=1e-5)
    assert out.p_value == pytest.approx(2 * stats.norm.sf(abs(b[0] / np.sqrt(cov[0, 0]))), rel=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
def test_mixed_lm_fixed_effect_matches_gls(seed):
    df = data(seed)
    out = REGISTRY["mixed_lm"].run(df, ctx(ModelSpec(outcome="y", exposure=["x"], covariates=["c"], group="g")))
    tau2, s2 = out.extra["group_var"], out.extra["resid_var"]  # statsmodels cov_re is on the data scale
    X = X_of(df, ["x", "c"]).to_numpy(float)
    y = df.y.to_numpy(float)
    XtVX = np.zeros((3, 3))
    XtVy = np.zeros(3)
    for _, idx in df.groupby("g").indices.items():
        m = len(idx)
        Vinv = (np.eye(m) - tau2 / (s2 + m * tau2) * np.ones((m, m))) / s2
        XtVX += X[idx].T @ Vinv @ X[idx]
        XtVy += X[idx].T @ Vinv @ y[idx]
    cov = np.linalg.inv(XtVX)
    b = cov @ XtVy
    assert out.effect == pytest.approx(b[1], rel=1e-4)
    # statsmodels takes the fixed-effect covariance from the full inverse Hessian (which couples the
    # variance parameters); this GLS check holds them fixed, so SEs agree to ~0.1%, not machine precision.
    assert out.extra["se"] == pytest.approx(float(np.sqrt(cov[1, 1])), rel=1e-2)


@pytest.mark.parametrize("seed", SEEDS)
def test_tobit_vs_second_optimizer(seed):
    df = data(seed)
    out = REGISTRY["tobit"].run(df, ctx(ModelSpec(outcome="yt", exposure=["x"], covariates=["c"])))
    y = df.yt.to_numpy(float)
    X = X_of(df, ["x", "c"]).to_numpy(float)
    cens = y <= 0

    def nll(th):
        b, s = th[:-1], np.exp(th[-1])
        mu = X @ b
        return -(
            stats.norm.logcdf(-mu[cens] / s).sum() + (stats.norm.logpdf((y[~cens] - mu[~cens]) / s) - np.log(s)).sum()
        )

    res = optimize.minimize(
        nll, np.zeros(4), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000}
    )
    assert out.effect == pytest.approx(res.x[1], abs=2e-4)


def test_m4_errors_are_clean():
    df = data(1)
    with pytest.raises(ToolError):
        REGISTRY["iv_2sls"].run(df, ctx(ModelSpec(outcome="yiv", exposure=["d"])))
    with pytest.raises(ToolError):
        REGISTRY["logistic"].run(df, ctx(ModelSpec(outcome="y", exposure=["x"])))
    with pytest.raises(ToolError):
        REGISTRY["ols_coef"].run(df, ctx(ModelSpec(outcome="y", exposure=["x"], se="cluster")))
    with pytest.raises(ToolError):
        REGISTRY["ols_coef"].run(df, RunContext(columns={}))
