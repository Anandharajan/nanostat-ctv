"""Every plugin matches an independent reference within tolerance on >= 5 fixtures (SPEC §13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanostat.schemas import Alternative
from nanostat.tools import REGISTRY, get_tool
from nanostat.tools.base import RunContext
from tests.fixtures import reference_impl as ref

ALTS = ["two-sided", "greater", "less", "two-sided", "greater"]
SEEDS = [11, 22, 33, 44, 55]
REL = 1e-7


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-10, REL * max(abs(a), abs(b)))


def two_group_df(seed: int, ties: bool = False):
    rng = np.random.default_rng(seed)
    na, nb = int(rng.integers(12, 30)), int(rng.integers(12, 30))
    a = rng.normal(0.4, 1.0, na)
    b = rng.normal(0.0, 1.5, nb)
    if ties:
        a, b = np.round(a, 1), np.round(b, 1)
    df = pd.DataFrame({"g": ["ctl"] * nb + ["trt"] * na, "y": np.r_[b, a]})
    return df, a, b


def ctx(alt: str, **cols: str) -> RunContext:
    return RunContext(columns=cols, alternative=Alternative(alt), group_order=["ctl", "trt"])


@pytest.mark.parametrize("i", range(5))
@pytest.mark.parametrize(
    "method,fn",
    [("welch_t", ref.welch), ("student_t", ref.student), ("yuen_trimmed_t", ref.yuen)],
)
def test_two_sample_t_family(method, fn, i):
    df, a, b = two_group_df(SEEDS[i])
    out = get_tool(method).run(df, ctx(ALTS[i], y="y", group="g"))
    ts, p = fn(a, b, ALTS[i])
    assert close(out.statistic, ts) and close(out.p_value, p)


@pytest.mark.parametrize("i", range(5))
def test_mann_whitney(i):
    df, a, b = two_group_df(SEEDS[i], ties=True)  # ties -> asymptotic branch with tie correction
    out = get_tool("mann_whitney").run(df, ctx(ALTS[i], y="y", group="g"))
    u1, p = ref.mann_whitney_asymptotic(a, b, ALTS[i])
    assert close(out.statistic, u1) and close(out.p_value, p)


def paired_df(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 25))
    pre = rng.normal(50, 10, n)
    post = pre + rng.normal(1.5, 3, n)
    ids = np.arange(n)
    df = pd.DataFrame({"id": np.r_[ids, ids], "g": ["ctl"] * n + ["trt"] * n, "y": np.r_[pre, post]})
    return df.sample(frac=1, random_state=seed), post, pre


@pytest.mark.parametrize("i", range(5))
def test_paired_t_long_and_wide(i):
    df, post, pre = paired_df(SEEDS[i])
    out = get_tool("paired_t").run(df, ctx(ALTS[i], y="y", group="g", pair_id="id"))
    ts, p = ref.paired(post, pre, ALTS[i])
    assert close(out.statistic, ts) and close(out.p_value, p)
    wide = pd.DataFrame({"pre": pre, "post": post})
    out2 = get_tool("paired_t").run(wide, ctx(ALTS[i], y1="pre", y2="post"))
    assert close(out2.p_value, p)


@pytest.mark.parametrize("i", range(5))
def test_wilcoxon_exact(i):
    df, post, pre = paired_df(SEEDS[i])
    out = get_tool("wilcoxon_signed_rank").run(df, ctx(ALTS[i], y="y", group="g", pair_id="id"))
    stat, p = ref.wilcoxon_exact(post - pre, ALTS[i])
    assert close(out.statistic, stat) and close(out.p_value, p)


def xy_df(seed: int, ties: bool = False):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(15, 60))
    x = rng.normal(size=n)
    y = 0.3 * x + rng.normal(size=n)
    if ties:
        x, y = np.round(x, 1), np.round(y, 1)
    return pd.DataFrame({"x": x, "y": y}), x, y


@pytest.mark.parametrize("i", range(5))
def test_pearson(i):
    df, x, y = xy_df(SEEDS[i])
    out = get_tool("pearson").run(df, ctx(ALTS[i], x="x", y="y"))
    r, p = ref.pearson(x, y, ALTS[i])
    assert close(out.statistic, r) and close(out.p_value, p)


@pytest.mark.parametrize("i", range(5))
def test_spearman_with_ties(i):
    df, x, y = xy_df(SEEDS[i], ties=True)
    out = get_tool("spearman").run(df, ctx(ALTS[i], x="x", y="y"))
    r, p = ref.spearman(x, y, ALTS[i])
    assert close(out.statistic, r) and close(out.p_value, p)


def k_df(seed: int):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 5))
    gs = [np.round(rng.normal(0.3 * j, 1, int(rng.integers(8, 20))), 2) for j in range(k)]
    df = pd.DataFrame({"g": sum(([f"l{j}"] * len(g) for j, g in enumerate(gs)), []), "y": np.concatenate(gs)})
    return df, gs


@pytest.mark.parametrize("i", range(5))
def test_anova_and_kruskal(i):
    df, gs = k_df(SEEDS[i])
    c = RunContext(columns={"y": "y", "group": "g"})
    out = get_tool("anova_oneway").run(df, c)
    F, p = ref.anova(gs)
    assert close(out.statistic, F) and close(out.p_value, p)
    out = get_tool("kruskal_wallis").run(df, c)
    H, p = ref.kruskal(gs)
    assert close(out.statistic, H) and close(out.p_value, p)


def table_df(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 40))
    ya = (rng.random(n) < 0.55).astype(int)
    yb = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({"g": ["trt"] * n + ["ctl"] * n, "y": np.r_[ya, yb]})
    tab = [[int(ya.sum()), int(n - ya.sum())], [int(yb.sum()), int(n - yb.sum())]]
    return df, tab


@pytest.mark.parametrize("i", range(5))
def test_chi2_and_fisher(i):
    df, tab = table_df(SEEDS[i])
    out = get_tool("chi2_independence").run(df, ctx("two-sided", y="y", group="g"))
    x2, p = ref.chi2_2x2(tab)
    assert close(out.statistic, x2) and close(out.p_value, p)
    out = get_tool("fisher_exact").run(df, ctx(ALTS[i], y="y", group="g"))
    assert close(out.p_value, ref.fisher(tab, ALTS[i]))


@pytest.mark.parametrize("i", range(5))
def test_binomial(i):
    rng = np.random.default_rng(SEEDS[i])
    n = int(rng.integers(15, 80))
    p0 = [0.2, 0.25, 0.3, 0.5, 0.4][i]
    y = (rng.random(n) < p0 + 0.1).astype(int)
    df = pd.DataFrame({"y": y})
    out = get_tool("binomial_test").run(df, RunContext(columns={"y": "y"}, alternative=Alternative(ALTS[i]), p0=p0))
    assert close(out.p_value, ref.binom(int(y.sum()), n, p0, ALTS[i]))


def test_every_registered_tool_has_fixture_coverage():
    covered = {
        "welch_t",
        "student_t",
        "yuen_trimmed_t",
        "mann_whitney",
        "paired_t",
        "wilcoxon_signed_rank",
        "pearson",
        "spearman",
        "anova_oneway",
        "kruskal_wallis",
        "chi2_independence",
        "fisher_exact",
        "binomial_test",
        "permutation_test",
    }
    assert set(REGISTRY) == covered


@pytest.mark.parametrize("i", range(5))
def test_permutation_exact_two_group_paired_correlation(i):
    rng = np.random.default_rng(SEEDS[i])
    alt = ALTS[i]
    a, b = rng.normal(0.8, 1, 5), rng.normal(0, 1, 4)
    df = pd.DataFrame({"g": ["ctl"] * 4 + ["trt"] * 5, "y": np.r_[b, a]})
    out = get_tool("permutation_test").run(df, ctx(alt, y="y", group="g"))
    stat, p = ref.perm_two_exact(a, b, alt)
    assert out.extra["exact"] and close(out.statistic, stat) and close(out.p_value, p)

    pre = rng.normal(10, 2, 8)
    post = pre + rng.normal(0.5, 1, 8)
    wide = pd.DataFrame({"pre": pre, "post": post})
    out = get_tool("permutation_test").run(wide, ctx(alt, y1="pre", y2="post"))
    stat, p = ref.perm_paired_exact(list(post - pre), alt)
    assert out.extra["exact"] and close(out.statistic, stat) and close(out.p_value, p)

    x = rng.normal(size=6)
    y = 0.6 * x + rng.normal(size=6)
    out = get_tool("permutation_test").run(pd.DataFrame({"x": x, "y": y}), ctx(alt, x="x", y="y"))
    stat, p = ref.perm_corr_exact(list(x), list(y), alt)
    assert out.extra["exact"] and close(out.statistic, stat) and close(out.p_value, p)


def test_permutation_monte_carlo_accuracy_and_determinism():
    df, a, b = two_group_df(SEEDS[0])
    t = get_tool("permutation_test")
    o1 = t.run(df, ctx("two-sided", y="y", group="g"))
    o2 = t.run(df, ctx("two-sided", y="y", group="g"))
    assert not o1.extra["exact"] and o1.p_value == o2.p_value  # seeded -> reproducible
    # independent high-precision Monte Carlo reference (different generator, 200k permutations)
    rng = np.random.default_rng(999)
    pooled, na = np.r_[a, b], len(a)
    obs = a.mean() - b.mean()
    hits = 0
    for _ in range(20):
        idx = np.argsort(rng.random((10_000, len(pooled))), axis=1)[:, :na]
        sa = pooled[idx].sum(axis=1)
        hits += int((np.abs(sa / na - (pooled.sum() - sa) / len(b)) >= abs(obs) - 1e-12).sum())
    p_ref = hits / 200_000
    se = np.sqrt(max(p_ref * (1 - p_ref), 1e-4) / 9_999)
    assert abs(o1.p_value - p_ref) < 4 * se + 1e-4
