"""Independent reference implementations written from textbook formulas (numpy + distribution
functions only, no scipy test routines). Used to check every tool plugin (SPEC §6.1, §13)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.stats import chi2, f, norm, t


def _p_from_t(tstat: float, df: float, alt: str) -> float:
    if alt == "greater":
        return float(t.sf(tstat, df))
    if alt == "less":
        return float(t.cdf(tstat, df))
    return float(min(1.0, 2 * t.sf(abs(tstat), df)))


def welch(a, b, alt):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    ts = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return ts, _p_from_t(ts, df, alt)


def student(a, b, alt):
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    ts = (a.mean() - b.mean()) / math.sqrt(sp2 * (1 / na + 1 / nb))
    return ts, _p_from_t(ts, na + nb - 2, alt)


def paired(y2, y1, alt):
    d = y2 - y1
    ts = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return ts, _p_from_t(ts, len(d) - 1, alt)


def _yuen_parts(x, tr=0.2):
    x = np.sort(x)
    n = len(x)
    g = int(math.floor(tr * n))
    tm = x[g : n - g].mean()
    w = x.copy()
    w[:g] = x[g]
    w[n - g :] = x[n - g - 1]
    sw2 = w.var(ddof=1)
    h = n - 2 * g
    d = (n - 1) * sw2 / (h * (h - 1))
    return tm, d, h


def yuen(a, b, alt):
    ta, da, ha = _yuen_parts(a)
    tb, db, hb = _yuen_parts(b)
    ts = (ta - tb) / math.sqrt(da + db)
    df = (da + db) ** 2 / (da**2 / (ha - 1) + db**2 / (hb - 1))
    return ts, _p_from_t(ts, df, alt)


def rankdata_avg(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def mann_whitney_asymptotic(a, b, alt):
    n1, n2 = len(a), len(b)
    r = rankdata_avg(np.r_[a, b])
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    n = n1 + n2
    _, counts = np.unique(np.r_[a, b], return_counts=True)
    tie = (counts**3 - counts).sum()
    s = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    mu = n1 * n2 / 2
    u = {"greater": u1, "less": u2}.get(alt, max(u1, u2))
    z = (u - mu - 0.5) / s
    p = norm.sf(z)
    if alt == "two-sided":
        p = min(1.0, 2 * p)
    return u1, float(p)


def wilcoxon_exact(d, alt):
    d = d[d != 0]
    n = len(d)
    r = rankdata_avg(np.abs(d))
    tplus = int(round(r[d > 0].sum()))
    tminus = int(round(r[d < 0].sum()))
    # distribution of T+ under H0 via subset-sum counts
    maxs = n * (n + 1) // 2
    cnt = np.zeros(maxs + 1, dtype=object)
    cnt[0] = 1
    for k in range(1, n + 1):
        cnt[k:] = cnt[k:] + cnt[: maxs + 1 - k].copy()
    total = 2**n
    pmf = np.array([Fraction(int(c), total) for c in cnt])
    ge = float(sum(pmf[tplus:]))
    le = float(sum(pmf[: tplus + 1]))
    if alt == "greater":
        return tplus, ge
    if alt == "less":
        return tplus, le
    return min(tplus, tminus), min(1.0, 2 * min(ge, le))


def pearson(x, y, alt):
    xm, ym = x - x.mean(), y - y.mean()
    r = (xm * ym).sum() / math.sqrt((xm**2).sum() * (ym**2).sum())
    n = len(x)
    ts = r * math.sqrt((n - 2) / (1 - r**2))
    return r, _p_from_t(ts, n - 2, alt)


def spearman(x, y, alt):
    return pearson(rankdata_avg(x), rankdata_avg(y), alt)


def anova(groups):
    allv = np.concatenate(groups)
    k, n = len(groups), len(allv)
    ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    F = (ssb / (k - 1)) / (ssw / (n - k))
    return F, float(f.sf(F, k - 1, n - k))


def kruskal(groups):
    allv = np.concatenate(groups)
    n = len(allv)
    r = rankdata_avg(allv)
    h, i = 0.0, 0
    for g in groups:
        h += r[i : i + len(g)].sum() ** 2 / len(g)
        i += len(g)
    H = 12 / (n * (n + 1)) * h - 3 * (n + 1)
    _, counts = np.unique(allv, return_counts=True)
    C = 1 - (counts**3 - counts).sum() / (n**3 - n)
    H /= C
    return H, float(chi2.sf(H, len(groups) - 1))


def chi2_2x2(tab):
    tab = np.asarray(tab, float)
    e = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
    x2 = ((tab - e) ** 2 / e).sum()
    return x2, float(chi2.sf(x2, 1))


def fisher(tab, alt):
    (a, b), (c, d) = tab
    r1, c1, N = a + b, a + c, a + b + c + d
    lo, hi = max(0, r1 - (N - c1)), min(r1, c1)
    den = math.comb(N, r1)
    pm = {x: Fraction(math.comb(c1, x) * math.comb(N - c1, r1 - x), den) for x in range(lo, hi + 1)}
    if alt == "greater":
        return float(sum(v for x, v in pm.items() if x >= a))
    if alt == "less":
        return float(sum(v for x, v in pm.items() if x <= a))
    pa = pm[a]
    return float(min(1, sum(v for v in pm.values() if v <= pa * Fraction(10000001, 10000000))))


def binom(k, n, p0, alt):
    p0f = Fraction(p0).limit_denominator(10**6)
    pm = [math.comb(n, i) * p0f**i * (1 - p0f) ** (n - i) for i in range(n + 1)]
    if alt == "greater":
        return float(sum(pm[k:]))
    if alt == "less":
        return float(sum(pm[: k + 1]))
    d = pm[k]
    return float(min(1, sum(v for v in pm if v <= d * Fraction(10000001, 10000000))))
