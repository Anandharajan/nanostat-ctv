"""Seeded data-generating processes (SPEC §6.4). Own implementation; no third-party data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

FAMILIES = ("two_group", "paired", "correlation", "k_group", "binary_two_group", "one_proportion")
EFFECTS = ("null", "borderline", "medium", "large")
N_REGIMES = {"small": (12, 20), "medium": (25, 45), "large": (60, 120)}
EFFECT_Z = {"null": 0.0, "borderline": 2.0, "medium": 3.2, "large": 4.5}

PERTURBATIONS: dict[str, tuple[str, ...]] = {
    "two_group": ("none", "outliers", "skewed", "unequal_variance"),
    "paired": ("none", "outliers", "skewed"),
    "correlation": ("none", "outliers", "skewed"),
    "k_group": ("none", "outliers", "skewed", "unequal_variance"),
    "binary_two_group": ("none", "small_counts"),
    "one_proportion": ("none",),
}


@dataclass(frozen=True)
class Scenario:
    family: str
    n_regime: str
    effect: str
    perturbation: str
    alternative: str  # two-sided | greater | less
    prompt_style: str  # hint | no_hint
    seed: int
    domain: int = 0
    k: int = 3
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def with_(self, **kw: Any) -> Scenario:
        return replace(self, **kw)


@dataclass
class Simulated:
    df: pd.DataFrame
    roles: dict[str, str]
    group_order: list[str] | None
    h0_true: bool
    p0: float | None = None
    true_effect: float = 0.0


def _sign(sc: Scenario, rng: np.random.Generator) -> float:
    if sc.alternative == "greater":
        return 1.0
    if sc.alternative == "less":
        return -1.0
    return float(rng.choice([-1.0, 1.0]))


def _errors(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Mean-zero, unit-variance noise."""
    if kind == "skewed":
        s = 1.0
        x = rng.lognormal(0.0, s, size)
        mean, var = math.exp(s**2 / 2), (math.exp(s**2) - 1) * math.exp(s**2)
        return (x - mean) / math.sqrt(var)
    return rng.standard_normal(size)


def inject_outliers(
    x: np.ndarray, rng: np.random.Generator, k: int | None = None, sign: float | None = None
) -> np.ndarray:
    """Replace k values by points 8-12 robust SDs away from the median."""
    x = x.copy()
    k = k or max(1, int(round(0.08 * len(x))))
    med = float(np.median(x))
    sd = float(np.median(np.abs(x - med)) / 0.6745) or float(np.std(x)) or 1.0
    idx = rng.choice(len(x), size=k, replace=False)
    # sign=None -> independent random sign per point (symmetric contamination, keeps a null a null)
    sgn = np.full(k, sign) if sign is not None else rng.choice([-1.0, 1.0], size=k)
    x[idx] = med + sgn * rng.uniform(8.0, 12.0, size=k) * sd
    return x


def _n(sc: Scenario, rng: np.random.Generator) -> int:
    lo, hi = N_REGIMES[sc.n_regime]
    return int(rng.integers(lo, hi + 1))


def _ids(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(np.arange(1000, 1000 + n))


def simulate(sc: Scenario, names: dict[str, Any]) -> Simulated:
    rng = np.random.default_rng(sc.seed)
    z = EFFECT_Z[sc.effect]
    sgn = _sign(sc, rng)
    h0 = sc.effect == "null"
    ref, comp = names["ref"], names["comp"]
    y, g, pid = names["y"], names["group"], names["id"]
    base, scale = float(rng.uniform(20, 80)), float(rng.uniform(2, 15))

    if sc.family == "two_group":
        n = _n(sc, rng)
        n_ref, n_comp = n, n
        sd_ref, sd_comp = 1.0, 1.0
        if sc.perturbation == "unequal_variance":
            n_ref, n_comp, sd_ref, sd_comp = 2 * n, n, 1.0, 3.0  # larger variance in smaller group
        se = math.sqrt(sd_ref**2 / n_ref + sd_comp**2 / n_comp)
        delta = sgn * z * se
        a = _errors(sc.perturbation, n_ref, rng) * sd_ref
        b = delta + _errors(sc.perturbation, n_comp, rng) * sd_comp
        if sc.perturbation == "outliers":
            if h0:  # same symmetric contamination in both groups: H0 (identical distributions) still holds
                a = inject_outliers(a, rng)
                b = inject_outliers(b, rng)
            else:
                b = inject_outliers(b, rng, sign=sgn)
        vals = np.r_[a, b] * scale + base
        df = pd.DataFrame({pid: _ids(n_ref + n_comp, rng), g: [ref] * n_ref + [comp] * n_comp, y: np.round(vals, 3)})
        return Simulated(df, {"y": y, "group": g}, [ref, comp], h0, true_effect=delta * scale)

    if sc.family == "paired":
        n = _n(sc, rng)
        delta = sgn * z / math.sqrt(n)
        subj = rng.standard_normal(n) * 2.0
        d = delta + _errors(sc.perturbation, n, rng)
        if sc.perturbation == "outliers":
            d = inject_outliers(d, rng, sign=sgn if not h0 else None)
        pre = subj + rng.standard_normal(n) * 0.3
        post = pre + d
        ids = _ids(n, rng)
        df = pd.DataFrame(
            {pid: np.r_[ids, ids], g: [ref] * n + [comp] * n, y: np.round(np.r_[pre, post] * scale + base, 3)}
        )
        df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
        return Simulated(df, {"y": y, "group": g, "pair_id": pid}, [ref, comp], h0, true_effect=delta * scale)

    if sc.family == "correlation":
        n = _n(sc, rng)
        r = sgn * math.tanh(z / math.sqrt(max(n - 3, 1)))
        x = _errors(sc.perturbation, n, rng)
        e = _errors(sc.perturbation, n, rng)
        yy = r * x + math.sqrt(1 - r**2) * e
        if sc.perturbation == "outliers":
            k = max(1, int(round(0.06 * n)))
            idx = rng.choice(n, size=k, replace=False)
            # under H0 the y-sign of each leverage point is independent of x: no induced association
            s_lev = np.full(k, sgn) if not h0 else rng.choice([-1.0, 1.0], size=k)
            x[idx] = rng.uniform(8, 12, size=k)
            yy[idx] = s_lev * rng.uniform(8, 12, size=k)
        df = pd.DataFrame({names["x"]: np.round(x * scale + base, 3), y: np.round(yy * scale + base, 3)})
        return Simulated(df, {"x": names["x"], "y": y}, None, h0, true_effect=r)

    if sc.family == "k_group":
        k, n = sc.k, _n(sc, rng)
        levels = [names["levels"][i] for i in range(k)]
        sds = [1.0] * k
        ns = [n] * k
        if sc.perturbation == "unequal_variance":
            sds[-1], ns[-1] = 3.0, max(6, n // 2)
        shift = z * math.sqrt(2.0 / n) * float(rng.choice([-1.0, 1.0]))
        parts, labs = [], []
        for i, lv in enumerate(levels):
            v = (shift if i == k - 1 else 0.0) + _errors(sc.perturbation, ns[i], rng) * sds[i]
            if sc.perturbation == "outliers" and (h0 or i == k - 1):
                v = inject_outliers(v, rng)
            parts.append(v)
            labs += [lv] * ns[i]
        vals = np.concatenate(parts) * scale + base
        df = pd.DataFrame({g: labs, y: np.round(vals, 3)})
        return Simulated(df, {"y": y, "group": g}, None, h0, true_effect=shift * scale)

    if sc.family == "binary_two_group":
        n = _n(sc, rng) * 3
        p_ref = float(rng.uniform(0.25, 0.5))
        if sc.perturbation == "small_counts":
            n, p_ref = int(rng.integers(8, 16)), float(rng.uniform(0.08, 0.2))
        pbar = p_ref
        se = math.sqrt(2 * pbar * (1 - pbar) / n)
        p_comp = float(np.clip(p_ref + sgn * z * se, 0.02, 0.98))
        a = (rng.random(n) < p_ref).astype(int)
        b = (rng.random(n) < p_comp).astype(int)
        df = pd.DataFrame({pid: _ids(2 * n, rng), g: [ref] * n + [comp] * n, y: np.r_[a, b]})
        return Simulated(df, {"y": y, "group": g}, [ref, comp], h0, true_effect=p_comp - p_ref)

    if sc.family == "one_proportion":
        n = _n(sc, rng) * 3
        p0 = float(rng.choice([0.2, 0.25, 0.3, 0.5]))
        p = float(np.clip(p0 + sgn * z * math.sqrt(p0 * (1 - p0) / n), 0.02, 0.98))
        df = pd.DataFrame({pid: _ids(n, rng), y: (rng.random(n) < p).astype(int)})
        return Simulated(df, {"y": y}, None, h0, p0=p0, true_effect=p - p0)

    raise ValueError(f"unknown family {sc.family}")
