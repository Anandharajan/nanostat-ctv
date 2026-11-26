"""Assumption diagnostics and runtime perturbation inference (CPU, deterministic).

These are computed from the user's data only; they never read answer keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nanostat.schemas import Design
from nanostat.tools.base import (
    RunContext,
    ToolError,
    k_groups,
    paired_arrays,
    two_by_two,
    two_groups,
    xy,
)

# Thresholds (versioned with the policy registry; see configs/policies)
OUTLIER_ROBUST_Z = 5.0
SKEW_TRIMMED = 1.0
VAR_RATIO = 4.0
MIN_EXPECTED = 5.0

PERTURBATIONS = ("small_expected_counts", "heavy_outliers", "skewed", "unequal_variance", "none")


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        mad = np.mean(np.abs(x - med)) * 1.2533 or 1.0
        return (x - med) / mad
    return 0.6745 * (x - med) / mad


def _trimmed_skew(x: np.ndarray) -> float:
    if len(x) < 8:
        return 0.0
    t = np.sort(x)[int(0.05 * len(x)) : len(x) - int(0.05 * len(x))]
    if len(t) < 3 or np.std(t) == 0:
        return 0.0
    return float(stats.skew(t))


def summarize(x: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(x))}
    if len(x) == 0:
        return out
    out.update(
        mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        median=float(np.median(x)),
        skew=float(stats.skew(x)) if len(x) > 2 and np.std(x) > 0 else 0.0,
        trimmed_skew=_trimmed_skew(x),
        max_abs_robust_z=float(np.max(np.abs(robust_z(x)))) if len(x) > 2 else 0.0,
        n_outliers=int((np.abs(robust_z(x)) > OUTLIER_ROBUST_Z).sum()) if len(x) > 2 else 0,
    )
    if 3 <= len(x) <= 5000 and np.ptp(x) > 0:
        out["shapiro_p"] = float(stats.shapiro(x).pvalue)
    return out


def compute_diagnostics(df: pd.DataFrame, ctx: RunContext, design: Design) -> dict[str, Any]:
    """Return a dict with per-sample summaries, flags and the inferred perturbation."""
    d: dict[str, Any] = {"design": design.value, "samples": {}}
    flags = {p: False for p in PERTURBATIONS if p != "none"}
    samples: dict[str, np.ndarray] = {}
    try:
        if design == Design.independent_two_group and _is_binary(df, ctx):
            t = two_by_two(df, ctx)
            exp = stats.contingency.expected_freq(t)
            d["table"] = t.tolist()
            d["min_expected"] = float(exp.min())
            flags["small_expected_counts"] = bool(exp.min() < MIN_EXPECTED)
        elif design == Design.independent_two_group:
            a, b = two_groups(df, ctx)
            samples = {"comparison": a, "reference": b}
        elif design == Design.k_group:
            gs = k_groups(df, ctx)
            samples = {f"group_{i}": g for i, g in enumerate(gs)}
        elif design == Design.paired:
            y2, y1 = paired_arrays(df, ctx)
            samples = {"difference": y2 - y1}
            d["n_pairs"] = int(len(y2))
        elif design == Design.correlation:
            x, y = xy(df, ctx)
            samples = {"x": x, "y": y}
        elif design == Design.one_sample:
            d["n"] = int(df[ctx.columns["y"]].notna().sum()) if "y" in ctx.columns else 0
    except (ToolError, KeyError) as e:
        d["error"] = str(e)
        d["perturbation"] = "none"
        d["flags"] = flags
        return d

    for k, x in samples.items():
        d["samples"][k] = summarize(x)
    if samples:
        summ = list(d["samples"].values())
        flags["heavy_outliers"] = any(s.get("n_outliers", 0) > 0 for s in summ)
        flags["skewed"] = any(abs(s.get("trimmed_skew", 0.0)) > SKEW_TRIMMED for s in summ)
        if design in (Design.independent_two_group, Design.k_group) and len(samples) >= 2:
            sds = [s["sd"] for s in summ if s.get("sd", 0) > 0]
            if len(sds) >= 2:
                d["var_ratio"] = float((max(sds) / min(sds)) ** 2)
                flags["unequal_variance"] = d["var_ratio"] > VAR_RATIO
            arrs = [x for x in samples.values() if np.ptp(x) > 0]
            if len(arrs) >= 2:
                d["levene_p"] = float(stats.levene(*arrs, center="median").pvalue)
    d["flags"] = flags
    d["perturbation"] = next((p for p in PERTURBATIONS if p != "none" and flags[p]), "none")
    return d


def _is_binary(df: pd.DataFrame, ctx: RunContext) -> bool:
    c = ctx.columns.get("y")
    if c is None or c not in df.columns:
        return False
    return int(df[c].dropna().nunique()) == 2
