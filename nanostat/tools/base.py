"""Tool plugin base class and data extraction helpers (SPEC §6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from nanostat.schemas import Alternative, Check, Design, OutcomeType


class ToolError(ValueError):
    """Raised when a tool cannot run on the given data/plan."""


@dataclass
class RunContext:
    """Non-answer-bearing context the executor passes to tools."""

    columns: dict[str, str]
    alternative: Alternative = Alternative.two_sided
    group_order: list[str] | None = None
    p0: float | None = None
    alpha: float = 0.05


@dataclass
class ToolOutput:
    statistic: float
    p_value: float
    effect: float | None
    effect_label: str
    n: int
    ci: tuple[float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Tool:
    name: ClassVar[str]
    estimands: ClassVar[tuple[str, ...]]
    designs: ClassVar[tuple[Design, ...]]
    outcomes: ClassVar[tuple[OutcomeType, ...]]
    two_sided_only: ClassVar[bool] = False
    required_checks: ClassVar[tuple[Check, ...]] = ()
    r_reference: ClassVar[str | None] = None
    tolerance: ClassVar[float] = 1e-9

    def run(self, df: pd.DataFrame, ctx: RunContext) -> ToolOutput:  # pragma: no cover
        raise NotImplementedError

    def diagnostics(self, df: pd.DataFrame, ctx: RunContext) -> dict[str, Any]:
        from nanostat.diagnostics import compute_diagnostics

        return compute_diagnostics(df, ctx, self.designs[0])


# ---- extraction helpers ----------------------------------------------------------------


def column(df: pd.DataFrame, ctx: RunContext, role: str) -> str:
    c = ctx.columns.get(role)
    if c is None:
        raise ToolError(f"missing column role {role!r}")
    if c not in df.columns:
        raise ToolError(f"unknown column {c!r}")
    return c


def numeric(s: pd.Series) -> np.ndarray:
    try:
        arr = pd.to_numeric(s, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        raise ToolError(f"column {s.name!r} is not numeric") from e
    return arr[np.isfinite(arr)]


def group_levels(df: pd.DataFrame, ctx: RunContext, k: int | None = None) -> list[str]:
    g = column(df, ctx, "group")
    present = sorted(df[g].dropna().astype(str).unique().tolist())
    if ctx.group_order:
        order = [str(v) for v in ctx.group_order]
        missing = [v for v in order if v not in present]
        if missing:
            raise ToolError(f"group levels {missing} not in data")
        levels = order + [v for v in present if v not in order]
    else:
        levels = present
    if k is not None and len(levels) != k:
        raise ToolError(f"expected {k} groups, found {len(levels)}")
    return levels


def two_groups(df: pd.DataFrame, ctx: RunContext) -> tuple[np.ndarray, np.ndarray]:
    """Return (comparison, reference). Effect sign = comparison - reference."""
    y, g = column(df, ctx, "y"), column(df, ctx, "group")
    ref, comp = group_levels(df, ctx, 2)
    gs = df[g].astype(str)
    a = numeric(df.loc[gs == comp, y])
    b = numeric(df.loc[gs == ref, y])
    if len(a) < 2 or len(b) < 2:
        raise ToolError("each group needs at least 2 observations")
    return a, b


def k_groups(df: pd.DataFrame, ctx: RunContext) -> list[np.ndarray]:
    y, g = column(df, ctx, "y"), column(df, ctx, "group")
    levels = group_levels(df, ctx)
    if len(levels) < 2:
        raise ToolError("need at least 2 groups")
    gs = df[g].astype(str)
    out = [numeric(df.loc[gs == lv, y]) for lv in levels]
    if any(len(x) < 2 for x in out):
        raise ToolError("each group needs at least 2 observations")
    return out


def paired_arrays(df: pd.DataFrame, ctx: RunContext) -> tuple[np.ndarray, np.ndarray]:
    """Return (y2, y1) aligned pairs, complete cases only. Effect = y2 - y1."""
    if "y1" in ctx.columns and "y2" in ctx.columns:
        c1, c2 = column(df, ctx, "y1"), column(df, ctx, "y2")
        sub = df[[c1, c2]].apply(pd.to_numeric, errors="coerce").dropna()
        return sub[c2].to_numpy(float), sub[c1].to_numpy(float)
    y, g, pid = column(df, ctx, "y"), column(df, ctx, "group"), column(df, ctx, "pair_id")
    ref, comp = group_levels(df, ctx, 2)
    tmp = df[[pid, g, y]].copy()
    tmp[g] = tmp[g].astype(str)
    if tmp.duplicated(subset=[pid, g]).any():
        raise ToolError("pair_id has duplicate rows within a group")
    wide = tmp.pivot(index=pid, columns=g, values=y).apply(pd.to_numeric, errors="coerce")
    wide = wide[[ref, comp]].dropna()
    if len(wide) < 3:
        raise ToolError("need at least 3 complete pairs")
    return wide[comp].to_numpy(float), wide[ref].to_numpy(float)


def xy(df: pd.DataFrame, ctx: RunContext) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = column(df, ctx, "x"), column(df, ctx, "y")
    sub = df[[cx, cy]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 4:
        raise ToolError("need at least 4 complete (x, y) pairs")
    return sub[cx].to_numpy(float), sub[cy].to_numpy(float)


def binary01(s: pd.Series) -> np.ndarray:
    """Map a binary column to 0/1. Numeric: values must be {0,1}. Else second sorted level = 1."""
    s = s.dropna()
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        vals = set(np.unique(num.to_numpy(float)).tolist())
        if not vals <= {0.0, 1.0}:
            raise ToolError(f"column {s.name!r} is not binary 0/1")
        return num.to_numpy(float).astype(int)
    levels = sorted(s.astype(str).unique().tolist())
    if len(levels) != 2:
        raise ToolError(f"column {s.name!r} has {len(levels)} levels, expected 2")
    return (s.astype(str) == levels[1]).to_numpy().astype(int)


def two_by_two(df: pd.DataFrame, ctx: RunContext) -> np.ndarray:
    """Rows = [comparison, reference]; cols = [success, failure]."""
    y, g = column(df, ctx, "y"), column(df, ctx, "group")
    ref, comp = group_levels(df, ctx, 2)
    sub = df[[y, g]].dropna()
    yy = binary01(sub[y])
    gs = sub[g].astype(str).to_numpy()
    t = np.array(
        [
            [int(((gs == comp) & (yy == 1)).sum()), int(((gs == comp) & (yy == 0)).sum())],
            [int(((gs == ref) & (yy == 1)).sum()), int(((gs == ref) & (yy == 0)).sum())],
        ]
    )
    if (t.sum(axis=1) == 0).any():
        raise ToolError("empty group in 2x2 table")
    if (t.sum(axis=0) == 0).any():
        raise ToolError("outcome has no variation (all observations in one category)")
    return t


def hodges_lehmann_two(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) * len(b) > 400_000:
        rng = np.random.default_rng(0)
        a = rng.choice(a, 600, replace=False) if len(a) > 600 else a
        b = rng.choice(b, 600, replace=False) if len(b) > 600 else b
    return float(np.median(np.subtract.outer(a, b)))


def pseudo_median(d: np.ndarray) -> float:
    if len(d) > 900:
        d = np.random.default_rng(0).choice(d, 900, replace=False)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def ci_tuple(ci: Any) -> tuple[float, float]:
    return (float(ci.low), float(ci.high))
