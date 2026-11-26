"""Reference solver: run every method in A(x) U B(x) on CPU and cache results (SPEC §6.4 step 3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from nanostat.diagnostics import compute_diagnostics
from nanostat.policy import PolicyCell, lookup
from nanostat.schemas import Alternative, Check, Design, MethodRun, OutcomeType
from nanostat.tools import REGISTRY, ToolError, get_tool
from nanostat.tools.base import RunContext


@dataclass
class Reference:
    cell: PolicyCell
    perturbation: str
    admissible: list[MethodRun]
    invalid: list[MethodRun]
    preferred: str | None
    reference_decision: str | None
    required_checks: list[Check]
    outcome: OutcomeType


def _run(method: str, df: pd.DataFrame, ctx: RunContext, reason: str | None = None) -> MethodRun:
    tool = REGISTRY[method]
    alt = Alternative.two_sided if tool.two_sided_only else ctx.alternative
    c2 = RunContext(ctx.columns, alt, ctx.group_order, ctx.p0, ctx.alpha)
    cfg = {"alternative": alt.value}
    try:
        out = get_tool(method).run(df, c2)
        p = out.p_value if math.isfinite(out.p_value) else None
        return MethodRun(method=method, config=cfg, p=p, statistic=out.statistic, effect=out.effect, reason=reason)
    except (ToolError, ValueError, FloatingPointError):
        return MethodRun(method=method, config=cfg, p=None, reason=reason)


def build_reference(
    df: pd.DataFrame,
    design: Design,
    columns: dict[str, str],
    alternative: Alternative,
    group_order: list[str] | None = None,
    p0: float | None = None,
    alpha: float = 0.05,
) -> Reference | None:
    ctx = RunContext(columns, alternative, group_order, p0, alpha)
    diag = compute_diagnostics(df, ctx, design)
    if "error" in diag:
        return None
    ycol = columns.get("y") or columns.get("y2")
    outcome = OutcomeType.binary if ycol and df[ycol].dropna().nunique() == 2 else OutcomeType.continuous
    pert = diag["perturbation"]
    cell = lookup(design, outcome, pert)
    if cell is None:
        return None
    a_methods = cell.admissible(alternative)
    adm = [_run(m, df, ctx) for m in a_methods]
    inv = [_run(m, df, ctx, reason=r) for m, r in cell.invalid.items()]
    preferred = a_methods[0] if a_methods else None
    pref_p = adm[0].p if adm else None
    decision = None if pref_p is None else ("reject" if pref_p < alpha else "fail_to_reject")
    checks = list(REGISTRY[preferred].required_checks) if preferred else []
    return Reference(cell, pert, adm, inv, preferred, decision, checks, outcome)
