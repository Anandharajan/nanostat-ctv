"""Reference solver: run every method in A(x,e) U B(x,e) on CPU and cache results (SPEC §6.4 step 3,
v3.1 §2.3). Sets are conditioned on the protocol's estimand."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from nanostat.diagnostics import compute_diagnostics
from nanostat.policy import PolicyCell, lookup
from nanostat.protocol import allowed_estimands, hypothesis_id
from nanostat.schemas import Alternative, Check, Design, MethodRun, OutcomeType, Protocol
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
    estimand: str | None
    hypothesis: str | None


def _run(method: str, df: pd.DataFrame, ctx: RunContext, reason: str | None = None) -> MethodRun:
    tool = REGISTRY[method]
    alt = Alternative.two_sided if tool.two_sided_only else ctx.alternative
    c2 = RunContext(ctx.columns, alt, ctx.group_order, ctx.p0, ctx.alpha, ctx.design)
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
    protocol: Protocol,
    group_order: list[str] | None = None,
) -> Reference | None:
    alt = protocol.alternative
    ctx = RunContext(columns, alt, group_order, protocol.p0, protocol.alpha, design)
    diag = compute_diagnostics(df, ctx, design)
    if "error" in diag:
        return None
    ycol = columns.get("y") or columns.get("y2")
    outcome = OutcomeType.binary if ycol and df[ycol].dropna().nunique() == 2 else OutcomeType.continuous
    pert = diag["perturbation"]
    cell = lookup(design, outcome, pert)
    if cell is None:
        return None
    allowed = allowed_estimands(design, protocol)
    a_methods = cell.admissible(alt, allowed)
    adm = [_run(m, df, ctx) for m in a_methods]
    inv = [_run(m, df, ctx, reason=r) for m, r in cell.invalid_for(alt, allowed).items()]
    preferred = a_methods[0] if a_methods else None
    pref_p = adm[0].p if adm else None
    decision = None if pref_p is None else ("reject" if pref_p < protocol.alpha else "fail_to_reject")
    checks = list(REGISTRY[preferred].required_checks) if preferred else []
    est = cell.estimand_of(preferred, allowed) if preferred else None
    return Reference(cell, pert, adm, inv, preferred, decision, checks, outcome, est, hypothesis_id(design, est))
