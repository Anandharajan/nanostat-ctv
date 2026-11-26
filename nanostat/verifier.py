"""Design/assumption verifier (SPEC §4).

The verifier checks a typed plan against the data profile, the study metadata and the
question. It never sees a target p-value and never changes the method; it only accepts,
rejects (with error codes) or reports which checks the chosen method requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nanostat.claims import infer_alternative, mentions_pairing
from nanostat.diagnostics import compute_diagnostics
from nanostat.policy import PolicyCell, lookup
from nanostat.profiler import column_info, column_names
from nanostat.schemas import (
    Action,
    Alternative,
    Check,
    Design,
    Filter,
    MethodStatus,
    OutcomeType,
    Plan,
    ReasonCode,
    Study,
)
from nanostat.tools import REGISTRY
from nanostat.tools.base import RunContext

OUTCOME_ROLES = ("y", "y1", "y2", "x")

REQUIRED_ROLES: dict[Design, list[set[str]]] = {
    Design.independent_two_group: [{"y", "group"}],
    Design.paired: [{"y1", "y2"}, {"y", "group", "pair_id"}],
    Design.correlation: [{"x", "y"}],
    Design.k_group: [{"y", "group"}],
    Design.one_sample: [{"y"}],
}


@dataclass
class VerifyReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    expected_reason: ReasonCode | None = None
    claim: Alternative = Alternative.two_sided
    cell: PolicyCell | None = None
    outcome: OutcomeType | None = None
    method_status: MethodStatus = MethodStatus.unknown
    status_reason: str | None = None
    required_checks: list[Check] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    ctx: RunContext | None = None


def pair_id_columns(df: pd.DataFrame, prof: dict[str, Any], study: Study) -> list[str]:
    """Columns whose values repeat (each id appears >= 2 times): candidate pairing IDs."""
    out = []
    declared = study.roles.get("pair_id")
    role_cols = {v for k, v in study.roles.items() if k != "pair_id"}
    for c in prof["columns"]:
        if c["name"] in role_cols:
            continue
        idl = c.get("id_like")
        if idl and idl["repeats"] >= 2:
            out.append(c["name"])
    if declared and declared in df.columns and declared not in out:
        vc = df[declared].dropna().astype(str).value_counts()
        if len(vc) and vc.min() >= 2:
            out.append(declared)
    return out


def expected_clarification(question: str, study: Study, df: pd.DataFrame, prof: dict[str, Any]) -> ReasonCode | None:
    """Facts the data/metadata cannot supply, so the right action is to ask (SPEC §7.3)."""
    roles = study.roles
    pairing_cue = study.design == Design.paired or mentions_pairing(question + " " + study.description)
    wide = "y1" in roles and "y2" in roles
    if pairing_cue and not wide and not pair_id_columns(df, prof, study):
        return ReasonCode.missing_pair_id
    if study.design == Design.one_sample and study.p0 is None:
        return ReasonCode.missing_null_value
    if not roles:
        return ReasonCode.design_unspecified
    return None


def outcome_type_of(df: pd.DataFrame, plan_cols: dict[str, str]) -> OutcomeType | None:
    for role in ("y", "y2", "x"):
        c = plan_cols.get(role)
        if c and c in df.columns:
            return OutcomeType.binary if df[c].dropna().nunique() == 2 else OutcomeType.continuous
    return None


def verify(plan: Plan, df: pd.DataFrame, prof: dict[str, Any], study: Study, question: str) -> VerifyReport:
    rep = VerifyReport(ok=True, claim=infer_alternative(question))
    rep.expected_reason = expected_clarification(question, study, df, prof)

    if plan.action != Action.execute:
        rep.messages.append(f"plan requests {plan.action.value}: {plan.reason_code}")
        return rep

    assert plan.design is not None and plan.method is not None  # enforced by schema
    err = rep.errors.append

    # --- columns -------------------------------------------------------------------
    names = column_names(prof)
    for role, col in plan.columns.items():
        if col not in names:
            err(f"unknown_column:{role}={col}")
    for op in plan.preprocess:
        pcol = getattr(op, "column", None)
        if pcol is not None and pcol not in names:
            err(f"unknown_column:preprocess={pcol}")
    if rep.errors:
        rep.ok = False
        return rep

    # --- method & design --------------------------------------------------------------
    tool = REGISTRY.get(plan.method)
    if tool is None:
        err(f"unsupported_method:{plan.method}")
        rep.ok = False
        return rep
    if plan.design not in tool.designs:
        err(f"method_design_mismatch:{plan.method} does not support {plan.design.value}")
    if study.design is not None and plan.design != study.design:
        err(f"design_mismatch:study declares {study.design.value}, plan uses {plan.design.value}")
    roles = set(plan.columns)
    if not any(req <= roles for req in REQUIRED_ROLES[plan.design]):
        if plan.design == Design.paired and {"y", "group"} <= roles:
            err("missing_pair_id")
        else:
            need = " or ".join(str(sorted(r)) for r in REQUIRED_ROLES[plan.design])
            err(f"missing_roles:{plan.design.value} needs {need}")

    pid_cols = pair_id_columns(df, prof, study)
    if plan.design == Design.independent_two_group and pid_cols:
        err(f"design_mismatch:paired data (repeated ids in {pid_cols[0]!r}) analysed as independent")
    if plan.design == Design.paired and "pair_id" in plan.columns:
        if plan.columns["pair_id"] not in pid_cols:
            err("design_mismatch:pair_id column does not repeat; data are not paired")
    if rep.expected_reason == ReasonCode.missing_pair_id:
        err("missing_pair_id")

    # --- outcome type -------------------------------------------------------------------
    otype = outcome_type_of(df, plan.columns)
    if study.outcome is not None and otype is not None and study.outcome != otype:
        rep.messages.append(f"study declares {study.outcome.value} outcome, data look {otype.value}")
    if otype is not None and otype not in tool.outcomes:
        err(f"outcome_type_mismatch:{plan.method} expects {[o.value for o in tool.outcomes]}, outcome is {otype.value}")
    rep.outcome = otype

    # --- groups -------------------------------------------------------------------------
    gcol = plan.columns.get("group")
    if gcol and plan.design in (Design.independent_two_group, Design.paired):
        info = column_info(prof, gcol) or {}
        if info.get("n_unique", 0) != 2:
            err(f"group_levels:{gcol} has {info.get('n_unique')} levels, design needs 2")
    if gcol and plan.design == Design.k_group:
        info = column_info(prof, gcol) or {}
        if info.get("n_unique", 0) < 3:
            err(f"group_levels:{gcol} has {info.get('n_unique')} levels, k_group needs >= 3")
    if study.group_order and gcol:
        levels = set((column_info(prof, gcol) or {}).get("levels", []))
        if levels and not set(study.group_order) <= levels:
            err(f"group_order:{study.group_order} not in levels of {gcol}")

    # --- tail ---------------------------------------------------------------------------
    if plan.alternative != rep.claim:
        err(f"wrong_tail:claim is {rep.claim.value}, plan uses {plan.alternative.value}")
    if tool.two_sided_only and plan.alternative != Alternative.two_sided:
        err(f"wrong_tail:{plan.method} is two-sided only")

    # --- one-sample null value --------------------------------------------------------------
    p0 = plan.params.get("p0", study.p0)
    if plan.design == Design.one_sample and p0 is None:
        err("missing_null_value")

    # --- forbidden preprocessing ------------------------------------------------------------
    outcome_cols = {plan.columns[r] for r in OUTCOME_ROLES if r in plan.columns}
    for op in plan.preprocess:
        if isinstance(op, Filter) and op.column in outcome_cols:
            err(f"forbidden:filter on outcome column {op.column!r}")

    if rep.errors:
        rep.ok = False
        return rep

    # --- diagnostics, policy cell, method status ---------------------------------------------
    rep.ctx = RunContext(
        columns=dict(plan.columns),
        alternative=plan.alternative,
        group_order=study.group_order,
        p0=p0,
        alpha=plan.alpha,
    )
    rep.diagnostics = compute_diagnostics(df, rep.ctx, plan.design)
    if "error" in rep.diagnostics:
        err(f"data_error:{rep.diagnostics['error']}")
        rep.ok = False
        return rep
    pert = rep.diagnostics.get("perturbation", "none")
    rep.cell = lookup(plan.design, otype or OutcomeType.continuous, pert)
    rep.required_checks = list(tool.required_checks)
    if rep.cell is not None:
        if plan.method in rep.cell.admissible(plan.alternative):
            rep.method_status = MethodStatus.admissible
        elif plan.method in rep.cell.invalid:
            rep.method_status = MethodStatus.invalid
            rep.status_reason = rep.cell.invalid[plan.method]
        if not rep.cell.confirmatory_ok:
            rep.messages.append(f"policy cell {rep.cell.key} is reviewed_by=PENDING (pilot use only)")
    return rep
