"""Design/assumption verifier (SPEC §4, v3.1 §2.1-§2.5).

Two gates:
  Gate 1 (static): the plan is checked against the LOCKED protocol (alpha, direction, estimand,
          null value, population), the tool registry and the data profile; preprocessing ops are
          validated against the protocol before anything runs.
  Gate 2 (analysed sample): preprocessing is executed, the analysed rows are re-profiled and
          re-diagnosed, and design, pairing, groups, estimand and method status are re-verified on
          exactly the rows the test will see.

In confirmatory mode a method that is invalid or of unknown status for the analysed data is
rejected (the controller re-plans, clarifies or abstains). Diagnostic mode may run it, but the
result never carries a confirmatory decision. The verifier never sees a reference p-value.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from nanostat.claims import mentions_pairing
from nanostat.diagnostics import compute_diagnostics, cpu_verdicts
from nanostat.dsl import PreprocessError, apply_preprocess
from nanostat.policy import PolicyCell, lookup
from nanostat.profiler import column_info, column_names, profile
from nanostat.protocol import allowed_estimands, build_protocol, hypothesis_id
from nanostat.schemas import (
    Action,
    Alternative,
    Check,
    Design,
    Estimand,
    Filter,
    MethodStatus,
    OutcomeType,
    Plan,
    Protocol,
    ReasonCode,
    SelectLevels,
    Study,
    TrimOp,
)
from nanostat.tools import REGISTRY
from nanostat.tools.base import RunContext

Mode = Literal["confirmatory", "diagnostic"]
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
    protocol: Protocol
    mode: Mode = "confirmatory"
    errors: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    expected_reason: ReasonCode | None = None
    claim: Alternative = Alternative.two_sided
    cell: PolicyCell | None = None
    outcome: OutcomeType | None = None
    allowed_estimands: set[Estimand] | None = None
    estimand: str | None = None
    hypothesis: str | None = None
    method_status: MethodStatus = MethodStatus.unknown
    status_reason: str | None = None
    required_checks: list[Check] = field(default_factory=list)
    cpu_verdicts: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    ctx: RunContext | None = None
    analysed: pd.DataFrame | None = None
    analysed_sha256: str | None = None
    preprocess_log: list[str] = field(default_factory=list)
    estimand_changed: bool = False


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


def _gate1(plan: Plan, study: Study, prof: dict[str, Any], rep: VerifyReport) -> None:
    """Static checks against protocol, registry and profile. Appends to rep.errors."""
    assert plan.design is not None and plan.method is not None  # enforced by schema
    err = rep.errors.append
    proto = rep.protocol
    names = column_names(prof)
    for role, col in plan.columns.items():
        if col not in names:
            err(f"unknown_column:{role}={col}")
    for op in plan.preprocess:
        pcol = getattr(op, "column", None)
        if pcol is not None and pcol not in names:
            err(f"unknown_column:preprocess={pcol}")
    tool = REGISTRY.get(plan.method)
    if tool is None:
        err(f"unsupported_method:{plan.method}")
    if rep.errors:
        return
    assert tool is not None

    # design and roles
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

    # protocol lock: the model may restate but never change these fields
    if plan.alpha is not None and abs(plan.alpha - proto.alpha) > 1e-12:
        err(f"protocol_violation:alpha {plan.alpha} != locked {proto.alpha}")
    if plan.alternative is not None and plan.alternative != proto.alternative:
        err(f"wrong_tail:claim is {proto.alternative.value}, plan uses {plan.alternative.value}")
    if tool.two_sided_only and proto.alternative != Alternative.two_sided:
        err(f"wrong_tail:{plan.method} is two-sided only")
    if "p0" in plan.params and (proto.p0 is None or abs(plan.params["p0"] - proto.p0) > 1e-12):
        err(f"protocol_violation:p0 {plan.params['p0']} != locked {proto.p0}")
    extra_params = set(plan.params) - {"p0"}
    if extra_params:
        err(f"protocol_violation:unknown params {sorted(extra_params)}")
    if plan.design == Design.one_sample and proto.p0 is None:
        err("missing_null_value")

    # estimand: plan, tool and protocol must agree
    rep.allowed_estimands = allowed_estimands(plan.design, proto)
    if plan.estimand is not None:
        if plan.estimand.value not in tool.estimands:
            err(f"method_estimand_mismatch:{plan.method} does not estimate {plan.estimand.value}")
        if rep.allowed_estimands is not None and plan.estimand not in rep.allowed_estimands:
            allowed = sorted(e.value for e in rep.allowed_estimands)
            err(f"estimand_mismatch:protocol allows {allowed}, plan uses {plan.estimand.value}")

    # preprocessing validated against the protocol before it runs
    outcome_cols = {plan.columns[r] for r in OUTCOME_ROLES if r in plan.columns}
    for op in plan.preprocess:
        if isinstance(op, Filter) and op.column in outcome_cols:
            err(f"forbidden:filter on outcome column {op.column!r}")
        elif isinstance(op, (Filter, SelectLevels)) and not proto.allow_population_filters:
            err(f"protocol_violation:population changed by {op.op}({op.column})")
        if isinstance(op, TrimOp) and op.column in outcome_cols:
            locked = rep.allowed_estimands
            if locked is not None and Estimand.trimmed_mean_difference not in locked:
                err(f"estimand_changed_by_preprocessing:{op.op} on {op.column!r}")


def _gate2(plan: Plan, raw: pd.DataFrame, study: Study, rep: VerifyReport) -> None:
    """Run preprocessing, then re-verify on the analysed rows."""
    assert plan.design is not None and plan.method is not None
    err = rep.errors.append
    tool = REGISTRY[plan.method]
    try:
        df, log, changed = apply_preprocess(raw, plan)
    except PreprocessError as e:
        err(f"preprocess_error:{e}")
        return
    rep.preprocess_log, rep.estimand_changed = log, changed
    aprof = profile(df)

    # pairing re-verified on the analysed rows
    pid_cols = pair_id_columns(df, aprof, study)
    if plan.design == Design.independent_two_group and pid_cols:
        err(f"design_mismatch:paired data (repeated ids in {pid_cols[0]!r}) analysed as independent")
    if plan.design == Design.paired and "pair_id" in plan.columns and plan.columns["pair_id"] not in pid_cols:
        err("design_mismatch:pair_id column does not repeat in the analysed rows; data are not paired")
    if rep.expected_reason == ReasonCode.missing_pair_id:
        err("missing_pair_id")

    otype = outcome_type_of(df, plan.columns)
    if study.outcome is not None and otype is not None and study.outcome != otype:
        rep.messages.append(f"study declares {study.outcome.value} outcome, data look {otype.value}")
    if otype is not None and otype not in tool.outcomes:
        err(f"outcome_type_mismatch:{plan.method} expects {[o.value for o in tool.outcomes]}, outcome is {otype.value}")
    rep.outcome = otype

    gcol = plan.columns.get("group")
    if gcol and gcol in df.columns:
        k = int(df[gcol].nunique())
        if plan.design in (Design.independent_two_group, Design.paired) and k != 2:
            err(f"group_levels:{gcol} has {k} levels in the analysed rows, design needs 2")
        if plan.design == Design.k_group and k < 3:
            err(f"group_levels:{gcol} has {k} levels in the analysed rows, k_group needs >= 3")
        if study.group_order:
            levels = set((column_info(aprof, gcol) or {}).get("levels", []))
            if levels and not set(study.group_order) <= levels:
                err(f"group_order:{study.group_order} not in levels of {gcol}")
    if rep.errors:
        return

    proto = rep.protocol
    rep.ctx = RunContext(
        columns=dict(plan.columns),
        alternative=proto.alternative,
        group_order=study.group_order,
        p0=proto.p0,
        alpha=proto.alpha,
        design=plan.design,
    )
    rep.diagnostics = compute_diagnostics(df, rep.ctx, plan.design)
    if "error" in rep.diagnostics:
        err(f"data_error:{rep.diagnostics['error']}")
        return
    rep.cpu_verdicts = cpu_verdicts(rep.diagnostics, plan.design)
    rep.required_checks = list(tool.required_checks)

    pert = rep.diagnostics.get("perturbation", "none")
    allowed = rep.allowed_estimands
    rep.cell = lookup(plan.design, otype or OutcomeType.continuous, pert)
    if rep.cell is not None:
        adm = rep.cell.admissible(proto.alternative, allowed)
        inv = rep.cell.invalid_for(proto.alternative, allowed)
        if plan.method in adm:
            rep.method_status = MethodStatus.admissible
            est = plan.estimand.value if plan.estimand else rep.cell.estimand_of(plan.method, allowed)
            if est is not None and est not in {e for e, ms in rep.cell.estimand_options.items() if plan.method in ms}:
                rep.method_status = MethodStatus.invalid
                rep.status_reason = "estimand_mismatch"
            rep.estimand = est
        elif plan.method in inv:
            rep.method_status = MethodStatus.invalid
            rep.status_reason = inv[plan.method]
        if not rep.cell.confirmatory_ok:
            rep.messages.append(f"policy cell {rep.cell.key} is reviewed_by=PENDING (pilot use only)")
    if rep.estimand is None:
        rep.estimand = plan.estimand.value if plan.estimand else (tool.estimands[0] if tool.estimands else None)
    rep.hypothesis = hypothesis_id(plan.design, rep.estimand)

    if rep.mode == "confirmatory":
        if rep.method_status == MethodStatus.invalid:
            err(f"invalid_method:{plan.method}:{rep.status_reason}")
        elif rep.method_status == MethodStatus.unknown:
            err(f"unknown_method_status:{plan.method} is not in the reviewed sets for this design/data")
    else:
        rep.messages.append(f"diagnostic mode: method status {rep.method_status.value}; no confirmatory decision")

    rep.analysed = df
    rep.analysed_sha256 = hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def verify(
    plan: Plan,
    df: pd.DataFrame,
    prof: dict[str, Any],
    study: Study,
    question: str,
    protocol: Protocol | None = None,
    mode: Mode = "confirmatory",
) -> VerifyReport:
    proto = protocol or build_protocol(question, study)
    rep = VerifyReport(ok=True, protocol=proto, mode=mode, claim=proto.alternative)
    rep.expected_reason = expected_clarification(question, study, df, prof)
    if plan.action != Action.execute:
        rep.messages.append(f"plan requests {plan.action.value}: {plan.reason_code}")
        return rep
    _gate1(plan, study, prof, rep)
    if not rep.errors:
        _gate2(plan, df, study, rep)
    rep.ok = not rep.errors
    return rep


def assumption_report(plan: Plan, rep: VerifyReport) -> list[dict[str, Any]]:
    """Per required check: what the plan claimed vs what the CPU observed."""
    claims = {c.check.value: c.verdict for c in plan.checks}
    rows = []
    for chk in rep.required_checks:
        cpu = rep.cpu_verdicts.get(chk.value)
        cl = claims.get(chk.value)
        rows.append({"check": chk.value, "claimed": cl, "cpu": cpu, "match": cpu is not None and cl == cpu})
    return rows


def assumption_score(plan: Plan, rep: VerifyReport) -> float:
    """Evidence-grounded r_assume in [0, 1] (SPEC v3.1 §2.5).

    Credit only for a required check that was requested AND whose typed verdict matches the CPU's
    reading of the executed diagnostic. Claims for checks the method does not need, or that the
    data cannot support, are penalised, so listing every check earns nothing."""
    required = {c.value for c in rep.required_checks if c.value in rep.cpu_verdicts}
    claims = {c.check.value: c.verdict for c in plan.checks}
    if not required:
        return 0.0 if claims else 1.0
    matched = sum(1 for c in required if claims.get(c) == rep.cpu_verdicts[c])
    irrelevant = sum(1 for c in claims if c not in required)
    score = matched / len(required) - 0.5 * irrelevant / max(1, len(claims))
    return float(min(1.0, max(0.0, score)))
