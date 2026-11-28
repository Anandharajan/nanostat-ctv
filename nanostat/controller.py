"""Controller: profile -> plan -> verify -> execute -> result (SPEC §4).

The LLM (or rule planner) never computes a statistic; every number comes from nanostat.tools.
This module must never import answer keys (see tests/unit/test_leakage.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nanostat.audit.events import AuditLog
from nanostat.dsl import PreprocessError
from nanostat.planner import Planner, RulePlanner, plan_errors
from nanostat.profiler import profile
from nanostat.schemas import (
    Action,
    AdmissibleRow,
    Budget,
    MethodStatus,
    Plan,
    ReasonCode,
    Result,
    Study,
)
from nanostat.tools import ToolError, get_tool
from nanostat.tools.base import RunContext
from nanostat.verifier import Mode, VerifyReport, assumption_report, verify


@dataclass
class RunRecord:
    question: str
    study: Study
    plan: Plan | None
    raw_outputs: list[str]
    report: VerifyReport | None
    result: Result
    audit: AuditLog
    profile: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    operational_failure: str | None = None


def _decision(p: float, alpha: float) -> str:
    return "reject" if p < alpha else "fail_to_reject"


def admissible_range(df: pd.DataFrame, rep: VerifyReport, plan: Plan) -> list[AdmissibleRow]:
    """Run every admissible method of the inferred cell on the user's data (multiverse pillar)."""
    rows: list[AdmissibleRow] = []
    if rep.cell is None or rep.ctx is None:
        return rows
    for m in rep.cell.admissible(rep.protocol.alternative, rep.allowed_estimands):
        try:
            out = get_tool(m).run(df, rep.ctx)
            rows.append(AdmissibleRow(method=m, statistic=out.statistic, p=out.p_value, effect=out.effect))
        except (ToolError, ValueError, FloatingPointError) as e:
            rows.append(AdmissibleRow(method=m, statistic=None, p=None, effect=None, error=str(e)))
    return rows


def execute_plan(df: pd.DataFrame, plan: Plan, rep: VerifyReport) -> Result:
    """Execute on exactly the analysed rows that gate 2 verified (never re-derived here)."""
    assert plan.method is not None and rep.ctx is not None and rep.analysed is not None
    del df  # the raw frame is intentionally unused: only verified analysed rows are tested
    df2 = rep.analysed
    tool = get_tool(plan.method)
    ctx: RunContext = rep.ctx
    out = tool.run(df2, ctx)
    if not np.isfinite(out.p_value):
        raise ToolError("test returned a non-finite p-value (degenerate data)")
    return Result(
        action=Action.execute,
        method=plan.method,
        design=plan.design,
        alternative=rep.protocol.alternative,
        statistic=out.statistic,
        p_value=out.p_value,
        effect=out.effect,
        effect_label=out.effect_label,
        ci=out.ci,
        n=out.n,
        alpha=rep.protocol.alpha,
        decision=_decision(out.p_value, rep.protocol.alpha) if rep.mode == "confirmatory" else None,  # type: ignore[arg-type]
        method_status=rep.method_status,
        status_reason=rep.status_reason,
        policy_cell=rep.cell.key if rep.cell else None,
        admissible_range=admissible_range(df2, rep, plan),
        diagnostics={**rep.diagnostics, "required_checks": [c.value for c in rep.required_checks], "extra": out.extra},
        preprocess_log=rep.preprocess_log,
        estimand_changed=rep.estimand_changed,
        messages=rep.messages,
        estimand=rep.estimand,
        hypothesis=rep.hypothesis,
        confirmatory=rep.mode == "confirmatory",
        protocol=rep.protocol.model_dump(mode="json"),
        analysed_sha256=rep.analysed_sha256,
        assumption_report=assumption_report(plan, rep),
    )


def run_analysis(
    question: str,
    study: Study,
    df: pd.DataFrame,
    planner: Planner | None = None,
    budget: Budget | None = None,
    deterministic_audit: bool = False,
    mode: Mode = "confirmatory",
) -> RunRecord:
    planner = planner or RulePlanner()
    budget = budget or Budget()
    audit = AuditLog(deterministic=deterministic_audit)
    prof = profile(df)
    audit.add("input", {"question": question, "study": study.model_dump(mode="json"), "n_rows": len(df)})
    audit.add("profile", prof)

    raws: list[str] = []
    feedback: list[str] | None = None
    plan: Plan | None = None
    rep: VerifyReport | None = None
    failure: str | None = None
    attempts = 0
    for attempts in range(1, budget.max_steps + 1):  # noqa: B007
        plan, raw = planner.plan(question, study, df, prof, feedback)
        raws.append(raw)
        if plan is None:
            failure = "bad_plan_json"
            audit.add("plan_rejected", {"attempt": attempts, "error": failure})
            feedback = plan_errors(raw) or ["output was not a valid typed plan JSON"]
            continue
        audit.add("plan", {"attempt": attempts, "planner": planner.name, "plan": plan.model_dump(mode="json")})
        rep = verify(plan, df, prof, study, question, mode=mode)
        audit.add(
            "verify",
            {
                "ok": rep.ok,
                "errors": rep.errors,
                "messages": rep.messages,
                "protocol": rep.protocol.model_dump(mode="json"),
                "analysed_sha256": rep.analysed_sha256,
            },
        )
        if rep.ok:
            failure = None
            break
        failure = "verifier_rejected"
        feedback = rep.errors

    if plan is None or rep is None or not rep.ok:
        code = ReasonCode.invalid_design
        if rep is not None and any(e.startswith(("invalid_method", "unknown_method_status")) for e in rep.errors):
            code = ReasonCode.no_admissible_method
        if rep is not None and any(e.startswith("unknown_column") for e in rep.errors):
            code = ReasonCode.unknown_column
        elif rep is not None and any(e.startswith("unsupported_method") for e in rep.errors):
            code = ReasonCode.unsupported_method
        result = Result(
            action=Action.abstain,
            reason_code=code,
            messages=(rep.errors if rep else []) + [f"no valid plan within budget ({failure})"],
        )
        audit.add("result", result.model_dump(mode="json"))
        return RunRecord(question, study, plan, raws, rep, result, audit, prof, attempts, failure)

    if plan.action != Action.execute:
        result = Result(action=plan.action, reason_code=plan.reason_code, messages=rep.messages)
        audit.add("result", result.model_dump(mode="json"))
        return RunRecord(question, study, plan, raws, rep, result, audit, prof, attempts, None)

    try:
        result = execute_plan(df, plan, rep)
    except (ToolError, PreprocessError, ValueError, FloatingPointError) as e:
        failure = "execution_error"
        result = Result(
            action=Action.abstain,
            reason_code=ReasonCode.insufficient_data,
            method=plan.method,
            method_status=MethodStatus.unknown,
            messages=[f"execution failed: {e}"],
        )
    audit.add("result", result.model_dump(mode="json"))
    return RunRecord(question, study, plan, raws, rep, result, audit, prof, attempts, failure)
