"""Planners: the rule-based B0 planner and an LLM planner wrapper.

Planners see only the question, study metadata and the data profile/diagnostics.
They never import from `keys` or `benchmark.keys` (enforced by tests/unit/test_leakage.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd
from pydantic import ValidationError

from nanostat.diagnostics import compute_diagnostics, cpu_verdicts
from nanostat.policy import lookup
from nanostat.profiler import column_info
from nanostat.prompting import render_prompt
from nanostat.protocol import allowed_estimands, build_protocol
from nanostat.schemas import Action, CheckClaim, Design, Estimand, OutcomeType, Plan, ReasonCode, Study
from nanostat.tools import REGISTRY
from nanostat.tools.base import RunContext
from nanostat.verifier import expected_clarification, pair_id_columns


class Planner(Protocol):
    name: str

    def plan(
        self,
        question: str,
        study: Study,
        df: pd.DataFrame,
        prof: dict[str, Any],
        feedback: list[str] | None = None,
    ) -> tuple[Plan | None, str]:
        """Return (plan or None if unparseable, raw text)."""
        ...


_ESTIMAND_FOR = {
    "welch_t": Estimand.mean_difference,
    "student_t": Estimand.mean_difference,
    "yuen_trimmed_t": Estimand.trimmed_mean_difference,
    "paired_t": Estimand.mean_difference_paired,
    "mann_whitney": Estimand.location_shift,
    "wilcoxon_signed_rank": Estimand.location_shift,
    "pearson": Estimand.correlation,
    "spearman": Estimand.rank_correlation,
    "anova_oneway": Estimand.group_means_equal,
    "kruskal_wallis": Estimand.group_distributions_equal,
    "chi2_independence": Estimand.association,
    "fisher_exact": Estimand.association,
    "binomial_test": Estimand.proportion,
    "permutation_test": Estimand.mean_difference,
    "ols_coef": Estimand.coefficient,
    "ols_wald": Estimand.linear_restriction,
    "logistic": Estimand.coefficient,
    "probit": Estimand.coefficient,
    "iv_2sls": Estimand.coefficient,
    "mixed_lm": Estimand.coefficient,
    "tobit": Estimand.coefficient,
    "cox_ph": Estimand.hazard_ratio,
    "log_rank": Estimand.survival_curves_equal,
}


def estimand_for(method: str) -> Estimand:
    return _ESTIMAND_FOR[method]


def infer_design(study: Study, df: pd.DataFrame, prof: dict[str, Any]) -> Design | None:
    if study.design is not None:
        return study.design
    r = study.roles
    if "x" in r:
        return Design.correlation
    if "y1" in r and "y2" in r:
        return Design.paired
    if "group" in r:
        if pair_id_columns(df, prof, study):
            return Design.paired
        info = column_info(prof, r["group"]) or {}
        k = int(info.get("n_unique", 0))
        if k == 2:
            return Design.independent_two_group
        if k >= 3:
            return Design.k_group
        return None
    if "y" in r:
        return Design.one_sample
    return None


def roles_for(design: Design, study: Study, df: pd.DataFrame, prof: dict[str, Any]) -> dict[str, str]:
    r = study.roles
    if design == Design.correlation:
        return {k: r[k] for k in ("x", "y") if k in r}
    if design == Design.paired:
        if "y1" in r and "y2" in r:
            return {k: r[k] for k in ("y1", "y2", "pair_id") if k in r}
        out: dict[str, str] = {k: r[k] for k in ("y", "group") if k in r}
        pid = r.get("pair_id") or next(iter(pair_id_columns(df, prof, study)), None)
        if pid:
            out["pair_id"] = pid
        return out
    if design == Design.one_sample:
        return {"y": r["y"]} if "y" in r else {}
    return {k: r[k] for k in ("y", "group") if k in r}


RATIONALE = {
    "none": "assumptions look reasonable for the {est} estimand",
    "heavy_outliers": "extreme outliers detected, so a rank/robust method is used",
    "skewed": "strong skew detected, so a rank-based method is used",
    "unequal_variance": "group variances differ, so pooled-variance methods are avoided",
    "small_expected_counts": "expected cell counts are below 5, so an exact test is used",
}


class RulePlanner:
    """Deterministic B0 baseline: registry-preferred admissible method for the inferred cell."""

    name = "rules"

    def _model_plan(self, question: str, study: Study, df: pd.DataFrame) -> Plan:
        return _model_plan_impl(question, study, df)

    def plan(
        self,
        question: str,
        study: Study,
        df: pd.DataFrame,
        prof: dict[str, Any],
        feedback: list[str] | None = None,
    ) -> tuple[Plan | None, str]:
        p = self._plan(question, study, df, prof)
        return p, p.model_dump_json(exclude_none=True)

    def _plan(self, question: str, study: Study, df: pd.DataFrame, prof: dict[str, Any]) -> Plan:
        reason = expected_clarification(question, study, df, prof)
        if reason is not None:
            return Plan(action=Action.clarify, reason_code=reason, rationale=f"cannot proceed: {reason.value}")
        design = infer_design(study, df, prof)
        if study.model is not None and design in (None, Design.regression, Design.survival):
            return self._model_plan(question, study, df)
        if design is None:
            return Plan(action=Action.abstain, reason_code=ReasonCode.unsupported_design)
        cols = roles_for(design, study, df, prof)
        proto = build_protocol(question, study)
        alt = proto.alternative
        y = cols.get("y") or cols.get("y2")
        if y and y in df.columns and df[y].dropna().nunique() < 2:
            return Plan(
                action=Action.abstain,
                reason_code=ReasonCode.insufficient_data,
                rationale=f"outcome column {y!r} has no variation",
            )
        outcome = (
            OutcomeType.binary if y and y in df.columns and df[y].dropna().nunique() == 2 else OutcomeType.continuous
        )
        ctx = RunContext(columns=cols, alternative=alt, group_order=study.group_order, p0=proto.p0, design=design)
        diag = compute_diagnostics(df, ctx, design)
        if "error" in diag:
            return Plan(action=Action.abstain, reason_code=ReasonCode.insufficient_data, rationale=diag["error"][:200])
        pert = diag["perturbation"]
        cell = lookup(design, outcome, pert)
        allowed = allowed_estimands(design, proto)
        method = cell.preferred(alt, allowed) if cell else None
        if method is None or cell is None:
            return Plan(
                action=Action.abstain,
                reason_code=ReasonCode.no_admissible_method,
                rationale=f"no reviewed admissible method for {design.value}/{pert} under this protocol",
            )
        est_name = cell.estimand_of(method, allowed)
        est = Estimand(est_name) if est_name else estimand_for(method)
        verdicts = cpu_verdicts(diag, design)
        checks = [
            CheckClaim(check=c, verdict=verdicts[c.value])  # type: ignore[arg-type]
            for c in REGISTRY[method].required_checks
            if c.value in verdicts
        ]
        return Plan(
            action=Action.execute,
            estimand=est,
            design=design,
            method=method,
            columns=cols,
            alternative=alt,
            checks=checks,
            rationale=f"{design.value} design; " + RATIONALE.get(pert, "").format(est=est.value),
        )


def _model_plan_impl(question: str, study: Study, df: pd.DataFrame) -> Plan:
    """Rule plan for a user-declared model: survival if time/event given, else regression;
    method = registry-preferred for the outcome type and protocol."""
    m = study.model
    assert m is not None
    design = Design.survival if (m.time and m.event) else Design.regression
    if design == Design.survival:
        method = "cox_ph" if m.exposure else "log_rank"
    else:
        y = m.outcome
        binary = bool(y and y in df.columns and df[y].dropna().nunique() == 2)
        if len(m.exposure) > 1 or m.restriction != "each":
            method = "ols_wald"
        elif m.instruments:
            method = "iv_2sls"
        elif m.group:
            method = "mixed_lm"
        else:
            method = "logistic" if binary else "ols_coef"
    return Plan(
        action=Action.execute,
        design=design,
        method=method,
        model=m,
        estimand=estimand_for(method),
        rationale="specification declared by the user",
    )


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_plan(text: str) -> tuple[Plan | None, str | None]:
    """Extract and validate the first JSON object in model output. No code is executed."""
    m = _JSON_RE.search(text)
    if not m:
        return None, "no_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "bad_json"
    try:
        return Plan.model_validate(obj), None
    except ValidationError as e:
        return None, f"schema_error:{e.error_count()}"


class LLMPlanner:
    """Wraps any text generator `generate(prompt) -> str` (HF model, mock, ...)."""

    name = "llm"

    def __init__(self, generate: Callable[[str], str]):
        self.generate = generate

    def plan(
        self,
        question: str,
        study: Study,
        df: pd.DataFrame,
        prof: dict[str, Any],
        feedback: list[str] | None = None,
    ) -> tuple[Plan | None, str]:
        prompt = render_prompt(question, study, prof, feedback)
        text = self.generate(prompt)
        plan, _err = parse_plan(text)
        return plan, text
