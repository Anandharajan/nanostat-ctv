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

from nanostat.claims import infer_alternative
from nanostat.diagnostics import compute_diagnostics
from nanostat.policy import lookup
from nanostat.profiler import column_info
from nanostat.prompting import render_prompt
from nanostat.schemas import Action, Design, Estimand, OutcomeType, Plan, ReasonCode, Study
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
        if design is None:
            return Plan(action=Action.abstain, reason_code=ReasonCode.unsupported_design)
        cols = roles_for(design, study, df, prof)
        alt = infer_alternative(question)
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
        ctx = RunContext(columns=cols, alternative=alt, group_order=study.group_order, p0=study.p0)
        diag = compute_diagnostics(df, ctx, design)
        if "error" in diag:
            return Plan(action=Action.abstain, reason_code=ReasonCode.insufficient_data, rationale=diag["error"][:200])
        pert = diag["perturbation"]
        cell = lookup(design, outcome, pert)
        method = cell.preferred(alt) if cell else None
        if method is None:
            return Plan(action=Action.abstain, reason_code=ReasonCode.unsupported_method)
        params = {"p0": study.p0} if design == Design.one_sample and study.p0 is not None else {}
        est = estimand_for(method)
        return Plan(
            action=Action.execute,
            estimand=est,
            design=design,
            method=method,
            columns=cols,
            alternative=alt,
            alpha=0.05,
            checks=list(REGISTRY[method].required_checks),
            params=params,
            rationale=f"{design.value} design; " + RATIONALE.get(pert, "").format(est=est.value),
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
