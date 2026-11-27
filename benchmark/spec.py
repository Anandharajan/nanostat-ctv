"""Specification-level admissibility for model-based designs (regression / survival). Evaluator-side.

For these designs the method name is not enough: OLS with classical SEs is INVALID on clustered data
while OLS with cluster-robust SEs at the right level is ADMISSIBLE. Each admissible MethodRun.config
carries REQUIREMENTS; each invalid MethodRun.config carries a BAD PATTERN:

  admissible config: {"exposure": [...], "covariates_required": [...], "instruments_required": [...],
                      "se_allowed": [...], "cluster_required": col | None, "group_required": col | None,
                      "restriction": "each" | "joint_zero" | "equal" | None, "model": <canonical spec>}
  invalid config:    {"exposure": [...], "pattern": "se_in" | "missing_covariate" | "any",
                      "se_in": [...], "covariates": [...], "model": <representative bad spec>}

A plan is admissible if ANY admissible rule matches, invalid if ANY bad pattern matches (and no
admissible rule does), otherwise unknown.
"""

from __future__ import annotations

from typing import Any

from nanostat.schemas import KeyRecord, MethodRun, ModelSpec

MODEL_DESIGNS = ("regression", "survival")


def _covs(m: ModelSpec) -> set[str]:
    return set(m.covariates) | set(m.categorical)


def matches_admissible(run: MethodRun, method: str, m: ModelSpec) -> bool:
    c: dict[str, Any] = run.config
    if run.method != method or set(m.exposure) != set(c.get("exposure", [])):
        return False
    if not set(c.get("covariates_required", [])) <= _covs(m):
        return False
    if not set(c.get("instruments_required", [])) <= set(m.instruments):
        return False
    if c.get("se_allowed") and m.se not in c["se_allowed"]:
        return False
    if c.get("cluster_required") and m.cluster != c["cluster_required"]:
        return False
    if c.get("group_required") and m.group != c["group_required"]:
        return False
    if c.get("restriction") and m.restriction != c["restriction"]:
        return False
    if c.get("time_required") and (m.time != c["time_required"] or m.event != c.get("event_required")):
        return False
    return True


def matches_invalid(run: MethodRun, method: str, m: ModelSpec) -> bool:
    c: dict[str, Any] = run.config
    if run.method != method:
        return False
    if c.get("exposure") and set(m.exposure) != set(c["exposure"]):
        return False
    pat = c.get("pattern", "any")
    if pat == "any":
        return True
    if pat == "se_in":
        return m.se in c.get("se_in", [])
    if pat == "missing_covariate":
        return not set(c.get("covariates", [])) <= _covs(m)
    return False


def spec_status(key: KeyRecord, method: str | None, model: ModelSpec | None) -> tuple[str, str | None]:
    """('admissible' | 'invalid' | 'unknown', reason) for a plan on a model-based task."""
    if method is None or model is None:
        return "unknown", "no model specification"
    for r in key.admissible:
        if matches_admissible(r, method, model):
            return "admissible", None
    for r in key.invalid:
        if matches_invalid(r, method, model):
            return "invalid", r.reason
    return "unknown", None


def is_model_key(key: KeyRecord) -> bool:
    return key.design.value in MODEL_DESIGNS
