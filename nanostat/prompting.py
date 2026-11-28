"""Prompt rendering for the typed-plan policy. Inputs are agent-visible fields only:
question, study metadata, data profile and the LOCKED protocol (derived from the user's own
study/question). Never answer keys, admissible sets, p-values or DGP truth."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nanostat.protocol import build_protocol
from nanostat.schemas import ROLE_NAMES, Check, Design, ReasonCode, Study
from nanostat.tools import REGISTRY

PROMPT_VERSION = "typed-plan-v3"
# v3 (2026-11-28, tuned on the DEV split only after the v2 B1 baseline, docs/LLM_B1_BASELINE.md): enumerate
# every allowed value instead of placeholders, copy column roles from Study.roles, omit the optional estimand,
# make execute the default and clarify conditional on a named missing fact, and ask for raw JSON.

_ROLES = ", ".join(sorted(ROLE_NAMES))
_CHECKS = ", ".join(c.value for c in Check)
_REASONS = ", ".join(r.value for r in ReasonCode)
_DESIGNS = "|".join(d.value for d in Design)

SYSTEM = (
    "You are NanoStat, a statistical planning assistant. Read the question, the study metadata, the "
    "locked protocol and the data profile, then output ONE JSON typed analysis plan. Output raw JSON only: "
    "no markdown, no code fences, no text before or after. Never compute or state statistics yourself "
    "(no p-values or test statistics): a deterministic executor runs the plan. The protocol (alpha, direction, "
    "estimand family, null value) is locked: do not restate or change it.\n"
    "Rules:\n"
    '1. action is "execute" whenever Study.roles names the columns your design needs. This is the normal case.\n'
    '2. Use "clarify" ONLY if a required fact is missing (e.g. repeated measures but no pair_id role: '
    'reason_code missing_pair_id; a one-sample question with no null value: missing_null_value). Use "abstain" '
    "with reason_code no_admissible_method only if no listed method is valid. clarify and abstain MUST include "
    f"reason_code, one of: {_REASONS}.\n"
    f"3. columns maps ROLE -> COLUMN. Copy it from Study.roles. Keys must be role names ({_ROLES}), never "
    'column names. Example: Study.roles {"y":"score","group":"arm"} gives columns {"y":"score","group":"arm"}.\n'
    '4. Omit "estimand". It is optional, and the locked estimand_family (e.g. "typical") is NOT an estimand value.\n'
    f"5. checks is optional. If you add checks, each check must be one of: {_CHECKS}; verdict is ok or violated, "
    "as read from the profile.\n"
    "6. method must be listed under your design in Methods by design."
)

PLAN_FIELDS = (
    f'{{"action": "execute|clarify|abstain", "design": "{_DESIGNS}", '
    '"method": "<one of Methods for that design>", '
    '"columns": {"<role>": "<column>"}, "checks": [{"check": "<check name>", "verdict": "ok|violated"}], '
    '"model": {"outcome": str, "exposure": [str], "covariates": [str], "categorical": [str], '
    '"se": "classical|HC0|HC1|HC2|HC3|cluster", "cluster": str, "weights": str, "instruments": [str], '
    '"time": str, "event": str, "group": str, "restriction": "each|joint_zero|equal", "ties": "efron|breslow"} '
    "(model is REQUIRED for design regression/survival; omit otherwise), "
    '"reason_code": "<clarify/abstain only>", '
    '"rationale": "<= 30 words"}'
)

EXAMPLE = (
    '{"action":"execute","design":"paired","method":"wilcoxon_signed_rank",'
    '"columns":{"y":"score","group":"visit","pair_id":"id"},'
    '"checks":[{"check":"pairing","verdict":"ok"}],"rationale":"same subjects twice; outliers in differences"}  '
    '{"action":"execute","design":"regression","method":"ols_coef","model":{"outcome":"wage","exposure":["treat"],'
    '"covariates":["age"],"categorical":["region"],"se":"cluster","cluster":"school_id"},"rationale":"clustered RCT"}'
)


def _methods_by_design() -> str:
    by: dict[str, list[str]] = {}
    for name, tool in sorted(REGISTRY.items()):
        for d in tool.designs:
            by.setdefault(d.value, []).append(name)
    return "; ".join(f"{d}: {', '.join(ms)}" for d, ms in sorted(by.items()))


def render_prompt(question: str, study: Study, prof: dict[str, Any], feedback: list[str] | None = None) -> str:
    proto = build_protocol(question, study)
    parts = [
        SYSTEM,
        f"Methods by design: {_methods_by_design()}",
        f"Plan schema: {PLAN_FIELDS}",
        f"Examples (format only; your columns come from YOUR Study.roles): {EXAMPLE}",
        f"Question: {question}",
        "Study: " + study.model_dump_json(exclude_none=True, exclude_defaults=True),
        "Locked protocol: " + proto.model_dump_json(exclude={"source"}),
        "Profile: " + json.dumps(prof, separators=(",", ":")),
    ]
    if feedback:
        parts.append("Verifier feedback on your previous plan (fix it): " + "; ".join(feedback))
    parts.append("Plan JSON:")
    return "\n".join(parts)


def prompt_hash() -> str:
    """Hash of the fixed prompt scaffold, recorded with every evaluation for reproducibility."""
    return hashlib.sha256(
        (PROMPT_VERSION + SYSTEM + PLAN_FIELDS + EXAMPLE + _methods_by_design()).encode()
    ).hexdigest()[:16]
