"""Prompt rendering for the typed-plan policy. Inputs are agent-visible fields only:
question, study metadata, data profile and the LOCKED protocol (derived from the user's own
study/question). Never answer keys, admissible sets, p-values or DGP truth."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nanostat.protocol import build_protocol
from nanostat.schemas import Study
from nanostat.tools import REGISTRY

PROMPT_VERSION = "typed-plan-v2"

SYSTEM = (
    "You are NanoStat, a statistical planning assistant. Read the question, the study metadata, the "
    "locked protocol and the data profile, then output ONE JSON typed analysis plan and nothing else. "
    "Never compute or state statistics yourself (no p-values or test statistics in your answer): a "
    "deterministic executor runs the plan. The protocol (alpha, direction, estimand family, null value) "
    "is locked: do not change it. Use the study's declared roles for `columns`. If a required fact is "
    "missing (e.g. repeated measures but no subject ID), output action=clarify with a reason_code; if no "
    "method is valid for this design and data, output action=abstain with reason_code=no_admissible_method."
)

PLAN_FIELDS = (
    '{"action": "execute|clarify|abstain", '
    '"design": "independent_two_group|paired|correlation|k_group|one_sample|regression|survival", '
    '"method": "<one of Methods>", "estimand": "<optional>", '
    '"columns": {"<role>": "<column>"}, "checks": [{"check": "<name>", "verdict": "ok|violated"}], '
    '"model": {"outcome": str, "exposure": [str], "covariates": [str], "categorical": [str], '
    '"se": "classical|HC0|HC1|HC2|HC3|cluster", "cluster": str, "weights": str, "instruments": [str], '
    '"time": str, "event": str, "group": str, "restriction": "each|joint_zero|equal", "ties": "efron|breslow"} '
    "(model is REQUIRED for design regression/survival; omit otherwise), "
    '"reason_code": "missing_pair_id|missing_null_value|no_admissible_method|... (clarify/abstain only)", '
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
        f"Example (format only): {EXAMPLE}",
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
