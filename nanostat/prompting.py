"""Prompt rendering for the typed-plan policy. Inputs are agent-visible fields only."""

from __future__ import annotations

import json
from typing import Any

from nanostat.schemas import Study
from nanostat.tools import REGISTRY

SYSTEM = (
    "You are NanoStat, a statistical planning assistant. Read the question, study metadata and "
    "data profile, then output ONE JSON typed analysis plan and nothing else. Never compute or "
    "state statistics yourself; a deterministic executor runs the plan. If a required fact is "
    "missing, output action=clarify with a reason_code; if the analysis is unsupported, "
    "action=abstain."
)

PLAN_FIELDS = (
    '{"action": "execute|clarify|abstain", "estimand": str, "design": '
    '"independent_two_group|paired|correlation|k_group|one_sample", "method": str, '
    '"columns": {role: column}, "alternative": "two-sided|greater|less", '
    '"preprocess": [], "alpha": 0.05, "checks": [str], "params": {}, '
    '"reason_code": str|null, "rationale": str}'
)


def render_prompt(question: str, study: Study, prof: dict[str, Any], feedback: list[str] | None = None) -> str:
    parts = [
        SYSTEM,
        f"Methods: {', '.join(sorted(REGISTRY))}",
        f"Plan schema: {PLAN_FIELDS}",
        f"Question: {question}",
        "Study: " + study.model_dump_json(exclude_none=True),
        "Profile: " + json.dumps(prof, separators=(",", ":")),
    ]
    if feedback:
        parts.append("Verifier feedback on previous plan: " + "; ".join(feedback))
    parts.append("Plan JSON:")
    return "\n".join(parts)
