"""CTV reward (SPEC §7.4) and the Fisher-R1-style comparator reward (arm R-F).

CPU only. Reads answer keys; must run in the reward process, never in the policy process.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from benchmark.keys import load_keys  # noqa: F401  (re-export for the reward process)
from benchmark.spec import is_model_key, spec_status
from nanostat.controller import execute_plan
from nanostat.dsl import PreprocessError
from nanostat.planner import parse_plan
from nanostat.profiler import profile
from nanostat.schemas import Action, KeyRecord, Study
from nanostat.tools import ToolError
from nanostat.verifier import assumption_score, verify
from nanostat.zscore import zc


@dataclass(frozen=True)
class RewardWeights:
    w_method: float = 0.45
    w_p: float = 0.35
    w_concl: float = 0.10
    w_assume: float = 0.10
    invalid_penalty: float = -0.2
    abstain_on_answerable: float = -0.1
    # A plan that cannot be parsed, verified or executed must score BELOW an invalid method. Otherwise the
    # policy can dodge invalid_penalty by emitting broken plans (reward hacking; docs/limitations.md, v0.4.1).
    malformed_penalty: float = -0.5


@dataclass
class Scored:
    reward: float
    valid: bool
    method: str | None = None
    p: float | None = None
    decision: str | None = None
    in_A: bool = False
    in_B: bool = False
    action: str | None = None
    parts: dict[str, float] = field(default_factory=dict)
    failure: str | None = None
    hypothesis: str | None = None


# Numbers the model states outside the typed plan count as fabricated (the model never computes
# statistics). Matches e.g. "p = 0.03", "p<.001", "t = 2.1", "statistic: 4.2".
_FABRICATED = re.compile(
    r"(\bp(-value)?\s*[=<>:]\s*\d*\.?\d+)|(\b(t|z|u|w|f|chi2|statistic)\s*[=:]\s*-?\d*\.?\d+)", re.I
)


def has_fabricated_number(text: str) -> bool:
    outside = re.sub(r"\{.*\}", " ", text, flags=re.S)
    return bool(_FABRICATED.search(outside))


def _decide(p: float, alpha: float) -> str:
    return "reject" if p < alpha else "fail_to_reject"


def score_plan(
    key: KeyRecord,
    raw_text: str,
    df: pd.DataFrame,
    study: Study,
    question: str,
    weights: RewardWeights | None = None,
) -> Scored:
    """CTV per-rollout reward (SPEC §7.4, v3.1 §2.2-§2.5)."""
    w = weights or RewardWeights()
    plan, err = parse_plan(raw_text)
    if plan is None:
        return Scored(w.malformed_penalty, False, failure=err)

    # --- abstention branch ------------------------------------------------------------------
    if key.abstain_target is not None:
        ok = (
            plan.action.value == key.abstain_target["action"]
            and plan.reason_code is not None
            and plan.reason_code.value == key.abstain_target["reason_code"]
        )
        return Scored(1.0 if ok else 0.0, True, action=plan.action.value, parts={"abstain_match": float(ok)})
    if plan.action != Action.execute:
        return Scored(w.abstain_on_answerable, True, action=plan.action.value)

    if has_fabricated_number(raw_text):
        return Scored(w.invalid_penalty, True, method=plan.method, action="execute", failure="fabricated_number")

    method = plan.method or ""
    if is_model_key(key):  # regression / survival: admissibility is judged at SPEC level
        status, _why = spec_status(key, plan.method, plan.model)
        in_B, spec_ok = status == "invalid", status == "admissible"
    else:
        in_B, spec_ok = method in {b.method for b in key.invalid}, True
    rep = verify(plan, df, profile(df), study, question, protocol=key.protocol)
    forbidden = any(e.startswith(("forbidden", "protocol_violation", "wrong_tail")) for e in rep.errors)
    if in_B or forbidden:
        return Scored(
            w.invalid_penalty,
            True,
            method=method,
            in_B=in_B,
            action="execute",
            failure="forbidden_or_protocol" if forbidden else "invalid_method",
        )
    if not rep.ok:
        return Scored(
            w.malformed_penalty, False, method=method, action="execute", failure="verifier:" + ";".join(rep.errors[:3])
        )
    try:
        res = execute_plan(df, plan, rep)
    except (ToolError, PreprocessError, ValueError, FloatingPointError) as e:
        return Scored(w.malformed_penalty, False, method=method, action="execute", failure=f"execution:{e}")
    p = res.p_value
    if p is None or math.isnan(p):
        return Scored(w.malformed_penalty, False, method=method, action="execute", failure="nan_p")

    in_A = method in {a.method for a in key.admissible} and spec_ok
    refs = [a.p for a in key.admissible if a.p is not None]  # max over A(x, e) only
    p_close = max((math.exp(-abs(zc(p) - zc(pa))) for pa in refs), default=0.0)
    concl = float(res.decision == key.reference_decision)
    r_assume = assumption_score(plan, rep)  # evidence-grounded, see nanostat.verifier
    parts = {"method": float(in_A), "p": p_close, "concl": concl, "assume": r_assume}
    R = w.w_method * parts["method"] + w.w_p * p_close + w.w_concl * concl + w.w_assume * r_assume
    return Scored(
        R,
        True,
        method=method,
        p=p,
        decision=res.decision,
        in_A=in_A,
        in_B=False,
        action="execute",
        parts=parts,
        hypothesis=rep.hypothesis,
    )


def score_fisher(key: KeyRecord, raw_text: str, df: pd.DataFrame, study: Study, question: str) -> Scored:
    """R-F comparator (Fisher-R1-style): 0.9 * z-closeness to the single canonical reference
    + 0.1 * conclusion, gated only on operational validity (the plan parses, columns exist and
    the test executes). It deliberately has no knowledge of A(x) or B(x): it runs in the
    verifier's diagnostic mode, which does not reject invalid or unknown methods."""
    plan, err = parse_plan(raw_text)
    if plan is None:
        return Scored(0.0, False, failure=err)
    if key.abstain_target is not None or plan.action != Action.execute:
        return Scored(0.0, True, action=plan.action.value)
    rep = verify(plan, df, profile(df), study, question, protocol=key.protocol, mode="diagnostic")
    if not rep.ok:
        return Scored(0.0, False, method=plan.method, failure="verifier:" + ";".join(rep.errors[:3]))
    try:
        res = execute_plan(df, plan, rep)
    except (ToolError, PreprocessError, ValueError, FloatingPointError) as e:
        return Scored(0.0, False, method=plan.method, failure=f"execution:{e}")
    ref = next((a for a in key.admissible if a.method == key.preferred), None)
    if ref is None or ref.p is None or res.p_value is None:
        return Scored(0.0, True, method=plan.method)
    decision = _decide(res.p_value, key.protocol.alpha)
    close = math.exp(-abs(zc(res.p_value) - zc(ref.p)))
    concl = float(decision == key.reference_decision)
    return Scored(
        0.9 * close + 0.1 * concl,
        True,
        method=plan.method,
        p=res.p_value,
        decision=decision,
        in_A=plan.method in {a.method for a in key.admissible},
        in_B=plan.method in {b.method for b in key.invalid},
        action="execute",
        hypothesis=rep.hypothesis,
    )


def hypothesis_true(key: KeyRecord, s: Scored) -> bool | None:
    """Truth of the null that the executed test actually tests (SPEC v3.1 §2.6)."""
    hyp = s.hypothesis or key.hypothesis
    if hyp is None:
        return None
    return key.dgp.nulls.get(hyp)


def batch_stats(keys: list[KeyRecord], scored: list[Scored]) -> dict[str, Any]:
    """Inputs for the validity constraint (SPEC §7.6)."""
    null = [s for k, s in zip(keys, scored) if s.decision is not None and hypothesis_true(k, s) is True]
    alt = [
        s
        for k, s in zip(keys, scored)
        if s.decision is not None and hypothesis_true(k, s) is False and k.dgp.effect in ("medium", "large")
    ]
    return {
        "null_rejects": [float(s.decision == "reject") for s in null],
        "alt_rejects": [float(s.decision == "reject") for s in alt],
    }
