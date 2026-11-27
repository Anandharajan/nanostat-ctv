"""NanoStat-Bench metrics (SPEC §9.1): method validity, invalid-method rate, twin consistency,
abstention correctness and operational failures. Evaluator-side: reads keys."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import beta

from benchmark.keys import load_keys
from nanostat.controller import run_analysis
from nanostat.planner import Planner, RulePlanner
from nanostat.schemas import Action, KeyRecord, TaskRecord
from nanostat.zscore import zc


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


@dataclass
class TaskOutcome:
    task_id: str
    answerable: bool
    action: str
    method: str | None
    valid: bool
    invalid: bool
    abstain_correct: bool | None
    operational_failure: str | None
    twin_id: str | None
    decision: str | None
    h0_true: bool
    family: str
    cell: str
    strict: bool | None = None


@dataclass
class Summary:
    n: int
    metrics: dict[str, Any] = field(default_factory=dict)


def evaluate_task(t: TaskRecord, key: KeyRecord, df: pd.DataFrame, planner: Planner) -> TaskOutcome:
    rec = run_analysis(t.question, t.study, df, planner=planner, budget=t.budget, deterministic_audit=True)
    return outcome_from_record(t, key, rec)


def strict_correct(key: KeyRecord, r: Any) -> bool | None:
    """Decision equals the reference AND |dz| < 0.5 versus the registry-preferred method's p."""
    if key.abstain_target is not None:
        return None
    ref = next((a.p for a in key.admissible if a.method == key.preferred), None)
    if ref is None or r.p_value is None or r.decision is None:
        return False
    return bool(r.decision == key.reference_decision and abs(zc(r.p_value) - zc(ref)) < 0.5)


def outcome_from_record(t: TaskRecord, key: KeyRecord, rec: Any) -> TaskOutcome:
    r = rec.result
    answerable = key.abstain_target is None
    alt_ok = r.alternative is not None and any(
        a.method == r.method and a.config.get("alternative") == r.alternative.value for a in key.admissible
    )
    valid = answerable and r.action == Action.execute and r.method in {a.method for a in key.admissible} and alt_ok
    invalid = answerable and r.action == Action.execute and r.method in {b.method for b in key.invalid}
    abst = None
    if not answerable:
        abst = (
            r.action.value == key.abstain_target["action"]  # type: ignore[index]
            and r.reason_code is not None
            and r.reason_code.value == key.abstain_target["reason_code"]  # type: ignore[index]
        )
    return TaskOutcome(
        task_id=t.task_id,
        answerable=answerable,
        action=r.action.value,
        method=r.method,
        valid=valid,
        invalid=invalid,
        abstain_correct=abst,
        operational_failure=rec.operational_failure,
        twin_id=key.twin_id,
        decision=r.decision,
        h0_true=key.dgp.h0_true,
        family=key.dgp.family,
        cell=key.policy_cell,
        strict=strict_correct(key, r),
    )


def summarize(outs: list[TaskOutcome]) -> Summary:
    ans = [o for o in outs if o.answerable]
    non = [o for o in outs if not o.answerable]
    by_id = {o.task_id: o for o in outs}
    pairs, seen = [], set()
    for o in outs:
        if o.twin_id and o.twin_id in by_id and o.task_id not in seen:
            seen |= {o.task_id, o.twin_id}
            pairs.append((o, by_id[o.twin_id]))

    def ok(o: TaskOutcome) -> bool:
        return o.valid if o.answerable else bool(o.abstain_correct)

    def rate(k: int, n: int) -> dict[str, Any]:
        lo, hi = clopper_pearson(k, n)
        return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}

    m: dict[str, Any] = {
        "method_validity_rate": rate(sum(o.valid for o in ans), len(ans)),
        "invalid_method_rate": rate(sum(o.invalid for o in ans), len(ans)),
        "abstention_correctness": rate(sum(bool(o.abstain_correct) for o in non), len(non)),
        "false_abstention_rate": rate(sum(o.action != "execute" for o in ans), len(ans)),
        "twin_consistency_accuracy": rate(sum(ok(a) and ok(b) for a, b in pairs), len(pairs)),
        "operational_failure_rate": rate(sum(o.operational_failure is not None for o in outs), len(outs)),
        "strict_single_reference": rate(sum(bool(o.strict) for o in ans), len(ans)),
        "task_level_reject_rate_true_null": rate(
            sum(o.decision == "reject" for o in ans if o.h0_true), sum(1 for o in ans if o.h0_true)
        ),
    }
    per_family: dict[str, Any] = {}
    for fam in sorted({o.family for o in ans}):
        fo = [o for o in ans if o.family == fam]
        per_family[fam] = rate(sum(o.valid for o in fo), len(fo))
    m["method_validity_by_family"] = per_family
    return Summary(n=len(outs), metrics=m)


def iter_tasks(data_dir: Path, split: str) -> Iterable[TaskRecord]:
    with (data_dir / "tasks" / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield TaskRecord.model_validate_json(line)


def evaluate_split(data_dir: Path, split: str, planner: Planner | None = None) -> tuple[Summary, list[TaskOutcome]]:
    planner = planner or RulePlanner()
    keys = load_keys(data_dir / "keys" / "keys.jsonl")
    outs = [
        evaluate_task(t, keys[t.task_id], pd.read_csv(data_dir / t.data_ref), planner)
        for t in iter_tasks(data_dir, split)
    ]
    return summarize(outs), outs
