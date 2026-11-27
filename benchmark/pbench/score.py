"""Clean-room P-Bench scorer (SPEC §9.2, v3.1 Step 3.3).

Written from the metric definitions only; no Fisher-R1 code was copied (their repository has no
licence). Definitions implemented:

  Raw      : the trial's decision at alpha = 0.05 equals the reference decision.
  Strict   : Raw AND |z(p) - z(p_ref)| < 0.5
  Relaxed  : Raw AND |z(p) - z(p_ref)| < 1.0
  z(p)     : Phi^{-1}(1 - p/2) = norm.isf(p/2)
  pass@1   : mean over the 3 trials of each task, averaged over tasks
  pass@3   : 1 if any of the 3 trials is correct, averaged over tasks

Boundary handling (DOCUMENTED, to be confirmed with the P-Bench authors before any external
comparison): p is clipped to [P_FLOOR, 1] before z, so p = 0 maps to z(P_FLOOR) ~= 8.29 rather
than infinity; p = 1 maps to z = 0. A trial whose status is not "ok", or with a missing / non-finite
p-value or an unknown decision, is INCORRECT. A task with fewer than 3 trials counts the missing
trials as incorrect. Unsupported, abstained, failed and timed-out tasks are therefore incorrect.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from scipy.stats import norm

P_FLOOR = 1e-16
N_TRIALS = 3
STRICT_DZ = 0.5
RELAXED_DZ = 1.0
ALPHA = 0.05


def z_score(p: float) -> float:
    p = min(max(float(p), P_FLOOR), 1.0)
    return float(norm.isf(p / 2.0))


@dataclass(frozen=True)
class Answer:
    task_id: str
    p_ref: float
    decision_ref: str  # "reject" | "fail_to_reject"
    difficulty: str  # "easy" | "hard"
    category: str
    perturbed: bool = False


@dataclass(frozen=True)
class Trial:
    task_id: str
    trial: int
    status: str  # "ok" | "unsupported" | "abstained" | "failed" | "timeout" | "malformed"
    p_value: float | None = None
    decision: str | None = None


def grade_trial(ans: Answer, t: Trial | None) -> dict[str, bool]:
    bad = {"raw": False, "strict": False, "relaxed": False}
    if t is None or t.status != "ok" or t.p_value is None or t.decision not in ("reject", "fail_to_reject"):
        return bad
    if not math.isfinite(t.p_value) or not 0.0 <= t.p_value <= 1.0:
        return bad
    raw = t.decision == ans.decision_ref
    dz = abs(z_score(t.p_value) - z_score(ans.p_ref))
    return {"raw": raw, "strict": raw and dz < STRICT_DZ, "relaxed": raw and dz < RELAXED_DZ}


def score(answers: Iterable[Answer], trials: Iterable[Trial]) -> dict[str, Any]:
    ans = {a.task_id: a for a in answers}
    by_task: dict[str, dict[int, Trial]] = defaultdict(dict)
    for t in trials:
        if t.task_id in ans and 0 <= t.trial < N_TRIALS:
            by_task[t.task_id][t.trial] = t  # last write wins for a duplicated trial id

    per_task: dict[str, dict[str, float]] = {}
    for tid, a in ans.items():
        grades = [grade_trial(a, by_task[tid].get(i)) for i in range(N_TRIALS)]
        row: dict[str, float] = {}
        for m in ("raw", "strict", "relaxed"):
            row[f"{m}_pass1"] = sum(g[m] for g in grades) / N_TRIALS
            row[f"{m}_pass3"] = float(any(g[m] for g in grades))
        per_task[tid] = row

    def agg(ids: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(ids)}
        for k in ("raw_pass1", "raw_pass3", "strict_pass1", "strict_pass3", "relaxed_pass1", "relaxed_pass3"):
            out[k] = 100.0 * sum(per_task[i][k] for i in ids) / len(ids) if ids else None
        return out

    groups: dict[str, list[str]] = defaultdict(list)
    for tid, a in ans.items():
        groups["overall"].append(tid)
        groups[f"difficulty:{a.difficulty}"].append(tid)
        groups[f"category:{a.category}"].append(tid)
        groups["perturbed" if a.perturbed else "clean"].append(tid)
    report: dict[str, Any] = {g: agg(ids) for g, ids in sorted(groups.items())}
    hard = report.get("difficulty:hard", {})
    report["composite_hard_strict_pass1"] = hard.get("strict_pass1")
    report["per_task"] = per_task
    return report


FIXTURE_SHEET: list[dict[str, Any]] = [
    # name, reference, trials -> expected strict pass@1 / pass@3 (hand-computed)
    {
        "name": "all correct, tiny dz",
        "p_ref": 0.01,
        "dec": "reject",
        "trials": [(0.012, "reject"), (0.011, "reject"), (0.009, "reject")],
        "strict_pass1": 1.0,
        "strict_pass3": 1.0,
    },
    {
        "name": "wrong decision",
        "p_ref": 0.01,
        "dec": "reject",
        "trials": [(0.2, "fail_to_reject")] * 3,
        "strict_pass1": 0.0,
        "strict_pass3": 0.0,
    },
    {
        "name": "right decision, dz too large",
        "p_ref": 0.04,
        "dec": "reject",
        "trials": [(1e-6, "reject")] * 3,
        "strict_pass1": 0.0,
        "strict_pass3": 0.0,
    },
    {
        "name": "one of three",
        "p_ref": 0.3,
        "dec": "fail_to_reject",
        "trials": [(0.31, "fail_to_reject"), ("malformed", None), ("abstained", None)],
        "strict_pass1": 1 / 3,
        "strict_pass3": 1.0,
    },
    {
        "name": "p=0 vs tiny reference",
        "p_ref": 1e-20,
        "dec": "reject",
        "trials": [(0.0, "reject")] * 3,
        "strict_pass1": 1.0,
        "strict_pass3": 1.0,
    },
    {
        "name": "p=1 vs p_ref=0.9",
        "p_ref": 0.9,
        "dec": "fail_to_reject",
        "trials": [(1.0, "fail_to_reject")] * 3,
        "strict_pass1": 1.0,
        "strict_pass3": 1.0,
    },
]
