"""P-Bench runner (SPEC §9.2, v3.1 Step 3.4): 425 tasks x N trials through the FULL NanoStat agent.

The agent receives only the question text (research question, hypotheses, design, glossary) and the
CSV, under an opaque id. It never sees the category, difficulty, answer or source path. Every trial
is appended to a resumable JSONL trace; scoring uses the clean-room scorer. Unsupported, abstained,
failed and timed-out trials count as incorrect.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.pbench.preflight import load_tasks
from benchmark.pbench.score import Answer, Trial, score
from nanostat.controller import run_analysis
from nanostat.planner import LLMPlanner, Planner
from nanostat.prompting import PROMPT_VERSION, prompt_hash
from nanostat.schemas import Budget, Study


def _token_counter(gen: object) -> object | None:
    """The CallStats of an OpenAICompatClient behind a bound `generate`, if any (for resource accounting)."""
    return getattr(getattr(gen, "__self__", None), "stats", None)


def _tok(stats: object | None) -> tuple[int, int, int]:
    if stats is None:
        return (0, 0, 0)
    return (
        int(getattr(stats, "prompt_tokens", 0)),
        int(getattr(stats, "completion_tokens", 0)),
        int(getattr(stats, "calls", 0)),
    )


def opaque_id(key: str) -> str:
    return "pb_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _status(rec: Any) -> str:
    r = rec.result
    if r.p_value is not None and r.decision is not None:
        return "ok"
    if rec.operational_failure:
        return "failed"
    return "abstained" if r.action.value in ("abstain", "clarify") else "unsupported"


def run_pbench(
    data: Path | None,
    generate_for_trial: Callable[[int], Callable[[str], str]] | None,
    label: str,
    traces: Path,
    trials: int = 3,
    max_steps: int = 3,
    limit: int | None = None,
    planner_for_trial: Callable[[int], Planner] | None = None,
) -> dict[str, Any]:
    tasks = load_tasks(None if data is None or str(data) == "data" else data)[: limit or None]
    traces.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if traces.exists():
        for line in traces.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d["label"] == label:
                    done.add((d["task"], d["trial"]))
    with traces.open("a", encoding="utf-8") as f:
        for trial in range(trials):
            if planner_for_trial is not None:
                planner = planner_for_trial(trial)
            else:
                assert generate_for_trial is not None
                gen = generate_for_trial(trial)
                planner = LLMPlanner(gen)
            for t in tasks:
                if (t["key"], trial) in done:
                    continue
                df = pd.read_csv(t["dir"] / "dataset.csv", low_memory=False)
                t0 = time.perf_counter()
                stats = _token_counter(getattr(planner, "generate", None))
                tok0 = _tok(stats)
                rec = run_analysis(
                    t["question"],
                    Study(),
                    df,
                    planner=planner,
                    budget=Budget(max_steps=max_steps),
                    deterministic_audit=True,
                )
                r = rec.result
                f.write(
                    json.dumps(
                        {
                            "label": label,
                            "task": t["key"],
                            "opaque_id": opaque_id(t["key"]),
                            "trial": trial,
                            "prompt_version": PROMPT_VERSION,
                            "prompt_hash": prompt_hash(),
                            "max_steps": max_steps,
                            "seconds": round(time.perf_counter() - t0, 3),
                            "prompt_tokens": _tok(stats)[0] - tok0[0],
                            "completion_tokens": _tok(stats)[1] - tok0[1],
                            "llm_calls": _tok(stats)[2] - tok0[2],
                            "status": _status(rec),
                            "method": r.method,
                            "p_value": r.p_value,
                            "decision": r.decision,
                            "reason_code": r.reason_code.value if r.reason_code else None,
                            "raw_outputs": rec.raw_outputs,
                            "errors": rec.report.errors if rec.report else None,
                        }
                    )
                    + "\n"
                )
                f.flush()
    return score_pbench_traces(traces, label)


def load_answers() -> list[Answer]:
    out = []
    for t in load_tasks():
        k = json.loads((t["dir"] / "answer_key.json").read_text(encoding="utf-8"))
        out.append(Answer(t["key"], float(k["p_value"]), k["decision"], t["difficulty"], t["category"], False))
    return out


def score_pbench_traces(traces: Path, label: str) -> dict[str, Any]:
    rows = [json.loads(x) for x in traces.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["label"] == label]
    seen = {r["task"] for r in rows}
    answers = [a for a in load_answers() if a.task_id in seen]
    trials = [Trial(r["task"], r["trial"], r["status"], r["p_value"], r["decision"]) for r in rows]
    rep = score(answers, trials)
    rep.pop("per_task", None)
    status: dict[str, int] = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    cost = {k: sum(r.get(k, 0) for r in rows) for k in ("prompt_tokens", "completion_tokens", "llm_calls", "seconds")}
    return {
        "label": label,
        "cost": cost,
        "tasks_scored": len(answers),
        "trial_rows": len(rows),
        "status_counts": status,
        "note": "subset of P-Bench" if len(answers) < 425 else "full P-Bench (425 tasks)",
        **rep,
    }
