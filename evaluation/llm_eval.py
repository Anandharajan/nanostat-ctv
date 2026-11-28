"""Evaluate a locally served LLM (e.g. via Magnitude) as the NanoStat planner (arm B1 / B1+verify).

For each task and trial the FULL agent runs: prompt -> model -> typed plan -> two-gate verifier ->
(re-plan with verifier feedback, up to `max_steps`) -> deterministic executor. Every raw model
output is saved to a resumable JSONL trace. Scoring reads keys evaluator-side only; the model sees
the agent-visible prompt alone.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.keys import load_keys
from evaluation.metrics import TaskOutcome, iter_tasks, outcome_from_record, summarize
from nanostat.controller import run_analysis
from nanostat.planner import LLMPlanner, parse_plan
from nanostat.prompting import PROMPT_VERSION, prompt_hash
from nanostat.schemas import Budget


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


def _done(traces: Path) -> set[tuple[str, int]]:
    if not traces.exists():
        return set()
    out = set()
    for line in traces.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.add((d["task_id"], d["trial"]))
    return out


def run_llm_eval(
    data_dir: Path,
    split: str,
    generate_for_trial: Callable[[int], Callable[[str], str]],
    model_label: str,
    traces: Path,
    trials: int = 1,
    max_steps: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    """`generate_for_trial(trial)` returns a text generator (one per trial, e.g. a different seed)."""
    keys = load_keys(data_dir / "keys" / "keys.jsonl")
    tasks = list(iter_tasks(data_dir, split))[: limit or None]
    traces.parent.mkdir(parents=True, exist_ok=True)
    done = _done(traces)
    with traces.open("a", encoding="utf-8") as f:
        for trial in range(trials):
            gen = generate_for_trial(trial)
            stats = _token_counter(gen)
            for t in tasks:
                if (t.task_id, trial) in done:
                    continue
                df = pd.read_csv(data_dir / t.data_ref)
                t0 = time.perf_counter()
                tok0 = _tok(stats)
                rec = run_analysis(
                    t.question,
                    t.study,
                    df,
                    planner=LLMPlanner(gen),
                    budget=Budget(max_steps=max_steps),
                    deterministic_audit=True,
                )
                first, _err = parse_plan(rec.raw_outputs[0]) if rec.raw_outputs else (None, "no_output")
                key = keys[t.task_id]
                f.write(
                    json.dumps(
                        {
                            "task_id": t.task_id,
                            "trial": trial,
                            "model": model_label,
                            "prompt_version": PROMPT_VERSION,
                            "prompt_hash": prompt_hash(),
                            "max_steps": max_steps,
                            "seconds": round(time.perf_counter() - t0, 3),
                            "prompt_tokens": _tok(stats)[0] - tok0[0],
                            "completion_tokens": _tok(stats)[1] - tok0[1],
                            "llm_calls": _tok(stats)[2] - tok0[2],
                            "raw_outputs": rec.raw_outputs,
                            "attempts": rec.attempts,
                            "operational_failure": rec.operational_failure,
                            "first_plan_parsed": first is not None,
                            "first_method": first.method if first else None,
                            "first_method_invalid": bool(first and first.method in {b.method for b in key.invalid}),
                            "first_method_admissible": bool(
                                first and first.method in {a.method for a in key.admissible}
                            ),
                            "result": rec.result.model_dump(mode="json", exclude={"diagnostics", "admissible_range"}),
                            "outcome": asdict(outcome_from_record(t, key, rec)),
                        }
                    )
                    + "\n"
                )
                f.flush()
    return score_traces(traces, model_label)


def score_traces(traces: Path, model_label: str | None = None) -> dict[str, Any]:
    rows = [json.loads(x) for x in traces.read_text(encoding="utf-8").splitlines() if x.strip()]
    if model_label:
        rows = [r for r in rows if r["model"] == model_label]
    by_trial: dict[int, list[TaskOutcome]] = {}
    for r in rows:
        by_trial.setdefault(r["trial"], []).append(TaskOutcome(**r["outcome"]))
    per_trial = {tr: summarize(outs).metrics for tr, outs in sorted(by_trial.items())}
    first_ok = [r["first_plan_parsed"] for r in rows]
    ans_rows = [r for r in rows if r["outcome"]["answerable"]]
    out: dict[str, Any] = {
        "model": model_label,
        # the version recorded in the traces, not the current constant (re-scoring old traces must not relabel them)
        "prompt_version": ",".join(sorted({str(r.get("prompt_version")) for r in rows})) or None,
        "prompt_hash": ",".join(sorted({str(r.get("prompt_hash")) for r in rows})) or None,
        "n_rows": len(rows),
        "trials": len(per_trial),
        "first_attempt_json_valid_rate": sum(first_ok) / len(first_ok) if first_ok else None,
        "first_attempt_invalid_method_rate": (sum(r["first_method_invalid"] for r in ans_rows) / len(ans_rows))
        if ans_rows
        else None,
        "first_attempt_admissible_rate": (sum(r["first_method_admissible"] for r in ans_rows) / len(ans_rows))
        if ans_rows
        else None,
        "mean_seconds_per_task": sum(r["seconds"] for r in rows) / len(rows) if rows else None,
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        "total_llm_calls": sum(r.get("llm_calls", 0) for r in rows),
        "total_seconds": sum(r["seconds"] for r in rows),
        "per_trial": per_trial,
    }
    # pass@1 = mean over trials of the per-trial rates
    for k in (
        "method_validity_rate",
        "strict_single_reference",
        "twin_consistency_accuracy",
        "abstention_correctness",
        "operational_failure_rate",
    ):
        vals = [m[k]["rate"] for m in per_trial.values() if m[k]["rate"] is not None]
        out[f"{k}_pass1"] = sum(vals) / len(vals) if vals else None
    return out
