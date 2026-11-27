"""Programmatic SFT data (SPEC §6.5): gold typed plan from the registry-preferred admissible
method plus a short template rationale. No teacher LLM, no API."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.keys import load_keys
from nanostat.planner import estimand_for
from nanostat.policy import lookup
from nanostat.profiler import profile
from nanostat.prompting import render_prompt
from nanostat.protocol import allowed_estimands
from nanostat.schemas import Action, CheckClaim, Estimand, KeyRecord, Plan, ReasonCode, TaskRecord
from nanostat.tools import REGISTRY
from nanostat.verifier import verify

_WHY = {
    "none": "assumptions look reasonable",
    "heavy_outliers": "extreme outliers present; use a rank/robust method",
    "skewed": "strong skew; use a rank-based method",
    "unequal_variance": "unequal group variances; avoid pooled variance",
    "small_expected_counts": "expected counts below 5; use an exact test",
}


def gold_plan(key: KeyRecord, task: TaskRecord | None = None, df: pd.DataFrame | None = None) -> Plan:
    """Gold typed plan. With the task and data, assumption claims carry the CPU's verdicts
    (the evidence-grounded target); without them the plan has no claims."""
    if key.abstain_target is not None:
        return Plan(
            action=Action(key.abstain_target["action"]),
            reason_code=ReasonCode(key.abstain_target["reason_code"]),
            rationale="a required design fact is missing or no reviewed method fits",
        )
    assert key.preferred is not None
    m = key.preferred
    if key.gold_model is not None:  # model-based design: the gold specification is the target
        return Plan(
            action=Action.execute,
            estimand=estimand_for(m),
            design=key.design,
            method=m,
            model=key.gold_model,
            rationale=f"{key.design.value}; specification from the study design",
        )
    pert = key.policy_cell.rsplit("/", 1)[-1]
    allowed = allowed_estimands(key.design, key.protocol)
    cell = lookup(key.design, key.outcome, pert)
    est_name = cell.estimand_of(m, allowed) if cell else None
    plan = Plan(
        action=Action.execute,
        estimand=Estimand(est_name) if est_name else estimand_for(m),
        design=key.design,
        method=m,
        columns=dict(key.columns),
        rationale=f"{key.design.value}; {_WHY.get(pert, '')}",
    )
    if task is not None and df is not None:
        rep = verify(plan, df, profile(df), task.study, task.question, protocol=key.protocol)
        claims = [
            CheckClaim(check=c, verdict=rep.cpu_verdicts[c.value])  # type: ignore[arg-type]
            for c in REGISTRY[m].required_checks
            if c.value in rep.cpu_verdicts
        ]
        plan = plan.model_copy(update={"checks": claims})
    return plan


def build_sft(data_dir: Path, out: Path, split: str = "train", limit: int | None = None) -> int:
    keys = load_keys(data_dir / "keys" / "keys.jsonl")
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with (data_dir / "tasks" / f"{split}.jsonl").open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as fo:
        for line in f:
            t = TaskRecord.model_validate_json(line)
            df = pd.read_csv(data_dir / t.data_ref)
            prompt = render_prompt(t.question, t.study, profile(df))
            completion = gold_plan(keys[t.task_id], t, df).model_dump_json(exclude_none=True)
            fo.write(json.dumps({"task_id": t.task_id, "prompt": prompt, "completion": completion}) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n
