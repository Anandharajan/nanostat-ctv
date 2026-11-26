"""Programmatic SFT data (SPEC §6.5): gold typed plan from the registry-preferred admissible
method plus a short template rationale. No teacher LLM, no API."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.keys import load_keys
from nanostat.planner import estimand_for
from nanostat.profiler import profile
from nanostat.prompting import render_prompt
from nanostat.schemas import Action, Alternative, KeyRecord, Plan, ReasonCode, TaskRecord
from nanostat.tools import REGISTRY

_WHY = {
    "none": "assumptions look reasonable",
    "heavy_outliers": "extreme outliers present; use a rank/robust method",
    "skewed": "strong skew; use a rank-based method",
    "unequal_variance": "unequal group variances; avoid pooled variance",
    "small_expected_counts": "expected counts below 5; use an exact test",
}


def gold_plan(key: KeyRecord) -> Plan:
    if key.abstain_target is not None:
        return Plan(
            action=Action(key.abstain_target["action"]),
            reason_code=ReasonCode(key.abstain_target["reason_code"]),
            rationale="a required design fact is missing",
        )
    assert key.preferred is not None
    m = key.preferred
    pert = key.policy_cell.rsplit("/", 1)[-1]
    params: dict[str, float] = {}
    alt = Alternative(next(a.config["alternative"] for a in key.admissible if a.method == m))
    return Plan(
        action=Action.execute,
        estimand=estimand_for(m),
        design=key.design,
        method=m,
        columns=dict(key.columns),
        alternative=alt,
        checks=list(REGISTRY[m].required_checks),
        params=params,
        rationale=f"{key.design.value}; {_WHY.get(pert, '')}",
    )


def build_sft(data_dir: Path, out: Path, split: str = "train", limit: int | None = None) -> int:
    keys = load_keys(data_dir / "keys" / "keys.jsonl")
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with (data_dir / "tasks" / f"{split}.jsonl").open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as fo:
        for line in f:
            t = TaskRecord.model_validate_json(line)
            df = pd.read_csv(data_dir / t.data_ref)
            prompt = render_prompt(t.question, t.study, profile(df))
            completion = gold_plan(keys[t.task_id]).model_dump_json(exclude_none=True)
            fo.write(json.dumps({"task_id": t.task_id, "prompt": prompt, "completion": completion}) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n
