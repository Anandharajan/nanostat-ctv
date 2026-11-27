"""Reward functions in the shape TRL's GRPOTrainer expects: f(prompts, completions, task_id=[...]) -> list[float].

Pure Python (no torch): unit-tested on CPU. Used by training/grpo_trl.py on a GPU (Colab/Kaggle).
The reward process reads answer keys; the policy only ever sees the prompt.

Arms supported with stock GRPO:
  R-F    fisher reward
  R-A    CTV admissible-set reward
  R-AD   + discordance-weighted sampling (dataset repetition, see build_rl_dataset)
  R-AV   + scenario validity constraint, applied as batch-level reward shaping inside the reward call
Arms needing a custom trainer (twin-joint advantage over paired rollouts): R-ADT, R-CTV -> NOT
implemented for the LLM; the CPU ablation implements them (training/ablation).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.keys import load_keys
from nanostat.schemas import TaskRecord
from training.constraint import ScenarioConstraint, scenario_of
from training.reward_server import hypothesis_true, score_fisher, score_plan


def _text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):  # conversational format
        return str(c[-1].get("content", ""))
    return str(c)


class TaskStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.keys = load_keys(data_dir / "keys" / "keys.jsonl")
        self.tasks: dict[str, TaskRecord] = {}
        for sp in ("train", "dev", "test"):
            p = data_dir / "tasks" / f"{sp}.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        t = TaskRecord.model_validate_json(line)
                        self.tasks[t.task_id] = t
        self._dfs: dict[str, pd.DataFrame] = {}

    def df(self, tid: str) -> pd.DataFrame:
        if tid not in self._dfs:
            self._dfs[tid] = pd.read_csv(self.data_dir / self.tasks[tid].data_ref)
        return self._dfs[tid]


def make_reward(store: TaskStore, kind: str = "ctv", constraint: ScenarioConstraint | None = None):  # type: ignore[no-untyped-def]
    def reward(prompts: list[Any], completions: list[Any], task_id: list[str], **_: Any) -> list[float]:
        scored = []
        for c, tid in zip(completions, task_id):
            t, k, df = store.tasks[tid], store.keys[tid], store.df(tid)
            s = (score_plan if kind == "ctv" else score_fisher)(k, _text(c), df, t.study, t.question)
            scored.append((k, s))
        rewards = [s.reward for _, s in scored]
        if constraint is not None:
            null_r: dict[str, list[float]] = {}
            alt_r: dict[str, list[float]] = {}
            for k, s in scored:
                if s.decision is None:
                    continue
                sc, ht = scenario_of(k.dgp.family, k.dgp.perturbation), hypothesis_true(k, s)
                if ht is True:
                    null_r.setdefault(sc, []).append(float(s.decision == "reject"))
                elif ht is False and k.dgp.effect in ("medium", "large"):
                    alt_r.setdefault(sc, []).append(float(s.decision == "reject"))
            constraint.update(null_r, alt_r)
            rewards = [
                constraint.shape(
                    scenario_of(k.dgp.family, k.dgp.perturbation),
                    r,
                    s.decision == "reject",
                    hypothesis_true(k, s),
                    k.dgp.effect,
                )
                if s.decision is not None
                else r
                for (k, s), r in zip(scored, rewards)
            ]
        return rewards

    reward.__name__ = f"{kind}_reward"
    return reward


def build_rl_dataset(
    sft_jsonl: Path, store: TaskStore, discordance_weighting: bool, eps: float = 0.1, max_copies: int = 4
) -> list[dict[str, Any]]:
    """Rows {prompt, task_id}. With discordance weighting each task is repeated
    round((eps + D) / eps') times (capped), approximating w(x) proportional to eps + D(x)."""
    rows = []
    for line in sft_jsonl.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        reps = 1
        if discordance_weighting:
            k = store.keys[d["task_id"]]
            dval = 1.0 if k.abstain_target is not None else k.discordance.score  # abstain-aware (see ablation)
            reps = max(1, min(max_copies, int(math.ceil((eps + dval) / (eps + 0.25)))))
        rows += [{"prompt": d["prompt"], "task_id": d["task_id"]}] * reps
    return rows
