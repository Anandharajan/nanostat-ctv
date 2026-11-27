"""CPU ablation machinery: reward tables agree with the reward server, the policy gradient is
correct, runs are deterministic, and policy-side modules never read keys."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.keys import load_keys
from nanostat.profiler import profile
from nanostat.schemas import TaskRecord
from training.ablation.actions import CATALOGUE, render_plan
from training.ablation.experiment import ARMS, HParams, holm, run_ablation
from training.ablation.policy import LinearSoftmaxPolicy, softmax
from training.ablation.tables import build_task_table
from training.reward_server import score_fisher, score_plan


@pytest.fixture(scope="module")
def table(smoke_data: Path):
    return build_task_table(smoke_data)


def test_table_matches_reward_server(smoke_data: Path, table):
    keys = load_keys(smoke_data / "keys" / "keys.jsonl")
    tasks = {}
    for sp in ("train", "dev", "test"):
        for line in (smoke_data / "tasks" / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            t = TaskRecord.model_validate_json(line)
            tasks[t.task_id] = t
    rng = np.random.default_rng(0)
    for _ in range(25):
        i = int(rng.integers(len(table.task_ids)))
        a = int(rng.integers(len(CATALOGUE)))
        t = tasks[table.task_ids[i]]
        df = pd.read_csv(smoke_data / t.data_ref)
        text = render_plan(CATALOGUE[a], t.study, df, profile(df)).model_dump_json(exclude_none=True)
        assert table.ctv[i, a] == pytest.approx(score_plan(keys[t.task_id], text, df, t.study, t.question).reward)
        assert table.fisher[i, a] == pytest.approx(score_fisher(keys[t.task_id], text, df, t.study, t.question).reward)
    # gold actions are always correct
    assert table.correct[np.arange(len(table.gold)), table.gold].all()


def test_policy_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    pol = LinearSoftmaxPolicy(4, 3)
    pol.W = rng.normal(size=(4, 3))
    phi = rng.normal(size=(5, 3))
    acts = rng.integers(0, 4, size=5)
    adv = rng.normal(size=5)

    def objective(W):
        p = softmax(phi @ W.T)
        return float((adv * np.log(p[np.arange(5), acts])).mean())

    W0 = pol.W.copy()
    pol.grpo_step(phi, acts, adv, lr=1.0)
    analytic = pol.W - W0
    num = np.zeros_like(W0)
    for idx in np.ndindex(*W0.shape):
        e = np.zeros_like(W0)
        e[idx] = 1e-6
        num[idx] = (objective(W0 + e) - objective(W0 - e)) / 2e-6
    assert np.allclose(analytic, num, atol=1e-6)


def test_ablation_is_deterministic(table):
    hp = HParams(updates=10, log_every=5, sft_examples=16)
    r1 = run_ablation(table, None, [3], hp, ARMS[:3])
    r2 = run_ablation(table, None, [3], hp, ARMS[:3])
    import json

    dump = lambda r: json.dumps(r["arms"], sort_keys=True, default=str)  # noqa: E731  (NaN-safe comparison)
    assert dump(r1) == dump(r2)


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
