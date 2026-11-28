"""Evaluator-side reward tables for the CPU ablation.

Each (task, action) pair is scored ONCE with the real reward server (training/reward_server.py)
and the real executor; training then samples from these deterministic tables. This module reads
answer keys and must never be imported by policy-side code.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.factory import render
from benchmark.factory.dgp import Scenario, simulate
from benchmark.factory.grid import FAMILY_DESIGN
from benchmark.keys import load_keys
from benchmark.nullcal import NULLCAL_SCENARIOS
from nanostat.controller import run_analysis
from nanostat.planner import LLMPlanner
from nanostat.profiler import profile
from nanostat.schemas import Budget, KeyRecord, Study, TaskRecord
from nanostat.zscore import zc
from training.ablation.actions import CATALOGUE, ActionSpec, render_plan
from training.ablation.features import featurize
from training.constraint import scenario_of
from training.reward_server import hypothesis_true, score_fisher, score_plan


@dataclass
class TaskTable:
    task_ids: list[str]
    phi: np.ndarray  # (N, F)
    ctv: np.ndarray  # (N, A)
    fisher: np.ndarray  # (N, A)
    reject: np.ndarray  # (N, A) 1/0, nan if not executed confirmatorily
    reject_any: np.ndarray  # (N, A) 1/0, nan if not executed even in diagnostic mode (for R-F)
    htrue: np.ndarray  # (N, A) truth of executed test's null, nan if unknown
    correct: np.ndarray  # (N, A) valid method in A(x,e) or the correct clarify/abstain
    invalid: np.ndarray  # (N, A) method in B(x,e)
    strict: np.ndarray  # (N, A) decision == reference and |dz| < 0.5 vs the preferred reference
    gold: np.ndarray  # (N,) action index of the registry-preferred gold plan
    meta: list[dict[str, Any]] = field(default_factory=list)
    broken: np.ndarray | None = None  # (N, A) 1 if the plan could not be verified or executed (reward-hacking audit)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            phi=self.phi,
            ctv=self.ctv,
            fisher=self.fisher,
            reject=self.reject,
            reject_any=self.reject_any,
            htrue=self.htrue,
            correct=self.correct,
            invalid=self.invalid,
            strict=self.strict,
            gold=self.gold,
            broken=self.broken if self.broken is not None else np.full_like(self.ctv, np.nan),
        )
        path.with_suffix(".meta.json").write_text(json.dumps({"task_ids": self.task_ids, "meta": self.meta}))

    @classmethod
    def load(cls, path: Path) -> TaskTable:
        z = np.load(path)
        m = json.loads(path.with_suffix(".meta.json").read_text())
        return cls(
            m["task_ids"],
            z["phi"],
            z["ctv"],
            z["fisher"],
            z["reject"],
            z["reject_any"],
            z["htrue"],
            z["correct"],
            z["invalid"],
            z["strict"],
            z["gold"],
            m["meta"],
            z["broken"] if "broken" in z.files else None,
        )


def gold_index(key: KeyRecord) -> int:
    for i, a in enumerate(CATALOGUE):
        if key.abstain_target is not None:
            if (
                a.kind == key.abstain_target["action"]
                and a.reason
                and a.reason.value == key.abstain_target["reason_code"]
            ):
                return i
        elif a.kind == "execute" and a.design == key.design and a.method == key.preferred:
            return i
    raise ValueError(f"no catalogue action for gold plan of {key.task_id}")


def _row(key: KeyRecord, t: TaskRecord, df: pd.DataFrame, prof: dict[str, Any], a: ActionSpec) -> dict[str, float]:
    text = render_plan(a, t.study, df, prof).model_dump_json(exclude_none=True)
    s = score_plan(key, text, df, t.study, t.question)
    f = score_fisher(key, text, df, t.study, t.question)
    nan = float("nan")
    executed = s.decision is not None
    ref = next((r.p for r in key.admissible if r.method == key.preferred), None)
    strict = False
    if executed and ref is not None and s.p is not None:
        strict = s.decision == key.reference_decision and abs(zc(s.p) - zc(ref)) < 0.5
    if key.abstain_target is not None:
        correct = s.reward >= 1.0
    else:
        correct = executed and s.in_A
    ht = hypothesis_true(key, s) if executed else None
    return {
        "ctv": s.reward,
        "fisher": f.reward,
        "reject": (1.0 if s.decision == "reject" else 0.0) if executed else nan,
        "reject_any": (1.0 if f.decision == "reject" else 0.0) if f.decision is not None else nan,
        "htrue": nan if ht is None else float(ht),
        "correct": float(correct),
        "invalid": float(s.in_B),
        "strict": float(strict),
        "broken": float(not s.valid),
    }


def build_task_table(data_dir: Path, splits: tuple[str, ...] = ("train", "dev", "test")) -> TaskTable:
    keys = load_keys(data_dir / "keys" / "keys.jsonl")
    ids, phis, golds, meta = [], [], [], []
    cols: dict[str, list[list[float]]] = {
        k: [] for k in ("ctv", "fisher", "reject", "reject_any", "htrue", "correct", "invalid", "strict", "broken")
    }
    for sp in splits:
        for line in (data_dir / "tasks" / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            t = TaskRecord.model_validate_json(line)
            k = keys[t.task_id]
            if k.gold_model is not None:
                continue  # the proxy policy has no specification head; model-based tasks are LLM-only
            df = pd.read_csv(data_dir / t.data_ref)
            prof = profile(df)
            rows = [_row(k, t, df, prof, a) for a in CATALOGUE]
            for c in cols:
                cols[c].append([r[c] for r in rows])
            ids.append(t.task_id)
            phis.append(featurize(t.question, t.study, df, prof))
            golds.append(gold_index(k))
            meta.append(
                {
                    "split": sp,
                    "twin_id": k.twin_id,
                    "twin_edit": k.twin_edit,
                    "twin_wording": k.twin_wording,
                    "scenario": scenario_of(k.dgp.family, k.dgp.perturbation),
                    "effect": k.dgp.effect,
                    "discordance": k.discordance.score,
                    "answerable": k.abstain_target is None,
                    "cell": k.policy_cell,
                }
            )
    arr = {c: np.asarray(v, float) for c, v in cols.items()}
    return TaskTable(
        ids,
        np.asarray(phis),
        arr["ctv"],
        arr["fisher"],
        arr["reject"],
        arr["reject_any"],
        arr["htrue"],
        arr["correct"],
        arr["invalid"],
        arr["strict"],
        np.asarray(golds),
        meta,
        arr["broken"],
    )


@dataclass
class NullTable:
    scenarios: list[str]
    phi: np.ndarray  # (S, F)
    reject: np.ndarray  # (S, A) confirmatory reject, nan if the action would not execute
    htrue: np.ndarray  # (S, A) truth of the executed test's null, nan if unknown/not executed

    def save(self, path: Path) -> None:
        np.savez_compressed(path, phi=self.phi, reject=self.reject, htrue=self.htrue)
        path.with_suffix(".meta.json").write_text(json.dumps({"scenarios": self.scenarios}))

    @classmethod
    def load(cls, path: Path) -> NullTable:
        z = np.load(path)
        return cls(
            json.loads(path.with_suffix(".meta.json").read_text())["scenarios"], z["phi"], z["reject"], z["htrue"]
        )


def _const(text: str) -> Callable[[str], str]:
    def gen(_prompt: str) -> str:
        return text

    return gen


def build_null_table(sims: int = 300, seed: int = 4242) -> NullTable:
    """Fresh effect-free simulations; every catalogue action run through the FULL confirmatory
    agent path (verify -> execute). Used for agent-level false-positive rates of each arm."""
    scen_names, phis, rej, ht = [], [], [], []
    for (fam, pert, nreg), child in zip(NULLCAL_SCENARIOS, np.random.SeedSequence(seed).spawn(len(NULLCAL_SCENARIOS))):
        rng = np.random.default_rng(child)
        for _ in range(sims):
            sc = Scenario(
                fam,
                nreg,
                "null",
                pert,
                "two-sided",
                "hint",
                int(rng.integers(1, 2**31 - 1)),
                0,
                3,
                {"estimand_word": "none"},
            )
            sim = simulate(sc, render.names_for(sc))
            study = Study(
                design=FAMILY_DESIGN[fam],
                roles=sim.roles,
                group_order=sim.group_order,
                description=render.description(sc),
                p0=sim.p0,
            )
            q = render.question(sc)
            prof = profile(sim.df)
            r_row, h_row = [], []
            for a in CATALOGUE:
                text = render_plan(a, study, sim.df, prof).model_dump_json(exclude_none=True)
                res = run_analysis(
                    q, study, sim.df, planner=LLMPlanner(_const(text)), budget=Budget(max_steps=1)
                ).result
                if res.decision is None:
                    r_row.append(math.nan)
                    h_row.append(math.nan)
                else:
                    r_row.append(1.0 if res.decision == "reject" else 0.0)
                    truth = sim.nulls.get(res.hypothesis or "")
                    h_row.append(math.nan if truth is None else float(truth))
            scen_names.append(f"{fam}/{pert}")
            phis.append(featurize(q, study, sim.df, prof))
            rej.append(r_row)
            ht.append(h_row)
    return NullTable(scen_names, np.asarray(phis), np.asarray(rej), np.asarray(ht))
