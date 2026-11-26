"""Agent-level null calibration (SPEC §9.1): re-run the FULL agent on fresh true-null
simulations per scenario and report the reject rate with a Clopper-Pearson 95% CI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.factory import render
from benchmark.factory.dgp import Scenario, simulate
from benchmark.factory.grid import FAMILY_DESIGN
from evaluation.metrics import clopper_pearson
from nanostat.controller import run_analysis
from nanostat.planner import Planner, RulePlanner
from nanostat.schemas import Study

NULLCAL_SCENARIOS: list[tuple[str, str, str]] = [
    ("two_group", "none", "small"),
    ("two_group", "outliers", "medium"),
    ("two_group", "skewed", "small"),
    ("two_group", "unequal_variance", "small"),
    ("paired", "none", "small"),
    ("paired", "outliers", "medium"),
    ("correlation", "none", "medium"),
    ("correlation", "outliers", "medium"),
    ("k_group", "skewed", "small"),
    ("binary_two_group", "small_counts", "small"),
]


@dataclass
class NullCalRow:
    family: str
    perturbation: str
    n_regime: str
    sims: int
    rejects: int
    executed: int
    fpr: float | None
    ci95: tuple[float, float]
    methods: dict[str, int]


def run_nullcal(
    sims: int = 1000,
    planner: Planner | None = None,
    scenarios: list[tuple[str, str, str]] | None = None,
    seed: int = 777,
) -> list[NullCalRow]:
    planner = planner or RulePlanner()
    rows = []
    ss = np.random.SeedSequence(seed)
    for (fam, pert, nreg), child in zip(scenarios or NULLCAL_SCENARIOS, ss.spawn(len(scenarios or NULLCAL_SCENARIOS))):
        rng = np.random.default_rng(child)
        rej = ex = 0
        methods: dict[str, int] = {}
        for _ in range(sims):
            sc = Scenario(fam, nreg, "null", pert, "two-sided", "hint", int(rng.integers(1, 2**31 - 1)), 0, 3, {})
            sim = simulate(sc, render.names_for(sc))
            study = Study(
                design=FAMILY_DESIGN[fam],
                roles=sim.roles,
                group_order=sim.group_order,
                description=render.description(sc),
                p0=sim.p0,
            )
            rec = run_analysis(render.question(sc), study, sim.df, planner=planner)
            r = rec.result
            if r.p_value is not None:
                ex += 1
                methods[r.method or "?"] = methods.get(r.method or "?", 0) + 1
                rej += int(r.decision == "reject")
        rows.append(
            NullCalRow(
                fam, pert, nreg, sims, rej, ex, rej / sims if sims else None, clopper_pearson(rej, sims), methods
            )
        )
    return rows
