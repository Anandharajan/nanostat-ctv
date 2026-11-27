"""Agent-level null calibration (SPEC §9.1, v3.1 §2.6).

For each scenario, simulate fresh effect-free datasets and
  * re-run the FULL agent (plan -> verify -> execute) on every simulation, and
  * run each fixed method on the same simulations (CPU-only reference arms).
A rejection counts as a false positive only if the null hypothesis of the test that was actually
run is TRUE for that DGP (e.g. Spearman tests independence, which is false under symmetric
leverage contamination even though rho = 0). Rates carry Clopper-Pearson 95% CIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from benchmark.factory import render
from benchmark.factory.dgp import Scenario, simulate
from benchmark.factory.grid import FAMILY_DESIGN
from evaluation.metrics import clopper_pearson
from nanostat.controller import run_analysis
from nanostat.planner import Planner, RulePlanner
from nanostat.protocol import hypothesis_id
from nanostat.schemas import Alternative, Estimand, Study
from nanostat.tools import REGISTRY, ToolError, get_tool
from nanostat.tools.base import RunContext

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

FIXED_METHODS: dict[str, tuple[str, ...]] = {
    "two_group": ("welch_t", "student_t", "mann_whitney", "permutation_test", "yuen_trimmed_t"),
    "paired": ("paired_t", "wilcoxon_signed_rank", "permutation_test"),
    "correlation": ("pearson", "spearman", "permutation_test"),
    "k_group": ("anova_oneway", "kruskal_wallis"),
    "binary_two_group": ("chi2_independence", "fisher_exact"),
    "one_proportion": ("binomial_test",),
}


@dataclass
class Rate:
    k: int = 0
    n: int = 0

    def add(self, reject: bool) -> None:
        self.k += int(reject)
        self.n += 1

    def summary(self) -> dict[str, Any]:
        lo, hi = clopper_pearson(self.k, self.n)
        return {"k": self.k, "n": self.n, "rate": self.k / self.n if self.n else None, "ci95": [lo, hi]}


@dataclass
class NullCalRow:
    family: str
    perturbation: str
    n_regime: str
    sims: int
    agent_executed: int
    agent_fpr: dict[str, Any]  # rejects among sims where the chosen test's null is true
    agent_raw_reject: dict[str, Any]  # rejects among all executed sims (any hypothesis)
    agent_methods: dict[str, int]
    fixed: dict[str, dict[str, Any]] = field(default_factory=dict)  # method -> {hypothesis, null_true, fpr}


def _method_estimand(method: str, design: Any) -> str:
    for e in REGISTRY[method].estimands:
        if hypothesis_id(design, Estimand(e)) is not None:
            return e
    return REGISTRY[method].estimands[0]


def run_nullcal(
    sims: int = 1000,
    planner: Planner | None = None,
    scenarios: list[tuple[str, str, str]] | None = None,
    seed: int = 777,
    fixed_methods: bool = True,
) -> list[NullCalRow]:
    planner = planner or RulePlanner()
    scen = scenarios or NULLCAL_SCENARIOS
    rows = []
    for (fam, pert, nreg), child in zip(scen, np.random.SeedSequence(seed).spawn(len(scen))):
        rng = np.random.default_rng(child)
        design = FAMILY_DESIGN[fam]
        fpr, raw = Rate(), Rate()
        methods: dict[str, int] = {}
        fixed: dict[str, Rate] = {m: Rate() for m in FIXED_METHODS.get(fam, ())} if fixed_methods else {}
        fixed_hyp: dict[str, str | None] = {}
        fixed_true: dict[str, bool | None] = {}
        executed = 0
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
                design=design,
                roles=sim.roles,
                group_order=sim.group_order,
                description=render.description(sc),
                p0=sim.p0,
            )
            r = run_analysis(render.question(sc), study, sim.df, planner=planner).result
            if r.p_value is not None and r.decision is not None:
                executed += 1
                methods[r.method or "?"] = methods.get(r.method or "?", 0) + 1
                rej = r.decision == "reject"
                raw.add(rej)
                if sim.nulls.get(r.hypothesis or "") is True:
                    fpr.add(rej)
            for m, rate in fixed.items():
                est = _method_estimand(m, design)
                hyp = hypothesis_id(design, Estimand(est))
                fixed_hyp[m], fixed_true[m] = hyp, sim.nulls.get(hyp or "")
                ctx = RunContext(dict(sim.roles), Alternative.two_sided, sim.group_order, sim.p0, 0.05, design)
                try:
                    p = get_tool(m).run(sim.df, ctx).p_value
                except (ToolError, ValueError, FloatingPointError):
                    continue
                if np.isfinite(p) and sim.nulls.get(hyp or "") is True:
                    rate.add(p < 0.05)
        rows.append(
            NullCalRow(
                fam,
                pert,
                nreg,
                sims,
                executed,
                fpr.summary(),
                raw.summary(),
                methods,
                {
                    m: {"hypothesis": fixed_hyp.get(m), "null_true": fixed_true.get(m), "fpr": rt.summary()}
                    for m, rt in fixed.items()
                },
            )
        )
    return rows


def rows_to_json(rows: list[NullCalRow]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
