"""Counterfactual design twins (SPEC §7.3).

A twin x' applies exactly one edit to x that should flip the correct method (or action).
Clustering twins need OLS cluster-robust SEs (an M4 plugin) and are not built in v0.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark.factory.dgp import Scenario, Simulated, inject_outliers, simulate

EDITS = ("design_flip", "add_outliers", "tail_flip", "small_counts", "remove_pair_id", "outcome_type")

# Which base families each edit applies to.
EDIT_FAMILIES: dict[str, tuple[str, ...]] = {
    "design_flip": ("two_group",),
    "add_outliers": ("two_group", "paired", "correlation", "k_group"),
    "tail_flip": ("two_group", "paired", "correlation", "binary_two_group", "one_proportion"),
    "small_counts": ("binary_two_group",),
    "remove_pair_id": ("paired",),
    "outcome_type": ("two_group",),
}

# Which wording kinds each edit supports.
EDIT_WORDINGS: dict[str, tuple[str, ...]] = {
    "design_flip": ("fixed", "changed"),
    "add_outliers": ("fixed", "changed"),
    "tail_flip": ("changed",),
    "small_counts": ("fixed",),
    "remove_pair_id": ("fixed",),
    "outcome_type": ("fixed", "changed"),
}


def base_scenario(edit: str, sc: Scenario, wording: str) -> Scenario:
    """Adjust the base scenario so the pair is well-defined for this edit/wording."""
    extra = dict(sc.extra)
    pert = sc.perturbation
    alt = sc.alternative
    if edit in ("design_flip", "outcome_type", "add_outliers", "small_counts"):
        pert = "none"
    if edit == "design_flip":
        extra["keep_arm_labels"] = True
        if wording == "fixed":
            extra["neutral_question"] = True
            alt = "two-sided"
    if edit == "outcome_type" and wording == "fixed":
        extra["neutral_question"] = True
        alt = "two-sided"
    if edit == "small_counts":
        alt = "two-sided" if alt == "two-sided" else alt
    if edit == "remove_pair_id":
        extra["force_description"] = True
        pert = "none"
    return Scenario(sc.family, sc.n_regime, sc.effect, pert, alt, sc.prompt_style, sc.seed, sc.domain, sc.k, extra)


def make_twin(edit: str, sc: Scenario, sim: Simulated, wording: str) -> tuple[Scenario, Simulated]:
    rng = np.random.default_rng(sc.seed + 7919)
    if edit == "design_flip":
        fam = "paired" if sc.family == "two_group" else "two_group"
        sc2 = Scenario(
            fam,
            sc.n_regime,
            sc.effect,
            sc.perturbation,
            sc.alternative,
            sc.prompt_style,
            sc.seed,
            sc.domain,
            sc.k,
            dict(sc.extra),
        )
        if wording == "changed" and sc2.prompt_style == "no_hint":
            sc2.extra["force_description"] = True
        return sc2, simulate(sc2, _names(sc2))

    if edit == "add_outliers":
        df = sim.df.copy()
        r = sim.roles
        if sc.family in ("two_group", "k_group"):
            levels = sorted(df[r["group"]].astype(str).unique())
            target = sim.group_order[1] if sim.group_order else levels[-1]
            # under H0 contaminate every group alike so the twin is still a true null
            for lv in levels if sim.h0_true else [target]:
                m = df[r["group"]].astype(str) == lv
                df.loc[m, r["y"]] = np.round(inject_outliers(df.loc[m, r["y"]].to_numpy(float), rng), 3)
        elif sc.family == "paired":
            m = df[r["group"]].astype(str) == sim.group_order[1]  # type: ignore[index]
            df.loc[m, r["y"]] = np.round(inject_outliers(df.loc[m, r["y"]].to_numpy(float), rng), 3)
        elif sc.family == "correlation":
            n = len(df)
            k = max(1, int(round(0.06 * n)))
            idx = rng.choice(n, size=k, replace=False)
            for col in (r["x"], r["y"]):
                x = df[col].to_numpy(float)
                med, sd = np.median(x), np.median(np.abs(x - np.median(x))) / 0.6745 or 1.0
                # under H0 the y-direction is independent of x (no induced association)
                sg = rng.choice([-1.0, 1.0], size=k) if (sim.h0_true and col == r["y"]) else 1.0
                df.loc[idx, col] = np.round(med + sg * rng.uniform(8, 12, size=k) * sd, 3)
        sc2 = Scenario(
            sc.family,
            sc.n_regime,
            sc.effect,
            "outliers",
            sc.alternative,
            sc.prompt_style,
            sc.seed,
            sc.domain,
            sc.k,
            {**sc.extra, "outlier_note": wording == "changed"},
        )
        return sc2, Simulated(df, dict(r), sim.group_order, sim.h0_true, sim.p0, sim.true_effect)

    if edit == "tail_flip":
        if sc.alternative == "two-sided":
            alt = "greater" if sim.true_effect >= 0 else "less"
        else:
            alt = "two-sided"
        sc2 = Scenario(
            sc.family,
            sc.n_regime,
            sc.effect,
            sc.perturbation,
            alt,
            sc.prompt_style,
            sc.seed,
            sc.domain,
            sc.k,
            dict(sc.extra),
        )
        return sc2, sim

    if edit == "small_counts":
        sc2 = Scenario(
            sc.family,
            sc.n_regime,
            sc.effect,
            "small_counts",
            sc.alternative,
            sc.prompt_style,
            sc.seed,
            sc.domain,
            sc.k,
            dict(sc.extra),
        )
        return sc2, simulate(sc2, _names(sc2))

    if edit == "remove_pair_id":
        r = {k: v for k, v in sim.roles.items() if k != "pair_id"}
        df = sim.df.drop(columns=[sim.roles["pair_id"]])
        sc2 = Scenario(
            sc.family,
            sc.n_regime,
            sc.effect,
            sc.perturbation,
            sc.alternative,
            sc.prompt_style,
            sc.seed,
            sc.domain,
            sc.k,
            {**sc.extra, "pair_id_removed": True},
        )
        return sc2, Simulated(df, r, sim.group_order, sim.h0_true, sim.p0, sim.true_effect)

    if edit == "outcome_type":
        df = sim.df.copy()
        y = sim.roles["y"]
        ref = df[sim.roles["group"]].astype(str) == sim.group_order[0]  # type: ignore[index]
        thr = float(np.median(df.loc[ref, y]))
        df[y] = (df[y].to_numpy(float) > thr).astype(int)
        sc2 = Scenario(
            sc.family,
            sc.n_regime,
            sc.effect,
            sc.perturbation,
            sc.alternative,
            sc.prompt_style,
            sc.seed,
            sc.domain,
            sc.k,
            {**sc.extra, "dichotomised": True},
        )
        return sc2, Simulated(df, dict(sim.roles), sim.group_order, sim.h0_true, sim.p0, sim.true_effect)

    raise ValueError(edit)


def _names(sc: Scenario) -> dict[str, Any]:
    from benchmark.factory.render import names_for

    return names_for(sc)
