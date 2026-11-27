"""Regression / survival task families: spec-level admissibility, gold plans, spec twins, and a
simulation check that the admissible / invalid SPEC labels are statistically right."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.factory.dgp import Scenario
from benchmark.factory.grid import build_reg_task
from benchmark.factory.regression import REG_FAMILIES, REG_PERTURBATIONS, run_spec, simulate_reg
from benchmark.factory.twins import base_scenario, make_twin
from benchmark.spec import spec_status
from nanostat.schemas import Alternative, ModelSpec
from training.reward_server import score_plan
from training.sft_data import gold_plan


def _task(fam: str, pert: str, seed: int = 11, effect: str = "medium", alt: str = "two-sided"):
    b = build_reg_task(Scenario(fam, "medium", effect, pert, alt, "no_hint", seed, 0, 3, {}), "sg")
    assert not isinstance(b, str), b
    return b


def test_clustered_requires_cluster_robust_se_at_the_right_level():
    b = _task("reg_linear", "clustered")
    k = b.key
    treat = k.gold_model.exposure[0]
    y = k.gold_model.outcome
    assert spec_status(k, "ols_coef", ModelSpec(outcome=y, exposure=[treat]))[0] == "invalid"
    assert spec_status(k, "ols_coef", ModelSpec(outcome=y, exposure=[treat], se="HC3"))[1] == "ignores_clustering"
    covs = k.gold_model.covariates  # the question may also require adjustment variables
    good = ModelSpec(outcome=y, exposure=[treat], covariates=covs, se="cluster", cluster=k.gold_model.cluster)
    assert spec_status(k, "ols_coef", good)[0] == "admissible"
    assert spec_status(k, "ols_coef", good.model_copy(update={"cluster": "age"}))[0] == "unknown"
    assert (
        spec_status(k, "mixed_lm", ModelSpec(outcome=y, exposure=[treat], covariates=covs, group=good.cluster))[0]
        == "admissible"
    )


def test_confounded_requires_the_confounder():
    k = _task("reg_linear", "confounded").key
    m = k.gold_model
    assert spec_status(k, "ols_coef", m)[0] == "admissible"
    assert spec_status(k, "ols_coef", m.model_copy(update={"covariates": []})) == ("invalid", "omitted_confounder")


def test_endogenous_requires_iv():
    k = _task("reg_iv", "none").key
    m = k.gold_model
    assert spec_status(k, "iv_2sls", m)[0] == "admissible"
    assert (
        spec_status(k, "ols_coef", ModelSpec(outcome=m.outcome, exposure=m.exposure, se="HC1"))[1]
        == "endogenous_regressor"
    )
    assert spec_status(k, "iv_2sls", m.model_copy(update={"instruments": []}))[0] != "admissible"


def test_multiarm_restriction_must_match():
    k = _task("reg_multiarm", "none").key
    m = k.gold_model
    other = "equal" if m.restriction == "joint_zero" else "joint_zero"
    assert spec_status(k, "ols_wald", m)[0] == "admissible"
    assert spec_status(k, "ols_wald", m.model_copy(update={"restriction": other}))[0] == "unknown"


@pytest.mark.parametrize("fam", REG_FAMILIES)
def test_gold_plan_scores_at_top_of_reward(fam):
    pert = REG_PERTURBATIONS[fam][-1]
    b = _task(fam, pert, seed=3)
    df = pd.read_csv(pd.io.common.StringIO(b.csv_bytes.decode()))
    plan = gold_plan(b.key, b.task, df)
    s = score_plan(b.key, plan.model_dump_json(exclude_none=True), df, b.task.study, b.task.question)
    assert s.in_A and s.reward >= 0.85, (fam, s)
    for run in b.key.invalid:  # every invalid spec is penalised
        bad = plan.model_copy(update={"method": run.method, "model": ModelSpec(**run.config["model"])})
        sb = score_plan(b.key, bad.model_dump_json(exclude_none=True), df, b.task.study, b.task.question)
        assert sb.reward == pytest.approx(-0.2), (fam, run.reason, sb)


@pytest.mark.parametrize("edit", ["add_clustering", "add_confounder", "make_endogenous"])
def test_spec_twins_change_the_required_specification(edit):
    sc = base_scenario(
        edit, Scenario("reg_linear", "medium", "medium", "none", "two-sided", "no_hint", 21, 1, 3, {}), "changed"
    )
    b1 = build_reg_task(sc, "sg", "a", edit, "changed")
    sc2, _ = make_twin(edit, sc, None, "changed")
    b2 = build_reg_task(sc2, "sg", "b", edit, "changed")
    assert not isinstance(b1, str) and not isinstance(b2, str)
    assert b1.gold != b2.gold
    # the base task's gold spec is NOT admissible on the twin
    st, _ = spec_status(b2.key, b1.key.preferred, b1.key.gold_model)
    assert st != "admissible"


@pytest.mark.parametrize(
    "fam,pert,good,bad",
    [
        ("reg_linear", "clustered", 0, 0),  # cluster-robust OLS vs classical OLS
        ("reg_linear", "confounded", 0, 0),  # adjusted vs unadjusted
        ("reg_iv", "none", 0, 0),  # 2SLS vs naive OLS
    ],
)
def test_spec_labels_hold_under_true_nulls(fam, pert, good, bad):
    """Simulate true nulls: the admissible spec keeps its rejection rate near alpha, the invalid one inflates it."""
    rej_good = rej_bad = 0
    sims = 120
    for s in range(sims):
        t = simulate_reg(Scenario(fam, "small", "null", pert, "two-sided", "no_hint", 1000 + s, 0, 3, {}))
        m_g, spec_g, _ = t.admissible[good]
        m_b, spec_b, _, _ = t.invalid[bad]
        pg = run_spec(m_g, spec_g, t.df, Alternative.two_sided, t.design)[0]
        pb = run_spec(m_b, spec_b, t.df, Alternative.two_sided, t.design)[0]
        rej_good += int(pg is not None and pg < 0.05)
        rej_bad += int(pb is not None and pb < 0.05)
    assert rej_good / sims < 0.12, (fam, pert, rej_good)
    assert rej_bad / sims > 2 * max(rej_good, 1) / sims, (fam, pert, rej_good, rej_bad)
    assert np.isfinite(rej_bad)
