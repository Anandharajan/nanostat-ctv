"""z-mapping, reward, twin-joint advantage and validity constraint (SPEC §7, §13)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from benchmark.keys import load_keys
from benchmark.reference import build_reference
from nanostat.schemas import (
    DGP,
    Alternative,
    Design,
    Discordance,
    KeyRecord,
    Lineage,
    MethodRun,
    OutcomeType,
    Study,
    TaskRecord,
)
from nanostat.zscore import z_of_p, zc
from training.constraint import ValidityConstraint, group_has_signal, twin_joint_advantage
from training.reward_server import has_fabricated_number, score_fisher, score_plan
from training.sft_data import gold_plan

# ---- z mapping -------------------------------------------------------------------------------


def test_z_mapping_values():
    assert z_of_p(1.0) == pytest.approx(0.0, abs=1e-12)
    assert z_of_p(0.5) == pytest.approx(0.6744897501960817)
    assert z_of_p(0.05) == pytest.approx(1.959963984540054)
    # independent check: solve erfc(z / sqrt 2) = 1e-10 by bisection
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if math.erfc(mid / math.sqrt(2)) > 1e-10 else (lo, mid)
    assert z_of_p(1e-10) == pytest.approx(lo, rel=1e-9)
    assert math.isinf(z_of_p(0.0))
    assert zc(1e-10) == 5.0 and zc(0.0) == 5.0 and zc(1.0) == 0.0


def test_z_mapping_monotonic():
    ps = np.logspace(-12, 0, 200)
    zs = [zc(p) for p in ps]
    assert all(a >= b for a, b in zip(zs, zs[1:]))
    with pytest.raises(ValueError):
        z_of_p(float("nan"))


# ---- reward ------------------------------------------------------------------------------------


@pytest.fixture
def outlier_task():
    rng = np.random.default_rng(8)
    n = 20
    a = rng.normal(0, 1, n)
    b = rng.normal(0.2, 1, n)
    b[:2] = [9.0, 11.0]
    df = pd.DataFrame({"arm": ["control"] * n + ["treatment"] * n, "y": np.r_[a, b]})
    study = Study(roles={"y": "y", "group": "arm"}, group_order=["control", "treatment"])
    q = "Is there a difference in y between control and treatment?"
    ref = build_reference(
        df, Design.independent_two_group, {"y": "y", "group": "arm"}, Alternative.two_sided, ["control", "treatment"]
    )
    assert ref is not None and ref.perturbation == "heavy_outliers"
    key = KeyRecord(
        task_id="t_x",
        dgp=DGP(family="two_group", h0_true=False, effect="borderline", perturbation="outliers"),
        design=Design.independent_two_group,
        outcome=OutcomeType.continuous,
        alternative=Alternative.two_sided,
        columns={"y": "y", "group": "arm"},
        admissible=ref.admissible,
        invalid=ref.invalid,
        preferred=ref.preferred,
        reference_decision=ref.reference_decision,  # type: ignore[arg-type]
        discordance=Discordance(decision_flip=False, z_gap=0, score=0),
        lineage=Lineage(scenario_group="s", template_group="t", seed=1, generator_version="x"),
        policy_cell=ref.cell.key,
        required_checks=ref.required_checks,
    )
    return key, df, study, q


def _plan_text(method: str, alt: str = "two-sided", checks=None, **extra) -> str:
    d = {
        "action": "execute",
        "design": "independent_two_group",
        "method": method,
        "columns": {"y": "y", "group": "arm"},
        "alternative": alt,
        "checks": checks or [],
    }
    d.update(extra)
    return json.dumps(d)


def test_method_in_A_beats_method_in_B(outlier_task):
    key, df, study, q = outlier_task
    good = score_plan(key, _plan_text("mann_whitney"), df, study, q)
    bad = score_plan(key, _plan_text("student_t"), df, study, q)
    assert good.in_A and bad.in_B
    assert good.reward > 0 > bad.reward == -0.2


def test_admissible_alternative_gets_full_method_credit(outlier_task):
    key, df, study, q = outlier_task
    alt = [a.method for a in key.admissible if a.method != key.preferred][0]
    s = score_plan(key, _plan_text(alt), df, study, q)
    assert s.in_A and s.parts["method"] == 1.0
    assert s.parts["p"] == pytest.approx(1.0)  # its own cached p is in the admissible reference set


def test_fabricated_number_penalised(outlier_task):
    key, df, study, q = outlier_task
    s = score_plan(key, "The result is significant, p = 0.003.\n" + _plan_text("mann_whitney"), df, study, q)
    assert s.reward == -0.2 and s.failure == "fabricated_number"
    assert not has_fabricated_number(_plan_text("mann_whitney"))


def test_invalid_output_scores_zero(outlier_task):
    key, df, study, q = outlier_task
    assert score_plan(key, "not json", df, study, q).reward == 0.0
    assert (
        score_plan(key, _plan_text("mann_whitney", columns={"y": "nope", "group": "arm"}), df, study, q).reward == 0.0
    )


def test_wrong_tail_not_rewarded_as_valid(outlier_task):
    key, df, study, q = outlier_task
    s = score_plan(key, _plan_text("mann_whitney", alt="greater"), df, study, q)
    assert s.reward == 0.0 and not s.valid


def test_assumption_credit(outlier_task):
    key, df, study, q = outlier_task
    lo = score_plan(key, _plan_text("mann_whitney"), df, study, q)
    hi = score_plan(key, _plan_text("mann_whitney", checks=["sample_size"]), df, study, q)
    assert hi.reward - lo.reward == pytest.approx(0.10)


def test_always_abstain_scores_below_uniform_valid(smoke_data):
    keys = load_keys(smoke_data / "keys" / "keys.jsonl")
    tasks = [
        TaskRecord.model_validate_json(x)
        for sp in ("train", "dev", "test")
        for x in (smoke_data / "tasks" / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    abstain_total, valid_total = 0.0, 0.0
    for t in tasks:
        k = keys[t.task_id]
        df = pd.read_csv(smoke_data / t.data_ref)
        abstain = json.dumps({"action": "abstain", "reason_code": "unsupported_design"})
        abstain_total += score_plan(k, abstain, df, t.study, t.question).reward
        valid_total += score_plan(k, gold_plan(k).model_dump_json(), df, t.study, t.question).reward
    assert valid_total > abstain_total
    assert abstain_total < 0


def test_gold_plan_scores_near_max(smoke_data):
    keys = load_keys(smoke_data / "keys" / "keys.jsonl")
    for line in (smoke_data / "tasks" / "train.jsonl").read_text(encoding="utf-8").splitlines()[:25]:
        t = TaskRecord.model_validate_json(line)
        k = keys[t.task_id]
        s = score_plan(k, gold_plan(k).model_dump_json(), pd.read_csv(smoke_data / t.data_ref), t.study, t.question)
        assert s.reward >= 0.99, (t.task_id, s)


def test_fisher_reward_penalises_admissible_alternative_but_ctv_does_not(outlier_task):
    key, df, study, q = outlier_task
    alt = [a for a in key.admissible if a.method != key.preferred][0]
    pref = [a for a in key.admissible if a.method == key.preferred][0]
    f_alt = score_fisher(key, _plan_text(alt.method), df, study, q)
    f_pref = score_fisher(key, _plan_text(pref.method), df, study, q)
    if abs(zc(alt.p) - zc(pref.p)) > 1e-3:  # type: ignore[arg-type]
        assert f_alt.reward < f_pref.reward
    c_alt = score_plan(key, _plan_text(alt.method), df, study, q)
    c_pref = score_plan(key, _plan_text(pref.method), df, study, q)
    assert c_alt.parts["method"] == c_pref.parts["method"] == 1.0


def test_abstention_branch():
    key = KeyRecord(
        task_id="t",
        dgp=DGP(family="paired", h0_true=True, effect="null", perturbation="none"),
        design=Design.paired,
        outcome=OutcomeType.continuous,
        alternative=Alternative.two_sided,
        columns={},
        admissible=[],
        invalid=[],
        preferred=None,
        reference_decision=None,
        discordance=Discordance(decision_flip=False, z_gap=0, score=0),
        lineage=Lineage(scenario_group="s", template_group="t", seed=1, generator_version="x"),
        abstain_target={"action": "clarify", "reason_code": "missing_pair_id"},
        policy_cell="c",
    )
    df = pd.DataFrame({"y": [1.0, 2.0]})
    st = Study()
    assert score_plan(key, '{"action":"clarify","reason_code":"missing_pair_id"}', df, st, "q").reward == 1.0
    assert score_plan(key, '{"action":"clarify","reason_code":"design_unspecified"}', df, st, "q").reward == 0.0


# ---- twin-joint advantage -----------------------------------------------------------------------


def test_twin_joint_advantage():
    adv = twin_joint_advantage([1.0, 0.0], [1.0, 1.0])
    assert adv[0] > 0 > adv[1]
    assert not group_has_signal(twin_joint_advantage([0.5, 0.5], [0.5, 0.5]))
    # joint bonus only when both members succeed
    a = twin_joint_advantage([0.9, 0.9, 0.0], [0.9, 0.0, 0.9])
    assert a[0] > a[1] and a[1] == pytest.approx(a[2])
    with pytest.raises(ValueError):
        twin_joint_advantage([1.0], [1.0, 0.0])


# ---- validity constraint --------------------------------------------------------------------------


def test_lambda1_rises_with_liberal_choices():
    c = ValidityConstraint()
    rng = np.random.default_rng(0)
    for _ in range(100):
        c.update(list((rng.random(16) < 0.25).astype(float)), list((rng.random(16) < 0.9).astype(float)))
    assert c.lam1 > 0.3
    assert 0 <= c.lam1 <= c.lambda_max and 0 <= c.lam2 <= c.lambda_max


def test_lambda1_stays_near_zero_with_valid_choices():
    c = ValidityConstraint()
    rng = np.random.default_rng(1)
    for _ in range(200):
        c.update(list((rng.random(32) < 0.05).astype(float)), list((rng.random(32) < 0.85).astype(float)))
    assert c.lam1 < 0.05
    assert c.lam2 < 0.05


def test_lambda2_rises_when_power_collapses_and_is_bounded():
    c = ValidityConstraint()
    for _ in range(500):
        c.update([0.0] * 8, [0.0] * 8)
    assert c.lam2 == c.lambda_max and c.lam1 == 0.0


def test_shaping():
    c = ValidityConstraint(lam1=0.3, lam2=0.2)
    assert c.shape(1.0, True, True, "null") == pytest.approx(0.7)
    assert c.shape(1.0, True, False, "large") == pytest.approx(1.2)
    assert c.shape(1.0, True, False, "borderline") == pytest.approx(1.0)
    assert c.shape(1.0, False, True, "null") == pytest.approx(1.0)
    s = c.state_dict()
    c2 = ValidityConstraint()
    c2.load_state_dict(s)
    assert c2.state_dict() == s


def test_discordance_score():
    from benchmark.discordance import discordance, sampling_weight

    A = [MethodRun(method="mann_whitney", p=0.08)]
    B = [MethodRun(method="student_t", p=0.003)]
    d = discordance(A, B)
    assert d.decision_flip and d.z_gap > 1 and 0.5 < d.score <= 1
    assert discordance(A, []).score == 0.0
    same = discordance([MethodRun(method="a", p=0.5)], [MethodRun(method="b", p=0.5)])
    assert same.score == 0.0 and not same.decision_flip
    assert sampling_weight(0.0) == pytest.approx(0.1)
