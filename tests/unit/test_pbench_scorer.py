"""Clean-room P-Bench scorer against hand-computed fixtures (SPEC §13, v3.1 Step 3.3)."""

from __future__ import annotations

import math

import pytest

from benchmark.pbench.score import FIXTURE_SHEET, Answer, Trial, grade_trial, score, z_score


def _trials(tid: str, spec):
    out = []
    for i, (p, dec) in enumerate(spec):
        if isinstance(p, str):
            out.append(Trial(tid, i, status=p))
        else:
            out.append(Trial(tid, i, status="ok", p_value=p, decision=dec))
    return out


@pytest.mark.parametrize("fx", FIXTURE_SHEET, ids=lambda f: f["name"])
def test_fixture_sheet(fx):
    a = Answer("t", fx["p_ref"], fx["dec"], "hard", "cat")
    rep = score([a], _trials("t", fx["trials"]))
    assert rep["overall"]["strict_pass1"] == pytest.approx(100 * fx["strict_pass1"])
    assert rep["overall"]["strict_pass3"] == pytest.approx(100 * fx["strict_pass3"])


def test_z_boundaries():
    assert z_score(1.0) == 0.0
    assert math.isfinite(z_score(0.0)) and z_score(0.0) == pytest.approx(z_score(1e-16))
    assert z_score(0.05) == pytest.approx(1.959963984540054)


@pytest.mark.parametrize("status", ["unsupported", "abstained", "failed", "timeout", "malformed"])
def test_non_ok_trials_are_incorrect(status):
    a = Answer("t", 0.01, "reject", "easy", "c")
    assert not any(grade_trial(a, Trial("t", 0, status=status, p_value=0.01, decision="reject")).values())


def test_missing_and_invalid_trials_count_as_incorrect():
    a = Answer("t", 0.01, "reject", "easy", "c")
    rep = score([a], [Trial("t", 0, "ok", 0.01, "reject")])  # 2 trials missing
    assert rep["overall"]["strict_pass1"] == pytest.approx(100 / 3)
    for bad in (float("nan"), -0.1, 1.5, None):
        assert not grade_trial(a, Trial("t", 0, "ok", bad, "reject"))["raw"]
    assert not grade_trial(a, Trial("t", 0, "ok", 0.01, "maybe"))["raw"]


def test_aggregation_and_composite():
    answers = [Answer("e1", 0.01, "reject", "easy", "ttest"), Answer("h1", 0.2, "fail_to_reject", "hard", "cox", True)]
    trials = _trials("e1", [(0.01, "reject")] * 3) + _trials("h1", [(0.6, "fail_to_reject")] * 3)
    rep = score(answers, trials)
    assert rep["difficulty:easy"]["strict_pass1"] == 100.0
    assert rep["difficulty:hard"]["raw_pass1"] == 100.0 and rep["difficulty:hard"]["strict_pass1"] == 0.0
    assert rep["composite_hard_strict_pass1"] == 0.0
    assert rep["perturbed"]["n"] == 1 and rep["clean"]["n"] == 1
    assert rep["category:cox"]["relaxed_pass1"] == 100.0  # dz = 0.757: fails Strict (< 0.5), passes Relaxed (< 1.0)
