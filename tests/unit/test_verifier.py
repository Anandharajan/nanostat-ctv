"""Verifier rejects design/tail/outcome/pairing/column errors (SPEC §13)."""

from __future__ import annotations

from nanostat.profiler import profile
from nanostat.schemas import Action, Plan, Study
from nanostat.verifier import verify

PAIRED_STUDY = Study(roles={"y": "score", "group": "time", "pair_id": "id"}, group_order=["pre", "post"])
TWO_STUDY = Study(roles={"y": "sbp", "group": "arm"}, group_order=["control", "treatment"])


def _plan(**kw):
    base = dict(
        action="execute",
        design="independent_two_group",
        method="welch_t",
        columns={"y": "sbp", "group": "arm"},
        alternative="two-sided",
    )
    base.update(kw)
    return Plan.model_validate(base)


def test_accepts_valid_paired(paired_long):
    p = _plan(
        design="paired",
        method="paired_t",
        columns={"y": "score", "group": "time", "pair_id": "id"},
        alternative="greater",
    )
    rep = verify(p, paired_long, profile(paired_long), PAIRED_STUDY, "Did the score increase from pre to post?")
    assert rep.ok, rep.errors
    assert rep.method_status.value == "admissible"


def test_rejects_paired_data_analysed_as_independent(paired_long):
    p = _plan(columns={"y": "score", "group": "time"}, alternative="greater")
    st = Study(roles={"y": "score", "group": "time"}, group_order=["pre", "post"])
    rep = verify(p, paired_long, profile(paired_long), st, "Did the score increase from pre to post?")
    assert not rep.ok
    assert any(e.startswith("design_mismatch") for e in rep.errors)


def test_rejects_wrong_tail(two_group):
    p = _plan(alternative="greater")
    rep = verify(p, two_group, profile(two_group), TWO_STUDY, "Does treatment reduce sbp compared with control?")
    assert not rep.ok
    assert any(e.startswith("wrong_tail") for e in rep.errors)


def test_rejects_one_sided_claim_tested_two_sided(two_group):
    p = _plan(alternative="two-sided")
    rep = verify(p, two_group, profile(two_group), TWO_STUDY, "Does treatment reduce sbp compared with control?")
    assert any(e.startswith("wrong_tail") for e in rep.errors)


def test_rejects_wrong_outcome_type(two_group):
    p = _plan(columns={"y": "event", "group": "arm"})
    st = Study(roles={"y": "event", "group": "arm"}, group_order=["control", "treatment"])
    rep = verify(p, two_group, profile(two_group), st, "Is the rate of event different between arms?")
    assert not rep.ok
    assert any(e.startswith("outcome_type_mismatch") for e in rep.errors)


def test_rejects_missing_pair_id(paired_long):
    df = paired_long.drop(columns="id")
    st = Study(
        roles={"y": "score", "group": "time"},
        group_order=["pre", "post"],
        description="The same participants were measured at pre and at post.",
    )
    p = _plan(design="paired", method="paired_t", columns={"y": "score", "group": "time"})
    rep = verify(p, df, profile(df), st, "Did the score change from pre to post?")
    assert not rep.ok
    assert "missing_pair_id" in rep.errors
    # and the correct action is to clarify
    clar = Plan(action=Action.clarify, reason_code="missing_pair_id")
    rep2 = verify(clar, df, profile(df), st, "Did the score change from pre to post?")
    assert rep2.ok and rep2.expected_reason is not None and rep2.expected_reason.value == "missing_pair_id"


def test_rejects_unknown_column(two_group):
    p = _plan(columns={"y": "systolic", "group": "arm"})
    rep = verify(p, two_group, profile(two_group), TWO_STUDY, "Is there a difference in sbp?")
    assert not rep.ok
    assert any(e.startswith("unknown_column") for e in rep.errors)


def test_rejects_unsupported_method(two_group):
    p = _plan(method="bayes_factor_magic")
    rep = verify(p, two_group, profile(two_group), TWO_STUDY, "Is there a difference in sbp?")
    assert not rep.ok and rep.errors[0].startswith("unsupported_method")


def test_forbids_filtering_on_outcome(two_group):
    p = _plan(preprocess=[{"op": "filter", "column": "sbp", "cmp": "<", "value": 150}])
    rep = verify(p, two_group, profile(two_group), TWO_STUDY, "Is there a difference in sbp?")
    assert any(e.startswith("forbidden") for e in rep.errors)


def test_marks_invalid_method_under_outliers(two_group):
    df = two_group.copy()
    df.loc[df.arm == "treatment", "sbp"] = df.loc[df.arm == "treatment", "sbp"].to_numpy()
    df.loc[df.index[-1], "sbp"] = 400.0
    df.loc[df.index[-2], "sbp"] = 420.0
    p = _plan(method="student_t")
    rep = verify(p, df, profile(df), TWO_STUDY, "Is there a difference in sbp between control and treatment?")
    assert rep.ok
    assert rep.method_status.value == "invalid"
    assert rep.status_reason == "outlier_driven_inference"


def test_plan_schema_is_strict():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Plan.model_validate({"action": "execute", "design": "paired", "method": "paired_t", "code": "import os"})
    with pytest.raises(ValidationError):
        Plan.model_validate({"action": "clarify"})  # clarify needs a reason_code
    with pytest.raises(ValidationError):
        Plan.model_validate(
            {
                "action": "execute",
                "design": "paired",
                "method": "paired_t",
                "preprocess": [{"op": "python", "src": "x"}],
            }
        )
