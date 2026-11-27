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


def _outlier_df(two_group):
    df = two_group.copy()
    df.loc[df.index[-1], "sbp"] = 400.0
    df.loc[df.index[-2], "sbp"] = 420.0
    return df


def test_invalid_method_rejected_in_confirmatory_mode(two_group):
    df = _outlier_df(two_group)
    q = "Is there a difference in sbp between control and treatment?"
    rep = verify(_plan(method="student_t"), df, profile(df), TWO_STUDY, q)
    assert not rep.ok
    assert any(e.startswith("invalid_method:student_t:outlier_driven_inference") for e in rep.errors)


def test_invalid_method_runs_only_in_diagnostic_mode_without_decision(two_group):
    from nanostat.controller import run_analysis
    from nanostat.planner import LLMPlanner

    df = _outlier_df(two_group)
    q = "Is there a difference in sbp between control and treatment?"
    rep = verify(_plan(method="student_t"), df, profile(df), TWO_STUDY, q, mode="diagnostic")
    assert rep.ok and rep.method_status.value == "invalid"
    plan = _plan(method="student_t").model_dump_json()
    rec = run_analysis(q, TWO_STUDY, df, planner=LLMPlanner(lambda _p: plan), mode="diagnostic")
    assert rec.result.p_value is not None and rec.result.decision is None and not rec.result.confirmatory
    rec2 = run_analysis(q, TWO_STUDY, df, planner=LLMPlanner(lambda _p: plan))
    assert rec2.result.action.value == "abstain" and rec2.result.reason_code.value == "no_admissible_method"


def test_estimand_mismatch_rejected_in_confirmatory_mode(two_group):
    q = "Is the typical sbp different between control and treatment?"
    rep = verify(_plan(method="welch_t"), two_group, profile(two_group), TWO_STUDY, q)
    assert not rep.ok and any(e.startswith("invalid_method:welch_t:estimand_mismatch") for e in rep.errors)


def test_estimand_lock_mean_vs_location(two_group):
    q_mean = "Is the mean sbp different between control and treatment?"
    rep = verify(_plan(method="mann_whitney"), two_group, profile(two_group), TWO_STUDY, q_mean)
    assert not rep.ok
    rep = verify(_plan(method="welch_t", estimand="mean_difference"), two_group, profile(two_group), TWO_STUDY, q_mean)
    assert rep.ok and rep.hypothesis == "mean_equal"
    rep = verify(_plan(method="welch_t", estimand="location_shift"), two_group, profile(two_group), TWO_STUDY, q_mean)
    assert not rep.ok and any(e.startswith("method_estimand_mismatch") for e in rep.errors)


def test_protocol_fields_are_locked(two_group):
    q = "Is there a difference in sbp between control and treatment?"
    for extra in ({"alpha": 0.1}, {"params": {"p0": 0.5}}, {"alternative": "less"}):
        rep = verify(_plan(**extra), two_group, profile(two_group), TWO_STUDY, q)
        assert not rep.ok, extra
    rep = verify(_plan(alpha=0.05), two_group, profile(two_group), TWO_STUDY, q)  # restating is fine
    assert rep.ok


def test_gate2_trimming_changes_estimand(two_group):
    q = "Is the mean sbp different between control and treatment?"
    p = _plan(preprocess=[{"op": "trim", "column": "sbp", "p": 0.1}])
    rep = verify(p, two_group, profile(two_group), TWO_STUDY, q)
    assert any(e.startswith("estimand_changed_by_preprocessing") for e in rep.errors)


def test_gate2_filter_changes_population(two_group):
    q = "Is there a difference in sbp between control and treatment?"
    p = _plan(preprocess=[{"op": "filter", "column": "pid", "cmp": "<", "value": 30}])
    rep = verify(p, two_group, profile(two_group), TWO_STUDY, q)
    assert any(e.startswith("protocol_violation:population") for e in rep.errors)
    open_study = TWO_STUDY.model_copy(update={"allow_population_filters": True})
    rep = verify(p, two_group, profile(two_group), open_study, q)
    assert rep.ok and rep.analysed is not None and len(rep.analysed) < len(two_group)
    assert rep.analysed_sha256 is not None


def test_gate2_recode_breaking_pairing(paired_long):
    q = "Did the score increase from pre to post?"
    cols = {"y": "score", "group": "time", "pair_id": "id"}
    # recoding ids on BOTH timepoints keeps pairing intact
    ok_map = {str(i): str(1000 + i) for i in range(100, 124)}
    p = _plan(
        design="paired",
        method="paired_t",
        columns=cols,
        alternative="greater",
        preprocess=[{"op": "recode", "column": "id", "mapping": ok_map}],
    )
    assert verify(p, paired_long, profile(paired_long), PAIRED_STUDY, q).ok
    # collapsing all ids to one value breaks pairing in the analysed rows -> gate 2 rejects
    bad_map = {str(i): "same" for i in range(100, 124)}
    p2 = _plan(
        design="paired",
        method="paired_t",
        columns=cols,
        alternative="greater",
        preprocess=[{"op": "recode", "column": "id", "mapping": bad_map}],
    )
    rep2 = verify(p2, paired_long, profile(paired_long), PAIRED_STUDY, q)
    assert not rep2.ok


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
