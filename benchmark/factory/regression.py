"""Regression and survival task families (model-based designs) with SPEC-level answer keys.

Families (perturbations):
  reg_linear  continuous outcome     (none, heteroskedastic, clustered, confounded)
  reg_binary  0/1 outcome            (none, clustered, confounded)
  reg_iv      endogenous treatment + randomised offer (instrument)   (none)
  reg_multiarm  several arms, joint / equality restriction          (none, clustered)
  reg_survgroup survival by group (log-rank / Cox with group dummy)  (none)
  reg_cox     survival, continuous exposure + covariates             (none, confounded)
  reg_panel   repeated measures on individuals                       (none)

Questions follow the P-Bench layout (research question, H0/H1, study design, variable glossary) so
the agent must read the specification from text: which variables to adjust for, the clustering
level, the instrument, the restriction. Nothing in the text names a method or an SE type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from benchmark.factory.dgp import EFFECT_Z, Scenario
from nanostat.schemas import Alternative, Design, MethodRun, ModelSpec, OutcomeType
from nanostat.tools import REGISTRY, ToolError
from nanostat.tools.base import RunContext

REG_FAMILIES = ("reg_linear", "reg_binary", "reg_iv", "reg_multiarm", "reg_survgroup", "reg_cox", "reg_panel")
REG_PERTURBATIONS: dict[str, tuple[str, ...]] = {
    "reg_linear": ("none", "heteroskedastic", "clustered", "confounded"),
    "reg_binary": ("none", "clustered", "confounded"),
    "reg_iv": ("none",),
    "reg_multiarm": ("none", "clustered"),
    "reg_survgroup": ("none",),
    "reg_cox": ("none", "confounded"),
    "reg_panel": ("none",),
}
REG_N = {"small": (120, 220), "medium": (300, 600), "large": (800, 1400)}
ALL_SE = ["classical", "HC0", "HC1", "HC2", "HC3"]
ROBUST_SE = ["HC0", "HC1", "HC2", "HC3"]

DOMAINS: list[dict[str, Any]] = [
    {
        "unit": "students",
        "y": "test_score",
        "y_desc": "Standardised test score",
        "yb": "passed_exam",
        "yb_desc": "Passed the end-of-year exam (1 = yes)",
        "treat": "tutoring",
        "treat_desc": "Assigned to tutoring (1 = yes)",
        "x": "study_hours",
        "x_desc": "Weekly study hours",
        "c1": "age",
        "c1_desc": "Age in years",
        "c2": "family_income",
        "c2_desc": "Family income (thousands)",
        "cl": "school_id",
        "cl_desc": "School identifier",
        "cl_unit": "school",
        "z": "lottery_offer",
        "z_desc": "Won the tutoring lottery offer (1 = yes)",
        "time": "months_to_dropout",
        "event": "dropped_out",
        "event_desc": "Dropped out during follow-up (1 = yes, 0 = censored)",
        "id": "student_id",
        "arm": "arm",
        "setting": "secondary schools",
    },
    {
        "unit": "patients",
        "y": "sbp",
        "y_desc": "Systolic blood pressure (mmHg)",
        "yb": "readmitted",
        "yb_desc": "Readmitted within 30 days (1 = yes)",
        "treat": "new_drug",
        "treat_desc": "Received the new drug (1 = yes)",
        "x": "dose_mg",
        "x_desc": "Daily dose (mg)",
        "c1": "age",
        "c1_desc": "Age in years",
        "c2": "bmi",
        "c2_desc": "Body-mass index",
        "cl": "clinic_id",
        "cl_desc": "Clinic identifier",
        "cl_unit": "clinic",
        "z": "trial_offer",
        "z_desc": "Randomly offered trial enrolment (1 = yes)",
        "time": "months_to_relapse",
        "event": "relapsed",
        "event_desc": "Relapsed during follow-up (1 = yes, 0 = censored)",
        "id": "patient_id",
        "arm": "arm",
        "setting": "outpatient clinics",
    },
    {
        "unit": "workers",
        "y": "log_earnings",
        "y_desc": "Log monthly earnings",
        "yb": "employed",
        "yb_desc": "Employed at follow-up (1 = yes)",
        "treat": "training",
        "treat_desc": "Completed job training (1 = yes)",
        "x": "experience",
        "x_desc": "Years of experience",
        "c1": "age",
        "c1_desc": "Age in years",
        "c2": "schooling",
        "c2_desc": "Years of schooling",
        "cl": "district_id",
        "cl_desc": "District identifier",
        "cl_unit": "district",
        "z": "voucher",
        "z_desc": "Randomly received a training voucher (1 = yes)",
        "time": "months_to_job",
        "event": "found_job",
        "event_desc": "Found a job during follow-up (1 = yes, 0 = censored)",
        "id": "worker_id",
        "arm": "arm",
        "setting": "a labour-market programme",
    },
]


@dataclass
class RegTask:
    df: pd.DataFrame
    question: str
    design: Design
    outcome: OutcomeType
    h0_true: bool
    nulls: dict[str, bool]
    admissible: list[tuple[str, ModelSpec, dict[str, Any]]]  # (method, canonical spec, requirements)
    invalid: list[tuple[str, ModelSpec, dict[str, Any], str]]  # (method, representative spec, pattern, reason)
    meta: dict[str, Any] = field(default_factory=dict)


def _n(sc: Scenario, rng: np.random.Generator) -> int:
    lo, hi = REG_N[sc.n_regime]
    return int(rng.integers(lo, hi + 1))


def _sign(sc: Scenario, rng: np.random.Generator) -> float:
    return {"greater": 1.0, "less": -1.0}.get(sc.alternative, float(rng.choice([-1.0, 1.0])))


def _h1(alt: str, two: str, up: str, down: str) -> str:
    return {"greater": up, "less": down}.get(alt, two)


def _render(
    d: dict[str, Any], rq: str, h0: str, h1: str, design_lines: list[str], glossary: list[tuple[str, str]]
) -> str:
    out = ["Research Question:", rq, "", "Hypothesis:", f"H0: {h0}", f"H1: {h1}", "", "STUDY DESIGN:"]
    out += [f"- {x}" for x in design_lines]
    out += ["", "VARIABLE GLOSSARY:"] + [f"- {c}: {desc}" for c, desc in glossary]
    return "\n".join(out)


def simulate_reg(sc: Scenario) -> RegTask:
    rng = np.random.default_rng(sc.seed)
    d = DOMAINS[sc.domain % len(DOMAINS)]
    z = EFFECT_Z[sc.effect]
    sgn = _sign(sc, rng)
    h0 = sc.effect == "null"
    n = _n(sc, rng)
    fam, pert = sc.family, sc.perturbation
    alt = sc.alternative
    adjust = bool(sc.extra.get("adjust", rng.random() < 0.5))

    if fam in ("reg_linear", "reg_binary"):
        clustered = pert == "clustered"
        G = int(rng.integers(30, 61)) if clustered else 0
        c1 = rng.normal(40, 10, n)
        c2 = rng.normal(0, 1, n)
        if clustered:
            g = rng.integers(0, G, n)
            treat_g = rng.permutation(np.r_[np.ones(G // 2), np.zeros(G - G // 2)])
            x = treat_g[g]
            u = rng.normal(0, 0.8, G)[g]
        else:
            g = None
            u = np.zeros(n)
            base = 0.08 * (c1 - 40) if pert == "confounded" else 0.0
            if pert == "heteroskedastic":  # unbalanced arms, larger variance in the SMALLER arm: classical SE fails
                base = math.log(0.25 / 0.75)
            x = (rng.random(n) < 1 / (1 + np.exp(-base))).astype(float)
        deff = 1 + (n / max(G, 1) - 1) * 0.4 if clustered else 1.0
        cols = {d["treat"]: x, d["c1"]: np.round(c1, 1), d["c2"]: np.round(c2, 3)}
        conf = 0.06 * (c1 - 40) if pert == "confounded" else 0.0
        if fam == "reg_linear":
            sd_e = np.where(x == 1, 3.0, 1.0) if pert == "heteroskedastic" else np.ones(n)
            se_b = math.sqrt(np.mean(sd_e**2) * deff * 4 / n)
            b = sgn * z * se_b
            y = 1.0 + b * x + 0.3 * c2 + conf + u + rng.normal(0, 1, n) * sd_e
            cols[d["y"]] = np.round(y, 4)
            yname, ydesc, otype = d["y"], d["y_desc"], OutcomeType.continuous
        else:
            se_b = math.sqrt(deff * 4 * 4.4 / n)
            b = sgn * z * se_b
            eta = -0.2 + b * x + 0.3 * c2 + conf + u
            cols[d["yb"]] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
            yname, ydesc, otype = d["yb"], d["yb_desc"], OutcomeType.binary
        if clustered:
            assert g is not None
            cols[d["cl"]] = np.array([f"{d['cl_unit'][0].upper()}{i:03d}" for i in g])
        df = pd.DataFrame(cols)
        required = [d["c1"]] if pert == "confounded" else ([d["c1"], d["c2"]] if adjust else [])
        covs = [d["c1"], d["c2"]] if (adjust or pert == "confounded") else []
        design_lines = [f"Unit of observation: {d['unit'][:-1]}", f"Setting: {d['setting']}"]
        if clustered:
            design_lines += [
                f"Randomisation level: {d['cl_unit']} (all {d['unit']} in a {d['cl_unit']} share the assignment)"
            ]
        else:
            design_lines += [
                "Assignment: "
                + (
                    "not randomised; older participants were more likely to be treated"
                    if pert == "confounded"
                    else "individually randomised"
                )
            ]
        if required:
            design_lines += [f"Analysis: adjusted for {', '.join(required)}"]
        gl = [
            (yname, f"{ydesc} [OUTCOME]"),
            (d["treat"], f"{d['treat_desc']} [TREATMENT]"),
            (d["c1"], d["c1_desc"]),
            (d["c2"], d["c2_desc"]),
        ]
        if clustered:
            gl.append((d["cl"], d["cl_desc"]))
        what = (
            "the probability of " + ydesc.split(" (")[0].lower()
            if fam == "reg_binary"
            else ydesc.split(" (")[0].lower()
        )
        q = _render(
            d,
            f"Does {d['treat'].replace('_', ' ')} change {what} among {d['unit']}?",
            f"{d['treat']} has no effect on {yname}",
            _h1(
                alt, f"{d['treat']} affects {yname}", f"{d['treat']} increases {yname}", f"{d['treat']} reduces {yname}"
            ),
            design_lines,
            gl,
        )
        se_ok = ["cluster"] if clustered else (ROBUST_SE if (pert == "heteroskedastic") else ALL_SE)
        req = {
            "exposure": [d["treat"]],
            "covariates_required": required,
            "se_allowed": se_ok,
            "cluster_required": d["cl"] if clustered else None,
        }
        canon = ModelSpec(
            outcome=yname,
            exposure=[d["treat"]],
            covariates=covs,
            **(
                {"se": "cluster", "cluster": d["cl"]}
                if clustered
                else ({"se": "HC1"} if pert == "heteroskedastic" else {})
            ),
        )
        adm: list[tuple[str, ModelSpec, dict[str, Any]]] = []
        inv: list[tuple[str, ModelSpec, dict[str, Any], str]] = []
        if fam == "reg_linear":
            adm.append(("ols_coef", canon, req))
            if clustered:
                adm.append(
                    (
                        "mixed_lm",
                        ModelSpec(outcome=yname, exposure=[d["treat"]], covariates=covs, group=d["cl"]),
                        {"exposure": [d["treat"]], "covariates_required": required, "group_required": d["cl"]},
                    )
                )
        else:
            for m in ("logistic", "probit"):
                adm.append((m, canon, {**req, "se_allowed": ["cluster"] if clustered else ALL_SE}))
            lpm = canon.model_copy(update={"se": "cluster" if clustered else "HC1"})
            adm.append(("ols_coef", lpm, {**req, "se_allowed": ["cluster"] if clustered else ROBUST_SE}))
            inv.append(("tobit", canon, {"exposure": [d["treat"]], "pattern": "any"}, "outcome_type_mismatch"))
        main = adm[0][0]
        if clustered:
            inv.append(
                (
                    main,
                    canon.model_copy(update={"se": "classical", "cluster": None}),
                    {"exposure": [d["treat"]], "pattern": "se_in", "se_in": ALL_SE},
                    "ignores_clustering",
                )
            )
        if pert == "heteroskedastic":
            inv.append(
                (
                    main,
                    canon.model_copy(update={"se": "classical"}),
                    {"exposure": [d["treat"]], "pattern": "se_in", "se_in": ["classical"]},
                    "ignores_heteroskedasticity",
                )
            )
        if pert == "confounded":
            inv.append(
                (
                    main,
                    canon.model_copy(update={"covariates": []}),
                    {"exposure": [d["treat"]], "pattern": "missing_covariate", "covariates": [d["c1"]]},
                    "omitted_confounder",
                )
            )
        return RegTask(
            df, q, Design.regression, otype, h0, {"coef_zero": h0}, adm, inv, {"clustered": clustered, "adjust": adjust}
        )

    if fam == "reg_iv":
        zz = (rng.random(n) < 0.5).astype(float)
        u = rng.normal(0, 1, n)
        dd = ((0.2 + 0.5 * zz + 0.3 * u + rng.normal(0, 0.3, n)) > 0.5).astype(float)
        c1 = rng.normal(40, 10, n)
        se_b = math.sqrt(4 / (n * 0.25))
        b = sgn * z * se_b
        y = 1 + b * dd + 0.02 * (c1 - 40) + 0.9 * u + rng.normal(0, 1, n)
        df = pd.DataFrame(
            {d["z"]: zz.astype(int), d["treat"]: dd.astype(int), d["c1"]: np.round(c1, 1), d["y"]: np.round(y, 4)}
        )
        gl = [
            (d["y"], f"{d['y_desc']} [OUTCOME]"),
            (d["treat"], f"{d['treat_desc']} (actual take-up) [TREATMENT]"),
            (d["z"], d["z_desc"]),
            (d["c1"], d["c1_desc"]),
        ]
        q = _render(
            d,
            f"What is the effect of actually taking up {d['treat'].replace('_', ' ')} on "
            f"{d['y_desc'].split(' (')[0].lower()}?",
            f"take-up of {d['treat']} has no effect on {d['y']}",
            _h1(
                alt,
                f"take-up of {d['treat']} affects {d['y']}",
                f"take-up of {d['treat']} increases {d['y']}",
                f"take-up of {d['treat']} reduces {d['y']}",
            ),
            [
                f"Unit of observation: {d['unit'][:-1]}",
                f"The offer ({d['z']}) was randomised; take-up was voluntary",
                "Take-up is likely related to unobserved motivation",
            ],
            gl,
        )
        canon = ModelSpec(outcome=d["y"], exposure=[d["treat"]], instruments=[d["z"]], se="HC1")
        adm = [("iv_2sls", canon, {"exposure": [d["treat"]], "instruments_required": [d["z"]], "se_allowed": ALL_SE})]
        inv = [
            (
                "ols_coef",
                ModelSpec(outcome=d["y"], exposure=[d["treat"]], se="HC1"),
                {"exposure": [d["treat"]], "pattern": "any"},
                "endogenous_regressor",
            )
        ]
        return RegTask(df, q, Design.regression, OutcomeType.continuous, h0, {"coef_zero": h0}, adm, inv)

    if fam == "reg_multiarm":
        clustered = pert == "clustered"
        G = int(rng.integers(30, 61)) if clustered else 0
        if clustered:
            g = rng.integers(0, G, n)
            arm_g = rng.integers(0, 3, G)
            arm = arm_g[g]
            u = rng.normal(0, 0.7, G)[g]
        else:
            g = None
            arm = rng.integers(0, 3, n)
            u = np.zeros(n)
        deff = 1 + (n / max(G, 1) - 1) * 0.35 if clustered else 1.0
        se_b = math.sqrt(deff * 2 * 3 / n)
        b = z * se_b
        mode = sc.extra.get("restriction") or str(rng.choice(["joint_zero", "equal"]))
        if h0:  # joint_zero: both zero; equal: B = C (a common non-zero effect keeps the null non-trivial)
            effB = effC = 0.0 if mode == "joint_zero" else 0.5 * se_b
        else:
            effB, effC = (b, 0.5 * b) if mode == "joint_zero" else (b / 2, -b / 2)
        y = 1 + effB * (arm == 1) + effC * (arm == 2) + u + rng.normal(0, 1, n)
        cols = {"arm_b": (arm == 1).astype(int), "arm_c": (arm == 2).astype(int), d["y"]: np.round(y, 4)}
        if clustered:
            assert g is not None
            cols[d["cl"]] = np.array([f"{d['cl_unit'][0].upper()}{i:03d}" for i in g])
        df = pd.DataFrame(cols)
        gl = [
            (d["y"], f"{d['y_desc']} [OUTCOME]"),
            ("arm_b", "Assigned to programme B (1 = yes) [TREATMENT]"),
            ("arm_c", "Assigned to programme C (1 = yes) [TREATMENT]"),
        ]
        if clustered:
            gl.append((d["cl"], d["cl_desc"]))
        if mode == "joint_zero":
            rq, h0s, h1s = (
                "Do programmes B and C change the outcome relative to the comparison arm A?",
                "the B and C coefficients are both zero",
                "at least one of the B and C coefficients is non-zero",
            )
        else:
            rq, h0s, h1s = (
                "Do programmes B and C have the same effect?",
                "the B and C coefficients are equal (B = C)",
                "the B and C coefficients differ",
            )
        lines = [f"Unit of observation: {d['unit'][:-1]}", "Three arms: A (comparison), B, C"]
        if clustered:
            lines.append(f"Randomisation level: {d['cl_unit']}")
        q = _render(d, rq, h0s, h1s, lines, gl)
        spec = ModelSpec(
            outcome=d["y"],
            exposure=["arm_b", "arm_c"],
            restriction=mode,  # type: ignore[arg-type]  # "joint_zero" | "equal" from a fixed choice
            **({"se": "cluster", "cluster": d["cl"]} if clustered else {}),
        )
        req = {
            "exposure": ["arm_b", "arm_c"],
            "restriction": mode,
            "se_allowed": ["cluster"] if clustered else ALL_SE,
            "cluster_required": d["cl"] if clustered else None,
        }
        adm = [("ols_wald", spec, req)]
        inv = []
        if clustered:
            inv.append(
                (
                    "ols_wald",
                    spec.model_copy(update={"se": "classical", "cluster": None}),
                    {"exposure": ["arm_b", "arm_c"], "pattern": "se_in", "se_in": ALL_SE},
                    "ignores_clustering",
                )
            )
        return RegTask(
            df,
            q,
            Design.regression,
            OutcomeType.continuous,
            h0,
            {"restriction_true": h0},
            adm,
            inv,
            {"restriction": mode},
        )

    if fam in ("reg_survgroup", "reg_cox"):
        c1 = rng.normal(55, 10, n)
        if fam == "reg_survgroup":
            x = (rng.random(n) < 0.5).astype(float)
        else:
            x = rng.normal(0, 1, n) + (0.05 * (c1 - 55) if pert == "confounded" else 0.0)
        conf = 0.04 * (c1 - 55) if pert == "confounded" else 0.0
        events_exp = 0.65 * n
        b = sgn * z * math.sqrt((4 if fam == "reg_survgroup" else 1) / events_exp)
        lam = 0.05 * np.exp(b * x + conf)
        t_ev = rng.exponential(1 / lam)
        t_c = rng.uniform(5, 40, n)
        t = np.round(np.minimum(t_ev, t_c), 1) + 0.1
        ev = (t_ev <= t_c).astype(int)
        xname = "high_risk_group" if fam == "reg_survgroup" else d["x"]
        df = pd.DataFrame(
            {
                d["time"]: t,
                d["event"]: ev,
                xname: x.astype(int) if fam == "reg_survgroup" else np.round(x, 3),
                d["c1"]: np.round(c1, 1),
            }
        )
        gl = [
            (d["time"], "Follow-up time in months [OUTCOME]"),
            (d["event"], f"{d['event_desc']} [OUTCOME]"),
            (
                xname,
                "Membership of the high-risk group (1 = yes) [EXPOSURE]"
                if fam == "reg_survgroup"
                else f"{d['x_desc']} (standardised) [EXPOSURE]",
            ),
            (d["c1"], d["c1_desc"]),
        ]
        lines = [f"Unit of observation: {d['unit'][:-1]}", "Observational cohort with right-censored follow-up"]
        if pert == "confounded":
            lines.append(f"Analysis: adjusted for {d['c1']}")
        q = _render(
            d,
            f"Is {xname.replace('_', ' ')} associated with time to {d['event'].replace('_', ' ')}?",
            f"time to {d['event']} does not depend on {xname}",
            _h1(
                alt if fam == "reg_cox" else "two-sided",
                f"time to {d['event']} differs by {xname}",
                f"{xname} is associated with a higher hazard of {d['event']}",
                f"{xname} is associated with a lower hazard of {d['event']}",
            ),
            lines,
            gl,
        )
        base_s = {"time": d["time"], "event": d["event"]}
        if fam == "reg_survgroup":
            adm = [
                (
                    "log_rank",
                    ModelSpec(**base_s, group=xname),
                    {"exposure": [], "group_required": xname, "time_required": d["time"], "event_required": d["event"]},
                ),
                (
                    "cox_ph",
                    ModelSpec(**base_s, exposure=[xname]),
                    {"exposure": [xname], "time_required": d["time"], "event_required": d["event"]},
                ),
            ]
            inv = [
                (
                    "ols_coef",
                    ModelSpec(outcome=d["time"], exposure=[xname]),
                    {"exposure": [xname], "pattern": "any"},
                    "ignores_censoring",
                )
            ]
            nulls = {"survival_equal": h0, "hr_one": h0}
        else:
            req_c = [d["c1"]] if pert == "confounded" else []
            adm = [
                (
                    "cox_ph",
                    ModelSpec(**base_s, exposure=[d["x"]], covariates=req_c),
                    {
                        "exposure": [d["x"]],
                        "covariates_required": req_c,
                        "time_required": d["time"],
                        "event_required": d["event"],
                    },
                )
            ]
            inv = [
                (
                    "ols_coef",
                    ModelSpec(outcome=d["time"], exposure=[d["x"]]),
                    {"exposure": [d["x"]], "pattern": "any"},
                    "ignores_censoring",
                )
            ]
            if pert == "confounded":
                inv.append(
                    (
                        "cox_ph",
                        ModelSpec(**base_s, exposure=[d["x"]]),
                        {"exposure": [d["x"]], "pattern": "missing_covariate", "covariates": [d["c1"]]},
                        "omitted_confounder",
                    )
                )
            nulls = {"hr_one": h0}
        return RegTask(df, q, Design.survival, OutcomeType.time_to_event, h0, nulls, adm, inv)

    if fam == "reg_panel":
        n_ind = int(rng.integers(40, 90)) if sc.n_regime != "large" else int(rng.integers(120, 200))
        T = 3
        ids = np.repeat(np.arange(n_ind), T)
        wave = np.tile(np.arange(T), n_ind)
        treat = np.repeat((rng.random(n_ind) < 0.5).astype(float), T)
        post = (wave > 0).astype(float)
        u = np.repeat(rng.normal(0, 1.2, n_ind), T)
        se_b = 2.3 * math.sqrt(2 * 2 * 1.0 / (n_ind * (T - 1)))  # calibrated so "large" lands near z = 4.5
        b = sgn * z * se_b
        y = 5 + 0.2 * wave + b * treat * post + u + rng.normal(0, 1, n_ind * T)
        df = pd.DataFrame(
            {
                d["id"]: ids,
                "wave": wave,
                d["treat"]: treat.astype(int),
                "post": post.astype(int),
                "treat_post": (treat * post).astype(int),
                d["y"]: np.round(y, 4),
            }
        )
        gl = [
            (d["y"], f"{d['y_desc']} [OUTCOME]"),
            ("treat_post", f"{d['treat']} x post-baseline wave [TARGET]"),
            (d["treat"], d["treat_desc"]),
            ("post", "Post-baseline wave (1 = yes)"),
            ("wave", "Survey wave (0, 1, 2)"),
            (d["id"], f"{d['unit'][:-1].capitalize()} identifier"),
        ]
        q = _render(
            d,
            f"Does {d['treat'].replace('_', ' ')} change {d['y_desc'].split(' (')[0].lower()} after baseline?",
            "the treatment-by-post coefficient is zero",
            _h1(
                alt,
                "the treatment-by-post coefficient is non-zero",
                "the treatment-by-post coefficient is positive",
                "the treatment-by-post coefficient is negative",
            ),
            [f"Each {d['unit'][:-1]} was measured in three waves", "Treatment assigned at the individual level"],
            gl,
        )
        covs = [d["treat"], "post"]
        adm = [
            (
                "mixed_lm",
                ModelSpec(outcome=d["y"], exposure=["treat_post"], covariates=covs, group=d["id"]),
                {"exposure": ["treat_post"], "covariates_required": covs, "group_required": d["id"]},
            ),
            (
                "ols_coef",
                ModelSpec(outcome=d["y"], exposure=["treat_post"], covariates=covs, se="cluster", cluster=d["id"]),
                {
                    "exposure": ["treat_post"],
                    "covariates_required": covs,
                    "se_allowed": ["cluster"],
                    "cluster_required": d["id"],
                },
            ),
        ]
        inv = [
            (
                "ols_coef",
                ModelSpec(outcome=d["y"], exposure=["treat_post"], covariates=covs),
                {"exposure": ["treat_post"], "pattern": "se_in", "se_in": ALL_SE},
                "ignores_repeated_measures",
            )
        ]
        return RegTask(df, q, Design.regression, OutcomeType.continuous, h0, {"coef_zero": h0}, adm, inv)

    raise ValueError(fam)


def run_spec(
    method: str, spec: ModelSpec, df: pd.DataFrame, alt: Alternative, design: Design
) -> tuple[float | None, float | None, float | None]:
    tool = REGISTRY[method]
    a = Alternative.two_sided if tool.two_sided_only else alt
    try:
        out = tool.run(df, RunContext(columns={}, alternative=a, design=design, model=spec))
    except (ToolError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None, None, None
    return out.p_value, out.statistic, out.effect


def reference_runs(t: RegTask, alt: Alternative) -> tuple[list[MethodRun], list[MethodRun]]:
    adm, inv = [], []
    for method, spec, req in t.admissible:
        p, s, e = run_spec(method, spec, t.df, alt, t.design)
        adm.append(
            MethodRun(
                method=method,
                config={**req, "model": spec.model_dump(exclude_defaults=True), "alternative": alt.value},
                p=p,
                statistic=s,
                effect=e,
            )
        )
    for method, spec, pat, reason in t.invalid:
        p, s, e = run_spec(method, spec, t.df, alt, Design.regression if method != "cox_ph" else t.design)
        inv.append(
            MethodRun(
                method=method,
                config={**pat, "model": spec.model_dump(exclude_defaults=True)},
                p=p,
                statistic=s,
                effect=e,
                reason=reason,
            )
        )
    return adm, inv
