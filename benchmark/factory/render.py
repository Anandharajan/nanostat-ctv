"""Question/metadata rendering from fixed templates. Answer-bearing wording is never used:
templates do not name methods, perturbations or the correct design label (no_hint)."""

from __future__ import annotations

from typing import Any

from benchmark.factory.dgp import Scenario

DOMAINS: list[dict[str, Any]] = [
    {
        "y": "sbp",
        "y_desc": "systolic blood pressure (mmHg)",
        "outcome": "systolic blood pressure",
        "group": "arm",
        "ref": "control",
        "comp": "treatment",
        "id": "patient_id",
        "x": "sodium_mg",
        "x_desc": "daily sodium intake",
        "event": "hospital readmission",
        "unit": "patients",
        "levels": ["clinic_a", "clinic_b", "clinic_c", "clinic_d"],
        "groups": "clinics",
        "pre": "baseline",
        "post": "week_8",
        "change": "the intervention",
    },
    {
        "y": "score",
        "y_desc": "exam score (points)",
        "outcome": "exam score",
        "group": "section",
        "ref": "lecture",
        "comp": "flipped",
        "id": "student_id",
        "x": "study_hours",
        "x_desc": "weekly study hours",
        "event": "course completion",
        "unit": "students",
        "levels": ["school_1", "school_2", "school_3", "school_4"],
        "groups": "schools",
        "pre": "term_start",
        "post": "term_end",
        "change": "the tutoring programme",
    },
    {
        "y": "yield_t",
        "y_desc": "crop yield (t/ha)",
        "outcome": "crop yield",
        "group": "fertiliser",
        "ref": "standard",
        "comp": "biochar",
        "id": "plot_id",
        "x": "rain_mm",
        "x_desc": "seasonal rainfall",
        "event": "pest infestation",
        "unit": "plots",
        "levels": ["north", "south", "east", "west"],
        "groups": "regions",
        "pre": "season_1",
        "post": "season_2",
        "change": "the new irrigation schedule",
    },
    {
        "y": "latency_ms",
        "y_desc": "response latency (ms)",
        "outcome": "response latency",
        "group": "build",
        "ref": "v1",
        "comp": "v2",
        "id": "server_id",
        "x": "cpu_load",
        "x_desc": "CPU load",
        "event": "request timeout",
        "unit": "servers",
        "levels": ["dc_1", "dc_2", "dc_3", "dc_4"],
        "groups": "data centres",
        "pre": "before_patch",
        "post": "after_patch",
        "change": "the patch",
    },
    {
        "y": "pain",
        "y_desc": "pain score (0-100 VAS)",
        "outcome": "pain score",
        "group": "therapy",
        "ref": "usual_care",
        "comp": "physio",
        "id": "subject_id",
        "x": "age",
        "x_desc": "age",
        "event": "symptom recurrence",
        "unit": "participants",
        "levels": ["site_a", "site_b", "site_c", "site_d"],
        "groups": "sites",
        "pre": "week_0",
        "post": "week_12",
        "change": "the therapy",
    },
]

N_TEMPLATE_VARIANTS = 2


def names_for(sc: Scenario) -> dict[str, Any]:
    d = dict(DOMAINS[sc.domain % len(DOMAINS)])
    if sc.family == "paired" and not sc.extra.get("keep_arm_labels"):
        d = {**d, "ref": d["pre"], "comp": d["post"], "group": "timepoint"}
    return d


def _dir(alt: str, up: str, down: str, neutral: str) -> str:
    return {"greater": up, "less": down}.get(alt, neutral)


def question(sc: Scenario, variant: int = 0) -> str:
    d = names_for(sc)
    a = sc.alternative
    o, ref, comp = d["outcome"], d["ref"], d["comp"]
    word = sc.extra.get("estimand_word")
    if word in ("mean", "typical") and sc.family in ("two_group", "paired", "k_group"):
        o = f"{word} {o}"  # locks the protocol estimand family (nanostat.protocol)
    if sc.family in ("two_group", "paired") and sc.extra.get("neutral_question"):
        return f"Is there a difference in {o} between {ref} and {comp}?"
    if sc.family == "two_group" and sc.extra.get("dichotomised"):
        e = f"{o} above the reference-group median"
        return _dir(
            a,
            f"Does {comp} increase the proportion of {d['unit']} with {e} compared with {ref}?",
            f"Does {comp} reduce the proportion of {d['unit']} with {e} compared with {ref}?",
            f"Is the proportion of {d['unit']} with {e} different between {ref} and {comp}?",
        )
    if sc.family == "two_group":
        if variant == 0:
            return _dir(
                a,
                f"Does {comp} increase {o} compared with {ref}?",
                f"Does {comp} reduce {o} compared with {ref}?",
                f"Is there a difference in {o} between {ref} and {comp}?",
            )
        return _dir(
            a,
            f"Is {o} higher under {comp} than under {ref}?",
            f"Is {o} lower under {comp} than under {ref}?",
            f"Does {o} differ between the {ref} and {comp} groups?",
        )
    if sc.family == "paired":
        if variant == 0:
            return _dir(
                a,
                f"Did {o} increase from {ref} to {comp}?",
                f"Did {o} decrease from {ref} to {comp}?",
                f"Did {o} change from {ref} to {comp}?",
            )
        return _dir(
            a,
            f"Is {o} higher at {comp} than at {ref}?",
            f"Is {o} lower at {comp} than at {ref}?",
            f"Does {o} differ between {ref} and {comp}?",
        )
    if sc.family == "correlation":
        return _dir(
            a,
            f"Is {d['x_desc']} positively associated with {o}?",
            f"Is {d['x_desc']} negatively associated with {o}?",
            f"Is {d['x_desc']} associated with {o}?",
        )
    if sc.family == "k_group":
        return (
            f"Does {o} differ across the {sc.k} {d['groups']}?"
            if variant == 0
            else f"Are there differences in {o} among the {d['groups']}?"
        )
    if sc.family == "binary_two_group":
        e = d["event"]
        if sc.extra.get("neutral_question"):
            return f"Is there a difference in {o} between {ref} and {comp}?"
        return _dir(
            a,
            f"Does {comp} increase the rate of {e} compared with {ref}?",
            f"Does {comp} reduce the rate of {e} compared with {ref}?",
            f"Is the rate of {e} different between {ref} and {comp}?",
        )
    if sc.family == "one_proportion":
        p0 = sc.extra.get("p0_text", "the reference rate")
        e = d["event"]
        return _dir(
            a,
            f"Is the rate of {e} higher than {p0}?",
            f"Is the rate of {e} lower than {p0}?",
            f"Is the rate of {e} different from {p0}?",
        )
    raise ValueError(sc.family)


def description(sc: Scenario) -> str:
    base = _description(sc)
    if sc.extra.get("outlier_note"):
        base = (base + " Some readings were taken during an equipment fault.").strip()
    return base


def _description(sc: Scenario) -> str:
    d = names_for(sc)
    if sc.prompt_style == "no_hint" and not sc.extra.get("force_description"):
        return ""
    u = d["unit"]
    if sc.family == "two_group" or sc.family == "binary_two_group":
        return f"{u.capitalize()} were randomly assigned to {d['ref']} or {d['comp']}; each was measured once."
    if sc.family == "paired":
        return f"The same {u} were measured at {d['ref']} and at {d['comp']}."
    if sc.family == "correlation":
        return f"One measurement per {u[:-1] if u.endswith('s') else u} of each variable."
    if sc.family == "k_group":
        return f"Independent {u} sampled from {sc.k} {d['groups']}."
    if sc.family == "one_proportion":
        return f"A single sample of {u}; each recorded whether {d['event']} occurred."
    return ""


def variables(sc: Scenario, roles: dict[str, str]) -> dict[str, str]:
    d = names_for(sc)
    out = {}
    for role, col in roles.items():
        if role == "y":
            if sc.family in ("binary_two_group", "one_proportion"):
                out[col] = d["event"] + " (1 = yes)"
            elif sc.extra.get("dichotomised"):
                out[col] = d["outcome"] + " above the reference-group median (1 = yes)"
            else:
                out[col] = d["y_desc"]
        elif role == "x":
            out[col] = d["x_desc"]
        elif role == "group":
            out[col] = "group label"
        elif role == "pair_id":
            out[col] = "identifier"
    return out
