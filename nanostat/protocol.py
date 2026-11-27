"""Protocol lock (SPEC v3.1 §2.2) and estimand/hypothesis bookkeeping (§2.3, §2.6).

The protocol (alpha, direction, estimand family, null value, population) is fixed by the task or
the user before any plan is made. It is derived deterministically from the study metadata and the
question text; the model can never change it.
"""

from __future__ import annotations

import re

from nanostat.claims import infer_alternative
from nanostat.schemas import Design, Estimand, EstimandFamily, Protocol, Study

_MEAN = (r"\bmeans?\b", r"\baverage\b")
_TYPICAL = (r"\btypical\b", r"\bmedian\b", r"\bdistributions?\b")


def infer_estimand_family(question: str) -> EstimandFamily | None:
    q = question.lower()
    mean = any(re.search(p, q) for p in _MEAN)
    typ = any(re.search(p, q) for p in _TYPICAL)
    if mean and not typ:
        return "mean"
    if typ and not mean:
        return "typical"
    return None


_H1 = re.compile(r"^\s*H1\s*[:.]\s*(.+)$", re.I | re.M)


def claim_text(question: str) -> str:
    """If the question states an alternative hypothesis ('H1: ...'), direction and estimand are read
    from that line only; otherwise from the whole question."""
    m = _H1.search(question)
    return m.group(1) if m else question


def build_protocol(question: str, study: Study) -> Protocol:
    question = claim_text(question)
    explicit = study.alternative is not None or study.estimand_family is not None
    return Protocol(
        alpha=study.alpha,
        alternative=study.alternative or infer_alternative(question),
        estimand_family=study.estimand_family or infer_estimand_family(question),
        p0=study.p0,
        allow_population_filters=study.allow_population_filters,
        source="declared" if explicit else "derived",
    )


# Estimands that answer each family, per design. None (family not locked) = every estimand.
FAMILY_ESTIMANDS: dict[Design, dict[str, set[Estimand]]] = {
    Design.independent_two_group: {
        "mean": {Estimand.mean_difference},
        "typical": {Estimand.location_shift, Estimand.trimmed_mean_difference},
    },
    Design.paired: {
        "mean": {Estimand.mean_difference_paired},
        "typical": {Estimand.location_shift},
    },
    Design.k_group: {
        "mean": {Estimand.group_means_equal},
        "typical": {Estimand.group_distributions_equal},
    },
}


def allowed_estimands(design: Design, protocol: Protocol) -> set[Estimand] | None:
    """Estimands the protocol permits for this design; None means unrestricted."""
    if protocol.estimand_family is None:
        return None
    fam = FAMILY_ESTIMANDS.get(design)
    if fam is None:
        return None  # family words carry no restriction for this design
    return fam[protocol.estimand_family]


# Hypothesis whose truth decides whether a rejection is a false positive (§2.6).
_HYP: dict[tuple[Design, Estimand], str] = {
    (Design.independent_two_group, Estimand.mean_difference): "mean_equal",
    (Design.independent_two_group, Estimand.location_shift): "distribution_equal",
    (Design.independent_two_group, Estimand.trimmed_mean_difference): "trimmed_mean_equal",
    (Design.independent_two_group, Estimand.association): "independence",
    (Design.independent_two_group, Estimand.odds_ratio): "independence",
    (Design.paired, Estimand.mean_difference_paired): "mean_diff_zero",
    (Design.paired, Estimand.location_shift): "symmetric_about_zero",
    (Design.correlation, Estimand.correlation): "rho_zero",
    (Design.correlation, Estimand.rank_correlation): "independence",
    (Design.k_group, Estimand.group_means_equal): "means_equal",
    (Design.k_group, Estimand.group_distributions_equal): "distributions_equal",
    (Design.one_sample, Estimand.proportion): "p_equals_p0",
    (Design.regression, Estimand.coefficient): "coef_zero",
    (Design.regression, Estimand.linear_restriction): "restriction_true",
    (Design.survival, Estimand.hazard_ratio): "hr_one",
    (Design.survival, Estimand.linear_restriction): "restriction_true",
    (Design.survival, Estimand.survival_curves_equal): "survival_equal",
}


def hypothesis_id(design: Design, estimand: Estimand | str | None) -> str | None:
    if estimand is None:
        return None
    return _HYP.get((design, Estimand(estimand)))
