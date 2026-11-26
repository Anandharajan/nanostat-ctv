"""Data contracts (SPEC §5). Pydantic v2, extra="forbid", strict enums."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class Action(str, Enum):
    execute = "execute"
    clarify = "clarify"
    abstain = "abstain"


class ReasonCode(str, Enum):
    missing_pair_id = "missing_pair_id"
    design_unspecified = "design_unspecified"
    unsupported_design = "unsupported_design"
    unsupported_method = "unsupported_method"
    insufficient_data = "insufficient_data"
    ambiguous_outcome = "ambiguous_outcome"
    unknown_column = "unknown_column"
    invalid_design = "invalid_design"
    missing_null_value = "missing_null_value"


class Design(str, Enum):
    independent_two_group = "independent_two_group"
    paired = "paired"
    correlation = "correlation"
    k_group = "k_group"
    one_sample = "one_sample"


class OutcomeType(str, Enum):
    continuous = "continuous"
    binary = "binary"


class Alternative(str, Enum):
    two_sided = "two-sided"
    greater = "greater"
    less = "less"


class Estimand(str, Enum):
    mean_difference = "mean_difference"
    mean_difference_paired = "mean_difference_paired"
    location_shift = "location_shift"
    trimmed_mean_difference = "trimmed_mean_difference"
    correlation = "correlation"
    rank_correlation = "rank_correlation"
    group_means_equal = "group_means_equal"
    group_distributions_equal = "group_distributions_equal"
    association = "association"
    odds_ratio = "odds_ratio"
    proportion = "proportion"


class Check(str, Enum):
    normality = "normality"
    equal_variance = "equal_variance"
    outliers = "outliers"
    expected_counts = "expected_counts"
    sample_size = "sample_size"
    pairing = "pairing"
    skewness = "skewness"


# ---- preprocess mini-DSL (SPEC §6.3) -------------------------------------------------


class DropMissing(_Strict):
    op: Literal["drop_missing"]
    scope: Literal["rows", "pairs"] = "rows"


class Filter(_Strict):
    op: Literal["filter"]
    column: str
    cmp: Literal["==", "!=", "<", "<=", ">", ">="]
    value: Union[float, int, str]


class LogOp(_Strict):
    op: Literal["log", "log1p", "standardize"]
    column: str


class Recode(_Strict):
    op: Literal["recode"]
    column: str
    mapping: dict[str, Union[str, int, float]]


class SelectLevels(_Strict):
    op: Literal["select_levels"]
    column: str
    levels: list[Union[str, int]]


class TrimOp(_Strict):
    op: Literal["winsorize", "trim"]
    column: str
    p: float = Field(gt=0.0, lt=0.5)


PreprocessOp = Annotated[Union[DropMissing, Filter, LogOp, Recode, SelectLevels, TrimOp], Field(discriminator="op")]


# ---- typed plan (SPEC §5.1) -----------------------------------------------------------

ROLE_NAMES = {"y", "group", "y1", "y2", "pair_id", "x", "cluster"}


class Plan(_Strict):
    action: Action
    estimand: Estimand | None = None
    design: Design | None = None
    method: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    alternative: Alternative = Alternative.two_sided
    preprocess: list[PreprocessOp] = Field(default_factory=list)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    checks: list[Check] = Field(default_factory=list)
    params: dict[str, float] = Field(default_factory=dict)
    reason_code: ReasonCode | None = None
    rationale: str | None = Field(default=None, max_length=600)

    @field_validator("columns")
    @classmethod
    def _roles(cls, v: dict[str, str]) -> dict[str, str]:
        bad = set(v) - ROLE_NAMES
        if bad:
            raise ValueError(f"unknown column roles: {sorted(bad)}")
        return v

    @model_validator(mode="after")
    def _action_fields(self) -> Plan:
        if self.action == Action.execute:
            if not self.method or self.design is None:
                raise ValueError("execute requires design and method")
            if self.reason_code is not None:
                raise ValueError("execute must not carry reason_code")
        else:
            if self.reason_code is None:
                raise ValueError(f"{self.action.value} requires reason_code")
        return self


# ---- study metadata + task records (SPEC §5.2) -----------------------------------------


class Study(_Strict):
    """Agent-visible design metadata. `design` is None for no_hint prompts."""

    design: Design | None = None
    outcome: OutcomeType | None = None
    roles: dict[str, str] = Field(default_factory=dict)  # role -> column, declared by the user
    variables: dict[str, str] = Field(default_factory=dict)  # column -> description
    group_order: list[str] | None = None  # [reference, comparison]
    description: str = ""
    p0: float | None = Field(default=None, gt=0.0, lt=1.0)

    @field_validator("roles")
    @classmethod
    def _roles(cls, v: dict[str, str]) -> dict[str, str]:
        bad = set(v) - ROLE_NAMES
        if bad:
            raise ValueError(f"unknown column roles: {sorted(bad)}")
        return v


class Budget(_Strict):
    max_steps: int = 3
    max_new_tokens: int = 256


class TaskRecord(_Strict):
    task_id: str
    question: str
    study: Study
    data_ref: str
    budget: Budget = Field(default_factory=Budget)


# ---- key record (SPEC §5.3, never agent-visible) ---------------------------------------


class MethodRun(_Strict):
    method: str
    config: dict[str, Any] = Field(default_factory=dict)
    p: float | None = None
    statistic: float | None = None
    effect: float | None = None
    reason: str | None = None


class Discordance(_Strict):
    decision_flip: bool
    z_gap: float
    score: float


class Lineage(_Strict):
    scenario_group: str
    template_group: str
    seed: int
    generator_version: str


class DGP(_Strict):
    family: str
    h0_true: bool
    effect: Literal["null", "borderline", "medium", "large"]
    perturbation: str


class KeyRecord(_Strict):
    task_id: str
    dgp: DGP
    design: Design
    outcome: OutcomeType
    alternative: Alternative
    columns: dict[str, str]
    admissible: list[MethodRun]
    invalid: list[MethodRun]
    preferred: str | None
    reference_decision: Literal["reject", "fail_to_reject"] | None
    discordance: Discordance
    twin_id: str | None = None
    twin_edit: str | None = None
    twin_wording: Literal["fixed", "changed"] | None = None
    lineage: Lineage
    abstain_target: dict[str, str] | None = None  # {"action": ..., "reason_code": ...}
    split: Literal["train", "dev", "test"] | None = None
    policy_cell: str
    required_checks: list[Check] = Field(default_factory=list)


# ---- result (SPEC §5.4) ---------------------------------------------------------------


class MethodStatus(str, Enum):
    admissible = "admissible"
    invalid = "invalid"
    unknown = "unknown"


class AdmissibleRow(_Strict):
    method: str
    statistic: float | None
    p: float | None
    effect: float | None
    error: str | None = None


class Result(_Strict):
    action: Action
    method: str | None = None
    design: Design | None = None
    alternative: Alternative | None = None
    statistic: float | None = None
    p_value: float | None = None
    effect: float | None = None
    effect_label: str | None = None
    ci: tuple[float, float] | None = None
    n: int | None = None
    alpha: float = 0.05
    decision: Literal["reject", "fail_to_reject"] | None = None
    method_status: MethodStatus = MethodStatus.unknown
    status_reason: str | None = None
    policy_cell: str | None = None
    admissible_range: list[AdmissibleRow] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    preprocess_log: list[str] = Field(default_factory=list)
    estimand_changed: bool = False
    reason_code: ReasonCode | None = None
    messages: list[str] = Field(default_factory=list)
