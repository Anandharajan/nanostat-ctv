"""Typed-plan action catalogue for the proxy policy. Agent-visible inputs only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from nanostat.planner import roles_for
from nanostat.schemas import Action, Design, Plan, ReasonCode, Study
from nanostat.tools import REGISTRY


@dataclass(frozen=True)
class ActionSpec:
    kind: str  # execute | clarify | abstain
    design: Design | None = None
    method: str | None = None
    reason: ReasonCode | None = None

    @property
    def name(self) -> str:
        if self.kind == "execute":
            assert self.design is not None
            return f"{self.design.value}:{self.method}"
        assert self.reason is not None
        return f"{self.kind}:{self.reason.value}"


def build_catalogue() -> list[ActionSpec]:
    # The proxy policy has no model-specification head, so model-based designs (regression / survival)
    # are not in its action space; those tasks are skipped by the ablation tables.
    model_designs = (Design.regression, Design.survival)
    acts = [
        ActionSpec("execute", d, m)
        for m, tool in sorted(REGISTRY.items())
        for d in tool.designs
        if d not in model_designs
    ]
    acts.append(ActionSpec("clarify", reason=ReasonCode.missing_pair_id))
    acts.append(ActionSpec("abstain", reason=ReasonCode.no_admissible_method))
    return acts


CATALOGUE: list[ActionSpec] = build_catalogue()


def render_plan(a: ActionSpec, study: Study, df: pd.DataFrame, prof: dict[str, Any]) -> Plan:
    """Turn an action into a typed plan. Columns come from the user's declared roles; the
    protocol fields (alpha, tail, estimand family, null value) are left to the locked protocol."""
    if a.kind != "execute":
        return Plan(action=Action(a.kind), reason_code=a.reason)
    assert a.design is not None
    return Plan(action=Action.execute, design=a.design, method=a.method, columns=roles_for(a.design, study, df, prof))
