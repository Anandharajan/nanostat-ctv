from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanostat.dsl import PreprocessError, apply_preprocess
from nanostat.policy import load_policies
from nanostat.schemas import Plan


def _plan(pre):
    return Plan.model_validate(
        {
            "action": "execute",
            "design": "independent_two_group",
            "method": "welch_t",
            "columns": {"y": "y", "group": "g"},
            "preprocess": pre,
        }
    )


def test_dsl_ops_logged_and_estimand_flag():
    df = pd.DataFrame({"g": list("aabb") * 5, "y": np.arange(20, dtype=float) + 1, "z": [1, None] * 10})
    out, log, changed = apply_preprocess(df, _plan([{"op": "drop_missing"}]))
    assert len(out) == 20 and not changed and log  # z is not a plan column
    out, log, changed = apply_preprocess(
        df, _plan([{"op": "winsorize", "column": "y", "p": 0.1}, {"op": "log", "column": "y"}])
    )
    assert changed and len(log) == 2
    with pytest.raises(PreprocessError):
        apply_preprocess(df.assign(y=df.y - 5), _plan([{"op": "log", "column": "y"}]))


def test_policy_registry_loads_and_is_consistent():
    cells = load_policies()
    assert len(cells) >= 15
    for c in cells.values():
        assert c.preferred() is not None
        assert not set(c.invalid) & set(c.admissible())
        assert c.reviewed_by == "PENDING" and not c.confirmatory_ok
    # directional claims drop two-sided-only tools from A(x)
    from nanostat.schemas import Alternative

    c = cells["independent_two_group/binary/none"]
    assert c.admissible(Alternative.greater) == ["fisher_exact"]


def test_binary_outcome_without_variation_abstains_cleanly():
    """Regression: chi2 on a table with an all-zero outcome column raised ValueError (pilot factory)."""
    from nanostat.controller import run_analysis
    from nanostat.schemas import Study
    from nanostat.tools import ToolError, get_tool
    from nanostat.tools.base import RunContext

    df = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 6, "y": [0] * 12})
    with pytest.raises(ToolError):
        get_tool("chi2_independence").run(df, RunContext(columns={"y": "y", "group": "g"}))
    df2 = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 6, "y": [0] * 11 + [1]})
    rec = run_analysis("Is the rate different between a and b?", Study(roles={"y": "y", "group": "g"}), df2)
    assert rec.result.p_value is not None or rec.result.action.value == "abstain"


def test_constant_outcome_abstains_instead_of_running_a_t_test():
    """Regression (null calibration): all-zero binary outcome fell through to welch_t with p = NaN."""
    from nanostat.controller import run_analysis
    from nanostat.schemas import Study

    df = pd.DataFrame({"g": ["a"] * 8 + ["b"] * 8, "y": [0] * 16})
    rec = run_analysis("Is the rate different between a and b?", Study(roles={"y": "y", "group": "g"}), df)
    assert rec.result.action.value == "abstain"
    assert rec.result.reason_code is not None and rec.result.reason_code.value == "insufficient_data"
    assert rec.result.p_value is None


def test_outlier_null_dgp_is_symmetric():
    """Regression (null calibration): one-sided contamination made 'null' outlier scenarios non-null."""
    from benchmark.factory import render
    from benchmark.factory.dgp import Scenario, simulate

    sc = Scenario("two_group", "medium", "null", "outliers", "two-sided", "hint", 5, 0, 3, {})
    sim = simulate(sc, render.names_for(sc))
    g = sim.df[sim.roles["group"]]
    y = sim.df[sim.roles["y"]]
    for lv in g.unique():
        x = y[g == lv]
        assert ((x - x.median()).abs() > 5 * (x - x.median()).abs().median()).any()  # every group contaminated
