from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.factory.grid import run_factory


@pytest.fixture(scope="session")
def smoke_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("smoke")
    run_factory("smoke", out, seed=20261126)
    return out


@pytest.fixture
def paired_long() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 24
    pre = rng.normal(50, 10, n)
    post = pre + rng.normal(3, 4, n)
    ids = np.arange(100, 100 + n)
    return pd.DataFrame({"id": np.r_[ids, ids], "time": ["pre"] * n + ["post"] * n, "score": np.r_[pre, post]})


@pytest.fixture
def two_group() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 25
    return pd.DataFrame(
        {
            "pid": np.arange(2 * n),
            "arm": ["control"] * n + ["treatment"] * n,
            "sbp": np.r_[rng.normal(140, 10, n), rng.normal(133, 10, n)],
            "event": np.r_[(rng.random(n) < 0.3).astype(int), (rng.random(n) < 0.5).astype(int)],
        }
    )
