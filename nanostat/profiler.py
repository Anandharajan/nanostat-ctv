"""Dataset profiler. Its output is agent-visible: summaries only, never p-values or keys."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nanostat.diagnostics import robust_z

MAX_LEVELS = 20


def column_kind(s: pd.Series) -> str:
    nn = s.dropna()
    if nn.nunique() == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(nn) and not pd.api.types.is_bool_dtype(nn):
        return "numeric"
    num = pd.to_numeric(nn, errors="coerce")
    if len(nn) and num.notna().all():
        return "numeric"
    return "categorical"


def id_like(s: pd.Series, n_rows: int) -> dict[str, Any] | None:
    """Detect identifier columns. Returns repeat structure, e.g. each value appears twice."""
    nn = s.dropna()
    if len(nn) < 4:
        return None
    vc = nn.astype(str).value_counts()
    if int(vc.max()) > 4 or len(vc) < max(4, n_rows / 4):
        return None
    if pd.api.types.is_float_dtype(nn):
        return None  # continuous measurements, not identifiers
    if int(vc.max()) == 1:
        return {"repeats": 1, "n_ids": int(len(vc))}
    rep = int(vc.median())
    if rep < 2:
        return None
    return {"repeats": rep, "n_ids": int(len(vc))}


def profile(df: pd.DataFrame) -> dict[str, Any]:
    cols: list[dict[str, Any]] = []
    for name in df.columns:
        s = df[name]
        kind = column_kind(s)
        info: dict[str, Any] = {
            "name": str(name),
            "kind": kind,
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique()),
        }
        if kind in ("binary", "categorical") and s.nunique() <= MAX_LEVELS:
            info["levels"] = sorted(s.dropna().astype(str).unique().tolist())
        if kind == "numeric":
            x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
            if len(x):
                info.update(
                    mean=round(float(x.mean()), 4),
                    sd=round(float(x.std(ddof=1)), 4) if len(x) > 1 else 0.0,
                    min=round(float(x.min()), 4),
                    max=round(float(x.max()), 4),
                    max_abs_robust_z=round(float(np.abs(robust_z(x)).max()), 2) if len(x) > 2 else 0.0,
                )
        idl = id_like(s, len(df))
        if idl and kind != "binary":
            info["id_like"] = idl
        cols.append(info)
    return {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": cols}


def column_names(prof: dict[str, Any]) -> set[str]:
    return {c["name"] for c in prof["columns"]}


def column_info(prof: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((c for c in prof["columns"] if c["name"] == name), None)
