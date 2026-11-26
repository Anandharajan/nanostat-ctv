"""Typed preprocess mini-DSL (SPEC §6.3). No free-form code is ever executed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanostat.schemas import DropMissing, Filter, LogOp, Plan, Recode, SelectLevels, TrimOp


class PreprocessError(ValueError):
    pass


_CMP = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
}


def _need(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise PreprocessError(f"unknown column {col!r}")


def apply_preprocess(df: pd.DataFrame, plan: Plan) -> tuple[pd.DataFrame, list[str], bool]:
    """Apply plan.preprocess in order. Returns (df, log lines, estimand_changed)."""
    out = df.copy()
    log: list[str] = []
    changed = False
    used = [c for c in plan.columns.values() if c in out.columns]
    for op in plan.preprocess:
        n0 = len(out)
        if isinstance(op, DropMissing):
            # 'pairs' on long-format data drops every row of a pair with any missing value
            pid = plan.columns.get("pair_id")
            if op.scope == "pairs" and pid and pid in out.columns:
                bad = out.loc[out[used].isna().any(axis=1), pid].unique()
                out = out[~out[pid].isin(bad)]
            else:
                out = out.dropna(subset=used)
            log.append(f"drop_missing({op.scope}): {n0} -> {len(out)} rows")
        elif isinstance(op, Filter):
            _need(out, op.column)
            out = out[_CMP[op.cmp](out[op.column], op.value)]
            log.append(f"filter({op.column} {op.cmp} {op.value!r}): {n0} -> {len(out)} rows")
        elif isinstance(op, LogOp):
            _need(out, op.column)
            x = pd.to_numeric(out[op.column], errors="raise").astype(float)
            if op.op == "log":
                if (x <= 0).any():
                    raise PreprocessError("log of non-positive values")
                out[op.column] = np.log(x)
            elif op.op == "log1p":
                if (x <= -1).any():
                    raise PreprocessError("log1p of values <= -1")
                out[op.column] = np.log1p(x)
            else:
                out[op.column] = (x - x.mean()) / x.std(ddof=1)
            changed = changed or op.op != "standardize"
            log.append(f"{op.op}({op.column})")
        elif isinstance(op, Recode):
            _need(out, op.column)
            out[op.column] = out[op.column].map(lambda v, m=op.mapping: m.get(str(v), v))
            log.append(f"recode({op.column}, {len(op.mapping)} keys)")
        elif isinstance(op, SelectLevels):
            _need(out, op.column)
            keep = {str(v) for v in op.levels}
            out = out[out[op.column].astype(str).isin(keep)]
            log.append(f"select_levels({op.column}, {sorted(keep)}): {n0} -> {len(out)} rows")
        elif isinstance(op, TrimOp):
            _need(out, op.column)
            x = pd.to_numeric(out[op.column], errors="raise").astype(float)
            lo, hi = x.quantile(op.p), x.quantile(1 - op.p)
            if op.op == "winsorize":
                out[op.column] = x.clip(lo, hi)
            else:
                out = out[(x >= lo) & (x <= hi)]
            changed = True
            log.append(f"{op.op}({op.column}, p={op.p}): {n0} -> {len(out)} rows; estimand changed")
        else:  # pragma: no cover - schema forbids other ops
            raise PreprocessError(f"unsupported op {op!r}")
    return out.reset_index(drop=True), log, changed
