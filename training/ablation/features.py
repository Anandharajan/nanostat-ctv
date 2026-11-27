"""Agent-visible features for the proxy policy: study metadata, the question text and data
summaries the profiler already exposes. Never reads keys, p-values or DGP truth."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from nanostat.claims import mentions_pairing
from nanostat.diagnostics import compute_diagnostics
from nanostat.planner import infer_design, roles_for
from nanostat.profiler import column_info
from nanostat.protocol import build_protocol
from nanostat.schemas import Design, Study
from nanostat.tools.base import RunContext
from nanostat.verifier import pair_id_columns

N_HASH = 32
_DESIGNS = [None, *Design]


def _onehot(v: Any, options: list[Any]) -> list[float]:
    return [1.0 if v == o else 0.0 for o in options]


def _bin(x: float | None, edges: list[float]) -> list[float]:
    out = [0.0] * (len(edges) + 2)  # last slot = missing
    if x is None or not np.isfinite(x):
        out[-1] = 1.0
    else:
        out[int(np.searchsorted(edges, x, side="right"))] = 1.0
    return out


def _hash_words(text: str) -> list[float]:
    v = np.zeros(N_HASH)
    for w in re.findall(r"[a-z]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
        v[h % N_HASH] += 1.0 if (h >> 8) & 1 else -1.0
    n = np.linalg.norm(v)
    return list(v / n) if n else list(v)


def featurize(question: str, study: Study, df: pd.DataFrame, prof: dict[str, Any]) -> np.ndarray:
    roles = study.roles
    proto = build_protocol(question, study)
    pid = bool(pair_id_columns(df, prof, study))
    gcol = roles.get("group")
    k = int((column_info(prof, gcol) or {}).get("n_unique", 0)) if gcol else 0
    ycol = roles.get("y") or roles.get("y2")
    binary = bool(ycol and ycol in df.columns and df[ycol].dropna().nunique() == 2)

    # summaries under the design the data most plausibly have (same profiler the LLM would see)
    d = infer_design(study, df, prof)
    diag: dict[str, Any] = {}
    if d is not None:
        ctx = RunContext(roles_for(d, study, df, prof), proto.alternative, study.group_order, proto.p0, design=d)
        diag = compute_diagnostics(df, ctx, d)
    samples = diag.get("samples", {}).values()
    max_rz = max((s.get("max_abs_robust_z", 0.0) for s in samples), default=None)
    max_skew = max((abs(s.get("trimmed_skew", 0.0)) for s in samples), default=None)

    f: list[float] = [1.0]
    f += _onehot(study.design, _DESIGNS)
    f += _onehot(study.outcome.value if study.outcome else None, [None, "continuous", "binary"])
    f += [float("x" in roles), float("pair_id" in roles), float("y1" in roles), float("group" in roles)]
    f += [float(pid), float(binary), float(mentions_pairing(question + " " + study.description))]
    f += _onehot(min(k, 3), [0, 1, 2, 3])
    f += _onehot(proto.estimand_family, [None, "mean", "typical"])
    f += _bin(max_rz, [3.0, 5.0, 8.0])
    f += _bin(max_skew, [0.5, 1.0, 2.0])
    f += _bin(diag.get("var_ratio"), [2.0, 4.0, 9.0])
    f += _bin(diag.get("min_expected"), [5.0, 10.0])
    f += _hash_words(question)
    return np.asarray(f, dtype=float)
