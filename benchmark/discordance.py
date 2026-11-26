"""Discordance score D(x) (SPEC §7.2), precomputed on CPU."""

from __future__ import annotations

from nanostat.schemas import Discordance, MethodRun
from nanostat.zscore import zc

EPS_D = 0.1


def _decision(p: float, alpha: float) -> bool:
    return p < alpha


def discordance(admissible: list[MethodRun], invalid: list[MethodRun], alpha: float = 0.05) -> Discordance:
    A = [r for r in admissible if r.p is not None]
    B = [r for r in invalid if r.p is not None]
    if not A or not B:
        return Discordance(decision_flip=False, z_gap=0.0, score=0.0)
    z_gap = min(abs(zc(a.p) - zc(b.p)) for a in A for b in B)  # type: ignore[arg-type]
    flip = any(_decision(a.p, alpha) != _decision(b.p, alpha) for a in A for b in B)  # type: ignore[arg-type]
    score = 0.5 * min(z_gap / 2.0, 1.0) + 0.5 * float(flip)
    return Discordance(decision_flip=flip, z_gap=round(z_gap, 6), score=round(score, 6))


def sampling_weight(d: float, eps: float = EPS_D) -> float:
    """w(x) proportional to eps_D + D(x)."""
    return eps + d
