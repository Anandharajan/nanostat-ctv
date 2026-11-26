"""p -> z mapping used by rewards and discordance (SPEC §7.1)."""

from __future__ import annotations

import math

from scipy.stats import norm

Z_CLIP = 5.0


def z_of_p(p: float) -> float:
    """z(p) = Phi^{-1}(1 - p/2) = norm.isf(p/2). p=1 -> 0, p=0 -> inf."""
    if p is None or math.isnan(p):
        raise ValueError("p is NaN/None")
    p = min(max(float(p), 0.0), 1.0)
    return float(norm.isf(p / 2.0))


def zc(p: float) -> float:
    """Clipped z in [0, 5]."""
    return float(min(max(z_of_p(p), 0.0), Z_CLIP))
