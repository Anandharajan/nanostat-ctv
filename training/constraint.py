"""Truth-based, batch-level validity constraint (SPEC §7.6) and twin-joint advantage (§7.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ValidityConstraint:
    alpha: float = 0.05
    eps: float = 0.02
    eta_lambda: float = 0.05
    lambda_max: float = 0.5
    ema: float = 0.9
    pwr_ref: float = 0.8
    lam1: float = 0.0
    lam2: float = 0.0
    fpr_hat: float | None = None
    pwr_hat: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def _ema(self, prev: float | None, x: float) -> float:
        return x if prev is None else self.ema * prev + (1 - self.ema) * x

    def update(self, null_rejects: list[float], alt_rejects: list[float]) -> None:
        if null_rejects:
            self.fpr_hat = self._ema(self.fpr_hat, float(np.mean(null_rejects)))
        if alt_rejects:
            self.pwr_hat = self._ema(self.pwr_hat, float(np.mean(alt_rejects)))
        if self.fpr_hat is not None:
            self.lam1 = float(
                np.clip(self.lam1 + self.eta_lambda * (self.fpr_hat - (self.alpha + self.eps)), 0, self.lambda_max)
            )
        if self.pwr_hat is not None:
            self.lam2 = float(
                np.clip(self.lam2 + self.eta_lambda * ((self.pwr_ref - self.eps) - self.pwr_hat), 0, self.lambda_max)
            )
        self.history.append(
            {
                "fpr_hat": float("nan") if self.fpr_hat is None else self.fpr_hat,
                "pwr_hat": float("nan") if self.pwr_hat is None else self.pwr_hat,
                "lambda1": self.lam1,
                "lambda2": self.lam2,
            }
        )

    def shape(self, reward: float, reject: bool, h0_true: bool, effect: str) -> float:
        """R'(o) = R(o) - lam1*1[reject & h0] + lam2*1[reject & ~h0 & effect >= medium]."""
        r = reward
        if reject and h0_true:
            r -= self.lam1
        if reject and not h0_true and effect in ("medium", "large"):
            r += self.lam2
        return r

    def state_dict(self) -> dict[str, float | None]:
        return {"lam1": self.lam1, "lam2": self.lam2, "fpr_hat": self.fpr_hat, "pwr_hat": self.pwr_hat}

    def load_state_dict(self, s: dict[str, float | None]) -> None:
        self.lam1, self.lam2 = float(s["lam1"] or 0.0), float(s["lam2"] or 0.0)
        self.fpr_hat, self.pwr_hat = s["fpr_hat"], s["pwr_hat"]


def twin_joint_advantage(r: list[float], r_twin: list[float], eta: float = 0.2, tau: float = 0.8) -> np.ndarray:
    """r_i = (R(o_i) + R(o'_i))/2 + eta*1[both >= tau];  A_i = (r_i - mean) / (std + 1e-6).
    Rollouts are paired by index. The same A_i applies to both o_i and o'_i."""
    a, b = np.asarray(r, float), np.asarray(r_twin, float)
    if a.shape != b.shape:
        raise ValueError("twin groups must have equal size")
    joint = 0.5 * (a + b) + eta * ((a >= tau) & (b >= tau))
    return (joint - joint.mean()) / (joint.std() + 1e-6)


def group_has_signal(adv: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.any(np.abs(adv) > tol))
