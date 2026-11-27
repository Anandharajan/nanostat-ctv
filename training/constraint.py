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


Z95 = 1.6448536269514722  # one-sided 95% normal quantile


def binomial_tolerance(rate: float, n_eff: float) -> float:
    """Prespecified tolerance eps_s = z_0.95 * sqrt(rate (1 - rate) / n_eff)."""
    n_eff = max(n_eff, 1.0)
    return Z95 * float(np.sqrt(max(rate * (1 - rate), 1e-12) / n_eff))


@dataclass
class _ScenarioState:
    lam1: float = 0.0
    lam2: float = 0.0
    fpr_hat: float | None = None
    pwr_hat: float | None = None
    n_null: float = 0.0  # EMA-weighted effective number of null observations
    n_alt: float = 0.0


@dataclass
class ScenarioConstraint:
    """Scenario-specific validity constraint (SPEC v3.1 §2.7):

        FPR_s <= alpha + eps_s      and      Power_s >= PowerRef_s - delta_s

    with eps_s, delta_s binomial tolerances on the effective sample behind each EMA estimate and
    PowerRef_s the power of the registry-preferred admissible method on scenario s, computed on CPU
    from the training keys (no blanket 80% power target)."""

    pwr_ref: dict[str, float]
    alpha: float = 0.05
    eta_lambda: float = 0.05
    lambda_max: float = 0.5
    ema: float = 0.9
    states: dict[str, _ScenarioState] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)

    def _st(self, s: str) -> _ScenarioState:
        return self.states.setdefault(s, _ScenarioState())

    def _ema(self, prev: float | None, x: float) -> float:
        return x if prev is None else self.ema * prev + (1 - self.ema) * x

    def eps(self, s: str) -> float:
        return binomial_tolerance(self.alpha, self._st(s).n_null)

    def delta(self, s: str) -> float:
        return binomial_tolerance(self.pwr_ref.get(s, 0.8), self._st(s).n_alt)

    def update(self, null_rejects: dict[str, list[float]], alt_rejects: dict[str, list[float]]) -> None:
        for s in set(null_rejects) | set(alt_rejects) | set(self.states):
            st = self._st(s)
            nr, ar = null_rejects.get(s, []), alt_rejects.get(s, [])
            st.n_null = self.ema * st.n_null + len(nr)
            st.n_alt = self.ema * st.n_alt + len(ar)
            if nr:
                st.fpr_hat = self._ema(st.fpr_hat, float(np.mean(nr)))
            if ar:
                st.pwr_hat = self._ema(st.pwr_hat, float(np.mean(ar)))
            if st.fpr_hat is not None:
                g = st.fpr_hat - (self.alpha + self.eps(s))
                st.lam1 = float(np.clip(st.lam1 + self.eta_lambda * g, 0, self.lambda_max))
            if st.pwr_hat is not None and s in self.pwr_ref:
                g = (self.pwr_ref[s] - self.delta(s)) - st.pwr_hat
                st.lam2 = float(np.clip(st.lam2 + self.eta_lambda * g, 0, self.lambda_max))
        self.history.append({f"lambda1/{s}": st.lam1 for s, st in self.states.items()})

    def shape(self, s: str, reward: float, reject: bool, h_true: bool | None, effect: str) -> float:
        st = self._st(s)
        r = reward
        if reject and h_true is True:
            r -= st.lam1
        if reject and h_true is False and effect in ("medium", "large"):
            r += st.lam2
        return r

    def state_dict(self) -> dict[str, dict[str, float | None]]:
        return {s: dict(vars(st)) for s, st in self.states.items()}

    def load_state_dict(self, d: dict[str, dict[str, float | None]]) -> None:
        self.states = {s: _ScenarioState(**v) for s, v in d.items()}  # type: ignore[arg-type]


def scenario_of(family: str, perturbation: str) -> str:
    return f"{family}/{perturbation}"
