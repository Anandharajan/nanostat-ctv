"""Linear-softmax proxy policy with GRPO-style updates (CPU, numpy only)."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LinearSoftmaxPolicy:
    def __init__(self, n_actions: int, n_features: int, seed: int = 0):
        self.W = np.zeros((n_actions, n_features))
        self.rng = np.random.default_rng(seed)

    def probs(self, phi: np.ndarray) -> np.ndarray:
        return softmax(phi @ self.W.T)

    def sample(self, phi: np.ndarray, g: int) -> np.ndarray:
        return self.rng.choice(self.W.shape[0], size=g, p=self.probs(phi))

    def greedy(self, phi: np.ndarray) -> np.ndarray:
        return np.argmax(phi @ self.W.T, axis=-1)

    def sft(self, phis: np.ndarray, targets: np.ndarray, epochs: int, lr: float) -> None:
        """Supervised cross-entropy on gold actions (programmatic SFT analogue)."""
        n = len(targets)
        for _ in range(epochs):
            p = self.probs(phis)
            y = np.zeros_like(p)
            y[np.arange(n), targets] = 1.0
            self.W -= lr * ((p - y).T @ phis) / n

    def grpo_step(
        self,
        phis: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        lr: float,
        ref_W: np.ndarray | None = None,
        kl_beta: float = 0.0,
    ) -> float:
        """One policy-gradient step on sampled (phi, action, advantage) triples.

        With a single inner epoch the PPO/GRPO ratio pi/pi_old is 1, so the clipped surrogate's
        gradient equals A * grad log pi(a|x). A KL(pi || pi_ref) penalty keeps the policy near SFT."""
        p = self.probs(phis)
        onehot = np.zeros_like(p)
        onehot[np.arange(len(actions)), actions] = 1.0
        grad = ((onehot - p) * advantages[:, None]).T @ phis / len(actions)
        kl = 0.0
        if ref_W is not None and kl_beta > 0:
            pr = softmax(phis @ ref_W.T)
            logr = np.log(p + 1e-12) - np.log(pr + 1e-12)
            kl_row = (p * logr).sum(axis=1)
            kl = float(kl_row.mean())
            # gradient of -beta * KL wrt logits: -beta * p * (logr - KL)
            gk = -(p * (logr - kl_row[:, None])).T @ phis / len(actions)
            grad = grad + kl_beta * gk
        self.W += lr * grad
        return kl
