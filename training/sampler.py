"""Discordance-weighted twin-pair sampler (SPEC §7.2, §7.7). Reads manifests, not keys."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class PairSampler:
    def __init__(self, weights_csv: Path, seed: int = 42):
        rows = list(csv.DictReader(weights_csv.open(encoding="utf-8")))
        seen: set[str] = set()
        self.units: list[tuple[str, str | None]] = []
        w: list[float] = []
        for r in rows:
            tid, twin = r["task_id"], r["twin_id"] or None
            if tid in seen:
                continue
            seen.add(tid)
            if twin:
                seen.add(twin)
            self.units.append((tid, twin))
            w.append(float(r["weight"]))
        p = np.asarray(w, float)
        self.p = p / p.sum()
        self.rng = np.random.default_rng(seed)

    def draw(self, k: int) -> list[tuple[str, str | None]]:
        idx = self.rng.choice(len(self.units), size=k, replace=len(self.units) < k, p=self.p)
        return [self.units[i] for i in idx]

    def state_dict(self) -> dict[str, object]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, s: dict[str, object]) -> None:
        self.rng.bit_generator.state = s["rng"]
