"""Answer-key store (SPEC §5.3). Read only by the reward server and the scorer.

Planner, controller and policy-training code must never import this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from nanostat.schemas import KeyRecord


def write_keys(path: Path, keys: Iterable[KeyRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for k in keys:
            f.write(k.model_dump_json() + "\n")
            n += 1
    return n


def load_keys(path: Path) -> dict[str, KeyRecord]:
    out: dict[str, KeyRecord] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                k = KeyRecord.model_validate_json(line)
                out[k.task_id] = k
    return out
