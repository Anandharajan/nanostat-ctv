"""Lineage-based split assignment (SPEC §6.4 step 8). Twins always share a split."""

from __future__ import annotations

import hashlib

HELD_OUT_EDITS = ("outcome_type", "small_counts")  # appear only in the locked test split
FRACTIONS = (("train", 0.8), ("dev", 0.1), ("test", 0.1))


def _u(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / float(16**12)


def assign_split(scenario_group: str, template_group: str, twin_edit: str | None) -> str:
    if twin_edit in HELD_OUT_EDITS:
        return "test"
    # Hash both lineage keys so a template never straddles train and test for one scenario group.
    u = _u(f"{scenario_group}|{template_group}")
    acc = 0.0
    for name, frac in FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return "test"
