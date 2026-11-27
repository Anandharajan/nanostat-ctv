"""Group-exclusive split assignment (SPEC §6.4 step 8, v3.1 §2.8). Twins always share a split.

Held out entirely for the locked test split:
  * twin edit types      HELD_OUT_EDITS
  * template domains     HELD_OUT_DOMAINS  (a whole wording/domain family never seen in training)
  * scenario families    HELD_OUT_SCENARIOS (family, perturbation)
"""

from __future__ import annotations

import hashlib
import re

HELD_OUT_EDITS = ("outcome_type", "small_counts")
HELD_OUT_DOMAINS = (4,)
HELD_OUT_SCENARIOS = (("correlation", "skewed"),)
FRACTIONS = (("train", 0.8), ("dev", 0.1), ("test", 0.1))

_DOMAIN = re.compile(r"_d(\d+)_")


def _u(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / float(16**12)


def held_out(template_group: str, twin_edit: str | None, family: str | None, perturbation: str | None) -> bool:
    if twin_edit in HELD_OUT_EDITS:
        return True
    m = _DOMAIN.search(template_group + "_")
    if m and int(m.group(1)) in HELD_OUT_DOMAINS:
        return True
    return (family, perturbation) in HELD_OUT_SCENARIOS


def assign_split(
    scenario_group: str,
    template_group: str,
    twin_edit: str | None,
    family: str | None = None,
    perturbation: str | None = None,
) -> str:
    if held_out(template_group, twin_edit, family, perturbation):
        return "test"
    u = _u(f"{scenario_group}|{template_group}")
    acc = 0.0
    for name, frac in FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return "test"
