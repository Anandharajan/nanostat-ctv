"""Admissible-set policy registry loader (SPEC §6.2).

The registry is public, versioned configuration. It is *not* an answer key: per-task
A(x)/B(x) with cached p-values live only in `keys/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from nanostat.schemas import Alternative, Design, OutcomeType
from nanostat.tools import REGISTRY

_REPO_POLICY_DIR = Path(__file__).resolve().parent.parent / "configs" / "policies"
_PKG_POLICY_DIR = Path(__file__).resolve().parent / "_policies"
DEFAULT_POLICY_DIR = _REPO_POLICY_DIR if _REPO_POLICY_DIR.is_dir() else _PKG_POLICY_DIR


@dataclass(frozen=True)
class PolicyCell:
    design: Design
    outcome: OutcomeType
    perturbation: str
    estimand_options: dict[str, list[str]]
    invalid: dict[str, str]  # method -> reason
    clarify_if: list[str] = field(default_factory=list)
    reviewed_by: str = "PENDING"
    version: str = "0.0.0"

    @property
    def key(self) -> str:
        return f"{self.design.value}/{self.outcome.value}/{self.perturbation}"

    def admissible(self, alternative: Alternative = Alternative.two_sided) -> list[str]:
        """Ordered, de-duplicated admissible methods. Two-sided-only tools drop out of A(x)
        when the claim is directional (their p-value answers a different question)."""
        out: list[str] = []
        for methods in self.estimand_options.values():
            for m in methods:
                if m in out:
                    continue
                if alternative != Alternative.two_sided and REGISTRY[m].two_sided_only:
                    continue
                out.append(m)
        return out

    def preferred(self, alternative: Alternative = Alternative.two_sided) -> str | None:
        a = self.admissible(alternative)
        return a[0] if a else None

    def estimand_of(self, method: str) -> str | None:
        return next((e for e, ms in self.estimand_options.items() if method in ms), None)

    @property
    def confirmatory_ok(self) -> bool:
        return self.reviewed_by.strip().upper() != "PENDING"


class PolicyError(ValueError):
    pass


def _parse(entry: dict[str, Any], src: str) -> PolicyCell:
    c = entry["cell"]
    cell = PolicyCell(
        design=Design(c["design"]),
        outcome=OutcomeType(c["outcome"]),
        perturbation=str(c["perturbation"]),
        estimand_options={k: list(v) for k, v in entry.get("estimand_options", {}).items()},
        invalid={d["method"]: d["reason"] for d in entry.get("invalid", [])},
        clarify_if=list(entry.get("clarify_if", [])),
        reviewed_by=str(entry.get("reviewed_by", "PENDING")),
        version=str(entry.get("version", "0.0.0")),
    )
    for m in [*cell.invalid, *(m for ms in cell.estimand_options.values() for m in ms)]:
        if m not in REGISTRY:
            raise PolicyError(f"{src}: cell {cell.key} names unknown method {m!r}")
    overlap = set(cell.invalid) & set(cell.admissible())
    if overlap:
        raise PolicyError(f"{src}: cell {cell.key} lists {overlap} as both admissible and invalid")
    return cell


@lru_cache(maxsize=4)
def load_policies(policy_dir: str | None = None) -> dict[str, PolicyCell]:
    d = Path(policy_dir) if policy_dir else DEFAULT_POLICY_DIR
    cells: dict[str, PolicyCell] = {}
    for f in sorted(d.glob("*.yaml")):
        for entry in yaml.safe_load(f.read_text(encoding="utf-8")) or []:
            cell = _parse(entry, f.name)
            if cell.key in cells:
                raise PolicyError(f"duplicate cell {cell.key}")
            cells[cell.key] = cell
    return cells


def lookup(design: Design, outcome: OutcomeType, perturbation: str) -> PolicyCell | None:
    cells = load_policies()
    cell = cells.get(f"{design.value}/{outcome.value}/{perturbation}")
    if cell is None and perturbation != "none":
        # Fall back to the most conservative matching cell for an unlisted perturbation.
        for alt in ("heavy_outliers", "skewed", "none"):
            cell = cells.get(f"{design.value}/{outcome.value}/{alt}")
            if cell is not None:
                break
    return cell
