"""Task factory orchestration (SPEC §6.4): grid sampling -> simulate -> reference -> discordance
-> twins -> render -> validate -> split -> write tasks, CSVs, keys and manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.discordance import discordance, sampling_weight
from benchmark.factory import render
from benchmark.factory.dgp import FAMILIES, N_REGIMES, PERTURBATIONS, Scenario, Simulated, null_labels, simulate
from benchmark.factory.twins import EDIT_FAMILIES, EDIT_WORDINGS, EDITS, base_scenario, make_twin
from benchmark.keys import write_keys
from benchmark.reference import build_reference
from benchmark.split import HELD_OUT_EDITS, assign_split, held_out
from nanostat.profiler import profile
from nanostat.protocol import build_protocol
from nanostat.schemas import (
    DGP,
    Action,
    Alternative,
    Design,
    KeyRecord,
    Lineage,
    OutcomeType,
    ReasonCode,
    Study,
    TaskRecord,
)
from nanostat.verifier import expected_clarification

GENERATOR_VERSION = "0.1.0"

PROFILES: dict[str, dict[str, int]] = {
    "smoke": {"singles": 40, "pairs": 12},
    "pilot": {"singles": 1000, "pairs": 200},
    "full": {"singles": 9000, "pairs": 800},
}

FAMILY_DESIGN = {
    "two_group": Design.independent_two_group,
    "paired": Design.paired,
    "correlation": Design.correlation,
    "k_group": Design.k_group,
    "binary_two_group": Design.independent_two_group,
    "one_proportion": Design.one_sample,
}


@dataclass
class Built:
    task: TaskRecord
    key: KeyRecord
    csv_bytes: bytes
    gold: tuple[str, ...]


@dataclass
class FactoryReport:
    profile: str
    seed: int
    attempted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    tasks: int = 0
    twin_pairs: int = 0
    per_split: Counter[str] = field(default_factory=Counter)
    per_family: Counter[str] = field(default_factory=Counter)
    per_cell: Counter[str] = field(default_factory=Counter)
    per_edit: Counter[str] = field(default_factory=Counter)
    per_wording: Counter[str] = field(default_factory=Counter)
    duplicates: int = 0


def _hid(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def sample_scenario(rng: np.random.Generator, seed: int, family: str | None = None) -> Scenario:
    fam = family or str(rng.choice(FAMILIES))
    alt = "two-sided" if fam == "k_group" else str(rng.choice(["two-sided", "two-sided", "greater", "less"]))
    return Scenario(
        family=fam,
        n_regime=str(rng.choice(list(N_REGIMES))),
        effect=str(rng.choice(["null", "borderline", "medium", "large"])),
        perturbation=str(rng.choice(PERTURBATIONS[fam])),
        alternative=alt,
        prompt_style=str(rng.choice(["hint", "no_hint"])),
        seed=seed,
        domain=int(rng.integers(0, len(render.DOMAINS))),
        k=int(rng.choice([3, 4])),
        extra={
            "variant": int(rng.integers(0, render.N_TEMPLATE_VARIANTS)),
            "estimand_word": (
                str(rng.choice(["none", "mean", "typical"])) if fam in ("two_group", "paired", "k_group") else "none"
            ),
        },
    )


def build_task(
    sc: Scenario,
    sim: Simulated,
    scenario_group: str,
    member: str = "single",
    edit: str | None = None,
    wording: str | None = None,
) -> Built | str:
    """Return a Built task or a rejection reason string."""
    variant = int(sc.extra.get("variant", 0))
    if sc.family == "one_proportion" and sim.p0 is not None:
        sc.extra["p0_text"] = f"{int(round(sim.p0 * 100))}%"
    design = FAMILY_DESIGN[sc.family]
    q = render.question(sc, variant)
    hint = sc.prompt_style == "hint"
    outcome = OutcomeType.binary if sim.df[sim.roles["y"]].dropna().nunique() == 2 else OutcomeType.continuous
    study = Study(
        design=design if hint else None,
        outcome=outcome if hint else None,
        roles=dict(sim.roles),
        variables=render.variables(sc, sim.roles),
        group_order=sim.group_order,
        description=render.description(sc),
        p0=sim.p0,
    )
    protocol = build_protocol(q, study)
    alt = protocol.alternative
    if alt != Alternative(sc.alternative):
        return "protocol_direction_mismatch"
    csv_bytes = sim.df.to_csv(index=False, lineterminator="\n").encode()
    tid = "t_" + _hid(GENERATOR_VERSION, asdict(sc) | {"extra": sorted(sc.extra.items())}, member, edit, wording)[:16]

    prof = profile(sim.df)
    clar = expected_clarification(q, study, sim.df, prof)
    template_group = f"tg_{sc.family}_d{sc.domain}_v{variant}"
    lineage = Lineage(
        scenario_group=scenario_group,
        template_group=template_group,
        seed=sc.seed,
        generator_version=GENERATOR_VERSION,
    )
    nulls = null_labels("binary_two_group" if sc.extra.get("dichotomised") else sc.family, sc.perturbation, sc.effect)

    def _dgp(hyp: str | None) -> DGP:
        h0 = nulls.get(hyp, sc.effect == "null") if hyp else sc.effect == "null"
        return DGP(family=sc.family, h0_true=h0, nulls=nulls, effect=sc.effect, perturbation=sc.perturbation)  # type: ignore[arg-type]

    common = {
        "task_id": tid,
        "columns": dict(sim.roles),
        "lineage": lineage,
        "protocol": protocol,
        "twin_edit": edit,
        "twin_wording": wording,
        "alternative": alt,
        "design": design,
    }

    if clar is not None:
        if clar != ReasonCode.missing_pair_id:
            return f"unexpected_clarify:{clar.value}"
        key = KeyRecord(
            **common,  # type: ignore[arg-type]
            dgp=_dgp(None),
            outcome=outcome,
            admissible=[],
            invalid=[],
            preferred=None,
            reference_decision=None,
            discordance=discordance([], []),
            abstain_target={"action": Action.clarify.value, "reason_code": clar.value},
            policy_cell=f"{design.value}/{outcome.value}/clarify",
        )
        gold: tuple[str, ...] = ("clarify", clar.value)
    else:
        ref = build_reference(sim.df, design, sim.roles, protocol, sim.group_order)
        if ref is None:
            return "reference_failed"
        if ref.preferred is None:
            # no reviewed admissible method under this protocol: the correct action is to abstain
            key = KeyRecord(
                **common,  # type: ignore[arg-type]
                dgp=_dgp(None),
                outcome=ref.outcome,
                admissible=[],
                invalid=ref.invalid,
                preferred=None,
                reference_decision=None,
                discordance=discordance([], []),
                abstain_target={"action": Action.abstain.value, "reason_code": ReasonCode.no_admissible_method.value},
                policy_cell=ref.cell.key,
            )
            gold = ("abstain", ReasonCode.no_admissible_method.value)
        else:
            if not ref.admissible or ref.admissible[0].p is None:
                return "no_admissible_p"
            if any(r.p is not None and not np.isfinite(r.p) for r in ref.admissible):
                return "nan_p"
            key = KeyRecord(
                **common,  # type: ignore[arg-type]
                dgp=_dgp(ref.hypothesis),
                outcome=ref.outcome,
                admissible=ref.admissible,
                invalid=ref.invalid,
                preferred=ref.preferred,
                reference_decision=ref.reference_decision,  # type: ignore[arg-type]
                discordance=discordance(ref.admissible, ref.invalid),
                policy_cell=ref.cell.key,
                required_checks=ref.required_checks,
                hypothesis=ref.hypothesis,
            )
            gold = ("execute", ",".join(sorted(r.method for r in ref.admissible)), alt.value)
    task = TaskRecord(task_id=tid, question=q, study=study, data_ref=f"csv/{tid}.csv")
    return Built(task, key, csv_bytes, gold)


def _near_dup_key(b: Built) -> str:
    return _hid(b.task.question, b.task.study.model_dump(mode="json"), hashlib.sha256(b.csv_bytes).hexdigest())


def run_factory(profile_name: str, out: Path, seed: int = 20261126) -> FactoryReport:
    cfg = PROFILES[profile_name]
    rng = np.random.default_rng(seed)
    rep = FactoryReport(profile=profile_name, seed=seed)
    built: list[Built] = []
    seen: set[str] = set()

    def _accept(b: Built) -> bool:
        k = _near_dup_key(b)
        if k in seen:
            rep.duplicates += 1
            return False
        seen.add(k)
        return True

    # --- singles ------------------------------------------------------------------------
    i = 0
    while sum(1 for b in built if b.key.twin_id is None) < cfg["singles"]:
        i += 1
        rep.attempted += 1
        s = int(rng.integers(1, 2**31 - 1))
        sc = sample_scenario(rng, s)
        sim = simulate(sc, render.names_for(sc))
        sg = "sg_" + _hid("single", sc.family, sc.n_regime, sc.effect, sc.perturbation, s)[:10]
        b = build_task(sc, sim, sg)
        if isinstance(b, str):
            rep.rejected[b] += 1
            continue
        if _accept(b):
            built.append(b)
        if i > cfg["singles"] * 5:
            break

    # --- twin pairs ------------------------------------------------------------------------
    j = 0
    pairs = 0
    while pairs < cfg["pairs"] and j < cfg["pairs"] * 20:
        j += 1
        rep.attempted += 1
        edit = EDITS[j % len(EDITS)]
        wording = str(rng.choice(EDIT_WORDINGS[edit]))
        fam = str(rng.choice(EDIT_FAMILIES[edit]))
        s = int(rng.integers(1, 2**31 - 1))
        sc0 = sample_scenario(rng, s, fam)
        sc = base_scenario(edit, sc0, wording)
        sim = simulate(sc, render.names_for(sc))
        sg = "sg_pair_" + _hid(edit, fam, s)[:10]
        b1 = build_task(sc, sim, sg, "a", edit, wording)
        if isinstance(b1, str):
            rep.rejected["twin_base:" + b1] += 1
            continue
        sc2, sim2 = make_twin(edit, sc, sim, wording)
        b2 = build_task(sc2, sim2, sg, "b", edit, wording)
        if isinstance(b2, str):
            rep.rejected["twin:" + b2] += 1
            continue
        if b1.gold == b2.gold:
            rep.rejected["twin_gold_unchanged:" + edit] += 1
            continue
        if wording == "fixed" and b1.task.question != b2.task.question:
            rep.rejected["twin_wording_not_fixed:" + edit] += 1
            continue
        if not (_accept(b1) and _accept(b2)):
            continue
        b1.key.twin_id, b2.key.twin_id = b2.task.task_id, b1.task.task_id
        built += [b1, b2]
        pairs += 1
        rep.per_edit[edit] += 1
        rep.per_wording[wording] += 1
    rep.twin_pairs = pairs

    # --- splits (twins share the base member's lineage) ----------------------------------------
    by_id = {b.task.task_id: b for b in built}
    for b in built:
        if b.key.split is not None:
            continue
        members = [b] + ([by_id[b.key.twin_id]] if b.key.twin_id else [])
        lin = b.key.lineage
        if any(
            held_out(m.key.lineage.template_group, m.key.twin_edit, m.key.dgp.family, m.key.dgp.perturbation)
            for m in members
        ):
            sp = "test"
        else:
            sp = assign_split(lin.scenario_group, lin.template_group, b.key.twin_edit)
        for m in members:  # twins always share a split
            m.key.split = sp  # type: ignore[assignment]

    _write(out, built, rep)
    return rep


def _write(out: Path, built: list[Built], rep: FactoryReport) -> None:
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for sp in ("train", "dev", "test"):
        files[sp] = (out / "tasks" / f"{sp}.jsonl").open("w", encoding="utf-8")
    hist: Counter[int] = Counter()
    try:
        with (
            (out / "manifests" / "split_manifest.csv").open("w", newline="", encoding="utf-8") as fm,
            (out / "manifests" / "sampling_weights.csv").open("w", newline="", encoding="utf-8") as fw,
        ):
            wm, ww = csv.writer(fm), csv.writer(fw)
            wm.writerow(["task_id", "split", "scenario_group", "template_group", "twin_id", "csv_sha256"])
            ww.writerow(["task_id", "twin_id", "weight"])
            for b in built:
                sp = b.key.split or "train"
                (out / b.task.data_ref).write_bytes(b.csv_bytes)
                files[sp].write(b.task.model_dump_json() + "\n")
                wm.writerow(
                    [
                        b.task.task_id,
                        sp,
                        b.key.lineage.scenario_group,
                        b.key.lineage.template_group,
                        b.key.twin_id or "",
                        hashlib.sha256(b.csv_bytes).hexdigest(),
                    ]
                )
                if sp == "train":
                    ww.writerow(
                        [b.task.task_id, b.key.twin_id or "", round(sampling_weight(b.key.discordance.score), 6)]
                    )
                rep.per_split[sp] += 1
                rep.per_family[b.key.dgp.family] += 1
                rep.per_cell[b.key.policy_cell] += 1
                hist[min(int(b.key.discordance.score * 10), 9)] += 1
    finally:
        for f in files.values():
            f.close()
    rep.tasks = len(built)
    write_keys(out / "keys" / "keys.jsonl", (b.key for b in built))
    with (out / "manifests" / "discordance_hist.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bin_low", "bin_high", "count"])
        for k in range(10):
            w.writerow([k / 10, (k + 1) / 10, hist.get(k, 0)])
    report = {
        "profile": rep.profile,
        "seed": rep.seed,
        "generator_version": GENERATOR_VERSION,
        "attempted": rep.attempted,
        "tasks": rep.tasks,
        "twin_pairs": rep.twin_pairs,
        "duplicates_removed": rep.duplicates,
        "rejected": dict(rep.rejected),
        "per_split": dict(rep.per_split),
        "per_family": dict(rep.per_family),
        "per_cell": dict(rep.per_cell),
        "per_edit": dict(rep.per_edit),
        "per_wording": dict(rep.per_wording),
        "held_out_edits": list(HELD_OUT_EDITS),
        "discordance_hist": {f"{k / 10:.1f}-{(k + 1) / 10:.1f}": hist.get(k, 0) for k in range(10)},
    }
    (out / "manifests" / "factory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    digests = []
    for p in sorted((out / "tasks").glob("*.jsonl")) + [
        out / "keys" / "keys.jsonl",
        out / "manifests" / "split_manifest.csv",
    ]:
        digests.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}")
    (out / "manifests" / "data_manifest.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
