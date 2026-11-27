"""CPU ablation of the CTV training signal: S0, R-F, R-A, R-AD, R-ADT, R-CTV (SPEC §8).

All arms share the initialisation (the same SFT'd proxy policy per seed), the number of updates,
the number of task instances per update, the group size, the learning rate and the KL weight.
Only the reward / sampling / twin / constraint switches differ. Hyperparameters are fixed a priori
(not tuned on test). Evaluation uses greedy actions on dev and the locked synthetic test split, plus
fresh null simulations for agent-level false-positive rates.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.split import HELD_OUT_EDITS
from training.ablation.actions import CATALOGUE
from training.ablation.policy import LinearSoftmaxPolicy
from training.ablation.tables import NullTable, TaskTable
from training.constraint import ScenarioConstraint, twin_joint_advantage


@dataclass(frozen=True)
class Arm:
    name: str
    reward: str = "ctv"  # "ctv" | "fisher"
    sampling: str = "uniform"  # "uniform" | "discordance" | "discordance_abstain"
    twins: bool = False
    constraint: bool = False
    rl: bool = True


ARMS: list[Arm] = [
    Arm("S0", rl=False),
    Arm("R-F", reward="fisher"),
    Arm("R-A"),
    Arm("R-AD", sampling="discordance"),
    Arm("R-ADT", sampling="discordance", twins=True),
    Arm("R-CTV", sampling="discordance", twins=True, constraint=True),
    # Post-hoc variant (selected from dev evidence, reported separately): clarify/abstain tasks are
    # maximally discordant (any executed answer is wrong), so D(x) = 1 instead of 0 for them.
    Arm("R-CTV-abs", sampling="discordance_abstain", twins=True, constraint=True),
]


@dataclass
class HParams:
    updates: int = 300
    instances_per_update: int = 16
    group_size: int = 4
    lr: float = 0.5
    kl_beta: float = 0.02
    sft_examples: int = 64
    sft_epochs: int = 50
    sft_lr: float = 0.5
    eps_d: float = 0.1
    eta_twin: float = 0.2
    tau_twin: float = 0.8
    log_every: int = 25
    max_resample: int = 2


@dataclass
class RunLog:
    arm: str
    seed: int
    curve: list[dict[str, float]] = field(default_factory=list)
    skipped_zero_var: int = 0
    instances: int = 0
    lambdas: dict[str, Any] = field(default_factory=dict)


def _split_idx(tb: TaskTable, split: str) -> np.ndarray:
    return np.array([i for i, m in enumerate(tb.meta) if m["split"] == split])


def _units(tb: TaskTable, idx: np.ndarray, twins: bool) -> list[tuple[int, ...]]:
    pos = {tb.task_ids[i]: i for i in idx}
    units: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for i in idx:
        if i in seen:
            continue
        tw = tb.meta[i]["twin_id"]
        if twins and tw in pos:
            j = pos[tw]
            units.append((i, j))
            seen |= {i, j}
        else:
            units.append((i,))
            seen.add(i)
    return units


def power_reference(tb: TaskTable, idx: np.ndarray) -> dict[str, float]:
    """PowerRef_s: power of the registry-preferred method on medium/large alternatives (CPU, train keys)."""
    acc: dict[str, list[float]] = {}
    for i in idx:
        g = tb.gold[i]
        if tb.meta[i]["effect"] in ("medium", "large") and tb.htrue[i, g] == 0 and not math.isnan(tb.reject[i, g]):
            acc.setdefault(tb.meta[i]["scenario"], []).append(tb.reject[i, g])
    return {s: float(np.mean(v)) for s, v in acc.items() if len(v) >= 5}


def train_arm(arm: Arm, tb: TaskTable, hp: HParams, seed: int) -> tuple[LinearSoftmaxPolicy, RunLog]:
    rng = np.random.default_rng(seed)
    train = _split_idx(tb, "train")
    dev = _split_idx(tb, "dev")
    pol = LinearSoftmaxPolicy(len(CATALOGUE), tb.phi.shape[1], seed=seed)
    # shared programmatic-SFT initialisation (identical across arms for a given seed)
    sft_rng = np.random.default_rng(10_000 + seed)
    sft_idx = sft_rng.choice(train, size=min(hp.sft_examples, len(train)), replace=False)
    pol.sft(tb.phi[sft_idx], tb.gold[sft_idx], hp.sft_epochs, hp.sft_lr)
    ref_W = pol.W.copy()
    log = RunLog(arm.name, seed)
    if not arm.rl:
        log.curve.append({"update": 0, **{f"dev_{k}": v for k, v in evaluate(pol, tb, dev).items()}})
        return pol, log

    table = tb.ctv if arm.reward == "ctv" else tb.fisher
    units = _units(tb, train, arm.twins)
    if arm.sampling in ("discordance", "discordance_abstain"):

        def d_of(i: int) -> float:
            if arm.sampling == "discordance_abstain" and not tb.meta[i]["answerable"]:
                return 1.0
            return float(tb.meta[i]["discordance"])

        w = np.array([np.mean([hp.eps_d + d_of(i) for i in u]) for u in units])
    else:
        w = np.ones(len(units))
    w = w / w.sum()
    cons = ScenarioConstraint(pwr_ref=power_reference(tb, train)) if arm.constraint else None

    for upd in range(1, hp.updates + 1):
        phis, acts, advs = [], [], []
        null_rej: dict[str, list[float]] = {}
        alt_rej: dict[str, list[float]] = {}
        n_inst = 0
        while n_inst < hp.instances_per_update:
            u = units[rng.choice(len(units), p=w)]
            n_inst += len(u)
            for _attempt in range(hp.max_resample + 1):
                samples = [pol.sample(tb.phi[i], hp.group_size) for i in u]
                rewards = []
                for i, a in zip(u, samples):
                    r = table[i, a].copy()
                    if cons is not None:
                        sc, eff = tb.meta[i]["scenario"], tb.meta[i]["effect"]
                        for k, ak in enumerate(a):
                            rej, ht = tb.reject[i, ak], tb.htrue[i, ak]
                            if not math.isnan(rej):
                                r[k] = cons.shape(sc, r[k], bool(rej), None if math.isnan(ht) else bool(ht), eff)
                    rewards.append(r)
                if len(u) == 2:
                    adv = twin_joint_advantage(list(rewards[0]), list(rewards[1]), hp.eta_twin, hp.tau_twin)
                    advs_u = [adv, adv]
                else:
                    r0 = rewards[0]
                    advs_u = [(r0 - r0.mean()) / (r0.std() + 1e-6)]
                if np.any(np.abs(advs_u[0]) > 1e-8):
                    break
            else:
                log.skipped_zero_var += 1
                continue
            for i, a, ad in zip(u, samples, advs_u):
                phis.append(np.repeat(tb.phi[i][None, :], len(a), axis=0))
                acts.append(a)
                advs.append(ad)
                if cons is not None:
                    sc, eff = tb.meta[i]["scenario"], tb.meta[i]["effect"]
                    for ak in a:
                        rej, ht = tb.reject[i, ak], tb.htrue[i, ak]
                        if math.isnan(rej) or math.isnan(ht):
                            continue
                        if ht == 1.0:
                            null_rej.setdefault(sc, []).append(rej)
                        elif eff in ("medium", "large"):
                            alt_rej.setdefault(sc, []).append(rej)
        log.instances += n_inst
        if phis:
            pol.grpo_step(np.concatenate(phis), np.concatenate(acts), np.concatenate(advs), hp.lr, ref_W, hp.kl_beta)
        if cons is not None:
            cons.update(null_rej, alt_rej)
        if upd % hp.log_every == 0 or upd == hp.updates:
            row = {"update": upd, **{f"dev_{k}": v for k, v in evaluate(pol, tb, dev).items()}}
            if cons is not None:
                row["max_lambda1"] = max((st.lam1 for st in cons.states.values()), default=0.0)
            log.curve.append(row)
    if cons is not None:
        log.lambdas = cons.state_dict()
    return pol, log


def _choose(pol: LinearSoftmaxPolicy, phi: np.ndarray) -> np.ndarray:
    return pol.greedy(phi)


def evaluate(pol: LinearSoftmaxPolicy, tb: TaskTable, idx: np.ndarray) -> dict[str, float]:
    if len(idx) == 0:
        return {}
    a = _choose(pol, tb.phi[idx])
    rows = np.arange(len(idx))
    ans = np.array([tb.meta[i]["answerable"] for i in idx])
    correct = tb.correct[idx, a][rows]
    out: dict[str, float] = {
        "method_validity": float(correct[ans].mean()) if ans.any() else float("nan"),
        "invalid_method_rate": float(tb.invalid[idx, a][rows][ans].mean()) if ans.any() else float("nan"),
        "abstention_correct": float(correct[~ans].mean()) if (~ans).any() else float("nan"),
        "strict": float(tb.strict[idx, a][rows][ans].mean()) if ans.any() else float("nan"),
        "ctv_reward": float(tb.ctv[idx, a][rows].mean()),
        "fisher_reward": float(tb.fisher[idx, a][rows].mean()),
    }
    pos = {tb.task_ids[i]: k for k, i in enumerate(idx)}
    pairs = []
    for k, i in enumerate(idx):
        tw = tb.meta[i]["twin_id"]
        if tw in pos and tb.task_ids[i] < tw:
            pairs.append(correct[k] * correct[pos[tw]])
    out["twin_consistency"] = float(np.mean(pairs)) if pairs else float("nan")
    held = [k for k, i in enumerate(idx) if tb.meta[i]["twin_edit"] in HELD_OUT_EDITS]
    out["held_out_edit_accuracy"] = float(correct[held].mean()) if held else float("nan")
    rej = tb.reject[idx, a][rows]
    ht = tb.htrue[idx, a][rows]
    m = (~np.isnan(rej)) & (ht == 1.0)
    out["task_fpr"] = float(rej[m].mean()) if m.any() else float("nan")
    return out


def null_fpr(pol: LinearSoftmaxPolicy, nt: NullTable) -> dict[str, Any]:
    a = _choose(pol, nt.phi)
    rows = np.arange(len(a))
    rej, ht = nt.reject[rows, a], nt.htrue[rows, a]
    m = (~np.isnan(rej)) & (ht == 1.0)
    per: dict[str, Any] = {}
    for s in sorted(set(nt.scenarios)):
        sel = np.array([x == s for x in nt.scenarios]) & m
        per[s] = {"k": int(rej[sel].sum()), "n": int(sel.sum())}
    executed = float((~np.isnan(rej)).mean())
    return {
        "pooled": {"k": int(rej[m].sum()), "n": int(m.sum())},
        "per_scenario": per,
        "executed_fraction": executed,
        "reject_vector": np.where(m, rej, np.nan).tolist(),
    }


def _per_task_correct(pol: LinearSoftmaxPolicy, tb: TaskTable, idx: np.ndarray) -> np.ndarray:
    a = _choose(pol, tb.phi[idx])
    return tb.correct[idx, a][np.arange(len(idx))]


def _bootstrap_diff(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int = 2000) -> dict[str, float]:
    """Paired bootstrap of mean(x) - mean(y) over units (rows); NaNs dropped pairwise."""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    d = x - y
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
        "p": float(min(1.0, p)),
        "n": int(len(d)),
    }


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for r, (k, p) in enumerate(items):
        adj = min(1.0, (len(items) - r) * p)
        running = max(running, adj)
        out[k] = running
    return out


def run_ablation(
    tb: TaskTable, nt: NullTable | None, seeds: list[int], hp: HParams | None = None, arms: list[Arm] | None = None
) -> dict[str, Any]:
    hp = hp or HParams()
    arms = arms or ARMS
    dev, test = _split_idx(tb, "dev"), _split_idx(tb, "test")
    results: dict[str, Any] = {"hparams": asdict(hp), "arms": {}, "logs": {}}
    per_task: dict[str, list[np.ndarray]] = {}
    pair_correct: dict[str, list[np.ndarray]] = {}
    null_vecs: dict[str, list[np.ndarray]] = {}
    test_pairs = [
        (k, m)
        for k, i in enumerate(test)
        for m, j in enumerate(test)
        if tb.meta[i]["twin_id"] == tb.task_ids[j] and tb.task_ids[i] < tb.task_ids[j]
    ]
    for arm in arms:
        runs = []
        for seed in seeds:
            pol, log = train_arm(arm, tb, hp, seed)
            ev = {"dev": evaluate(pol, tb, dev), "test": evaluate(pol, tb, test)}
            if nt is not None:
                nf = null_fpr(pol, nt)
                ev["null"] = {k: v for k, v in nf.items() if k != "reject_vector"}
                null_vecs.setdefault(arm.name, []).append(np.asarray(nf["reject_vector"], float))
            c = _per_task_correct(pol, tb, test)
            per_task.setdefault(arm.name, []).append(c)
            pair_correct.setdefault(arm.name, []).append(np.array([c[k] * c[m] for k, m in test_pairs]))
            runs.append(ev)
            results["logs"][f"{arm.name}/{seed}"] = asdict(log)
        results["arms"][arm.name] = {"runs": runs, "summary": _summarise(runs)}

    # primary comparisons: R-CTV vs R-F on the locked test split (seed-averaged per unit)
    rng = np.random.default_rng(2026)
    if "R-CTV" in per_task and "R-F" in per_task:
        ctv_c, f_c = np.mean(per_task["R-CTV"], axis=0), np.mean(per_task["R-F"], axis=0)
        ans = np.array([tb.meta[i]["answerable"] for i in test])
        comps = {
            "method_validity": _bootstrap_diff(ctv_c[ans], f_c[ans], rng),
            "twin_consistency": _bootstrap_diff(
                np.mean(pair_correct["R-CTV"], axis=0), np.mean(pair_correct["R-F"], axis=0), rng
            ),
        }
        if null_vecs:
            # lower FPR is better: report F - CTV so a positive diff favours CTV
            comps["agent_fpr_reduction"] = _bootstrap_diff(
                np.nanmean(null_vecs["R-F"], axis=0), np.nanmean(null_vecs["R-CTV"], axis=0), rng
            )
        adj = holm({k: v["p"] for k, v in comps.items() if not math.isnan(v["p"])})
        for k in comps:
            comps[k]["p_holm"] = adj.get(k, float("nan"))
        results["primary_R-CTV_vs_R-F"] = comps
    return results


def _summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("dev", "test"):
        keys = runs[0][split].keys()
        out[split] = {
            k: {
                "mean": float(np.nanmean([r[split][k] for r in runs])),
                "sd": float(np.nanstd([r[split][k] for r in runs])),
            }
            for k in keys
        }
    if "null" in runs[0]:
        ks = sum(r["null"]["pooled"]["k"] for r in runs)
        ns = sum(r["null"]["pooled"]["n"] for r in runs)
        out["null_fpr_pooled"] = {"k": ks, "n": ns, "rate": ks / ns if ns else float("nan")}
        out["null_executed_fraction"] = float(np.mean([r["null"]["executed_fraction"] for r in runs]))
    return out


METRICS = [
    ("method_validity", "Method validity"),
    ("invalid_method_rate", "Invalid-method rate"),
    ("twin_consistency", "Twin consistency"),
    ("abstention_correct", "Abstention correct"),
    ("held_out_edit_accuracy", "Held-out edits"),
    ("strict", "Strict (single ref.)"),
    ("task_fpr", "Task-level FPR"),
]


def _fmt(m: dict[str, float]) -> str:
    if m is None or math.isnan(m["mean"]):
        return "n/a"
    return f"{m['mean']:.3f} ± {m['sd']:.3f}"


def markdown_report(res: dict[str, Any], header: str) -> str:
    lines = [header, ""]
    for split in ("test", "dev"):
        lines += [f"### {split} split (greedy policy; mean ± sd over seeds)", ""]
        lines.append("| Arm | " + " | ".join(t for _, t in METRICS) + " | Agent FPR on null sims |")
        lines.append("|---" * (len(METRICS) + 2) + "|")
        for arm, v in res["arms"].items():
            sm = v["summary"]
            nf = sm.get("null_fpr_pooled")
            nfs = f"{nf['rate']:.3f} ({nf['k']}/{nf['n']})" if nf and nf["n"] else "n/a"
            lines.append(
                f"| {arm} | "
                + " | ".join(_fmt(sm[split].get(k)) for k, _ in METRICS)
                + f" | {nfs if split == 'test' else '-'} |"
            )
        lines.append("")
    if "primary_R-CTV_vs_R-F" in res:
        lines += [
            "### Primary comparisons: R-CTV minus R-F (locked test split, paired bootstrap 2,000, Holm)",
            "",
            "| Comparison | Difference | 95% CI | p | p (Holm) | n units |",
            "|---|---|---|---|---|---|",
        ]
        for k, c in res["primary_R-CTV_vs_R-F"].items():
            lines.append(
                f"| {k} | {c['diff']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p']:.3f} | {c.get('p_holm', float('nan')):.3f} | {c['n']} |"
            )
        lines.append("")
    return chr(10).join(lines)


def write_report(res: dict[str, Any], out: Path, header: str = "## CPU ablation") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_results.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "ablation_results.md").write_text(markdown_report(res, header), encoding="utf-8")
