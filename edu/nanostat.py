"""
nanostat.py - the smallest complete version of NanoStat-CTV, in one file, for learning.

A tiny GPT learns to *plan* a hypothesis test (which design, which test, which tail) from a question
and a data summary. Deterministic code runs the test; the model never computes a number. It is trained
like a modern agent: supervised fine-tuning, then reinforcement learning with verifiable rewards (GRPO).
Two rewards are compared at the SAME compute:
  fisher : "your p-value should be close to the reference p-value"  (Fisher-R1 style)
  ctv    : "your method must be in the admissible set; invalid methods are penalised"  (CTV style)

Sections:  1 statistics from scratch   2 a tiny world of tasks with answer keys   3 typed plans + verifier
           4 rewards   5 a tiny GPT   6 SFT   7 GRPO   8 evaluation   9 the experiment   10 CLI
Run:       python edu/nanostat.py compare      (about 6 minutes on a laptop CPU, 3 seeds)
Lessons:   https://anandharajan.github.io/nanostat-ctv/  (site/lessons/, one command + exercises each)
"""

from __future__ import annotations

import argparse
import copy
import math

import numpy as np
from scipy.special import ndtr, stdtr  # normal and Student-t CDFs; everything else is written out below

# ----------------------------------------------------------------------------------------------
# 1. Statistics from scratch. Each test returns a p-value for alt in {"two-sided", "greater", "less"}.


def p_from_t(t, df, alt):
    if alt == "greater":
        return float(stdtr(df, -t))
    if alt == "less":
        return float(stdtr(df, t))
    return float(2 * stdtr(df, -abs(t)))


def p_from_z(z, alt):
    if alt == "greater":
        return float(1 - ndtr(z))
    if alt == "less":
        return float(ndtr(z))
    return float(2 * (1 - ndtr(abs(z))))


def ranks(x):
    """Average ranks (1-based); ties share the mean of their positions."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def student_t(a, b, alt):  # pooled variance: assumes equal variances
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return p_from_t((a.mean() - b.mean()) / math.sqrt(sp2 * (1 / na + 1 / nb)), na + nb - 2, alt)


def welch_t(a, b, alt):  # separate variances + Welch-Satterthwaite df
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return p_from_t((a.mean() - b.mean()) / math.sqrt(va + vb), df, alt)


def mann_whitney(a, b, alt):  # rank-sum, normal approximation with tie correction and continuity
    n1, n2 = len(a), len(b)
    allv = np.r_[a, b]
    u = ranks(allv)[:n1].sum() - n1 * (n1 + 1) / 2
    _, t = np.unique(allv, return_counts=True)
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - (t**3 - t).sum() / (n * (n - 1))))
    d = u - n1 * n2 / 2
    return p_from_z((d - 0.5 * np.sign(d)) / sd, alt)


def paired_t(d, alt):
    return p_from_t(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))), len(d) - 1, alt)


def wilcoxon(d, alt):  # signed-rank, normal approximation
    d = d[d != 0]
    n = len(d)
    r = ranks(np.abs(d))
    w = r[d > 0].sum() - n * (n + 1) / 4
    _, t = np.unique(np.abs(d), return_counts=True)
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - (t**3 - t).sum() / 48)
    return p_from_z((w - 0.5 * np.sign(w)) / sd, alt)


def pearson(x, y, alt):
    r = np.corrcoef(x, y)[0, 1]
    return p_from_t(r * math.sqrt((len(x) - 2) / (1 - r * r)), len(x) - 2, alt)


def spearman(x, y, alt):
    return pearson(ranks(x), ranks(y), alt)


# method -> (design it belongs to, function)
METHODS = {
    "student_t": ("two_group", student_t),
    "welch_t": ("two_group", welch_t),
    "mann_whitney": ("two_group", mann_whitney),
    "paired_t": ("paired", paired_t),
    "wilcoxon": ("paired", wilcoxon),
    "pearson": ("correlation", pearson),
    "spearman": ("correlation", spearman),
}

# ----------------------------------------------------------------------------------------------
# 2. A tiny world. Each world = (data kind, perturbation). RULES is the "policy registry":
#    A = admissible methods (first = conventional reference), B = invalid methods with a reason.

WORLDS = [
    ("two_group", "none"),
    ("two_group", "unequal_var"),
    ("two_group", "outliers"),
    ("paired", "none"),
    ("paired", "outliers"),
    ("correlation", "none"),
    ("correlation", "outliers"),
]
IGNORE_PAIRS = {"student_t": "ignores pairing", "welch_t": "ignores pairing", "mann_whitney": "ignores pairing"}
RULES = {
    ("two_group", "none"): (["welch_t", "student_t", "mann_whitney"], {}),
    ("two_group", "unequal_var"): (["welch_t"], {"student_t": "pooled variance, unequal variances"}),
    ("two_group", "outliers"): (["mann_whitney"], {"student_t": "outlier-driven", "welch_t": "outlier-driven"}),
    ("paired", "none"): (["paired_t", "wilcoxon"], IGNORE_PAIRS),
    ("paired", "outliers"): (["wilcoxon"], {**IGNORE_PAIRS, "paired_t": "outlier-driven"}),
    ("correlation", "none"): (["pearson", "spearman"], {}),
    ("correlation", "outliers"): (["spearman"], {"pearson": "leverage-driven"}),
}
EFFECT_Z = {"null": 0.0, "small": 2.5, "large": 4.0}
WORDS = {"greater": ("increase", "raise"), "less": ("reduce", "lower"), "two-sided": ("change", "affect")}


def outliers(x, rng, k=2, sign=None):
    x = x.copy()
    idx = rng.choice(len(x), k, replace=False)
    s = rng.choice([-1, 1], k) if sign is None else np.full(k, sign)
    x[idx] = np.median(x) + s * rng.uniform(8, 12, k) * x.std()
    return x


def make_task(rng, world=None, effect=None, alt=None):
    kind, pert = world or WORLDS[rng.integers(len(WORLDS))]
    effect = effect or ["null", "small", "large"][rng.integers(3)]
    alt = alt or ["two-sided", "two-sided", "greater", "less"][rng.integers(4)]
    z, sgn = EFFECT_Z[effect], {"greater": 1, "less": -1}.get(alt, rng.choice([-1, 1]))
    n = int(rng.integers(15, 41))
    if kind == "two_group":
        na, nb, sb = (2 * n, n, 3.0) if pert == "unequal_var" else (n, n, 1.0)  # big variance in the SMALL group
        delta = sgn * z * math.sqrt(1 / na + sb**2 / nb)
        a, b = rng.normal(0, 1, na), rng.normal(delta, sb, nb)
        if pert == "outliers":  # under the null, contaminate BOTH groups alike so H0 stays true
            b = outliers(b, rng, sign=None if effect == "null" else sgn)
            a = outliers(a, rng) if effect == "null" else a
        data = {"a": b, "b": a}  # effect = comparison - reference
        q = f"does the treatment {WORDS[alt][rng.integers(2)]} the score compared with control ?"
    elif kind == "paired":
        d = sgn * z / math.sqrt(n) + rng.normal(0, 1, n)
        if pert == "outliers":
            d = outliers(d, rng, sign=None if effect == "null" else sgn)
        pre = rng.normal(50, 5, n)
        data = {"pre": pre, "post": pre + d}
        q = f"did the score {WORDS[alt][rng.integers(2)]} from before to after ?"
    else:
        r = sgn * math.tanh(z / math.sqrt(n - 3))
        x = rng.normal(0, 1, n)
        y = r * x + math.sqrt(1 - r * r) * rng.normal(0, 1, n)
        if pert == "outliers":
            k = 2
            x[:k] = rng.uniform(8, 12, k)
            y[:k] = (sgn if effect != "null" else rng.choice([-1, 1], k)) * rng.uniform(8, 12, k)
        data = {"x": x, "y": y}
        q = f"does study time {WORDS[alt][rng.integers(2)]} the score ?"
    A, B = RULES[(kind, pert)]
    return {
        "kind": kind,
        "pert": pert,
        "effect": effect,
        "alt": alt,
        "data": data,
        "question": q,
        "obs": observe(kind, data, q),
        "A": A,
        "B": B,
        "gold": [kind, A[0], alt],
        "textbook": [kind, TEXTBOOK[kind], alt],
    }


# A "textbook analyst" always reaches for the classic parametric test, whatever the data look like.
# Fine-tuning on these demonstrations teaches a realistic bad habit; RL then has to un-learn it.
TEXTBOOK = {"two_group": "student_t", "paired": "paired_t", "correlation": "pearson"}


def observe(kind, data, q):
    """What the agent may see: the question words plus a data PROFILE (no p-values, no answers)."""

    def rz(v):
        mad = np.median(np.abs(v - np.median(v))) or 1.0
        return np.max(np.abs(0.6745 * (v - np.median(v)) / mad))

    samples = {"two_group": ["a", "b"], "paired": [], "correlation": ["x", "y"]}[kind]
    vals = [data[s] for s in samples] or [data["post"] - data["pre"]]
    toks = (
        q.split()
        + {"two_group": ["cols:group"], "paired": ["cols:group", "ids:repeat"], "correlation": ["cols:xy"]}[kind]
    )
    toks.append("outl:yes" if max(rz(v) for v in vals) > 5 else "outl:no")
    if kind == "two_group":
        ratio = max(v.var() for v in vals) / min(v.var() for v in vals)
        toks.append("var:high" if ratio > 4 else "var:low")
    return toks


# ----------------------------------------------------------------------------------------------
# 3. Typed plans + verifier. A plan is three tokens: design, method, tail. No free text, no code.

TAILS = ["two-sided", "greater", "less"]


def execute(task, plan):
    """Verifier + executor. Returns a p-value, or None if the plan is malformed or cannot run."""
    if len(plan) != 3 or plan[1] not in METHODS or plan[2] not in TAILS:
        return None
    design, (m_design, fn) = plan[0], METHODS[plan[1]]
    if design != m_design:
        return None  # method does not belong to the stated design
    d, kind = task["data"], task["kind"]
    if design == "two_group" and kind in ("two_group", "paired"):
        a, b = (d["a"], d["b"]) if kind == "two_group" else (d["post"], d["pre"])  # paired data CAN be misanalysed
        return fn(a, b, plan[2])
    if design == "paired" and kind == "paired":
        return fn(d["post"] - d["pre"], plan[2])
    if design == "correlation" and kind == "correlation":
        return fn(d["x"], d["y"], plan[2])
    return None  # e.g. "paired" on data without repeated ids


# ----------------------------------------------------------------------------------------------
# 4. Rewards. zc(p) = Phi^-1(1 - p/2) clipped to [0, 5]: "how significant", on a z scale.


def zc(p):
    return float(min(5.0, -_norm_ppf(p / 2))) if p > 0 else 5.0


def _norm_ppf(q):  # inverse normal CDF by bisection on ndtr (from scratch, plenty accurate here)
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ndtr(mid) < q else (lo, mid)
    return (lo + hi) / 2


def reference_p(task, method):
    return execute(task, [METHODS[method][0], method, task["alt"]])


def fisher_reward(task, plan):
    """Fisher-R1 style: 0.9 * closeness to the single reference p + 0.1 * same decision."""
    p, ref = execute(task, plan), reference_p(task, task["A"][0])
    if p is None:
        return 0.0
    return 0.9 * math.exp(-abs(zc(p) - zc(ref))) + 0.1 * float((p < 0.05) == (ref < 0.05))


def ctv_reward(task, plan):
    """CTV style: method must be admissible (any of A); invalid methods (B) and a changed tail are penalised.
    A plan that cannot run scores LOWER than an invalid method; otherwise the policy learns to dodge the
    penalty by emitting broken plans (try MALFORMED = 0.0 in lesson 4 and watch it happen)."""
    p = execute(task, plan)
    if p is None:
        return MALFORMED
    if plan[1] in task["B"] or plan[2] != task["alt"]:
        return -0.2
    if plan[1] not in task["A"]:
        return 0.0
    refs = [reference_p(task, m) for m in task["A"]]
    close = max(math.exp(-abs(zc(p) - zc(r))) for r in refs)
    return 0.5 + 0.4 * close + 0.1 * float((p < 0.05) == (refs[0] < 0.05))


MALFORMED = -0.5  # reward for a plan that cannot run (CTV). Lesson 4: set it to 0.0 and retrain.
REWARDS = {"fisher": fisher_reward, "ctv": ctv_reward}

# ----------------------------------------------------------------------------------------------
# 5. A tiny GPT (decoder-only transformer), written out like nanoGPT. ~100k parameters.

import torch  # noqa: E402  (sections 1-4 need only numpy/scipy)
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PLAN_VOCAB = ["two_group", "paired", "correlation", *METHODS, *TAILS]
PROMPT_LEN = 16  # prompts are left-padded to this length; the plan always starts at the same position


def build_vocab(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    words = set()
    for _ in range(n):
        words |= set(make_task(rng)["obs"])
    return ["<pad>", "<plan>"] + sorted(words) + PLAN_VOCAB


class SelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.qkv, self.proj = heads, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.h)
        att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))  # causal mask
        return self.proj((F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.attn, self.ln2 = nn.LayerNorm(d), SelfAttention(d, heads), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d=64, heads=4, layers=2, block=PROMPT_LEN + 3):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab_size, d), nn.Embedding(block, d)
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying, as in GPT-2

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.ln(self.blocks(x)))  # (B, T, vocab) next-token logits


SLOTS = [["two_group", "paired", "correlation"], list(METHODS), TAILS]  # plan grammar: design, method, tail


class Agent:
    """Tokenizer + model + grammar-constrained sampling: everything that turns an observation into a plan."""

    def __init__(self, vocab, seed=0, **kw):
        torch.manual_seed(seed)
        self.vocab, self.stoi = vocab, {w: i for i, w in enumerate(vocab)}
        self.model = TinyGPT(len(vocab), **kw)
        self.slot_ids = [torch.tensor([self.stoi[w] for w in slot]) for slot in SLOTS]

    def encode(self, obs):
        ids = [self.stoi[w] for w in obs][-(PROMPT_LEN - 1) :] + [self.stoi["<plan>"]]
        return [0] * (PROMPT_LEN - len(ids)) + ids

    @torch.no_grad()
    def act(self, prompts, greedy=False):
        x = torch.tensor(prompts)
        for ids in self.slot_ids:  # each slot may only emit words of its own type
            logits = self.model(x)[:, -1, ids]
            nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)
            x = torch.cat([x, ids[nxt][:, None]], dim=1)
        return x  # prompt + 3 plan tokens

    def slot_logprobs(self, seqs):
        """Per slot: (log-probs over that slot's words, index of the word actually in seqs)."""
        logits = self.model(seqs[:, :-1])[:, PROMPT_LEN - 1 :, :]
        out = []
        for k, ids in enumerate(self.slot_ids):
            lp = torch.log_softmax(logits[:, k, ids], -1)
            target = (seqs[:, PROMPT_LEN + k, None] == ids[None, :]).float().argmax(-1)
            out.append((lp, target))
        return out

    def logprob(self, seqs):
        """log pi(plan | prompt) = sum of the 3 slot log-probabilities."""
        return sum(lp.gather(-1, t[:, None]).squeeze(-1) for lp, t in self.slot_logprobs(seqs))

    def decode(self, seqs):
        return [[self.vocab[i] for i in s[PROMPT_LEN:].tolist()] for s in seqs]


# ----------------------------------------------------------------------------------------------
# 6. Supervised fine-tuning on demonstrations. Label smoothing keeps some probability on alternatives,
#    so that RL can still explore afterwards (a fully confident policy never samples anything new).


def sft(agent, tasks, steps=150, lr=3e-3, batch=32, seed=0, teacher="textbook", smoothing=0.3):
    """teacher="textbook": classic parametric habits; teacher="gold": the registry-preferred plan."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(agent.model.parameters(), lr=lr)
    data = torch.tensor([agent.encode(t["obs"]) + [agent.stoi[w] for w in t[teacher]] for t in tasks])
    for _ in range(steps):
        b = data[torch.tensor(rng.integers(0, len(data), batch))]
        loss = sum(F.cross_entropy(lp, t, label_smoothing=smoothing) for lp, t in agent.slot_logprobs(b)) / 3
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# ----------------------------------------------------------------------------------------------
# 7. GRPO: sample G plans per task, score them, normalise rewards within the group, push up good ones.


def grpo(agent, tasks, reward, rollouts=25600, group=8, batch=8, lr=3e-4, beta=0.02, seed=0, log=None):
    rng = np.random.default_rng(seed)
    ref = copy.deepcopy(agent.model).eval()  # frozen SFT reference for the KL penalty
    opt = torch.optim.AdamW(agent.model.parameters(), lr=lr)
    used = 0
    while used < rollouts:  # compute budget = number of generated plans, identical for every reward
        picked = [tasks[i] for i in rng.integers(0, len(tasks), batch)]
        prompts = [agent.encode(t["obs"]) for t in picked for _ in range(group)]
        seqs = agent.act(prompts)
        plans = agent.decode(seqs)
        r = torch.tensor([REWARDS[reward](picked[i // group], p) for i, p in enumerate(plans)]).view(batch, group)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1)
        used += len(plans)
        lp = agent.logprob(seqs)
        with torch.no_grad():
            ref_model, agent.model = agent.model, ref
            lp_ref = agent.logprob(seqs)
            agent.model = ref_model
        kl = torch.exp(lp_ref - lp) - (lp_ref - lp) - 1  # k3 estimator of KL(policy || ref)
        loss = -(adv * lp).mean() + beta * kl.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log is not None:
            log.append(float(r.mean()))
    return used


# ----------------------------------------------------------------------------------------------
# 8. Evaluation: validity, invalid-method rate, tail accuracy, and the agent's Type I error rate.


def evaluate(agent, tasks):
    plans = agent.decode(agent.act([agent.encode(t["obs"]) for t in tasks], greedy=True))
    out = {"valid": 0, "invalid": 0, "tail_ok": 0, "null_runs": 0, "false_pos": 0}
    for t, p in zip(tasks, plans):
        out["valid"] += p[0] == t["kind"] and p[1] in t["A"] and p[2] == t["alt"]
        out["invalid"] += p[1] in t["B"]
        out["tail_ok"] += p[2] == t["alt"]
        pv = execute(t, p)
        # Type I error only where the chosen test's null is TRUE. Exception: correlation + leverage points,
        # where rho = 0 holds but x and y are not independent (Spearman's null is false) - see lesson 2.
        if t["effect"] == "null" and pv is not None and (t["kind"], t["pert"]) != ("correlation", "outliers"):
            out["null_runs"] += 1
            out["false_pos"] += pv < 0.05
    n = len(tasks)
    return {
        "validity": out["valid"] / n,
        "invalid_rate": out["invalid"] / n,
        "tail_acc": out["tail_ok"] / n,
        "agent_FPR": out["false_pos"] / max(out["null_runs"], 1),
    }


# ----------------------------------------------------------------------------------------------
# 9. The experiment: SFT, then GRPO with each reward at the same rollout budget, several seeds.


def experiment(seeds=(1, 2, 3), n_sft=40, rollouts=25600, quiet=False):
    vocab = build_vocab()
    test = [make_task(np.random.default_rng(10_000 + i)) for i in range(600)]
    nulls = [make_task(np.random.default_rng(20_000 + i), effect="null") for i in range(700)]
    rows = {"S0 (SFT only)": [], "GRPO fisher": [], "GRPO ctv": []}
    for s in seeds:
        rng = np.random.default_rng(s)
        train = [make_task(rng) for _ in range(2000)]
        base = Agent(vocab, seed=s)
        sft(base, train[:n_sft], seed=s)  # short SFT on textbook habits; RL must fix them
        for name, reward in [("S0 (SFT only)", None), ("GRPO fisher", "fisher"), ("GRPO ctv", "ctv")]:
            agent = copy.deepcopy(base)
            if reward:
                grpo(agent, train, reward, rollouts=rollouts, seed=s)
            ev = evaluate(agent, test)
            ev["agent_FPR"] = evaluate(agent, nulls)["agent_FPR"]
            rows[name].append(ev)
            if not quiet:
                print(f"seed {s} {name:15s} " + "  ".join(f"{k} {v:.3f}" for k, v in ev.items()))
    print(f"\n{'arm':16s}" + "".join(f"{k:>18s}" for k in rows["S0 (SFT only)"][0]))
    for name, evs in rows.items():
        print(
            f"{name:16s}"
            + "".join(f"{np.mean([e[k] for e in evs]):>10.3f} +/-{np.std([e[k] for e in evs]):.3f}" for k in evs[0])
        )
    print(f"(each GRPO arm generated exactly {rollouts} plans per seed; same SFT start per seed)")
    return rows


# ----------------------------------------------------------------------------------------------
# 10. CLI: one sub-command per lesson.


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cmd", choices=["stats", "null", "tasks", "rewards", "model", "sft", "grpo", "compare"])
    ap.add_argument("--reward", default="ctv", choices=list(REWARDS))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--rollouts", type=int, default=25600)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    if a.cmd == "stats":  # Lesson 1: our from-scratch tests vs scipy
        from scipy import stats as S

        x, y = rng.normal(0.5, 1, 25), rng.normal(0, 2, 30)
        for name, ours, ref in [
            ("student_t", student_t(x, y, "two-sided"), S.ttest_ind(x, y).pvalue),
            ("welch_t", welch_t(x, y, "two-sided"), S.ttest_ind(x, y, equal_var=False).pvalue),
            ("pearson", pearson(x[:25], y[:25], "two-sided"), S.pearsonr(x[:25], y[:25])[1]),
        ]:
            print(f"{name:10s} ours {ours:.6f}  scipy {ref:.6f}")
    elif a.cmd == "null":  # Lesson 2: how often does each test reject when nothing is going on?
        for world in WORLDS:
            hits, runs = {}, 600
            for i in range(runs):
                t = make_task(np.random.default_rng(i), world=world, effect="null", alt="two-sided")
                for m, (des, _) in METHODS.items():
                    p = execute(t, [des, m, "two-sided"])
                    if p is not None:
                        hits[m] = hits.get(m, 0) + (p < 0.05)
            print(f"{world[0]:12s} {world[1]:12s} " + "  ".join(f"{m} {v / runs:.3f}" for m, v in hits.items()))
    elif a.cmd == "tasks":  # Lesson 3: what a task, its answer key and a plan look like
        for _ in range(3):
            t = make_task(rng)
            print("observation:", " ".join(t["obs"]))
            print("  gold plan:", t["gold"], "| A:", t["A"], "| B:", t["B"], "| effect:", t["effect"], "\n")
    elif a.cmd == "rewards":  # Lesson 4: score good, invalid and broken plans under both rewards
        t = make_task(rng, world=("two_group", "outliers"), effect="small")
        for plan in [
            t["gold"],
            ["two_group", "welch_t", t["alt"]],
            ["two_group", "student_t", t["alt"]],
            ["two_group", t["gold"][1], "greater" if t["alt"] != "greater" else "less"],
            ["paired", "wilcoxon", t["alt"]],
        ]:
            print(
                f"{str(plan):50s} p={execute(t, plan)}  fisher={fisher_reward(t, plan):.3f}  ctv={ctv_reward(t, plan):.3f}"
            )
    elif a.cmd == "model":  # Lesson 5: the model itself
        agent = Agent(build_vocab())
        print(agent.model)
        print("parameters:", sum(p.numel() for p in agent.model.parameters()))
        print("untrained plans:", agent.decode(agent.act([agent.encode(make_task(rng)["obs"])] * 3)))
    elif a.cmd in ("sft", "grpo"):  # Lessons 6-7
        vocab, train = build_vocab(), [make_task(rng) for _ in range(2000)]
        test = [make_task(np.random.default_rng(10_000 + i)) for i in range(600)]
        agent = Agent(vocab)
        print("SFT (textbook teacher) final loss", round(sft(agent, train[:40]), 3), "| eval", evaluate(agent, test))
        if a.cmd == "grpo":
            log = []
            grpo(agent, train, a.reward, rollouts=a.rollouts, log=log)
            print(
                f"GRPO[{a.reward}] mean reward first/last 20 steps: {np.mean(log[:20]):.3f} -> {np.mean(log[-20:]):.3f}"
            )
            print("eval", evaluate(agent, test))
    else:  # Lesson 8: the compute-matched experiment
        experiment(tuple(int(s) for s in a.seeds.split(",")), rollouts=a.rollouts)


if __name__ == "__main__":
    main()
