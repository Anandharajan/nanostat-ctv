# NanoStat-CTV

**A local statistical co-pilot that tells you which tests are defensible for your design, runs them deterministically, and leaves an audit trail showing it did not fish for significance.**

NanoStat-CTV turns a research question + a CSV into a **typed analysis plan** (JSON), checks the plan against your design and data, runs it with a whitelisted CPU executor, and returns a result, a clarification request, or a justified abstention — plus a reproducible audit bundle.

> **Status — v0.4.1 (CPU release).** Executor, two-gate verifier with a locked analysis protocol, rule-based planner, CLI, local API, audit bundle, synthetic benchmark factory, the CTV reward/constraint library, a clean-room P-Bench scorer and pinned P-Bench harness (coverage ceiling 97.9%), nine model-based tools (OLS/robust/cluster, Wald/F, logistic, probit, 2SLS, log-rank, Cox, mixed, Tobit), a local-LLM planner via [Magnitude](docs/LOCAL_LLM_VALIDATION.md), and a measured CPU ablation of the training signal are implemented and tested. **The 1.5B LLM planner has not been trained** (GPU milestones: NOT RUN). Today the planner is a deterministic rule system. See [STATUS.md](STATUS.md), [docs/AUDIT.md](docs/AUDIT.md) and [docs/limitations.md](docs/limitations.md).

## Learn how it works: nanostat (no install needed)

**[Try the in-browser demo and 9 lessons →](https://anandharajan.github.io/nanostat-ctv/)**

A nanoGPT-style teaching track: one file ([`edu/nanostat.py`](edu/nanostat.py)) with hypothesis tests written from scratch, answer keys, typed plans, two verifiable rewards, a 104k-parameter GPT, SFT, GRPO and a compute-matched experiment. It includes a lesson that reproduces reward hacking on purpose. Run it in [Colab](https://colab.research.google.com/github/Anandharajan/nanostat-ctv/blob/main/edu/nanostat_colab.ipynb) or locally with `pip install numpy scipy torch`. See [edu/README.md](edu/README.md).

## Quickstart (≈2 minutes, CPU only)

```bash
pip install git+https://github.com/Anandharajan/nanostat-ctv
git clone https://github.com/Anandharajan/nanostat-ctv && cd nanostat-ctv   # for the example data

nanostat analyze \
  --question "Did systolic blood pressure decrease from baseline to week 8?" \
  --csv examples/bp_paired.csv --study examples/bp_paired_study.json --rules-only
```

Real output:

```text
action      : execute
method      : paired_t  [admissible]
design      : paired  (paired/continuous/none)
alternative : less
statistic   : -5.69148
p-value     : 1.86236e-06   (alpha = 0.05) -> reject
effect      : -6.8  (mean paired difference (comparison - reference))
CI          : [-inf, -4.76994]
n           : 30
admissible methods on your data (sensitivity range):
  - paired_t               p = 1.86236e-06
  - permutation_test       p = 0.0001
  - wilcoxon_signed_rank   p = 1.81071e-05
note        : policy cell paired/continuous/none is reviewed_by=PENDING (pilot use only)
run id      : e86fd9625895a724   (nanostat bundle --run e86fd9625895a724 --out bundle.zip)
```

With outliers in the data, Student/Welch t become invalid, and **what you ask decides the test** (the estimand is locked from your question, never chosen by the model):

```bash
nanostat analyze -q "Does the flipped section increase the typical exam score compared with lecture?" \
  --csv examples/exam_outliers.csv --study examples/exam_outliers_study.json --rules-only
# method : mann_whitney [admissible]       (estimand: location shift)
nanostat analyze -q "Does the flipped section increase the mean exam score compared with lecture?" \
  --csv examples/exam_outliers.csv --study examples/exam_outliers_study.json --rules-only
# method : permutation_test [admissible]   (estimand: mean difference)
```

Export the audit bundle and re-run it anywhere with only numpy/pandas/scipy:

```bash
nanostat bundle --run e86fd9625895a724 --out bundle.zip
unzip bundle.zip -d b && cd b && python analysis.py     # prints the same p-value
```

### The study file

`--study` tells NanoStat which column plays which role. Everything is optional except the roles.

```json
{
  "roles": {"y": "sbp", "group": "visit", "pair_id": "patient_id"},
  "group_order": ["baseline", "week_8"],
  "description": "The same patients were measured at baseline and at week 8."
}
```

| Role | Meaning |
|---|---|
| `y` | outcome |
| `group` | group / time-point label (first in `group_order` is the reference) |
| `pair_id` | subject ID for paired (long-format) data |
| `y1`, `y2` | paired data in wide format |
| `x` | second variable for correlation |

Optional: `design` (`independent_two_group`, `paired`, `correlation`, `k_group`, `one_sample`), `outcome` (`continuous`/`binary`), `p0` (null proportion for one-sample tests), and the protocol fields `alpha` (default 0.05), `alternative`, `estimand_family` (`mean` / `typical`), `allow_population_filters` (default false).

**The analysis protocol is locked before planning** (`nanostat/protocol.py`): alpha, claim direction, estimand family, null value and population come from your study file or, if absent, deterministically from the question ("increase/higher" → *greater*, "reduce/lower" → *less*, otherwise two-sided; "mean/average" → mean estimand, "typical/median" → location). A plan that tries to change any of them is rejected. Effects are *comparison − reference*.

**Two-gate verification:** the plan and its preprocessing are checked against the protocol, preprocessing runs, and the *analysed* rows are re-profiled and re-verified (design, pairing, estimand, method status) before the test runs; their hash is recorded. An invalid method never gets a confirmatory decision (`nanostat analyze --diagnostic` returns the p-value without a decision).

If a needed fact is missing — e.g. the description says the same people were measured twice but there is no ID column — NanoStat **asks** (`action: clarify, reason: missing_pair_id`) instead of guessing.

## What it supports (v0.2)

| Design | Methods (whitelisted) |
|---|---|
| Independent two-group, continuous | Welch t, Student t, permutation (means), Mann–Whitney U, Yuen 20% trimmed t |
| Independent two-group, binary | χ² test of independence, Fisher exact |
| Paired | paired t, permutation (sign-flip), Wilcoxon signed-rank |
| Correlation | Pearson, permutation (Pearson r), Spearman |
| k groups | one-way ANOVA, Kruskal–Wallis |
| One-sample proportion | exact binomial |
| Regression (`design: regression`, typed `model`) | OLS coefficient (classical, HC0–HC3, cluster-robust, WLS), Wald/F restriction, logistic, probit, 2SLS (IV), linear mixed model, Tobit |
| Survival (`design: survival`) | log-rank, Cox PH (Efron/Breslow) |

The synthetic factory also generates **regression/survival tasks with specification-level answer keys** (right covariates, SE type, clustering level, instrument) and spec twins: see [docs/REGRESSION_FAMILIES.md](docs/REGRESSION_FAMILIES.md).

Every method is checked against an **independent textbook implementation** on ≥5 fixtures (`tests/unit/test_tools_fixtures.py`).

The **policy registry** (`configs/policies/m0_m1.yaml`, v0.2.0) lists, per *(design, outcome, data perturbation)* cell and **per estimand**, which methods are admissible and which are invalid with a reason (e.g. `student_t: outlier_driven_inference`, `mann_whitney: estimand_mismatch` for a mean question). Every cell is currently `reviewed_by: PENDING` — no statistician has signed it off yet.

## Local API

```bash
nanostat serve            # binds 127.0.0.1:8765
curl -s localhost:8765/health
curl -s -X POST localhost:8765/v1/analyze -H 'content-type: application/json' \
  -d '{"question":"Is there a difference in score between A and B?","csv":"g,score\nA,1\nA,2\nA,3\nB,4\nB,5\nB,6\n","study":{"roles":{"y":"score","group":"g"}}}'
```

Endpoints: `POST /v1/analyze`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/bundle`, `GET /health`. CSV is uploaded in the body (no server paths); limits 20 MB / 100k rows / 200 columns.

## Research side: CTV rewards and the synthetic benchmark

The research question ([SPEC.md](SPEC.md)): at matched compute, do verifiable rewards that know the **admissible method set (A)**, sample by **discordance (D)**, train on **counterfactual design twins (T)** and constrain **agent-level Type I error (V)** give a 1.5B agent better method validity and error control than Fisher-R1-style p-value-matching rewards?

v0.2 contains the CPU half of that pipeline and a measured ablation of the training signal:

```bash
nanostat factory --profile pilot --out data          # tasks, CSVs, answer keys, twins, manifests
nanostat eval --arms B0 --suite nullcal --sims 1000   # agent-level false-positive rate (hypothesis-specific)
nanostat ablate --data data --seeds 1,2,3,4,5         # CPU ablation: S0, R-F, R-A, R-AD, R-ADT, R-CTV
nanostat train --stage sft-data --data data --out data/sft.jsonl
```

- `benchmark/factory` — seeded DGPs with **per-hypothesis null labels**, 7 twin edits (incl. estimand change), group-exclusive splits.
- `benchmark/reference.py`, `benchmark/discordance.py` — run A(x,e) ∪ B(x,e), compute D(x).
- `training/reward_server.py` — CTV reward with evidence-grounded assumption credit, and the Fisher-style comparator (R-F).
- `training/constraint.py` — scenario-specific Lagrangian FPR/power constraint and twin-joint advantage.
- `training/ablation/` — the CPU ablation (proxy policy). `benchmark/pbench/score.py` — clean-room P-Bench scorer.
- LLM training and P-Bench evaluation are **not run** (no GPU here; P-Bench download awaits approval).

### Measured results

**CPU ablation of the training signal** ([docs/ABLATION_RESULTS.md](docs/ABLATION_RESULTS.md)) — 1,400 synthetic tasks, linear-softmax *proxy policy* (not the LLM), **compute-matched at 19,200 generated rollouts per arm**, 5 seeds, locked test split (505 tasks, 88 twin pairs):

| Arm | Method validity | Invalid-method rate | Twin consistency | Abstention correct | Strict (single ref.) |
|---|---|---|---|---|---|
| S0 (SFT only) | 0.795 ± 0.073 | 0.109 | 0.570 | 0.236 | 0.787 |
| R-F (Fisher-style reward) | 0.880 ± 0.074 | 0.116 | 0.580 | 0.000 | 0.876 |
| R-A | 0.922 ± 0.068 | 0.011 | 0.661 | 0.440 | 0.890 |
| R-AD | 0.939 ± 0.083 | 0.014 | 0.643 | 0.012 | 0.900 |
| R-ADT | 0.933 ± 0.095 | 0.015 | 0.657 | 0.044 | 0.898 |
| **R-CTV** | **0.938 ± 0.086** | 0.015 | **0.664** | 0.048 | **0.908** |
| R-CTV-abs (post-hoc) | 0.877 ± 0.107 | 0.018 | 0.648 | 0.488 | 0.842 |

R-CTV − R-F (paired bootstrap, Holm; v0.4.1 reward): validity **+0.064 [+0.050, +0.079]**, twin consistency **+0.091 [+0.050, +0.141]**, agent FPR no difference. The admissible-set reward (A) carries the effect; D, T and V were not separable from A at this scale; discordance sampling starves abstention tasks. *An earlier version of this table was not compute-matched (uncounted re-samples); see the correction notices in the results document. A second correction (v0.4.1) fixed a reward ordering that let broken plans score above invalid ones.*

**Agent-level null calibration of B0** (1,000 fresh null simulations per scenario, judged against the null of the test actually run): every agent false-positive rate is within or below 0.05 ± CI (two-group clean small: 0.063 [0.049, 0.080], the same as fixed Welch on identical simulations). The registry's invalid labels are confirmed by the fixed-method arms: Student t 14.1% and permutation 14.5% under unequal variances; Pearson 48.8% under leverage points; the agent avoids them. Full table in [docs/ABLATION_RESULTS.md](docs/ABLATION_RESULTS.md).

**B0 on NanoStat-Bench** is circular by construction (keys and rule planner share the registry): 455/455 valid on the locked test, 0 invalid, 88/88 twin pairs, 50/50 abstentions — a consistency check, not evidence of statistical skill.

No LLM arm and no P-Bench evaluation has been run. No comparison with Fisher-R1 is claimed; their published values are recorded in [docs/FISHER_R1_BASELINE.md](docs/FISHER_R1_BASELINE.md).

## Development

```bash
pip install -e ".[dev]"
pytest            # 197 tests, ~35 s
ruff check . && mypy
```

Rules for contributors (and coding agents) are in [CLAUDE.md](CLAUDE.md); the acceptance contract is [SPEC.md](SPEC.md).

## Licence and citation

Code: Apache-2.0 ([LICENSE](LICENSE)). Third-party notes: [NOTICE.md](NOTICE.md). Cite via [CITATION.cff](CITATION.cff).

Not for autonomous clinical or regulatory decisions.
