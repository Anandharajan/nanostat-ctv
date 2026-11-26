# NanoStat-CTV

**A local statistical co-pilot that tells you which tests are defensible for your design, runs them deterministically, and leaves an audit trail showing it did not fish for significance.**

NanoStat-CTV turns a research question + a CSV into a **typed analysis plan** (JSON), checks the plan against your design and data, runs it with a whitelisted CPU executor, and returns a result, a clarification request, or a justified abstention — plus a reproducible audit bundle.

> **Status — v0.1.0 (CPU release).** The executor, verifier, rule-based planner, CLI, local API, audit bundle, synthetic benchmark factory and the CTV reward/constraint library are implemented and tested. **The 1.5B LLM planner has not been trained** (GPU milestones M2–M7: NOT RUN). Today the planner is a deterministic rule system. See [STATUS.md](STATUS.md) and [docs/limitations.md](docs/limitations.md).

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
  - wilcoxon_signed_rank   p = 1.81071e-05
note        : policy cell paired/continuous/none is reviewed_by=PENDING (pilot use only)
run id      : 112b739e6f83d343   (nanostat bundle --run 112b739e6f83d343 --out bundle.zip)
```

With outliers in the data, the same command picks a rank/robust method and marks Student/Welch t as invalid:

```bash
nanostat analyze -q "Does the flipped section increase exam score compared with lecture?" \
  --csv examples/exam_outliers.csv --study examples/exam_outliers_study.json --rules-only
# method : mann_whitney [admissible]   cell: independent_two_group/continuous/heavy_outliers
```

Export the audit bundle and re-run it anywhere with only numpy/pandas/scipy:

```bash
nanostat bundle --run 112b739e6f83d343 --out bundle.zip
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

Optional: `design` (`independent_two_group`, `paired`, `correlation`, `k_group`, `one_sample`), `outcome` (`continuous`/`binary`), `p0` (null proportion for one-sample tests).

**Claim direction** comes from your question: "increase/higher/improve" → one-sided *greater*; "reduce/decrease/lower" → *less*; otherwise two-sided. Effects are *comparison − reference*.

If a needed fact is missing — e.g. the description says the same people were measured twice but there is no ID column — NanoStat **asks** (`action: clarify, reason: missing_pair_id`) instead of guessing.

## What it supports (v0.1)

| Design | Methods (whitelisted) |
|---|---|
| Independent two-group, continuous | Welch t, Student t, Mann–Whitney U, Yuen 20% trimmed t |
| Independent two-group, binary | χ² test of independence, Fisher exact |
| Paired | paired t, Wilcoxon signed-rank |
| Correlation | Pearson, Spearman |
| k groups | one-way ANOVA, Kruskal–Wallis |
| One-sample proportion | exact binomial |

Every method is checked against an **independent textbook implementation** on ≥5 fixtures (`tests/unit/test_tools_fixtures.py`).

The **policy registry** (`configs/policies/m0_m1.yaml`) lists, per *(design, outcome, data perturbation)* cell, which methods are admissible and which are invalid with a reason (e.g. `student_t: outlier_driven_inference`). Every cell is currently `reviewed_by: PENDING` — no statistician has signed it off yet.

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

v0.1 contains the CPU half of that pipeline:

```bash
nanostat factory --profile pilot --out data          # tasks, CSVs, answer keys, twins, manifests
nanostat eval --arms B0 --suite nanostat --data data --split dev
nanostat eval --arms B0 --suite nullcal --sims 1000   # agent-level false-positive rate
nanostat train --stage sft-data --data data --out data/sft.jsonl
```

- `benchmark/factory` — seeded DGPs, perturbations, 6 twin edits (wording-fixed and wording-changed), lineage splits, held-out edit types.
- `benchmark/reference.py`, `benchmark/discordance.py` — run A(x) ∪ B(x), compute D(x).
- `training/reward_server.py` — CTV reward (§7.4) and the Fisher-style comparator (arm R-F).
- `training/constraint.py` — Lagrangian FPR/power constraint (§7.6) and twin-joint advantage (§7.5).
- The GRPO loop, SFT training and P-Bench evaluation are **not implemented yet**.

### Measured results (CPU, rule planner B0 only)

Pilot factory (`--profile pilot --seed 20261126`): **1,400 validated tasks** (train 1,026 / dev 132 / test 242), **200 twin pairs** (109 wording-fixed, 91 wording-changed), 0 duplicates, 49 candidates rejected by validation, 29.5 s on a laptop CPU. 63% of tasks have discordance D(x) < 0.1 (methods agree), 6.4% have D(x) ≥ 0.9.

| B0 on NanoStat-Bench | dev (n = 132) | test (n = 242) |
|---|---|---|
| Method-validity rate | 127/127 = 1.000 [0.971, 1.000] | 239/239 = 1.000 [0.985, 1.000] |
| Invalid-method rate | 0/127 | 0/239 |
| Twin-consistency accuracy | 15/15 | 74/74 |
| Abstention correctness | 5/5 | 3/3 |
| Operational failure rate | 0/132 | 0/242 |

**These B0 numbers are circular by construction**: the answer keys are built from the same registry and diagnostics the rule planner uses. They verify implementation consistency and set a ceiling for trained arms; they are *not* evidence of statistical skill. 95% Clopper–Pearson intervals in brackets.

**Agent-level null calibration (B0)** — full agent re-run on 1,000 fresh true-null simulations per scenario (`nanostat eval --arms B0 --suite nullcal --sims 1000`, seed 777, 3 min 19 s). Target α = 0.05; 95% Clopper–Pearson CI.

| Scenario (n regime) | Rejects / 1000 | FPR [95% CI] | Methods the agent chose |
|---|---|---|---|
| two-group, clean (small) | 65 | **0.065 [0.051, 0.082]** | Welch 881, Mann–Whitney 119 |
| two-group, outliers (medium) | 45 | 0.045 [0.033, 0.060] | Mann–Whitney 1000 |
| two-group, skewed (small) | 51 | 0.051 [0.038, 0.067] | Mann–Whitney 934, Welch 66 |
| two-group, unequal variance (small) | 49 | 0.049 [0.036, 0.064] | Welch 921, Mann–Whitney 79 |
| paired, clean (small) | 49 | 0.049 [0.036, 0.064] | paired t 940, Wilcoxon 60 |
| paired, outliers (medium) | 55 | 0.055 [0.042, 0.071] | Wilcoxon 1000 |
| correlation, clean (medium) | 43 | 0.043 [0.031, 0.057] | Pearson 985, Spearman 15 |
| correlation, leverage points (medium) | 72 | **0.072 [0.057, 0.090]** | Spearman 1000 |
| k groups, skewed (small) | 29 | 0.029 [0.020, 0.041] | Kruskal–Wallis 993, ANOVA 7 |
| binary, small expected counts (small) | 8 | 0.008 [0.003, 0.016] | Fisher 946, χ² 2 (52 runs abstained: no events) |

Two scenarios are liberal and are reported, not tuned away: (1) on clean small samples, choosing the test *after* looking at the data (Welch vs Mann–Whitney by diagnostics) inflates the false-positive rate to ~6.5%, a known cost of assumption pre-testing; (2) with symmetric leverage points, x and y are uncorrelated but not independent, which Spearman's null assumes. An earlier run of this table exposed two bugs (one-sided outlier contamination in "null" scenarios; a constant binary outcome falling through to Welch with p = NaN); both are fixed and covered by regression tests.

No LLM arm (B1, S0, R-F, R-A, R-AD, R-ADT, R-CTV) and no P-Bench evaluation has been run. No comparison with Fisher-R1 is claimed.

## Development

```bash
pip install -e ".[dev]"
pytest            # 151 tests, ~20-40 s
ruff check . && mypy
```

Rules for contributors (and coding agents) are in [CLAUDE.md](CLAUDE.md); the acceptance contract is [SPEC.md](SPEC.md).

## Licence and citation

Code: Apache-2.0 ([LICENSE](LICENSE)). Third-party notes: [NOTICE.md](NOTICE.md). Cite via [CITATION.cff](CITATION.cff).

Not for autonomous clinical or regulatory decisions.
