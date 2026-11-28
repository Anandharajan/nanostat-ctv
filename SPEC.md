# NanoStat-CTV — Implementation Specification (v3.0)

**Counterfactual-Twin, Validity-constrained GRPO for small, auditable statistical agents**

| Field | Value |
|---|---|
| Version | 3.0 (supersedes NanoStat-R1 PRD v2.0 §§10.5, 12.1 R-arms, 26.3–26.4) |
| Date | 26 November 2026 |
| Owner | Dr. Anandharajan TRV, Dept. of ECE, St. Joseph College of Engineering |
| Implementation | Claude Code, milestone by milestone, against this file as the acceptance contract |
| Status | Specification only. No experiment has been run; no number in this file is a result. |
| Resource envelope | 1 × T4 16 GB (Colab) for GPU stages; CPU for all statistics, data, rewards, evaluation; ₹0 required API spend |

> **Research question.** At matched compute, do verifiable rewards that know the *admissible method set*, train on *counterfactual design twins*, and constrain *agent-level Type I error against the true data-generating process* give a 1.5B agent better method validity and error control than Fisher-R1-style p-value-matching rewards, without losing P-Bench Strict accuracy?

---

## 0. How to use this file (instructions for Claude Code)

1. Put this file at the repository root as `SPEC.md`. Copy §19 into `CLAUDE.md`.
2. Implement **one milestone at a time** (§14). Do not start a milestone until the previous one's exit tests pass.
3. CPU first. No GPU code before M2. No P-Bench evaluation before the model and protocol are frozen (M5).
4. Never invent scores, logs, citations or platform statuses. Unrun = `NOT RUN`. Failed = report the failure.
5. Keep `STATUS.md` updated after every session: milestone, commands run, test results, blockers, next action, artifact hashes.
6. Ask the maintainer before adding any dependency not listed in §15, any network call, or anything that publishes.

---

## 1. Summary

NanoStat-CTV is an open-source Python framework and a 1.5B QLoRA adapter that answer a research question over a tabular dataset by emitting a **typed analysis plan**. A deterministic CPU executor runs the plan, and the agent returns an auditable result, a clarification request or a justified abstention.

The research contribution is a training signal, **CTV-GRPO**, built from four parts:

| Part | Fixes this weakness in Fisher-R1's reward | Mechanism |
|---|---|---|
| **A. Admissible-set reference** | Only one canonical method gets credit; defensible alternatives are penalised | Reward method membership in a reviewed set A(x); score p-value closeness against the nearest admissible reference |
| **D. Discordance-targeted sampling** | z-space closeness is flat when all methods agree (null, or effect clipped at z = 5) | Precompute, on CPU, how far admissible and invalid methods disagree; sample mostly where method choice changes the answer |
| **T. Counterfactual twins** | Policy can learn a fixed default ("always rank test") | Paired tasks differing by one design edit that flips the correct method; joint reward over the pair |
| **V. Validity constraint** | Reward agrees with the *reference analysis*, never with the *truth* of H0 | Lagrangian constraint on batch false-positive rate under true-null DGPs, with a power floor |

The small-model bet: Fisher-R1's 7B backbone lost 43.5% of tasks to operational failures (code that does not run). A typed plan + whitelisted executor removes that failure class, so a 1.5B model spends its capacity only on design and method reasoning.

---

## 2. Scope

**In scope (this spec):** CPU tool registry and executor; admissible-set policy registry; synthetic task factory with discordance scoring and twins; programmatic SFT; CTV-GRPO on a T4; evaluation on NanoStat-Bench, twin-consistency, agent-level null calibration and full P-Bench; minimal audit bundle; CLI/local API; release package.

**Out of scope for v1.0:** multimodal input, retrieval, multi-GPU, Bayesian testing, causal identification beyond declared design, autonomous clinical use, a hosted public service.

**Carried over from PRD v2.0 unchanged:** §§7–9 data contracts and leakage rules, §24 four pillars at minimum scope, §27.6 publication gates. Where this spec and v2.0 disagree, this spec wins.

---

## 3. Ideal customer profile (ICP) and product framing

| Segment | Profile | Job to be done | What they get from NanoStat-CTV |
|---|---|---|---|
| **Primary ICP** | Empirical researcher (PhD student, postdoc, faculty) in biomedical, social or engineering sciences; has a CSV and a hypothesis; not a statistician; often cannot send data to cloud APIs | "Pick a defensible test and show a reviewer I didn't p-hack" | Local analysis, method rationale, assumption checks, sensitivity range across admissible methods, reproducible ZIP |
| Secondary | Statisticians, journal reviewers, research-integrity offices | Audit someone else's analysis quickly | Typed plan, admissible vs invalid method table, claim-to-result check |
| Secondary | Labs with private data (hospitals, industry R&D) | Statistical help without data leaving the machine | Offline CLI/API, no API keys |
| Developers | Scientific-software and ML researchers | Extend or benchmark statistical agents | CPU-only plugins, twin benchmark, reward library |

**Positioning line:** *"A local statistical co-pilot that tells you which tests are defensible for your design, runs them deterministically, and proves it did not fish for significance."*

**Pilot metrics (hypotheses, measure in M8):** 30 researchers recruited; first supported analysis ≤ 5 min after install; ≥ 70% complete one analysis; ≥ 50% of completers download the audit bundle; ≥ 40% 30-day return. No product-market-fit claim without these observed.

---

## 4. Architecture

```mermaid
flowchart TD
  subgraph CPU
    Q[Question + study metadata + CSV] --> P[Profiler]
    P --> V[Design/assumption verifier]
    V -->|valid| X[Whitelisted executor]
    X --> R[Result + audit bundle]
    V -->|missing facts| C[Clarify]
    V -->|invalid/unsupported| AB[Abstain]
    F[Task factory] --> REF[Reference solver: run A(x) ∪ B(x)]
    REF --> DS[Discordance scorer]
    DS --> TW[Twin builder]
    TW --> M[Split manifests + key store]
    M --> RW[Reward server: A, D, T, V]
  end
  subgraph GPU[T4, one model]
    L[Qwen2.5-Coder-1.5B NF4 + LoRA] --> PL[Typed plan JSON]
  end
  P --> L
  PL --> V
  M --> SFT[Programmatic SFT]
  SFT --> G[CTV-GRPO]
  RW --> G
  G --> L
```

**Invariants**
- The LLM never computes a statistic. All numbers come from the executor.
- The policy never sees A(x), B(x), cached p-values, DGP truth labels, category labels or file names that encode answers. These live only in `keys/` and are read only by the reward server and the scorer.
- The verifier never uses the target p-value to choose or change a method.

---

## 5. Data contracts

### 5.1 Typed plan (model output)

```json
{
  "estimand": "mean_difference_paired",
  "design": "paired",
  "method": "paired_t",
  "columns": {"y1": "pre", "y2": "post", "pair_id": "id"},
  "alternative": "greater",
  "preprocess": [{"op": "drop_missing", "scope": "pairs"}],
  "alpha": 0.05,
  "action": "execute"
}
```

- `action ∈ {execute, clarify, abstain}`; `clarify`/`abstain` carry `reason_code` from a fixed enum.
- `preprocess` uses a typed mini-DSL (§6.3); free-form code is rejected.
- Pydantic v2, `extra="forbid"`, strict enums, column names validated against the profile.

### 5.2 Task record (training/evaluation, agent-visible part)

`task_id` (opaque), `question`, `study` (design metadata), `data_ref`, `budget`.

### 5.3 Key record (never agent-visible; `keys/` store)

```json
{
  "task_id": "t_8f2c…",
  "dgp": {"family": "two_group", "h0_true": false, "effect": "borderline", "perturbation": "outliers"},
  "admissible": [{"method": "mann_whitney", "config": {...}, "p": 0.084},
                 {"method": "welch_t_trimmed", "config": {...}, "p": 0.071}],
  "invalid":    [{"method": "student_t", "config": {...}, "p": 0.003, "reason": "outlier_driven"}],
  "reference_decision": "fail_to_reject",
  "discordance": {"decision_flip": true, "z_gap": 1.83, "score": 0.91},
  "twin_id": "t_19ab…",
  "lineage": {"scenario_group": "sg_041", "template_group": "tg_07", "seed": 123, "generator_version": "0.3.1"},
  "abstain_target": null
}
```

### 5.4 Result

As PRD v2.0 §8.3, plus `admissible_range` (min/max p and effect across A(x) executed on the user's data) and `method_status ∈ {admissible, invalid, unknown}`.

---

## 6. CPU components

### 6.1 Tool registry (`nanostat/tools/`)

Each plugin is a class with: `name`, `estimands`, `designs`, `config_schema`, `run(df, plan) -> Result`, `diagnostics(df, plan)`, `r_reference` (template name), `fixtures`, `tolerance`.

| Gate | Methods |
|---|---|
| M0 | Welch t, Student t, paired t, Mann–Whitney U, Wilcoxon signed-rank, Pearson, Spearman |
| M1 | One-way ANOVA, Kruskal–Wallis, χ² independence, Fisher exact, binomial/proportion test |
| M4 | OLS coefficient (HC1/HC3, cluster-robust), logistic, probit, log-rank, Cox PH, IV/2SLS, Tobit, linear mixed model, F-test; any other category found in the pinned P-Bench manifest |

A plugin is enabled only when its CPU fixtures pass against an independent reference (analytic value, R via `Rscript`, or a second implementation).

### 6.2 Policy registry (`configs/policies/*.yaml`)

For each `(design, outcome_type, perturbation)` cell, list admissible methods/configs, invalid methods with reason codes, and conditions requiring clarification or abstention. Example:

```yaml
- cell: {design: independent_two_group, outcome: continuous, perturbation: heavy_outliers}
  estimand_options:
    location_shift: [mann_whitney, welch_t_trimmed_20]
    mean_difference: [welch_t_bootstrap]
  invalid:
    - {method: student_t, reason: outlier_driven_inference}
    - {method: paired_t, reason: design_mismatch}
  clarify_if: [pair_id_present_but_design_unspecified]
  reviewed_by: PENDING
  version: 0.1.0
```

`reviewed_by` must name a statistician before the cell is used in a confirmatory run. Cells with `PENDING` are allowed only in pilot runs and are logged.

### 6.3 Preprocess DSL

Allowed ops: `drop_missing`, `filter(column, op, value)`, `log`, `log1p`, `recode`, `select_levels`, `winsorize(p)`, `trim(p)`, `standardize`. Each op is logged; `winsorize`/`trim` mark the estimand as changed.

### 6.4 Task factory (`benchmark/factory/`)

Grid (Fisher-R1 taxonomy, own implementation, no reuse of their data):

`method family (M) × scenario (S) × sample-size regime (N) × effect regime {null, borderline, medium, large} × prompt style {hint, no_hint} × perturbation (P) × seed (K)`

Steps per candidate:
1. Sample parameters; simulate data with a seeded NumPy generator; record `h0_true`.
2. Look up the policy cell → A(x), B(x).
3. Run **every** method in A(x) ∪ B(x) on CPU; cache statistic, p, effect, CI.
4. Compute discordance (§7.2).
5. Build twin(s) (§7.3).
6. Render question/metadata from templates; strip answer-bearing wording.
7. Validate: schema, reference fixtures, no NaN p, duplicate/near-duplicate hash check.
8. Assign split by `scenario_group`/`template_group` lineage; twins always share a split.

Targets: ≥ 8,000 unique validated training tasks after filtering; dev 200; locked test 400 (200 twin pairs); null-calibration scenarios ≥ 1,000 simulations each.

### 6.5 Programmatic SFT data

The target for each SFT example is the gold typed plan produced from A(x) (the registry's preferred admissible method) plus a ≤ 60-token rationale rendered from templates. No teacher LLM, no paid API. Abstention/clarification examples come from tasks with `abstain_target`.

---

## 7. CTV-GRPO

### 7.1 Notation

- x: task; o: rollout (typed plan, possibly multi-step within budget); m̂, ĉ: chosen method and config; p̂: executor p-value; d̂: decision at α.
- z(p) = Φ⁻¹(1 − p/2), computed with `scipy.stats.norm.isf(p/2)`; clip to [0, 5] for reward.
- A(x), B(x): admissible / invalid sets with cached p_a, p_b.

### 7.2 Discordance score (CPU, precomputed)

```
z_gap(x)   = min_{a∈A, b∈B} | zc(p_a) − zc(p_b) |          # zc = clipped z
flip(x)    = 1[ any decision(a) ≠ decision(b) ]
D(x)       = 0.5·min(z_gap/2, 1) + 0.5·flip(x)             # ∈ [0, 1]
```

If B(x) is empty, D(x) = 0. Sampling weight for RL: `w(x) ∝ ε_D + D(x)` with ε_D = 0.1 (keeps a floor of easy tasks). Report the corpus distribution of D(x) as a finding.

### 7.3 Counterfactual twins

A twin x′ applies exactly one edit from:

| Edit | Correct method should change from → to |
|---|---|
| independent ↔ paired (metadata and data) | Welch ↔ paired t |
| add high-leverage outliers | Pearson → Spearman; Student t → rank/robust |
| introduce clustering (unit ≠ observation) | OLS SE → cluster-robust |
| two-sided ↔ one-sided claim | tail flip |
| expected counts < 5 | χ² → Fisher exact |
| remove the pairing ID | paired → clarify |
| outcome binary ↔ continuous | t/rank → proportion/logistic |

Two twin kinds, both required: **wording-changed** (metadata text changes) and **wording-fixed** (only data/metadata values change). Hold out at least two edit types entirely for the test set.

### 7.4 Per-rollout reward

```
R(o) = I_valid(o) · [ 0.45·1[m̂ ∈ A]
                    + 0.35·max_{a∈A} exp(−| zc(p̂) − zc(p_a) |)
                    + 0.10·1[d̂ = reference_decision]
                    + 0.10·r_assume(o) ]
R(o) = −0.2                 if m̂ ∈ B(x) or forbidden action or fabricated number
R(o) = 0                    if I_valid(o) = 0 (bad JSON, unknown column, budget exceeded)
```

- `r_assume` ∈ [0,1]: fraction of the registry's required checks for m̂ that the plan requested (the verifier, not the model, decides which checks are required).
- Abstention branch: on tasks with `abstain_target`, `R = 1.0` for the correct `clarify`/`abstain` with a matching `reason_code`, else 0. On answerable tasks, `abstain` gets `−0.1`.
- Weights are starting hypotheses; tune only on dev; freeze before confirmatory runs.

### 7.5 Twin-joint advantage

For a twin pair (x, x′) sample G rollouts each and pair them by index:

```
r_i = ½·(R(o_i) + R(o′_i)) + η·1[R(o_i) ≥ τ ∧ R(o′_i) ≥ τ]      η = 0.2, τ = 0.8
Â_i = (r_i − mean(r)) / (std(r) + 1e-6)
```

Apply Â_i to the assistant tokens of both o_i and o′_i. Groups with zero variance: resample up to 2 times, then skip and log.

### 7.6 Validity constraint (truth-based, batch-level)

Per update, over the null-DGP and alternative-DGP tasks in the batch:

```
FPR_hat ← EMA_0.9( mean 1[d̂ = reject]  over tasks with h0_true )
PWR_hat ← EMA_0.9( mean 1[d̂ = reject]  over tasks with ¬h0_true and effect ∈ {medium, large} )
λ1 ← clip( λ1 + η_λ·(FPR_hat − (α + ε)),       0, λ_max )
λ2 ← clip( λ2 + η_λ·((PWR_ref − ε) − PWR_hat),  0, λ_max )
R′(o) = R(o) − λ1·1[reject ∧ h0_true] + λ2·1[reject ∧ ¬h0_true ∧ effect ≥ medium]
```

Defaults: α = 0.05, ε = 0.02, η_λ = 0.05, λ_max = 0.5. `PWR_ref` = power of the preferred admissible method on the same scenarios (precomputed on CPU). Log FPR_hat, PWR_hat, λ1, λ2 every update.

Why batch-level: a valid test rejects about α of true-null datasets by chance, so a per-task truth reward would push the policy to over-conservative tests. The constraint only bites on systematic liberal choices.

### 7.7 Objective and loop

GRPO (critic-free) with clipped importance ratio; optional DAPO-style asymmetric clip (ε_low 0.20, ε_high 0.28) as a **separately named** arm. KL to the SFT reference: keep the TRL default unless profiling forces a change, and log any change.

```python
for step in range(num_updates):
    pairs = sampler.draw(batch_pairs, weights=discordance_weights)  # §7.2
    rollouts = []
    for x, x_twin in pairs:
        o = policy.generate(prompt(x), n=G, max_new_tokens=256)
        o2 = policy.generate(prompt(x_twin), n=G, max_new_tokens=256)
        r = reward_server.score(x, o)  # CPU, separate process, §7.4
        r2 = reward_server.score(x_twin, o2)
        rollouts.append(joint(o, o2, r, r2))  # §7.5
    constraint.update(rollouts)  # §7.6
    loss = grpo_loss(rollouts, constraint.shape_rewards)
    optimizer_step(loss)  # adapter params only
    checkpoint_if_due(step, every=25)  # policy, optim, sched, RNG, sampler, λ, rollout IDs
```

### 7.8 T4 configuration (`configs/t4_ctv.yaml`)

```yaml
model_id: Qwen/Qwen2.5-Coder-1.5B-Instruct
model_revision: PIN_AFTER_SMOKE
quant: {type: nf4, double_quant: true, compute_dtype: float16}
lora: {r: 8, alpha: 16, dropout: 0.05, targets: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}
sft: {max_seq: 1024, micro_batch: 1, grad_accum: 8, lr: 2.0e-4, epochs: 1, examples: 3000}
grpo:
  group_size: 2            # per twin side; raise to 4 only after profiling
  pairs_per_update: 4
  max_prompt_tokens: 1024
  max_new_tokens: 256
  lr: 5.0e-6
  updates_probe: 50
  updates_pilot: 300
  checkpoint_every: 25
  kl_beta: trl_default
reward: {w_method: 0.45, w_p: 0.35, w_concl: 0.10, w_assume: 0.10, invalid_penalty: -0.2, eta_twin: 0.2, tau_twin: 0.8}
constraint: {alpha: 0.05, eps: 0.02, eta_lambda: 0.05, lambda_max: 0.5, ema: 0.9}
stop_rules: {max_reserved_gib: 14, on_oom: [reduce_new_tokens_128, group_size_2, abort_and_report]}
seeds: [42, 123, 456]
```

---

## 8. Experimental arms (all on the same backbone and budget)

| ID | Training | Purpose |
|---|---|---|
| B0 | none, rule-based planner | CPU baseline |
| B1 | base 1.5B, typed plan, single shot | tool-enabled baseline |
| S0 | programmatic SFT | SFT baseline |
| R-F | SFT + GRPO, **Fisher reward** (0.9 z-closeness to single reference + 0.1 conclusion) | comparator |
| R-A | SFT + GRPO, admissible-set reward (A) | part A |
| R-AD | + discordance sampling (D) | part D |
| R-ADT | + twins (T) | part T |
| R-CTV | + validity constraint (V) | full method |
| R-CTV-DAPO | R-CTV with DAPO clipping and dynamic sampling | optional |

Match across R-arms: initialisation, optimizer updates, sampled prompts, group size and generated-token budget (report the realised value; target within 5%). Run 3 seeds for R-F and R-CTV at minimum.

**Primary comparisons (Holm-corrected):** R-CTV vs R-F on (1) method-validity rate, (2) agent-level FPR, (3) twin-consistency accuracy. **Non-inferiority:** R-CTV vs R-F on P-Bench Strict pass@1, margin −3 pp, prespecified.

---

## 9. Evaluation

### 9.1 Metrics

| Metric | Definition |
|---|---|
| Method-validity rate | fraction of answerable tasks with m̂ ∈ A(x) |
| Invalid-method rate | fraction with m̂ ∈ B(x) |
| Twin-consistency accuracy | fraction of twin pairs where both members get a valid method (primary) |
| Agent-level FPR | reject rate over ≥ 1,000 fresh true-null simulations per scenario, full agent re-run each time; Clopper–Pearson 95% CI |
| Agent-level power | reject rate on medium/large-effect simulations |
| Abstention correctness / false abstention | on invalid-design vs answerable tasks |
| P-Bench Raw/Strict pass@1, pass@3 | Fisher-R1 protocol: 425 tasks × 3 trials, Strict = correct decision ∧ \|Δz\| < 0.5, Easy 203 / Hard 222 |
| Operational failure rate | bad JSON, unknown column, unsupported method, timeout |
| Efficiency | correct tasks per GPU-hour; generated tokens per correct task; peak VRAM |

### 9.2 P-Bench protocol

- Pin dataset revision and hashes (`May2222/P-Bench`, CC BY 4.0).
- Evaluator-only preflight builds `manifests/pbench_method_mapping.csv` (17 categories, counts sum to 425). The agent never sees category, answer or source path.
- Unsupported, abstained, failed or timed-out answerable tasks count as **incorrect**.
- Implement the scorer cleanly from the paper's definition; check it on hand-made fixtures; ask the authors to confirm on a few examples.
- Run once on the frozen model. No tuning on P-Bench. Pilot exposure (≤ 50 tasks) is recorded and disclosed.
- Report against Fisher-R1 Table 1 as **published reference values**, labelled as not re-run.

### 9.3 Statistics

Paired, cluster bootstrap over scenario/source groups (2,000 resamples) for all arm differences; McNemar only for independent task pairs; Holm across primary comparisons; report seed variance. No significance claim from smoke or pilot runs.

---

## 10. Audit bundle (minimum four-pillar slice)

ZIP containing `protocol.json`, `audit_events.jsonl` (hash-chained), `analysis.py` generated from the method template, `result.json`, `admissible_range.csv` (every admissible method run on the user's data — this is the multiverse pillar), `parity.json` (Python vs R where available), `assumptions_report.json`, `README.md` with the rerun command, `manifest.sha256`. Acceptance: a second machine reproduces the primary result within tolerance from the ZIP alone.

---

## 11. CLI and local API

```bash
nanostat analyze --question "Does the intervention reduce the outcome?" \
  --csv data.csv --study study.json [--adapter path] [--rules-only]
nanostat bundle  --run RUN_ID --out bundle.zip
nanostat factory --profile smoke|pilot|full --out data/
nanostat train   --stage sft|grpo --arm R-CTV --config configs/t4_ctv.yaml [--resume RUN_ID]
nanostat eval    --arms B0,S0,R-F,R-CTV --suite nanostat|twins|nullcal|pbench --trials 3
nanostat report  --runs results/
```

Local API (FastAPI): `POST /v1/analyze`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/bundle`, `GET /health`. Uploads only; no server paths; 20 MB / 100k rows / 200 columns default limits; localhost bind by default.

---

## 12. Repository layout

```text
nanostat-ctv/
├── SPEC.md                    # this file
├── CLAUDE.md                  # §19
├── STATUS.md                  # §18 template
├── README.md                  # quickstart, ICP, actual results only, limitations
├── LICENSE                    # Apache-2.0 (code)
├── NOTICE.md                  # third-party attributions (P-Bench CC BY 4.0, Qwen Apache-2.0)
├── CITATION.cff
├── pyproject.toml
├── configs/{t4_ctv.yaml, experiments.yaml, policies/*.yaml}
├── nanostat/
│   ├── schemas.py  profiler.py  verifier.py  planner.py  controller.py
│   ├── tools/                 # one file per method plugin
│   ├── dsl.py                 # preprocess ops
│   ├── audit/{events.py, bundle.py}
│   ├── parity/r_backend.py
│   ├── api.py  cli.py
├── benchmark/
│   ├── factory/{grid.py, dgp.py, perturb.py, twins.py, render.py}
│   ├── reference.py  discordance.py  split.py  keys.py
│   ├── pbench/{preflight.py, adapter.py, score.py}
│   └── nullcal.py
├── training/{sft.py, grpo_ctv.py, reward_server.py, constraint.py, sampler.py, checkpoint.py}
├── evaluation/{metrics.py, bootstrap.py, report.py, inspect_tasks/}
├── notebooks/{01_quickstart.ipynb, 02_sft_t4.ipynb, 03_ctv_grpo_t4.ipynb, 04_eval.ipynb}
├── tests/{unit, integration, fixtures}
├── keys/                      # git-ignored; answer keys
├── manifests/  results/  docs/{model_card.md, data_card.md, limitations.md}
└── .github/workflows/ci.yml   # CPU only: lint, mypy, pytest, fixtures
```

---

## 13. Required tests (selected; each milestone adds its own)

| Area | Test |
|---|---|
| Executor | every plugin matches its independent reference within tolerance on ≥ 5 fixtures |
| Verifier | rejects paired data analysed as independent, wrong tail, wrong outcome type, missing pair ID, unknown column |
| Security | no `eval`/`exec`/subprocess outside the R worker; path traversal rejected; network blocked in eval mode |
| Leakage | grep-based and import-graph test that `planner`, `controller`, `training/grpo_ctv.py` never import `keys` |
| z-mapping | p ∈ {1, 0.5, 0.05, 1e-10, 0} map correctly; clipping at 5; monotonic |
| Reward | method in A beats method in B on the same p; admissible alternative gets full method credit; fabricated number → penalty; always-abstain policy scores below a uniform-valid policy |
| Twins | both members share split; edit changes the gold method; wording-fixed twins have identical question text |
| Constraint | on a synthetic stream with liberal choices λ1 rises; with valid choices λ1 stays near 0; λ bounded |
| Training | adapter-only gradients; resume reproduces next-step loss within 1e-6 (FP32 tiny model) |
| Scorer | Strict scorer reproduces hand-computed fixtures (success, wrong decision, malformed output, p=0, p=1) |

---

## 14. Milestones (dependencies gate advancement; calendar is indicative)

| ID | Work | Exit evidence |
|---|---|---|
| **M0** CPU slice (days 1–3) | schemas, profiler, verifier, 7 tools, rule planner, controller with mocked LLM, CLI `analyze --rules-only` | CI green; verifier tests; 30 smoke tasks run end-to-end |
| **M1** Registry + factory (week 1–2) | policy YAML for M0/M1 methods, factory, reference runner, discordance, twins, splits, key store, minimal audit bundle | 1,000 validated tasks; D(x) histogram; leakage test passes; ZIP reproduces on a second machine |
| **M2** Base model + SFT (week 2–3) | HF loader (NF4), typed-plan prompting, programmatic SFT, profiling notebook | measured VRAM and tokens/s on T4; B1 and S0 results on dev; resume test |
| **M3** Scale data (week 3–4) | ≥ 8,000 unique training tasks; dev 200; locked test 400 (200 twin pairs); null-cal scenarios | counts after filtering; duplicate rate; per-family table |
| **M4** CTV-GRPO probe (week 4–5) | reward server, constraint, twin-joint GRPO; 50-update probe for R-F and R-CTV | peak memory ≤ 14 GiB or documented blocker; reward/λ curves; no NaN |
| **M5** Pilot ablations (week 5–8) | R-F, R-A, R-AD, R-ADT, R-CTV × 3 seeds × 300 updates | dev metrics with bootstrap CIs; freeze config and adapter hash |
| **M6** Advanced methods (parallel to M4–M5) | P-Bench preflight, 17-category mapping, M4 plugins with R parity | mapping sums to 425; plugin fixtures pass |
| **M7** Frozen evaluation (week 8–10) | locked test, twins, agent null calibration, full P-Bench 425×3, StatQA | complete result tables, raw traces, cost report |
| **M8** Release + pilot (week 10–14) | disclosure gate (§16), HF model + traces, GitHub public, arXiv, researcher pilot | gate checklist signed; release tag; pilot metrics |

Fallbacks: if M4 cannot fit on a T4, run an offline-preference (DPO on CTV-ranked pairs) arm and label it **not GRPO**; keep the CTV reward/sampling/twin contributions and report the blocker.

---

## 15. Dependencies (pin after smoke test)

Core CPU: `python>=3.10`, `numpy`, `pandas`, `scipy`, `statsmodels`, `pydantic>=2`, `pyyaml`, `typer`, `fastapi`, `uvicorn`, `jinja2`, `pytest`, `ruff`, `mypy`.
Optional CPU: `lifelines`, `linearmodels`, R 4.x with `survival`, `AER`, `sandwich`, `lmtest`, `lme4`, `lmerTest` via `Rscript` templates.
GPU: `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `trl`, `datasets`.
Evaluation (optional): `inspect-ai`.
Nothing else without maintainer approval.

---

## 16. Release, leaderboard and disclosure plan

| Step | Action | Rule |
|---|---|---|
| 1 | Private repo, private results | default until gate passes |
| 2 | **IP check** | if a patent filing is planned, file the provisional before *any* public step below; HF uploads, public GitHub, leaderboard entries and arXiv all count as disclosure |
| 3 | Contact Fisher-R1 / P-Bench authors privately | ask to confirm scorer fixtures and whether P-Bench will be registered as an HF benchmark |
| 4 | Independent rerun | non-author reruns the frozen adapter + evaluator; signed note in `release/` |
| 5 | Public release | GitHub (Apache-2.0), HF model with `.eval_results/` (self-reported unless verified), HF dataset for NanoStat twins/null-cal, arXiv preprint, Zenodo DOI |
| 6 | Venues | workshop (AI-for-science / agents) for feedback → TMLR or MLST (method), JOSS or SoftwareX (library) |

**Claim wording template:** "Under the Fisher-R1 P-Bench protocol (425 tasks × 3 trials, released traces), NanoStat-CTV-1.5B achieves X% Strict pass@1 [CI] and an agent-level false-positive rate of Y% [CI]; Fisher-R1-7B published values are shown for reference and were not re-run." Never "ranked #1" without a real listing.

---

## 17. Risks

| Risk | Mitigation |
|---|---|
| Typed plans cannot express some P-Bench preprocessing | preprocess DSL; count misses as failures; report coverage |
| Admissible sets encode the author's opinion | statistician review, versioned YAML, publish sets |
| Lagrangian oscillation | EMA, λ cap, logged curves, freeze after pilot |
| Twins teach surface cues | wording-fixed twins; held-out edit types |
| T4 unavailability/OOM | resumable checkpoints every 25 updates; stop rules; DPO fallback labelled |
| 1.5B too weak for Hard categories | report per-category; headline = validity metrics + efficiency, not raw Strict superiority |

---

## 18. `STATUS.md` template

```markdown
# STATUS
Milestone: M0 | Date: YYYY-MM-DD
## Done this session
- ...
## Commands run (exact)
- `pytest -q` → N passed, M failed
## Measured (never estimated)
- ...
## Blockers
- ...
## Next action
- ...
## Artifact hashes
- data manifest: sha256:...
```

---

## 19. `CLAUDE.md` (copy verbatim)

```markdown
# CLAUDE.md — NanoStat-CTV

You are implementing SPEC.md. It is the acceptance contract.

Rules
1. Work on the current milestone in STATUS.md only. Do not skip ahead.
2. CPU first. Run `pytest -q` before and after every change; paste the real summary into STATUS.md.
3. The model never computes statistics. All numbers come from nanostat/tools.
4. Never let planner/controller/training code import from `keys/` or `benchmark/keys.py`. The leakage test must stay green.
5. No eval/exec/shell on model output. R runs only through fixed templates in nanostat/parity/r_backend.py.
6. Never invent results, scores, citations, platform statuses or verification badges. Unrun = NOT RUN.
7. No new dependency, network call, upload or publish action without asking the maintainer.
8. Commits and PRs: author is Anandharajan only. Do not add Claude as co-author or contributor; no Co-Authored-By or "Generated with" lines.
9. Small commits, one concern each, message `M<k>: <what>`.
10. When a test fails, fix the cause; do not loosen tolerances or delete tests without recording why in STATUS.md.
11. Keep notebooks thin: they call package code only.
12. End every session by updating STATUS.md (done, commands, measured numbers, blockers, next action).

First task: M0 — create the repo skeleton from SPEC.md §12, implement §5 schemas, the seven M0 tools with fixtures, the verifier tests in §13, the rule planner and `nanostat analyze --rules-only`. Stop when CI is green and report.
```

---

## 20. References

1. Miao, Mu, Chen, Zou. *Fisher-R1: Training LLM Agents for Reliable Hypothesis Testing*. arXiv:2608.07437v2, NeurIPS 2026. Code: github.com/jmu27/FisherR1 (check licence before reusing any code; implement clean-room otherwise).
2. P-Bench dataset, huggingface.co/datasets/May2222/P-Bench (CC BY 4.0; no leaderboard at time of writing).
3. Shao et al. *DeepSeekMath* (GRPO), arXiv:2402.03300; Yu et al. *DAPO*, arXiv:2503.14476.
4. Dettmers et al. *QLoRA*, arXiv:2305.14314.
5. Huang et al. *POPPER: Automated Hypothesis Validation with Agentic Sequential Falsifications*, arXiv:2502.09858.
6. *The Agentic Garden of Forking Paths*, arXiv:2607.01507; *Mitigating LLM-based p-Hacking by Preregistering for the Next LLM*, arXiv:2606.27687 — position against both in related work.
7. Xie et al. *DoReMi*; Chen et al. *Skill-it!*; Fan et al. *Dynamic Gradient Alignment*, arXiv:2410.02498 — related work for discordance-weighted sampling.
8. Zhu et al. *Are Large Language Models Good Statisticians?* (StatQA), NeurIPS 2024.
9. Hugging Face Community Evals / eval results docs, huggingface.co/docs/hub/eval-results.

Verify every reference against the primary record before manuscript submission.

---

## 21. Version 3.1 addendum (27 November 2026)

This addendum supersedes conflicting text above. Source: maintainer's master plan of 27 Nov 2026.

### 21.1 Scientific-correctness requirements (P0, implemented)
1. **Invalid/unknown gating.** In confirmatory mode a method in B(x,e) or of unknown status is rejected by the verifier; the controller re-plans and finally abstains (`no_admissible_method`). Diagnostic mode may execute it but returns no decision (`Result.confirmatory = false`).
2. **Protocol lock.** `Protocol(alpha, alternative, estimand_family, p0, allow_population_filters)` is fixed by the task/user (`nanostat/protocol.py`). Plan fields that restate it must match; any difference is rejected and penalised (−0.2) by the reward.
3. **Estimand-conditioned admissible sets.** Policy cells key methods by estimand; A(x,e) = methods of the protocol's estimands; B(x,e) adds `estimand_mismatch`. Rp uses max over A(x,e) only. A permutation test was added so mean-difference protocols remain answerable under outliers/skew.
4. **Two-gate verification.** Gate 1 validates the plan and the preprocessing ops against the protocol; gate 2 executes preprocessing, re-profiles and re-verifies design, pairing, groups, estimand and method status on the analysed rows, which are hashed (`analysed_sha256`) and are exactly the rows tested.
5. **Evidence-grounded assumption reward.** `r_assume` credits a required check only when the plan's typed verdict (`ok`/`violated`) matches the CPU's reading of the executed diagnostic; irrelevant claims are penalised.
6. **Hypothesis-specific nulls.** DGPs carry `nulls: {hypothesis: bool}`; a rejection is a false positive only for the hypothesis of the executed test (e.g. Spearman ↔ independence, Welch ↔ equal means).
7. **Scenario-specific constraints.** `FPR_s ≤ α + ε_s`, `Power_s ≥ PowerRef_s − δ_s`; ε_s, δ_s are one-sided 95% binomial tolerances on the EMA-effective sample; PowerRef_s is the preferred method's power per scenario from training keys.
8. **Group-exclusive splits.** Held out for the locked test split: twin edits `outcome_type`, `small_counts`; template domain 4; scenario family (correlation, skewed). Twins always co-located.

### 21.2 Twins
Seven edits: design flip, add outliers, tail flip, small counts, remove pair ID, outcome type, **estimand change** (mean ↔ typical). Clustering twin remains blocked on the M4 OLS plugin.

### 21.3 Targets
- PRIMARY: R-CTV vs R-F on method validity, agent-level FPR, twin consistency (Holm); non-inferiority on Strict pass@1, margin −3 pp; operational failure ≤ 5%; peak reserved VRAM ≤ 14 GiB.
- STRETCH (not expected): Easy/Hard Strict pass@1 70/40; pass@3 80/50.
- Two comparisons are kept separate: internal (R-CTV vs R-F, matched compute) and external (published Fisher-R1 values, labelled not re-run; optionally Fisher-R1-7B re-run on our harness).

### 21.4 Experiment matrix and compute (GPU, NOT RUN)
Arms B0, B1, B2 (iterative diagnostics), B3 (method-doc retrieval), S0, R0, R-F, R-A, R-AD, R-ADT, R-CTV, R-CTV-Agent; optional B4, S1, R-CTV-DAPO. Seeds: 3 for R-F and R-CTV(-Agent), 1 otherwise. Full P-Bench 425 × 3 only for B1, S0, R-F, R-CTV-Agent. Budget ≈ 120–170 GPU-h on Colab + Kaggle free tiers. Profile FP16 LoRA vs NF4 QLoRA first and record the choice. Fallback: CTV-ranked DPO, labelled NOT GRPO.

### 21.5 CPU ablation (implemented)
`nanostat ablate` trains a linear-softmax proxy policy over an 18-action typed-plan catalogue with the six arms S0, R-F, R-A, R-AD, R-ADT, R-CTV at matched budget, using the real reward server, sampler, twin-joint advantage and scenario constraint. It isolates the effect of the training signal; it is not the language model. Results: `docs/ABLATION_RESULTS.md`.

### 21.6 Reward ordering (v0.4.1, 28 November 2026)
The CTV reward must order outcomes: admissible > not in A (0) > abstaining on an answerable task (−0.1) > invalid method / protocol violation / fabricated number (−0.2) > **broken plan** (cannot be parsed, verified or executed; −0.5, `RewardWeights.malformed_penalty`). Before v0.4.1, broken plans scored 0, so a policy could avoid the invalid penalty by emitting broken plans (measured: the CTV-family CPU arms kept 4–7% broken plans on average). Every evaluation reports the **broken-plan rate** next to the invalid-method rate. The R-F comparator keeps its original definition (broken = 0, its minimum).
