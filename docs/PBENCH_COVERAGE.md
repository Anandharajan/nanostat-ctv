# P-Bench preflight and coverage ceiling

Data: `May2222/P-Bench` (CC BY 4.0) at pinned revision `402205db5904b45b3f1f0aca895980f712fd61d3`, downloaded with maintainer approval on 2026-11-27 (1,279 files, 292 MB, SHA-256 manifest verified). Stored git-ignored under `data/pbench/`; answers are read only by the evaluator. Reproduce: `nanostat pbench --download`, then `nanostat pbench --preflight`.

Verified: 425 tasks = 203 Easy + 222 Hard; 17 categories. Full mapping: [`manifests/pbench_method_mapping.csv`](../manifests/pbench_method_mapping.csv); report: [`manifests/pbench_coverage.json`](../manifests/pbench_coverage.json).

## Coverage ceiling (upper bound: a NanoStat plugin exists for the task's category)

| Tool set | All | Easy | Hard |
|---|---:|---:|---:|
| v0.2.0 (M0/M1 tools + permutation) | 215/425 = 50.6% | 72.9% | 30.2% |
| **v0.3.0 (+ 9 M4 plugins)** | **416/425 = 97.9%** | **100%** | **95.9%** |
| v0.3.0, excluding questions with spec flags* | 359/425 = 84.5% | 94.6% | 75.2% |

\* Keyword heuristic over question text for specification features the typed plan/DSL cannot yet express (constructed weights: 32 tasks, subset restrictions: 32, stacking: 2, lags: 3, interactions: 1, …). The 9 `other` tasks (stacked / custom estimators) are not expressible.

**A ceiling is not accuracy.** Strict requires reproducing the reference p-value within |Δz| < 0.5, which needs the *paper's exact specification* (covariate set, fixed effects, clustering level, weights, ties, df convention). The model has to get that specification right from the question text.

| Category | n | Easy | Hard | NanoStat plugin(s) |
|---|---:|---:|---:|---|
| t_test | 76 | 35 | 41 | welch_t / student_t / paired_t |
| spearman_correlation | 55 | 35 | 20 | spearman |
| ols_coefficient_test | 50 | 0 | 50 | ols_coef (classical / HC0–HC3 / cluster CR1, covariates, factors, weights) |
| fisher_exact_test | 35 | 35 | 0 | fisher_exact |
| log_rank_test | 35 | 35 | 0 | log_rank |
| logistic_regression | 35 | 0 | 35 | logistic |
| mann_whitney_test | 35 | 35 | 0 | mann_whitney |
| iv_2sls | 24 | 0 | 24 | iv_2sls (first-stage F reported) |
| cox_regression | 20 | 0 | 20 | cox_ph (Efron / Breslow ties) |
| f_test | 15 | 0 | 15 | ols_wald (joint zero / equality) |
| probit | 13 | 13 | 0 | probit |
| other | 9 | 0 | 9 | — (not expressible) |
| mixed_effects | 7 | 7 | 0 | mixed_lm (REML random intercept) |
| kruskal_wallis_test | 6 | 0 | 6 | kruskal_wallis |
| chi_square_test | 5 | 5 | 0 | chi2_independence |
| binomial_test | 3 | 3 | 0 | binomial_test |
| tobit | 2 | 0 | 2 | tobit |

## Measured P-Bench runs
| Arm | Trials | Raw pass@1 | Strict pass@1 | Hard Strict pass@1 | Status |
|---|---|---:|---:|---:|---|
| B0 (rule planner) | 1 | 0.0 | 0.0 | 0.0 | 425/425 clarify: P-Bench declares no column roles, so the rule planner asks instead of guessing. Floor, by design. |
| B1 (base LLM via Magnitude) | 3 | — | — | — | NOT RUN (Magnitude not installed) |

Raw summary: `results/pbench/b0_rules_summary.json`. Fisher-R1 published values for comparison are in [FISHER_R1_BASELINE.md](FISHER_R1_BASELINE.md) (published, not re-run).
