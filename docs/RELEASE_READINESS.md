# Release and publication readiness (2026-11-28)

Status words: **PASS** · **FAIL** · **BLOCKED** (needs a person or resource) · **NOT RUN**. Every row cites evidence in this repository. Leaderboard facts come from `docs/LEADERBOARDS.md` (as of 2026-11-27, not re-verified).

## 1. Verdict

| Target | Publishable now? | Why |
|---|---|---|
| **BenchmarkList P-Bench leaderboard** | **No** | It ranks by Hard Strict pass@1 of a model on all 425 × 3 trials, sourced from an arXiv paper. The only P-Bench run here is the rule planner B0 (0.0: it asks for clarification on every task). No LLM has been run. |
| **Hugging Face (model card / Community Evals)** | **No** for P-Bench; possible but low-value for NanoStat-Bench | P-Bench has no `eval.yaml`. Registering our own NanoStat-Bench is allowed but self-referential. Our B0 is ~100% there by construction (circular), so it shows consistency, not skill. |
| **Workshop paper / technical report** | **Yes, with honest scope** | Measured, compute-matched, statistically tested results exist (Section 2). The contribution is a method plus a benchmark plus a controlled proxy ablation. The limitations (proxy learner, synthetic tasks, unreviewed registry) must be stated in the abstract. |
| **Software paper (JOSS / SoftwareX)** | **Nearly** | Open licence, documented CLI and API, 303 tests, CI on Linux and Windows, audit bundle. Missing: a statement of need with evidence of use, contribution guidelines, and R parity (NOT RUN). |
| **Main ML venue (TMLR / NeurIPS / ICLR)** | **No, not yet** | Reviewers will require the LLM experiment the SPEC promises: R-CTV vs R-F with the 1.5B model at matched GPU budget, ≥ 3 seeds, plus P-Bench non-inferiority. All NOT RUN. |

## 2. What the current results can and cannot support

**Can claim (measured, compute-matched, Holm-corrected; `docs/ABLATION_RESULTS.md`):**
- An admissible-set reward beats a Fisher-R1-style p-value-matching reward on method validity (+0.064 [+0.050, +0.079]) and twin consistency (+0.091 [+0.050, +0.141]) at equal generated rollouts (19,200 per arm, 5 seeds; v0.4.1 reward). The invalid-method rate falls from ~12% to ~2–3%, with no broken plans (the v0.4.0 reward let broken plans hide invalid choices; see correction notice 2).
- Hypothesis-specific null calibration: the agent's false-positive rate stays at or below 5% within its CI, and the registry's invalid labels are confirmed by fixed-method arms (e.g. Student t 14.1% and the permutation test 14.5% under unequal variances; Pearson 48.8% under leverage).
- Spec-level labels for regression/survival are statistically right: e.g. omitted confounder 60% vs 4%, naive OLS under endogeneity 98.8% vs 4.4% (`docs/REGRESSION_FAMILIES.md`).
- Engineering: P-Bench tool coverage ceiling 97.9% (Hard 95.9%), clean-room scorer, a verifier with a locked protocol and two gates, and reproducible audit bundles.

**Cannot claim:**
- Any P-Bench accuracy, any rank, or any comparison with Fisher-R1 performance. Their published numbers are for 7B/14B models trained with far more compute; they are **not** resource-matched to anything here.
- That discordance sampling (D), twins (T) or the validity constraint (V) help. At matched compute they were not separable from A, D starved abstention, and V had nothing to correct.
- That validity reflects expert judgement: the registry is `reviewed_by: PENDING`, and the same registry defines both the keys and the metric.

## 3. Resource parity: are comparisons fair for the resources used?

| Comparison | Matched on | Status |
|---|---|---|
| CPU ablation arms | Same SFT initialisation per seed; same hyperparameters; **same number of generated rollouts (19,200, re-samples included)**; updates allowed to differ and logged (127–147) | **PASS** (fixed 2026-11-28; the v0.2.0 run was not matched because re-sampled rollouts were not counted; see the correction notice) |
| B0 null calibration vs fixed methods | Same simulations for agent and every fixed method | **PASS** |
| LLM evaluation runs (B1, P-Bench) | Prompt version and hash, attempts, seconds, **prompt and completion tokens and LLM calls per trial** now logged; cost summarised per run | **PASS** for instrumentation; runs **NOT RUN** |
| LLM training arms (R-F vs R-CTV) | Must match generated tokens, not only optimiser steps: twin pairs and verifier retries change token counts. `training/train_gpu.py` fixes steps and group size but does not yet enforce a token budget | **FAIL** until a token-budget stop or a post-hoc token report is added and checked on the first GPU run |
| NanoStat-1.5B vs published Fisher-R1 | Different model size, data and compute | **Not matched by design.** Report as "published, not re-run" plus efficiency columns (correct per GPU-hour, tokens per correct task). Never "beats". |
| Optional Fisher-R1-7B re-run on our harness | Same scorer and tasks; different training compute | Allowed as an *evaluation-matched* comparison only; label it so |

## 4. Gate checklist (SPEC §16, master plan Step 9)

| Gate | Status | Evidence / blocker |
|---|---|---|
| Tests and CI green | PASS | 303 tests; CI run 4: Linux and Windows × 3.10/3.12 |
| Leakage tests (policy never reads keys) | PASS | `tests/unit/test_security_leakage.py` |
| No fabricated numbers; unrun = NOT RUN | PASS | STATUS.md, traceability |
| Compute-matched ablation | PASS | `results/ablation_cpu/ablation_results.json` (per-seed rollout logs) |
| Statistician review of the admissible registry | **BLOCKED** | All 20 cells `reviewed_by: PENDING`; LLM pre-review tool exists, but no sign-off |
| Independent (non-circular) test set | **NOT RUN** | Needs sets authored separately from the training registry |
| LLM B1 baseline (NanoStat-Bench, null calibration, P-Bench pilot) | **NOT RUN** | Magnitude not installed |
| LLM SFT + GRPO arms, ≥ 3 seeds, matched tokens | **NOT RUN** | Needs a T4 (Colab or Kaggle); token-budget enforcement pending |
| Full P-Bench 425 × 3 with a frozen model | **NOT RUN** | Depends on the two rows above |
| Scorer confirmation by P-Bench authors | **NOT RUN** | Fixture sheet ready (`benchmark/pbench/score.py::FIXTURE_SHEET`); email not sent |
| Independent re-run by a non-author | **NOT RUN** | — |
| R parity for model tools | **NOT RUN** | No `Rscript` on the development machine |
| IP gate before arXiv / leaderboard | **BLOCKED** | Maintainer decision. The code and CPU ablation are already public on GitHub (public disclosure has occurred for those) |

## 5. Shortest paths

**A. Workshop paper (weeks, ₹0):** run the LLM pre-review and have one statistician review the 20 cells → write up the method, benchmark and compute-matched proxy ablation with the negative findings. Add B1 numbers from Magnitude if available (strengthens it a lot).

**B. Leaderboard entry (≈ 120–170 free GPU-hours plus your time, per the master plan):**
1. Magnitude: B1 (Qwen2.5-Coder-1.5B, 7B) on NanoStat dev, then a P-Bench pilot (≤ 50 tasks, disclosed). This measures where the base model starts.
2. Kaggle/Colab: SFT on the full factory (now including regression/survival families with spec-level keys), then GRPO R-F vs R-CTV at a **matched token budget**, 3 seeds.
3. Freeze, run P-Bench 425 × 3 through Magnitude, release traces and cost columns.
4. Independent re-run → arXiv (after the IP gate) → email the P-Bench authors (scorer fixtures) → BenchmarkList Feedback link with the evidence pack. A listing would read "Self-reported" and is at their discretion.

**Entry bar (published values, not re-run):** the lowest listed Hard Strict pass@1 is 18.2 (GPT-OSS-120B); the top is 33.0 (Fisher-R1-14B). There is no basis yet to predict where a 1.5B NanoStat model would land. The first B1 run will tell.
