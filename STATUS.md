# STATUS
Milestone: SPEC v3.1 Step 2 (P0 correctness) done; Step 3 scorer done (loader/preflight BLOCKED); CPU ablation done · Release: v0.2.0 · Date: 2026-11-27

## Done this session (master plan of 27 Nov 2026)
- **Step 1:** `docs/AUDIT.md`, `docs/FISHER_R1_BASELINE.md` (published values, not re-run, transcribed from the master plan), `docs/LEADERBOARDS.md` (not re-verified), `VALIDATION_PLAN.md`, `REQUIREMENTS_TRACEABILITY.md`, SPEC §21 (v3.1 addendum), CLAUDE.md permanent policy lines.
- **Step 2.1:** invalid/unknown methods rejected in confirmatory mode; diagnostic mode (`nanostat analyze --diagnostic`) returns no decision.
- **Step 2.2:** `Protocol` lock (alpha, direction, estimand family, p0, population); plan changes are rejected and penalised.
- **Step 2.3:** estimand-keyed policy registry v0.2.0; `permutation_test` tool (exact + seeded Monte Carlo) so mean protocols stay answerable under outliers/skew; k-group + unequal variance now has no admissible method (abstain target).
- **Step 2.4:** two-gate verification; the executor runs exactly the verified analysed rows; `analysed_sha256` in results and the audit log.
- **Step 2.5:** evidence-grounded assumption reward (typed verdicts must match the CPU reading).
- **Step 2.6:** per-hypothesis null labels; null calibration judges each rejection against the null of the executed test; fixed-method arms on the same simulations.
- **Step 2.7:** `ScenarioConstraint` with binomial tolerances and per-scenario power references.
- **Step 2.8:** group-exclusive splits (edits, template domain 4, scenario correlation/skewed).
- **Twins:** 7th edit `estimand_change`.
- **Step 3.3:** clean-room P-Bench scorer + fixture sheet.
- **Ablation:** `training/ablation/` + `nanostat ablate`: S0, R-F, R-A, R-AD, R-ADT, R-CTV, R-CTV-abs with a CPU proxy policy.

## Commands run (exact)
- `pytest -p no:cacheprovider` → 197 passed
- `ruff check .` → All checks passed; `ruff format --check .` → clean; `mypy` → no issues in 38 source files
- `nanostat eval --arms B0 --suite nullcal --sims 1000 --out results/nullcal/b0_nullcal_v3_1.json` (9 min 18 s)
- `nanostat factory --profile pilot --out <tmp>/pilot31` → tasks=1400 twin_pairs=200 splits={train: 794, test: 505, dev: 101} rejected=49 duplicates=0
- reward/null table build (1,400 tasks × 18 actions; 3,000 null sims × 18 actions), then `nanostat ablate --data <tmp>/pilot31 --seeds 1,2,3,4,5 --updates 300 --out results/ablation_cpu` (≈ 31 s)
- `nanostat eval --arms B0 --suite nanostat --data <tmp>/pilot31 --split dev|test --out results/b0_<split>.json`

## Measured (never estimated)
- CPU ablation (proxy policy, locked test, 5 seeds): method validity R-F 0.891 ± 0.065 → R-CTV 0.977 ± 0.012; invalid-method rate 0.105 → 0.010; twin consistency 0.605 → 0.691; Strict (single reference) 0.889 → 0.941. R-CTV − R-F (Holm): validity +0.086 [+0.067, +0.106], twins +0.086 [+0.045, +0.136], agent FPR +0.001 [−0.002, +0.004] (no difference).
- Negative: discordance sampling starves abstention (R-A 0.440 → R-AD 0.064). Post-hoc R-CTV-abs restores 0.508 at a validity cost (0.922).
- Constraint V: no measurable effect in this setting (all RL arms ≈ 4.4% agent FPR).
- B0 null calibration: all agent FPRs within or below 0.05 ± CI; the earlier 6.5% flag resolved as Monte Carlo variation (fixed Welch identical on the same sims, CI 0.049–0.080). Registry invalid labels confirmed: Student t 0.141 and permutation 0.145 under unequal variance; Pearson 0.488 under leverage.
- B0 locked test (circular): 455/455 valid, 0 invalid, 88/88 twin pairs, 50/50 abstentions, 0/505 operational failures.

## NOT RUN / BLOCKED
- P-Bench download, preflight, 17-category mapping, coverage ceiling → **BLOCKED: needs maintainer approval to download `May2222/P-Bench`.**
- M4 plugins and clustering twin (largest P-Bench Hard lever) → MISSING.
- Step 5 typed inspect/diagnose loop, schema-constrained decoding → PLANNED.
- All LLM work (SFT, GRPO arms, FP16 LoRA vs NF4 profiling) → NOT RUN (no GPU; Colab/Kaggle notebooks still to be written).
- Statistician sign-off of policy cells → BLOCKED. Independent (non-circular) test set → PLANNED.
- CI for this push: result to be recorded after the GitHub Actions run.

## Next action
1. Approve the P-Bench download → preflight + coverage ceiling (decides the M4 plugin order).
2. Implement M4 plugins in coverage order with R parity.
3. Write the Kaggle/Colab notebooks for SFT + CTV-GRPO on the code paths the CPU ablation already exercises.

## Artifact hashes (pilot, seed 20261126, generator 0.1.0 + v3.1)
- tasks/train.jsonl: sha256:2ab5390cb85794fd50f03f4d3d94308a04badf531f88b5d2fa4800bb7aafdb98
- tasks/dev.jsonl: sha256:c50e66d7242cb4c7d6423d17c4adcbac9e02aedba1274c202555870f8f564860
- tasks/test.jsonl: sha256:f329570b959db7513a491dcffaa454d34800f0fba49d4ec4e0b2073d421a431d
- keys/keys.jsonl: sha256:dd58bae10577541b5e1fd610ee71241e1c1c4fb3386cd647b1acaa8cf4bcc8b0
- manifests/split_manifest.csv: sha256:ab18ceea3ffa604e8bbc5ec4f33f056f7eab02f3e30029246c255fbe592e2c0d
