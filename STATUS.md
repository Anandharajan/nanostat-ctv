# STATUS
Milestone: v3.1 Steps 2–4 done on CPU (P0 fixes, P-Bench preflight + runner, M4 tools); LLM harness ready, LLM runs NOT RUN · Release: v0.3.0 (local; v0.2.0 + v0.3.0 not pushed) · Date: 2026-11-27

## Done this session
### Earlier today (v0.2.0)
P0 correctness fixes (protocol lock, estimand-keyed sets, two-gate verifier, invalid gating, evidence-grounded assumptions), per-hypothesis nulls, scenario constraint, group-exclusive splits, estimand twin, clean-room P-Bench scorer, CPU ablation (S0, R-F, R-A, R-AD, R-ADT, R-CTV, R-CTV-abs). See docs/AUDIT.md and docs/ABLATION_RESULTS.md.

### v0.3.0
- **Local LLM planner via Magnitude** (`nanostat/llm.py`): stdlib OpenAI-compatible client, default `http://127.0.0.1:10100/inference/v1`, refuses non-local hosts. `nanostat analyze --llm-model`, `nanostat eval --arms B1 --llm-model … [--suite nanostat|nullcal|pbench]`, resumable JSONL traces with prompt version + hash. Prompt v2 includes the locked protocol and a model-spec schema.
- **P-Bench** (approved download): pinned revision `402205db…`, 1,279 files / 292 MB, SHA-256 verified; preflight asserts 425 / 203 / 222 / 17 categories; mapping + coverage report; runner with sanitised inputs (question + CSV only); B0 run on all 425.
- **M4 tools** (9): `ols_coef` (classical / HC0–HC3 / cluster CR1, WLS, factors), `ols_wald` (joint / equality F), `logistic`, `probit`, `iv_2sls` (+ first-stage F), `log_rank`, `cox_ph` (Efron/Breslow), `mixed_lm` (REML random intercept), `tobit`; new designs `regression` / `survival`; typed `ModelSpec`; policy cells v0.3.0 (PENDING).
- Protocol reads direction/estimand from the **H1 line** when hypotheses are stated.
- **Policy pre-review** command (`nanostat review-policies --llm-model`) → worksheet; never changes `reviewed_by`.
- **GPU training path** (`training/train_gpu.py`, `training/rl_rewards.py`, `notebooks/02_train_t4.ipynb`) for SFT and GRPO arms R-F/R-A/R-AD/R-AV with stock TRL; reward adapter CPU-tested.

## Commands run (exact)
- `pytest -p no:cacheprovider` → 286 passed
- `ruff check .` → All checks passed; `ruff format --check .` → clean; `mypy` → no issues in 46 source files
- `nanostat pbench --download` equivalent (`benchmark.pbench.loader.download`) → 1,279 files, 292,347,012 bytes, verify True (2 min 49 s)
- `nanostat pbench --preflight` → category ceiling 416/425
- B0 on P-Bench: `run_pbench(..., planner=RulePlanner(), trials=1)` → 425 rows, all `abstained` (200 s)

## Measured (never estimated)
- P-Bench coverage ceiling: v0.2.0 tools 50.6% (Easy 72.9%, Hard 30.2%) → v0.3.0 tools **97.9% (Easy 100%, Hard 95.9%)**; excluding spec-flagged questions 84.5% (Easy 94.6%, Hard 75.2%). Upper bounds, not accuracy.
- B0 on P-Bench: Raw / Strict pass@1 = 0.0 (clarifies on every task: no declared roles). Floor by design.
- M4 tools agree with independent references on 5 seeds each (OLS/HC/cluster/WLS/F vs statsmodels to 1e-7; logit/probit vs own Newton to 1e-5; 2SLS vs statsmodels IV2SLS to 1e-8; log-rank vs own implementation to 1e-6; Cox (Breslow) vs own partial likelihood; mixed fixed effects vs GLS (SE within 1%, reason recorded in the test); Tobit vs a second optimiser).
- CPU ablation and B0 null calibration: unchanged from v0.2.0 (docs/ABLATION_RESULTS.md).

## NOT RUN / BLOCKED
- **Every LLM number (B1 on NanoStat-Bench, LLM null calibration, P-Bench with an LLM, policy pre-review): NOT RUN.** Magnitude is not installed on this machine; the maintainer installs it and pulls models (I did not run its install script or download models).
- GPU training (SFT, GRPO): NOT RUN (no CUDA here). R-ADT / R-CTV twin-joint trainer for the LLM: MISSING (CPU version exists).
- Statistician sign-off: BLOCKED. All 20 policy cells `reviewed_by: PENDING`.
- R parity for M4 tools: NOT RUN (no Rscript). Clustering twin: MISSING.
- Push of v0.2.0 / v0.3.0 to GitHub: awaiting the maintainer's decision (IP gate).

## Next action
1. Maintainer: install Magnitude, `magnitude models pull` Qwen2.5-Coder-1.5B-Instruct (and 7B if hardware allows), `magnitude serve`, then tell me the model ids.
2. Run: B1 on NanoStat dev → prompt fixes on dev only → B1 on locked test + LLM null calibration → P-Bench pilot (≤ 50 tasks, disclosed) → full P-Bench 425 × 3 with the frozen prompt; `review-policies` worksheet for the statistician.
3. GPU: run `notebooks/02_train_t4.ipynb` on Kaggle/Colab; implement the twin-joint GRPO trainer.

## Artifact hashes
- P-Bench revision: 402205db5904b45b3f1f0aca895980f712fd61d3 (manifest `data/pbench/402205db5904/MANIFEST.sha256`, git-ignored)
- Pilot synthetic data (seed 20261126): the v0.2.0 hashes in this file's git history no longer apply — task files now serialise the new `Study.model` field, so regenerate before the next ablation or LLM run and record the new hashes.
