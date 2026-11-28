# Limitations (v0.2.0)

Read this before using NanoStat-CTV for anything that matters.

## What is and is not built

| Area | v0.1 status |
|---|---|
| CPU executor, 14 whitelisted tests, typed plan, two-gate verifier with locked protocol, rule planner (B0), CLI, local API, audit bundle | Built and tested |
| Policy registry (17 cells, estimand-keyed, M0/M1 methods) | Built; **every cell is `reviewed_by: PENDING`**. No statistician has reviewed the admissible/invalid sets. Pilot use only. |
| Task factory, reference solver, discordance, twins (7 edit types; clustering twin missing), group-exclusive splits, key store | Built and tested |
| CTV reward, Fisher-style comparator reward, scenario validity constraint, twin-joint advantage, sampler, SFT data builder | Built and unit-tested; used only in the **CPU proxy-policy ablation**, never to train an LLM |
| LLM planner (Qwen2.5-Coder-1.5B + LoRA), SFT, CTV-GRPO, arms B1/S0/R-* | **NOT RUN** (milestones M2–M5 need a GPU) |
| M4 methods (OLS/HC/cluster, Wald/F, logistic, probit, 2SLS, log-rank, Cox, mixed, Tobit) | Built and reference-tested; **R parity NOT RUN**; whether a *specification* (covariates, SE type, clustering) is right is not judged by the registry |
| Clustering twins | **Not implemented** |
| P-Bench scorer | Built and fixture-tested (clean-room) |
| P-Bench download, preflight, coverage ceiling | Done (pinned revision, approved 2026-11-27); B0 run: 0/425 (clarifies) |
| P-Bench / NanoStat-Bench with an LLM planner (Magnitude) | Harness built and mock-tested; **NOT RUN** (Magnitude not installed) |
| Python–R parity | Templates exist for 3 methods; **NOT RUN** on the development machine (no `Rscript`) |
| Second-machine reproduction of the audit bundle | Tested in a fresh, isolated Python process on the same machine only |

## Methodological caveats

- **B0 results are circular.** Answer keys are built from the same policy registry and the same deterministic diagnostics the rule planner uses, so B0's method-validity rate on NanoStat-Bench is ~100% by construction. It checks implementation consistency, not statistical judgement. It is a ceiling reference for trained arms, not evidence that the rules are right.
- **Admissible sets are opinions.** They encode one reading of common practice (e.g. Welch vs Student, rank tests under outliers). Different statisticians will disagree on some cells.
- **The CPU ablation uses a proxy policy.** A linear-softmax learner over 18 typed-plan actions shows the effect of the training signal; it says nothing direct about LLM accuracy.
- **Perturbation detection is heuristic.** Outliers = robust z > 5; skew = 5%-trimmed skewness > 1; unequal variance = SD ratio² > 4; small counts = min expected < 5. Heavy-tailed data are often classified as `heavy_outliers`.
- **Claim direction is parsed from wording** with a keyword list (`increase`, `reduce`, …). Unusual phrasing can be misread; the verifier will then flag a tail mismatch rather than silently proceed.
- **Synthetic data only.** All benchmark tasks are simulated; real datasets have missingness patterns, clustering and measurement issues the factory does not model.
- **Not for autonomous clinical or regulatory decisions.**

## Open issue: reward ordering for broken plans (found 2026-11-28)

In `training/reward_server.py::score_plan`, a plan that cannot be parsed, fails the verifier, or fails execution scores **0.0**. A plan that uses an invalid (B) method scores **-0.2**. A policy that cannot yet tell valid from invalid methods can therefore raise its reward by emitting broken plans. This is classic reward hacking, and the teaching track reproduces it (`edu/nanostat.py`, `MALFORMED`; lesson 4).

- **Effect on published results: NOT MEASURED.** The CPU proxy learner cannot emit unparseable text, but its catalogue does contain actions that fail the verifier on some tasks (e.g. a paired design on unpaired data), and these score 0.0 through the real `score_plan`. The ablation reports validity and the invalid rate, but not the share of chosen actions that failed verification. Measure that share before relying on the invalid-rate result.
- **Effect on LLM GRPO: likely.** A broken-plan reward below the invalid penalty (for example -0.5) should be set and tested before the first GPU run. Changing it now would change the reward definition after the ablation, so it is left as-is and recorded here.
