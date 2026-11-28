---
n: 9
title: From nano to real
goal: Map each piece of the 580-line file to the full NanoStat-CTV system and to how large agent models are trained, and see what changes at scale.
prev: 08-the-experiment
prev_title: The experiment
---

## The same pieces, bigger

| nanostat (this course) | NanoStat-CTV (the full repo) | Frontier-scale analogue |
|---|---|---|
| 7 tests written from scratch | 23 tools in `nanostat/tools/` (OLS with robust and cluster SEs, logit, probit, 2SLS, Cox, log-rank, mixed models, Tobit…), each tested against an independent implementation | Tool use: calculators, code sandboxes, search. The model calls them; it does not imitate them |
| 3-token plan, grammar-constrained | Typed `AnalysisPlan` (Pydantic): design, method, model spec, tail, alpha, estimand | Structured outputs, JSON schemas, function calling |
| `execute()` | Two-gate verifier plus a whitelisted executor; **locked protocol** (alpha, direction and estimand fixed from the question) | Sandboxes and validators between a model and the world |
| 7 worlds, `RULES` | Task factory with 13 families including regression and survival, estimand-keyed sets A(x, e) and B(x, e), spec-level keys, **twins** (minimal edits that should change the answer) | Large banks of auto-generated problems with checkable answers (maths, code, tests) |
| `fisher_reward` / `ctv_reward` | `training/reward_server.py`, plus discordance sampling, twin-joint advantages and a Lagrangian validity constraint | Verifiable rewards (unit tests, answer checkers); reward models where nothing can be verified |
| `MALFORMED = -0.5` | Reward ordering, rate limits and audit traces | Reward-hacking audits: what does the policy *actually* do for reward? |
| TinyGPT, 104k params | Qwen2.5-Coder-1.5B through stock TRL (`training/train_gpu.py`) on a free T4 | Billions of parameters, many GPUs |
| `grpo()` | GRPO with matched rollout and token budgets | GRPO and PPO variants with KL control, at scale |
| `evaluate()`, 3 seeds | Null calibration with Clopper–Pearson intervals; P-Bench (425 tasks) through a clean-room scorer; ≥ 3 seeds | Held-out benchmarks, contamination checks, many seeds when affordable |

## What does *not* survive scaling down

Be honest about the toy:

- **The observation is a clean summary.** A real model must read raw CSVs and messy questions. Most of its errors happen *before* the method choice.
- **The answer key is small and certain here.** Real admissible sets need expert review. NanoStat-CTV's 20 policy cells are all still awaiting a statistician's sign-off.
- **The action space is 63 plans** (3 × 7 × 3). Real plans include variable roles, covariates and model specifications, so exploration is far harder.
- **Status of the full system (measured vs not):** the compute-matched CPU ablation, the null calibration and the
  P-Bench *tool-coverage ceiling* are measured. No LLM has been trained or evaluated yet, so there is **no** P-Bench accuracy and **no** leaderboard rank.
  See [`docs/RELEASE_READINESS.md`](https://github.com/Anandharajan/nanostat-ctv/blob/main/docs/RELEASE_READINESS.md).

## Where to go next

1. **Read the real reward.** `training/reward_server.py`, then `training/constraint.py`. Find the lines that play the role of
   `MALFORMED`: unparseable or unverifiable plans score `0.0`, *above* the invalid penalty `-0.2`. That is lesson 4's hack 1.
   It is listed as an open issue in `docs/limitations.md`. Whether the CPU proxy learner exploited it has not been measured.
   An LLM trained with GRPO could. Measuring it and fixing it before the first GPU run is a good first contribution.
2. **Read a real verifier.** `nanostat/verifier.py`: why two gates, and what does the "locked protocol" stop the model from changing?
3. **Run the full CPU ablation.** `nanostat ablate`: the same experiment as lesson 8 with 18 actions, 5 seeds and paired confidence intervals.
4. **Run an actual LLM, locally.** `nanostat eval --arms B1 --llm-model <id>` against a local OpenAI-compatible server (e.g. Magnitude). See `docs/LOCAL_LLM_VALIDATION.md`.
5. **Train one.** `notebooks/02_train_t4.ipynb` runs SFT plus GRPO on a 1.5B model on a free Colab or Kaggle T4.
6. **Extend the toy.** Add a world (lesson 3, exercise 2), a third reward (lesson 4, exercise 3), or PPO clipping (lesson 7, exercise 3). Send a pull request.

> The goal of nanostat is the one nanoGPT set for language models: a file small enough to read in an evening that
> still contains *every* moving part, so that the big system stops being magic.
