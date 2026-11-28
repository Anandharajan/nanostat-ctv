---
n: 8
title: The experiment
goal: Compare two rewards the way a paper should, at equal compute, across seeds, with held-out evaluation, and say only what the numbers support.
prev: 07-grpo
prev_title: GRPO
next: 09-next-steps
next_title: From nano to real
---

## The protocol

`experiment()` in section 9:

- **3 seeds.** For each seed, one SFT model is trained (textbook teacher, 40 demonstrations), then copied, so both RL arms start from **the same weights**.
- **Same compute.** Each GRPO arm generates exactly **25,600 plans**. The hyperparameters are the same too.
- **Held-out evaluation.** 600 fresh test tasks, plus 700 null-only tasks for the false-positive rate. These are never trained on.
- **Metrics come from the answer key, not from either reward.** *Validity* = right design, an admissible method and the right tail.

```bash
python edu/nanostat.py compare        # about 6 minutes on a laptop CPU
```

## Measured result

Output saved in the repository as `results/edu/compare_3seeds.txt` (mean ± standard deviation over 3 seeds):

| Arm | Validity | Invalid-method rate | Tail accuracy | Agent false-positive rate |
|---|---|---|---|---|
| S0, SFT only | 0.436 ± 0.024 | 0.532 ± 0.000 | 0.949 ± 0.037 | 0.051 ± 0.001 |
| GRPO, Fisher-style reward | 0.462 ± 0.009 | 0.532 ± 0.000 | 0.976 ± 0.034 | 0.051 ± 0.002 |
| GRPO, CTV reward | **0.688 ± 0.118** | **0.267 ± 0.107** | 1.000 ± 0.000 | 0.045 ± 0.006 |

Per seed, CTV validity was 0.613, 0.855 and 0.595. Fisher-style validity was 0.468, 0.468 and 0.450.

## What you may and may not conclude

**Supported:** at equal compute, from the same start, the CTV reward cut the invalid-method rate on every seed.
The Fisher-style reward did not lower it at all (0.532 → 0.532): invalid methods that produce near-reference
p-values are paid almost as well as valid ones (lesson 4). Its small validity gain came from tail accuracy.

**Not supported:**
- *"CTV gives fewer false positives."* The agent false-positive rates overlap. In this world most invalid
  methods do not inflate Type I error on these null tasks (lesson 2). The exception is Student's t under unequal variances.
- *"CTV gets 69%."* The spread across seeds is large (0.595 to 0.855). Three seeds give the sign, not a precise size. Run more (exercise 1).
- *"CTV beats Fisher-R1."* This is a re-implementation of the *idea* of p-value matching in a toy world. It is not
  the Fisher-R1 system, its data or its compute. That comparison needs the full stack (lesson 9).

Single runs can mislead. The lesson 7 seed-0 run showed Fisher-style validity at 0.605. With three other
seeds it stayed near 0.46. Always report the seeds you ran, not the seed you liked.

## Exercises

1. Run `--seeds 1,2,3,4,5,6,7,8,9,10`. Compute a paired difference (CTV minus Fisher per seed) and a 95% interval for it. Is it still clearly above 0?
2. **Break compute parity on purpose.** Give the Fisher-style arm `2 * rollouts`. Does it catch up? How much extra compute would you have to give it, and would a reader of the paper notice?
3. Count compute in **tokens** instead of plans. Here they are proportional, because every plan is 3 tokens. Name two things in a real LLM agent that make tokens per rollout differ between arms. (Hint: retries, and longer reasoning.)
4. Start from `teacher="gold"` instead. Does the difference between rewards shrink? What does that say about *when* reward design matters?

> **How this mirrors the full repo.** NanoStat-CTV's own CPU ablation was first run without counting re-sampled
> rollouts, so the arms were not compute-matched. Fixing that dropped the measured gain from +0.086 to +0.058.
> The correction is published in `docs/ABLATION_RESULTS.md`. Compute parity is not a formality.
