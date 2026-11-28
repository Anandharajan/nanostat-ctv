---
n: 4
title: Rewards and reward hacking
goal: Write two verifiable rewards, find out what each one can and cannot see, then break both on purpose.
prev: 03-tasks-and-plans
prev_title: Tasks, answer keys, typed plans
next: 05-tiny-gpt
next_title: A tiny GPT
---

## Two rewards

Both rewards are *verifiable*: code computes them from the task and the plan, with no human or learned judge.

**Fisher-style** (in the spirit of Fisher-R1, re-implemented from its description) compares the plan's p-value to
**one** reference p-value, on a z scale `zc(p)`:

```python
reward = 0.9 * exp(-|zc(p) - zc(p_ref)|) + 0.1 * [same decision at 0.05]      # 0 if the plan can't run
```

**CTV** (*Calibrated Test Validity*) checks the **method** against the answer key first:

```python
if plan can't run:                       reward = MALFORMED  (-0.5)
elif method in B or tail is wrong:       reward = -0.2
elif method not in A:                    reward = 0
else:                                    reward = 0.5 + 0.4 * closeness to ANY admissible p + 0.1 * same decision
```

## Run it

```bash
python edu/nanostat.py rewards     # one task in the two_group / outliers world, small real effect
```

```
['two_group', 'mann_whitney', 'less']      p=0.1343  fisher=1.000  ctv=1.000     <- the reference plan
['two_group', 'welch_t', 'less']           p=0.0532  fisher=0.682  ctv=-0.200    <- invalid here
['two_group', 'student_t', 'less']         p=0.0515  fisher=0.674  ctv=-0.200    <- invalid here
['two_group', 'mann_whitney', 'greater']   p=0.8657  fisher=0.338  ctv=-0.200    <- wrong tail
['paired', 'wilcoxon', 'less']             p=None    fisher=0.000  ctv=-0.500    <- cannot run
```

**The Fisher-style reward pays an invalid method 0.67.** It only sees "how far is your p from the reference p".
An invalid test that happens to give a similar p-value looks *almost right*. Worse, the signal depends on luck in
the data: on another seed the invalid test's p could land right next to the reference. The CTV reward is
negative for invalid methods **whatever p-value they happen to produce**. Same data, same tools, a different question.

You can try this live in panel 2 of the [demo](../index.html).

## Reward hacking, on purpose

An RL policy optimises *the reward you wrote*, not the one you meant. Here are two classic ways to get it wrong:

**Hack 1: dodge the penalty.** If `MALFORMED` were 0.0, a broken plan (0) would score *higher* than an invalid one
(−0.2). A policy that cannot yet tell valid from invalid can lift its average reward by emitting plans that
cannot run. That is why `MALFORMED = -0.5` sits below every runnable plan.

Measured (seed 0, SFT on the textbook teacher, then CTV-GRPO with 12,800 rollouts, 600 held-out tasks; the
"Hack it yourself" cell in the Colab notebook):

| `MALFORMED` | Broken greedy plans | Invalid-method rate | Validity |
|---|---|---|---|
| −0.5 | 0 / 600 | 0.433 | 0.567 |
| **0.0** | **240 / 600** | **0.028** | 0.572 |

Read the second row the way a careless paper would: *"the invalid-method rate fell from 43% to 3%!"* Nothing
improved. Validity is the same. The policy learned to **hide** from the penalty by emitting plans that cannot run.
A metric that only counts invalid methods cannot see that. Always report the broken-plan rate next to it.

**Hack 2: game a single reference.** The Fisher-style reward has one right answer per task. In worlds with
several admissible methods (`two_group / none` admits three), the policy is pushed toward whatever happens to
match the reference, not toward the whole set of defensible choices.

## Exercises

1. Reproduce the table above, then try 5 seeds. Add a `broken_rate` metric to `evaluate()` so every later experiment reports it.
   (Lesson 7 explains GRPO; you can run this first and read later.)
2. Find a seed where an **invalid** method gets a Fisher-style reward above 0.9. Hint: loop `make_task` over
   seeds in the `correlation / outliers` world and score `pearson`.
3. Design a third reward that also penalises tests that are *too conservative* (lesson 2's paired example).
   What would you need to know about each world to compute it?

> **The general lesson.** "Verifiable" does not mean "correct". A verifier checks exactly what you told it to
> check. The hard part of RL with verifiable rewards is deciding *what* to verify. NanoStat-CTV's answer is the
> admissible set A and the invalid set B, built from null simulation and awaiting a statistician's review.
