---
n: 3
title: Tasks, answer keys, typed plans
goal: Turn lesson 2's table into a task generator with answer keys, and give the agent an action space too small to cheat in.
prev: 02-null-simulation
prev_title: Simulate the null
next: 04-rewards
next_title: Rewards and reward hacking
---

## Three pieces

**1. A task** (`make_task`) is a dataset, a question and a hidden truth: which world, whether the effect is real,
and which tail the question asks for.

**2. An answer key** (`RULES`) is the *policy registry*. For each world it says:
- `A`: the **admissible** methods, any of which a careful statistician would accept. The first one is the conventional reference.
- `B`: the **invalid** methods, each with a reason.

```python
("two_group", "unequal_var"): (["welch_t"], {"student_t": "pooled variance, unequal variances"}),
("correlation", "outliers"):  (["spearman"], {"pearson": "leverage-driven"}),
("paired", "none"):           (["paired_t", "wilcoxon"], IGNORE_PAIRS),
```

These entries are exactly the findings of lesson 2, written down. In the full system this registry is versioned,
and every cell needs a statistician's sign-off. They are all still `PENDING`, which is one of the honest limitations.

**3. A typed plan** is three tokens: `[design, method, tail]`. No free text and no code.
`execute(task, plan)` is the verifier plus the executor. It returns a p-value, or `None` if the plan cannot run:
a method from the wrong design, or a paired test on data with no repeated IDs.

## What the agent sees

The agent never sees the data, the p-values or the answer key. `observe()` gives it the question words plus a
coarse **data profile**:

```bash
python edu/nanostat.py tasks
```

```
observation: does the treatment change the score compared with control ? cols:group outl:yes var:high
  gold plan: ['two_group', 'mann_whitney', 'two-sided'] | A: ['mann_whitney'] | B: {'student_t': 'outlier-driven', 'welch_t': 'outlier-driven'} | effect: large

observation: does study time change the score ? cols:xy outl:yes
  gold plan: ['correlation', 'spearman', 'two-sided'] | A: ['spearman'] | B: {'pearson': 'leverage-driven'} | effect: large
```

Profile tokens: `cols:group` / `cols:xy` (data shape), `ids:repeat` (the same units measured twice), `outl:yes/no`
(a robust z-score above 5), and `var:high/low` (variance ratio above 4). The **tail** comes only from the question
words: *increase / raise* means `greater`, *reduce / lower* means `less`, *change / affect* means `two-sided`.

## Why "the model never computes a number"

The model's whole output is 3 tokens from a 13-word plan vocabulary. It cannot make up a p-value, hide a
calculation, or run code. Anything it says can be checked. This is the core safety idea of NanoStat-CTV: the
**policy plans, deterministic tools compute, a verifier gates**.

## Exercises

1. Print 20 tasks and look for one where `outl:yes` but the world has no outliers, or the reverse.
   How often is the profile wrong? (The profile is a noisy summary, just as it would be for a real analyst.)
2. Add a world: `("two_group", "skewed")`, using exponential data. What belongs in `A` and in `B`? Use lesson 2's simulator to *decide*, not to guess.
3. Try to build a plan that `execute` accepts but that makes no sense. Can you? What does that say about how big the action space should be?
