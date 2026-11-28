---
n: 1
title: Tests from scratch
goal: Write seven hypothesis tests yourself, and check each one against scipy before you trust it.
next: 02-null-simulation
next_title: Simulate the null
---

## Setup (once)

```bash
git clone https://github.com/Anandharajan/nanostat-ctv && cd nanostat-ctv
pip install numpy scipy torch        # torch is only needed from lesson 5 on (CPU build is fine)
```

No install at all? Use the [Colab notebook](https://colab.research.google.com/github/Anandharajan/nanostat-ctv/blob/main/edu/nanostat_colab.ipynb).

## The idea

An AI agent that does statistics must never make up a number. In nanostat the model only **chooses** a test.
Plain code computes the p-value. So the code has to be correct, and "correct" means *checked against an
independent implementation*.

Open `edu/nanostat.py`, section 1. Every test is a few lines:

```python
def welch_t(a, b, alt):  # separate variances + Welch-Satterthwaite df
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return p_from_t((a.mean() - b.mean()) / math.sqrt(va + vb), df, alt)
```

| Test | Design | Assumes |
|---|---|---|
| `student_t` | two groups | normal-ish, **equal variances** |
| `welch_t` | two groups | normal-ish, variances may differ |
| `mann_whitney` | two groups | ranks only: robust to outliers |
| `paired_t` | before/after on the same units | differences normal-ish |
| `wilcoxon` | before/after | ranks of differences |
| `pearson` | two numeric variables | linear, no leverage points |
| `spearman` | two numeric variables | ranks: monotone, robust |

Each returns a p-value for `alt` in `two-sided`, `greater` or `less`. The **tail** belongs to the question,
not to the data. "Does it increase?" is one-sided. "Does it change?" is two-sided.

## Run it

```bash
python edu/nanostat.py stats
```

```
student_t  ours 0.948617  scipy 0.948617
welch_t    ours 0.945496  scipy 0.945496
pearson    ours 0.877674  scipy 0.877674
```

## Exercises

1. Add `mann_whitney`, `wilcoxon` and `spearman` to the `stats` comparison. The matching scipy functions are
   `mannwhitneyu(..., method="asymptotic")`, `wilcoxon(..., correction=True, method="approx")` and
   `spearmanr`. Do they match to 6 decimals? If one doesn't, find out why: ties, the continuity correction, or the normal approximation.
2. Change `alt` to `"greater"` in one comparison. Check that `p_greater + p_less == 1` for the t-tests.
3. `ranks()` gives tied values their average rank. Remove that, rerun exercise 1 on data with ties, and watch it break.

> **Why this matters for agents.** The full system (`nanostat/tools/`) does the same for 23 tools. Every one
> is tested against statsmodels, scipy or a second implementation (`tests/`). A wrong tool would silently
> poison every reward computed from it.
