---
n: 2
title: Simulate the null
goal: Find out, by simulation, which tests can be trusted in which situations. This is the ground truth every reward is built on.
prev: 01-statistics
prev_title: Tests from scratch
next: 03-tasks-and-plans
next_title: Tasks, answer keys, typed plans
---

## The idea

A p-value threshold of 0.05 promises something specific. **If nothing is going on, the test says
"significant" about 5% of the time.** That rate is the *Type I error rate*. You don't have to take it on trust:
generate many datasets where the null hypothesis is true by construction, run the test on each, and count.

nanostat has 7 small "worlds". Each is a data kind plus one complication:

| World | What is different |
|---|---|
| `two_group / none` | two clean groups |
| `two_group / unequal_var` | the **smaller** group has 3x the standard deviation |
| `two_group / outliers` | a few extreme values |
| `paired / none` | the same units measured before and after |
| `paired / outliers` | paired, with a few extreme differences |
| `correlation / none` | two numeric variables |
| `correlation / outliers` | two high-leverage points far from the rest |

## Run it

```bash
python edu/nanostat.py null       # 600 null datasets per world, every applicable test
```

Measured output (seeded, so you should get the same numbers):

```
two_group    none         student_t 0.055  welch_t 0.055  mann_whitney 0.060
two_group    unequal_var  student_t 0.157  welch_t 0.043  mann_whitney 0.107
two_group    outliers     student_t 0.045  welch_t 0.043  mann_whitney 0.052
paired       none         student_t 0.000  welch_t 0.000  mann_whitney 0.000  paired_t 0.055  wilcoxon 0.047
paired       outliers     student_t 0.000  welch_t 0.000  mann_whitney 0.000  paired_t 0.018  wilcoxon 0.063
correlation  none         pearson 0.043  spearman 0.048
correlation  outliers     pearson 0.480  spearman 0.073
```

With 600 runs, a true 5% rate lands between about 3.3% and 6.8% (95% interval). So read the table like this:

- **Student's t under unequal variances: 15.7%.** That is three times the promised rate. Welch's test fixes it.
  Mann–Whitney (10.7%) is *not* a fix either. Its variance formula assumes both groups have the same
  shape under the null, and here their spreads differ.
- **Pearson with leverage points: 48%.** Two points decide the answer. Spearman's ranks are close to nominal.
  Its 7.3% is slightly high, because x and y are not truly independent in this world, even though the correlation is zero.
- **Two-group tests on paired data: 0.0%.** They are not "too liberal". They throw away the pairing, so they
  are hopelessly *conservative* and would miss real effects. Being wrong in the safe direction is still wrong.
- **Outliers in two-group data: Student t 4.5%.** No Type I inflation here, because both groups are contaminated
  alike. The registry still marks t-tests as invalid in this world, because a couple of points drive the estimate and its power.
  "Invalid" is a judgement about the method. It does not always show up as a false-positive rate. Paired t with outliers (1.8%) is the same story.

## Exercises

1. Raise `runs` to 5,000 for one world. Does `welch_t` under `unequal_var` settle near 0.05?
2. Edit `make_task` so the *larger* group has the big variance (swap `na` and `nb`). Predict first: does
   Student's t become liberal or conservative? Run it to check.
3. Measure **power**: use `effect="small"` instead of `"null"` and count rejections. In `two_group / outliers`,
   which test finds the effect most often? That is the real reason the t-tests are marked invalid there.

> **Why this matters for agents.** A reward that only asks "is the p-value close to *the* reference
> p-value?" cannot see any of this table. Lesson 4 makes that concrete. The full system runs the same check at
> scale: `nanostat`'s null calibration with Clopper–Pearson intervals, in `docs/ABLATION_RESULTS.md` and
> `docs/REGRESSION_FAMILIES.md`.
