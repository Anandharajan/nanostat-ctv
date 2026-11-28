# CPU ablation of the CTV training signal (compute-matched)

**What this is.** A controlled, measured comparison of the SPEC §8 training signals (S0, R-F, R-A, R-AD, R-ADT, R-CTV, plus the post-hoc R-CTV-abs) at **matched compute**. It uses the project's real reward server, discordance sampler, twin-joint advantage and scenario constraint on 1,400 synthetic tasks. The learner is a **linear-softmax proxy policy** over an 18-action typed-plan catalogue, trained with group-normalised policy gradients (GRPO with one inner epoch; KL to the SFT reference).

**What this is not.** It is not the 1.5B language model, not P-Bench, and not evidence about LLM accuracy. It isolates the effect of the *training signal* on a learner of fixed capacity. Model-based (regression/survival) tasks are excluded because the proxy has no specification head.

## Correction notice (2026-11-28)
The first version of this document (v0.2.0) matched arms on optimiser updates × task instances. Zero-variance groups were re-sampled up to twice *without counting those rollouts*, and twin units overshot the batch by 0.7–1.3%. From the logged skip counts, the twin-based arms generated at least ~35% more rollouts than R-F (lower bound; re-samples that succeeded were not logged). The comparison was therefore **not** compute-matched, contrary to SPEC §8. The ablation now matches arms on **rollouts generated** (every sampled action, re-samples included; 19,200 per arm). The number of optimiser updates each arm reaches is logged, not fixed. All numbers below are from the corrected run. The earlier, larger R-CTV advantage (+0.086 validity) and R-CTV's low seed variance were partly artefacts of the extra rollouts.

## Setup (fixed a priori, identical across arms)
- Data: pilot profile, seed 20261126 (generator before the v0.4 regression families): 1,400 tasks (train 794 / dev 101 / locked test 505), 200 twin pairs; held out for test only: edits `outcome_type`, `small_counts`, template domain 4, scenario (correlation, skewed).
- Each (task, action) is scored once by `training/reward_server.py` (CTV reward and Fisher-style reward) and the confirmatory executor. Fresh null simulations for agent FPR: 10 scenarios × 300.
- Initialisation: programmatic SFT on 64 random training tasks (50 epochs), the same per seed for every arm.
- **Budget: 19,200 generated rollouts per arm** (= 300 × 16 task instances × group size 4), re-samples included; up to 2 re-samples of a zero-variance group; lr 0.5; KL β 0.02; η_twin 0.2, τ_twin 0.8; ε_D 0.1. Seeds 1–5. Greedy evaluation.
- No arm models assumption claims (r_assume = 0 for all arms; a constant offset).
- Arms: **S0** SFT only · **R-F** Fisher-R1-style reward (0.9 z-closeness to the single preferred reference + 0.1 conclusion; operational gate only; no knowledge of A/B; no abstention credit) · **R-A** CTV admissible-set reward · **R-AD** + discordance-weighted sampling · **R-ADT** + twin-joint advantage · **R-CTV** + scenario validity constraint · **R-CTV-abs** post-hoc variant (D = 1 for clarify/abstain tasks).
- Reproduce: `nanostat ablate --data <pilot dir> --seeds 1,2,3,4,5 --updates 300` (≈ 20 s once tables exist). Raw output: `results/ablation_cpu/ablation_results.json` (per-seed logs include rollouts generated, rollouts with gradient, updates).

## Results

### test split (greedy policy; mean ± sd over seeds)

| Arm | Method validity | Invalid-method rate | Twin consistency | Abstention correct | Held-out edits | Strict (single ref.) | Task-level FPR | Agent FPR on null sims |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.795 ± 0.073 | 0.109 ± 0.047 | 0.570 ± 0.023 | 0.236 ± 0.050 | 0.757 ± 0.027 | 0.787 ± 0.074 | 0.059 ± 0.005 | 0.044 (459/10356) |
| R-F | 0.880 ± 0.074 | 0.116 ± 0.074 | 0.580 ± 0.054 | 0.000 ± 0.000 | 0.770 ± 0.028 | 0.876 ± 0.075 | 0.048 ± 0.004 | 0.046 (524/11373) |
| R-A | 0.922 ± 0.068 | 0.011 ± 0.005 | 0.661 ± 0.026 | 0.440 ± 0.067 | 0.796 ± 0.004 | 0.890 ± 0.065 | 0.051 ± 0.004 | 0.044 (583/13122) |
| R-AD | 0.939 ± 0.083 | 0.014 ± 0.003 | 0.643 ± 0.017 | 0.012 ± 0.016 | 0.794 ± 0.005 | 0.900 ± 0.072 | 0.050 ± 0.005 | 0.043 (549/12644) |
| R-ADT | 0.933 ± 0.095 | 0.015 ± 0.004 | 0.657 ± 0.042 | 0.044 ± 0.037 | 0.791 ± 0.009 | 0.898 ± 0.090 | 0.047 ± 0.000 | 0.044 (562/12757) |
| R-CTV | 0.938 ± 0.086 | 0.015 ± 0.004 | 0.664 ± 0.036 | 0.048 ± 0.039 | 0.794 ± 0.005 | 0.908 ± 0.086 | 0.047 ± 0.001 | 0.044 (570/12926) |
| R-CTV-abs | 0.877 ± 0.107 | 0.018 ± 0.008 | 0.648 ± 0.044 | 0.488 ± 0.037 | 0.791 ± 0.005 | 0.842 ± 0.104 | 0.052 ± 0.002 | 0.045 (549/12191) |

### dev split (greedy policy; mean ± sd over seeds)

| Arm | Method validity | Invalid-method rate | Twin consistency | Abstention correct | Held-out edits | Strict (single ref.) | Task-level FPR | Agent FPR on null sims |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.758 ± 0.073 | 0.123 ± 0.030 | 0.800 ± 0.100 | 0.500 ± 0.316 | n/a | 0.743 ± 0.077 | 0.048 ± 0.026 | - |
| R-F | 0.893 ± 0.050 | 0.105 ± 0.050 | 0.900 ± 0.094 | 0.000 ± 0.000 | n/a | 0.879 ± 0.053 | 0.044 ± 0.022 | - |
| R-A | 0.899 ± 0.036 | 0.024 ± 0.012 | 0.975 ± 0.050 | 0.800 ± 0.245 | n/a | 0.883 ± 0.030 | 0.048 ± 0.000 | - |
| R-AD | 0.933 ± 0.079 | 0.026 ± 0.015 | 0.975 ± 0.050 | 0.000 ± 0.000 | n/a | 0.905 ± 0.080 | 0.049 ± 0.003 | - |
| R-ADT | 0.913 ± 0.083 | 0.040 ± 0.006 | 0.975 ± 0.050 | 0.000 ± 0.000 | n/a | 0.885 ± 0.084 | 0.049 ± 0.003 | - |
| R-CTV | 0.927 ± 0.055 | 0.040 ± 0.009 | 0.975 ± 0.050 | 0.000 ± 0.000 | n/a | 0.893 ± 0.063 | 0.049 ± 0.002 | - |
| R-CTV-abs | 0.842 ± 0.099 | 0.038 ± 0.010 | 0.975 ± 0.050 | 1.000 ± 0.000 | n/a | 0.824 ± 0.095 | 0.040 ± 0.020 | - |

### Compute parity (per arm, mean over seeds)

| Arm | Rollouts generated | Rollouts contributing a gradient | Optimiser updates | Task instances |
|---|---|---|---|---|
| S0 | 0 | 0 | 0 | 0 |
| R-F | 19,200 | 4,050 | 133 | 2,119 |
| R-A | 19,200 | 4,415 | 135 | 2,158 |
| R-AD | 19,200 | 3,485 | 127 | 2,028 |
| R-ADT | 19,201 | 4,738 | 136 | 2,193 |
| R-CTV | 19,200 | 4,713 | 136 | 2,192 |
| R-CTV-abs | 19,201 | 5,980 | 147 | 2,363 |

### Primary comparisons: R-CTV minus R-F (locked test split, paired bootstrap 2,000, Holm)

| Comparison | Difference | 95% CI | p | p (Holm) | n units |
|---|---|---|---|---|---|
| method_validity | +0.058 | [+0.043, +0.073] | 0.000 | 0.000 | 455 |
| twin_consistency | +0.084 | [+0.048, +0.125] | 0.000 | 0.000 | 88 |
| agent_fpr_reduction | +0.000 | [-0.003, +0.004] | 0.842 | 0.842 | 2619 |

Metric notes: *method validity* = executed method ∈ A(x, e) on answerable tasks; *twin consistency* = both members correct; *Strict (single ref.)* = decision matches the reference and |Δz| < 0.5 versus the registry-preferred method; *agent FPR on null sims* = rejections among fresh null simulations where the executed test's null hypothesis is true, pooled over seeds.

## Findings (compute-matched)
1. **The admissible-set reward (A) carries the effect.** At equal rollouts, R-F still chooses an invalid method in ~12% of answerable test tasks; every CTV-family arm is at 1.1–1.8%. Validity: R-F 0.880 → R-A 0.922.
2. **R-CTV vs R-F (locked test, Holm):** method validity **+0.058 [+0.043, +0.073]**, twin consistency **+0.084 [+0.048, +0.125]**, agent FPR no detectable difference (+0.000 [−0.003, +0.004]). R-CTV is also higher on the single-reference Strict score (0.908 vs 0.876).
3. **D, T and V add no measurable benefit over A in this proxy setting.** R-A, R-AD, R-ADT and R-CTV are within seed noise of each other on validity (0.922–0.939, sd 0.07–0.09). The paper must not claim that each component helps. The defensible claim is "the admissible-set reward helps; D, T and V were not separable from A at this scale".
4. **Negative: discordance sampling starves abstention.** Abstention correctness falls from 0.44 (R-A) to 0.01 (R-AD). The post-hoc D = 1 fix (R-CTV-abs) restores 0.49, at a validity cost (0.877).
5. **The validity constraint has nothing to correct here:** all RL arms are at ≈ 4.4% agent FPR.
6. **Re-sampling consumes budget:** arms reach 127–147 optimiser updates within the same rollout budget; only 18–31% of generated rollouts contribute a gradient (zero-variance groups are frequent with group size 4). The twin-joint advantage *reduces* zero-variance waste (R-ADT/R-CTV ≈ 4,700 vs R-AD 3,485 gradient rollouts). That is a real efficiency property, but it did not translate into accuracy here.

## Agent-level null calibration of the rule planner (B0), hypothesis-specific
1,000 fresh simulations per scenario, full agent re-run each time (`nanostat eval --arms B0 --suite nullcal --sims 1000`). Raw: `results/nullcal/b0_nullcal_v3_1.json`.

| Scenario | Agent FPR (tested null true) | Agent methods | Fixed methods on the same sims (FPR where their null is true) |
|---|---|---|---|
| two_group / none (small) | 0.063 [0.049, 0.080] | welch_t 881, permutation_test 119 | welch_t 0.063 [0.049, 0.080]; student_t 0.063 [0.049, 0.080]; mann_whitney 0.054 [0.041, 0.070]; permutation_test 0.063 [0.049, 0.080]; yuen_trimmed_t 0.052 [0.039, 0.068] |
| two_group / outliers (medium) | 0.053 [0.040, 0.069] | permutation_test 1000 | welch_t 0.052 [0.039, 0.068]; student_t 0.053 [0.040, 0.069]; mann_whitney 0.045 [0.033, 0.060]; permutation_test 0.053 [0.040, 0.069]; yuen_trimmed_t 0.043 [0.031, 0.057] |
| two_group / skewed (small) | 0.051 [0.038, 0.067] | permutation_test 934, welch_t 66 | welch_t 0.037 [0.026, 0.051]; student_t 0.040 [0.029, 0.054]; mann_whitney 0.046 [0.034, 0.061]; permutation_test 0.051 [0.038, 0.067]; yuen_trimmed_t 0.037 [0.026, 0.051] |
| two_group / unequal_variance (small) | 0.048 [0.036, 0.063] | welch_t 921, permutation_test 79 | welch_t 0.037 [0.026, 0.051]; student_t 0.141 [0.120, 0.164]; mann_whitney - (null false); permutation_test 0.145 [0.124, 0.168]; yuen_trimmed_t 0.045 [0.033, 0.060] |
| paired / none (small) | 0.048 [0.036, 0.063] | paired_t 940, permutation_test 60 | paired_t 0.048 [0.036, 0.063]; wilcoxon_signed_rank 0.041 [0.030, 0.055]; permutation_test 0.048 [0.036, 0.063] |
| paired / outliers (medium) | 0.052 [0.039, 0.068] | permutation_test 1000 | paired_t 0.038 [0.027, 0.052]; wilcoxon_signed_rank 0.055 [0.042, 0.071]; permutation_test 0.052 [0.039, 0.068] |
| correlation / none (medium) | 0.043 [0.031, 0.057] | pearson 985, spearman 10, permutation_test 5 | pearson 0.043 [0.031, 0.057]; spearman 0.051 [0.038, 0.067]; permutation_test 0.044 [0.032, 0.059] |
| correlation / outliers (medium) | n/a (tested null is false) | spearman 1000 | pearson 0.488 [0.457, 0.519]; spearman - (null false); permutation_test 0.465 [0.434, 0.496] |
| k_group / skewed (small) | 0.028 [0.019, 0.040] | kruskal_wallis 988, anova_oneway 7 | anova_oneway 0.033 [0.023, 0.046]; kruskal_wallis 0.029 [0.020, 0.041] |
| binary_two_group / small_counts (small) | 0.008 [0.004, 0.017] | fisher_exact 946, chi2_independence 2 | chi2_independence 0.021 [0.013, 0.032]; fisher_exact 0.008 [0.004, 0.017] |

## Threats to validity
- Proxy learner (linear capacity, hand-built features): effect sizes on an LLM will differ.
- Synthetic tasks; the admissible registry is `reviewed_by: PENDING`, and A/B and the evaluation share it (validity is measured against our own definition, not an independent test set).
- Hyperparameters were fixed a priori and not tuned per arm. Only R-CTV-abs was introduced after looking at dev results.
- 5 seeds; bootstrap over test tasks averaged over seeds.
