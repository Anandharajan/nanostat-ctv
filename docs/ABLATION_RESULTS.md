# CPU ablation of the CTV training signal

**What this is.** A controlled, measured comparison of the training signals in SPEC §8 (S0, R-F, R-A, R-AD, R-ADT, R-CTV) at matched compute, using the project's real reward server, discordance sampler, twin-joint advantage and scenario constraint on 1,400 synthetic tasks. The learner is a **linear-softmax proxy policy** over an 18-action typed-plan catalogue, trained with group-normalised policy gradients (GRPO with one inner epoch; KL to the SFT reference).

**What this is not.** It is not the 1.5B language model, not P-Bench, and not evidence about LLM accuracy. It isolates the effect of the *training signal* on a learner of fixed capacity. LLM arms are NOT RUN (no GPU in this environment).

## Setup (fixed a priori, identical across arms)
- Data: `nanostat factory --profile pilot --seed 20261126`, generator 0.1.0 + v3.1 changes: 1,400 tasks (train 794 / dev 101 / locked test 505), 200 twin pairs over 7 edit types; held out for test only: edits `outcome_type`, `small_counts`, template domain 4, scenario (correlation, skewed).
- Each (task, action) is scored once by `training/reward_server.py` (CTV reward and Fisher-style reward) and the confirmatory executor; tables: `ablation_tasks.npz`. Fresh null simulations for agent FPR: 10 scenarios × 300 (`ablation_null.npz`).
- Initialisation: programmatic SFT on 64 random training tasks (50 epochs), the same per seed for every arm.
- Budget: 300 updates × 16 task instances × group size 4; lr 0.5; KL β 0.02; η_twin 0.2, τ_twin 0.8; ε_D 0.1. Seeds 1–5. Policy evaluation is greedy.
- Assumption head not modelled: no arm emits assumption claims, so r_assume = 0 for all arms (a constant offset).
- Arms: **S0** SFT only · **R-F** Fisher-R1-style reward (0.9 z-closeness to the single preferred reference + 0.1 conclusion; operational gate only, no knowledge of A/B, no abstention credit) · **R-A** CTV admissible-set reward · **R-AD** + discordance-weighted sampling · **R-ADT** + twin-joint advantage · **R-CTV** + scenario validity constraint · **R-CTV-abs** post-hoc variant (see findings).
- Reproduce: `nanostat ablate --data <pilot dir> --seeds 1,2,3,4,5 --updates 300` (≈ 30 s on a laptop CPU once tables exist). Raw output: `results/ablation_cpu/ablation_results.json`.

## Results

### test split (greedy policy; mean ± sd over seeds)

| Arm | Method validity | Invalid-method rate | Twin consistency | Abstention correct | Held-out edits | Strict (single ref.) | Task-level FPR | Agent FPR on null sims |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.795 ± 0.073 | 0.109 ± 0.047 | 0.570 ± 0.023 | 0.236 ± 0.050 | 0.757 ± 0.027 | 0.787 ± 0.074 | 0.059 ± 0.005 | 0.044 (459/10356) |
| R-F | 0.891 ± 0.065 | 0.105 ± 0.065 | 0.605 ± 0.048 | 0.000 ± 0.000 | 0.785 ± 0.021 | 0.889 ± 0.065 | 0.047 ± 0.002 | 0.045 (545/12035) |
| R-A | 0.945 ± 0.068 | 0.008 ± 0.012 | 0.698 ± 0.049 | 0.440 ± 0.104 | 0.794 ± 0.005 | 0.910 ± 0.061 | 0.049 ± 0.003 | 0.044 (580/13271) |
| R-AD | 0.943 ± 0.089 | 0.004 ± 0.003 | 0.675 ± 0.034 | 0.064 ± 0.050 | 0.796 ± 0.004 | 0.914 ± 0.080 | 0.049 ± 0.003 | 0.044 (573/12973) |
| R-ADT | 0.953 ± 0.066 | 0.011 ± 0.007 | 0.702 ± 0.030 | 0.112 ± 0.041 | 0.798 ± 0.000 | 0.924 ± 0.060 | 0.048 ± 0.003 | 0.044 (582/13313) |
| R-CTV | 0.977 ± 0.012 | 0.010 ± 0.004 | 0.691 ± 0.040 | 0.084 ± 0.057 | 0.791 ± 0.009 | 0.941 ± 0.017 | 0.046 ± 0.003 | 0.044 (578/13224) |
| R-CTV-abs | 0.922 ± 0.093 | 0.018 ± 0.015 | 0.698 ± 0.038 | 0.508 ± 0.072 | 0.789 ± 0.012 | 0.887 ± 0.091 | 0.052 ± 0.006 | 0.044 (568/12941) |

### dev split (greedy policy; mean ± sd over seeds)

| Arm | Method validity | Invalid-method rate | Twin consistency | Abstention correct | Held-out edits | Strict (single ref.) | Task-level FPR | Agent FPR on null sims |
|---|---|---|---|---|---|---|---|---|
| S0 | 0.758 ± 0.073 | 0.123 ± 0.030 | 0.800 ± 0.100 | 0.500 ± 0.316 | n/a | 0.743 ± 0.077 | 0.048 ± 0.026 | - |
| R-F | 0.923 ± 0.039 | 0.075 ± 0.038 | 0.975 ± 0.050 | 0.000 ± 0.000 | n/a | 0.915 ± 0.038 | 0.030 ± 0.024 | - |
| R-A | 0.952 ± 0.025 | 0.002 ± 0.004 | 0.975 ± 0.050 | 0.800 ± 0.245 | n/a | 0.933 ± 0.027 | 0.048 ± 0.000 | - |
| R-AD | 0.927 ± 0.082 | 0.006 ± 0.008 | 0.975 ± 0.050 | 0.100 ± 0.200 | n/a | 0.903 ± 0.075 | 0.049 ± 0.003 | - |
| R-ADT | 0.956 ± 0.026 | 0.024 ± 0.012 | 0.975 ± 0.050 | 0.000 ± 0.000 | n/a | 0.937 ± 0.022 | 0.048 ± 0.000 | - |
| R-CTV | 0.952 ± 0.012 | 0.024 ± 0.008 | 1.000 ± 0.000 | 0.000 ± 0.000 | n/a | 0.919 ± 0.028 | 0.048 ± 0.000 | - |
| R-CTV-abs | 0.915 ± 0.054 | 0.026 ± 0.008 | 0.975 ± 0.050 | 0.900 ± 0.200 | n/a | 0.887 ± 0.048 | 0.048 ± 0.000 | - |

### Primary comparisons: R-CTV minus R-F (locked test split, paired bootstrap 2,000, Holm)

| Comparison | Difference | 95% CI | p | p (Holm) | n units |
|---|---|---|---|---|---|
| method_validity | +0.086 | [+0.067, +0.106] | 0.000 | 0.000 | 455 |
| twin_consistency | +0.086 | [+0.045, +0.136] | 0.000 | 0.000 | 88 |
| agent_fpr_reduction | +0.001 | [-0.002, +0.004] | 0.703 | 0.703 | 2672 |

Metric notes: *method validity* = executed method ∈ A(x, e) on answerable tasks; *twin consistency* = both twin members correct (valid method or correct clarify/abstain); *Strict (single ref.)* = decision equals the reference and |Δz| < 0.5 versus the registry-preferred method (a P-Bench-like single-reference score); *agent FPR on null sims* = rejections among fresh null simulations where the executed test's null hypothesis is true, pooled over seeds.

## Findings
1. **The admissible-set reward (A) is the largest effect.** Against R-F at matched budget, validity rises and the invalid-method rate drops from ~10% to ~1%. The Fisher-style reward gives near-full credit to an invalid method whenever its p-value happens to sit close to the reference, so it keeps choosing invalid methods on non-discordant tasks.
2. **CTV does not trade away the single-reference score.** R-CTV is higher than R-F on Strict (single reference) as well (0.941 vs 0.889 test mean).
3. **R-CTV vs R-F primary comparisons (locked test, Holm):** method validity +0.086 [+0.067, +0.106]; twin consistency +0.086 [+0.045, +0.136]; agent FPR no detectable difference (+0.001 [−0.002, +0.004]).
4. **The validity constraint (V) had no measurable effect here.** Every RL arm already sits at ≈ 4.4% agent FPR on null simulations, so the Lagrangian has nothing to correct. Its only visible effect is lower seed variance of validity (R-CTV sd 0.012 vs R-ADT 0.066). A setting with liberal-but-attractive methods would be needed to test V; that is future work, not a claim.
5. **Negative finding: discordance sampling starves abstention.** D(x) = 0 for clarify/abstain tasks (no admissible/invalid pair), so ε_D-only weight makes them rare. Abstention correctness falls from 0.44 (R-A) to 0.06 (R-AD) on test, and 0.80 → 0.10 on dev.
6. **Post-hoc fix, chosen from dev and reported separately (R-CTV-abs):** treat clarify/abstain tasks as maximally discordant (D = 1). Abstention correctness recovers (test 0.508), but validity drops (0.922 vs 0.977) at this fixed budget. This is a real trade-off, not a free win; the paper should present both arms.
7. **Twins:** twin-joint advantage gives small, not clearly separable gains over R-AD in this proxy (0.702 vs 0.675); its main value may need a learner that can exploit surface cues, which a linear policy over these features cannot do much of.

## Agent-level null calibration of the rule planner (B0), hypothesis-specific
1,000 fresh simulations per scenario, full agent re-run each time (`nanostat eval --arms B0 --suite nullcal --sims 1000`, 9 min 18 s). Raw: `results/nullcal/b0_nullcal_v3_1.json`. Clopper–Pearson 95% CI.

| Scenario | Agent FPR (tested null true) | Agent methods | Fixed methods on the same sims (FPR where their null is true) |
|---|---|---|---|
| two_group / none (small) | 0.063 [0.049, 0.080] | welch_t 881, permutation_test 119 | welch_t 0.063 [0.049, 0.080]; student_t 0.063 [0.049, 0.080]; mann_whitney 0.054 [0.041, 0.070]; permutation_test 0.063 [0.049, 0.080]; yuen_trimmed_t 0.052 [0.039, 0.068] |
| two_group / outliers (medium) | 0.053 [0.040, 0.069] | permutation_test 1000 | welch_t 0.052 [0.039, 0.068]; student_t 0.053 [0.040, 0.069]; mann_whitney 0.045 [0.033, 0.060]; permutation_test 0.053 [0.040, 0.069]; yuen_trimmed_t 0.043 [0.031, 0.057] |
| two_group / skewed (small) | 0.051 [0.038, 0.067] | permutation_test 934, welch_t 66 | welch_t 0.037 [0.026, 0.051]; student_t 0.040 [0.029, 0.054]; mann_whitney 0.046 [0.034, 0.061]; permutation_test 0.051 [0.038, 0.067]; yuen_trimmed_t 0.037 [0.026, 0.051] |
| two_group / unequal_variance (small) | 0.048 [0.036, 0.063] | welch_t 921, permutation_test 79 | welch_t 0.037 [0.026, 0.051]; student_t 0.141 [0.120, 0.164]; mann_whitney — (null false); permutation_test 0.145 [0.124, 0.168]; yuen_trimmed_t 0.045 [0.033, 0.060] |
| paired / none (small) | 0.048 [0.036, 0.063] | paired_t 940, permutation_test 60 | paired_t 0.048 [0.036, 0.063]; wilcoxon_signed_rank 0.041 [0.030, 0.055]; permutation_test 0.048 [0.036, 0.063] |
| paired / outliers (medium) | 0.052 [0.039, 0.068] | permutation_test 1000 | paired_t 0.038 [0.027, 0.052]; wilcoxon_signed_rank 0.055 [0.042, 0.071]; permutation_test 0.052 [0.039, 0.068] |
| correlation / none (medium) | 0.043 [0.031, 0.057] | pearson 985, spearman 10, permutation_test 5 | pearson 0.043 [0.031, 0.057]; spearman 0.051 [0.038, 0.067]; permutation_test 0.044 [0.032, 0.059] |
| correlation / outliers (medium) | n/a (tested null is false) | spearman 1000 | pearson 0.488 [0.457, 0.519]; spearman — (null false); permutation_test 0.465 [0.434, 0.496] |
| k_group / skewed (small) | 0.028 [0.019, 0.040] | kruskal_wallis 988, anova_oneway 7 | anova_oneway 0.033 [0.023, 0.046]; kruskal_wallis 0.029 [0.020, 0.041] |
| binary_two_group / small_counts (small) | 0.008 [0.004, 0.017] | fisher_exact 946, chi2_independence 2 | chi2_independence 0.021 [0.013, 0.032]; fisher_exact 0.008 [0.004, 0.017] |

Reading: the earlier 6.5% flag on clean small two-group data is **not** selection inflation. On the same simulations, fixed Welch also rejects 6.3%, and the CI now includes 0.05. The registry's invalid labels are vindicated: under unequal variances Student t gives 14.1% and the permutation test 14.5%; under leverage points Pearson gives 48.8%. The agent avoids all of them.

## Threats to validity
- Proxy learner: linear capacity, hand-built features. Effect sizes on an LLM will differ.
- Synthetic tasks and an admissible-set registry that is `reviewed_by: PENDING`. A, B and the evaluation share the registry, so validity is measured against our own definition of admissible (not an independent test set).
- Hyperparameters were fixed a priori and not tuned per arm. Only R-CTV-abs was introduced after looking at dev results.
- 5 seeds; bootstrap over test tasks averaged over seeds.
