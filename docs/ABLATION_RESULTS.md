# CPU ablation of the CTV training signal (compute-matched)

**What this is.** A controlled, measured comparison of the SPEC §8 training signals (S0, R-F, R-A, R-AD, R-ADT, R-CTV, plus the post-hoc R-CTV-abs) at **matched compute**. It uses the project's real reward server, discordance sampler, twin-joint advantage and scenario constraint on 1,400 synthetic tasks. The learner is a **linear-softmax proxy policy** over an 18-action typed-plan catalogue, trained with group-normalised policy gradients (GRPO with one inner epoch; KL to the SFT reference).

**What this is not.** It is not the 1.5B language model, not P-Bench, and not evidence about LLM accuracy. It isolates the effect of the *training signal* on a learner of fixed capacity. Model-based (regression/survival) tasks are excluded because the proxy has no specification head.

## Correction notice (2026-11-28)
The first version of this document (v0.2.0) matched arms on optimiser updates × task instances. Zero-variance groups were re-sampled up to twice *without counting those rollouts*, and twin units overshot the batch by 0.7–1.3%. From the logged skip counts, the twin-based arms generated at least ~35% more rollouts than R-F (lower bound; re-samples that succeeded were not logged). The comparison was therefore **not** compute-matched, contrary to SPEC §8. The ablation now matches arms on **rollouts generated** (every sampled action, re-samples included; 19,200 per arm). The number of optimiser updates each arm reaches is logged, not fixed. All numbers below are from the corrected run. The earlier, larger R-CTV advantage (+0.086 validity) and R-CTV's low seed variance were partly artefacts of the extra rollouts.

## Correction notice 2 (2026-11-28): reward ordering for broken plans
Until v0.4.0 the CTV reward (`training/reward_server.py::score_plan`) scored a plan that could not be parsed, verified or executed **0.0**, *above* the invalid-method penalty (−0.2). A policy can then lower its penalty by choosing broken plans instead of learning valid ones (reward hacking; reproduced in the teaching track, lesson 4). We added a broken-plan rate to the evaluation and re-ran the published configuration on regenerated, identical data. The re-run reproduces every published number exactly, which confirms the data are the same (`results/ablation_cpu/v0.4.0_reward/`). It shows:
- R-F never keeps broken plans (0.000). The old CTV-family arms kept **4.0–6.7%** on average, concentrated in seed 2: 21.5% broken at the SFT start, 22.2% after R-CTV, 28.8% after R-CTV-abs. Their low invalid-method rate (1.1–1.8%) was therefore partly flattered by plans that never ran.
- v0.4.1 scores broken plans **−0.5** (`RewardWeights.malformed_penalty`), below the invalid penalty. The R-F comparator is unchanged: a broken plan is already its minimum (0). All tables below are from the v0.4.1 re-run. Broken plans fall to 0.000–0.002 in every CTV arm; the CTV-family invalid-method rate is now 1.7–2.9%; R-CTV − R-F validity is +0.064 (was +0.058).

## Setup (fixed a priori, identical across arms)
- Data: pilot profile, seed 20261126 (generator before the v0.4 regression families): 1,400 tasks (train 794 / dev 101 / locked test 505), 200 twin pairs; held out for test only: edits `outcome_type`, `small_counts`, template domain 4, scenario (correlation, skewed).
- Each (task, action) is scored once by `training/reward_server.py` (CTV reward and Fisher-style reward) and the confirmatory executor. Fresh null simulations for agent FPR: 10 scenarios × 300.
- Initialisation: programmatic SFT on 64 random training tasks (50 epochs), the same per seed for every arm.
- **Budget: 19,200 generated rollouts per arm** (= 300 × 16 task instances × group size 4), re-samples included; up to 2 re-samples of a zero-variance group; lr 0.5; KL β 0.02; η_twin 0.2, τ_twin 0.8; ε_D 0.1. Seeds 1–5. Greedy evaluation.
- No arm models assumption claims (r_assume = 0 for all arms; a constant offset).
- Arms: **S0** SFT only · **R-F** Fisher-R1-style reward (0.9 z-closeness to the single preferred reference + 0.1 conclusion; operational gate only; no knowledge of A/B; no abstention credit) · **R-A** CTV admissible-set reward · **R-AD** + discordance-weighted sampling · **R-ADT** + twin-joint advantage · **R-CTV** + scenario validity constraint · **R-CTV-abs** post-hoc variant (D = 1 for clarify/abstain tasks).
- Reward: v0.4.1 (`malformed_penalty` −0.5). Reproduce: `nanostat ablate --data <pilot dir> --seeds 1,2,3,4,5 --updates 300` (≈ 20 s once tables exist). Raw output: `results/ablation_cpu/ablation_results.json` (per-seed logs include rollouts generated, rollouts with gradient, updates).

## Results

### test split (greedy policy; mean ± sd over seeds)

| Arm | Method validity | Invalid-method rate | Broken-plan rate | Twin consistency | Abstention correct | Held-out edits | Strict (single ref.) | Task-level FPR | Agent FPR on null sims |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 0.795 ± 0.073 | 0.109 ± 0.047 | 0.073 ± 0.082 | 0.570 ± 0.023 | 0.236 ± 0.050 | 0.757 ± 0.027 | 0.787 ± 0.074 | 0.059 ± 0.005 | 0.044 (459/10356) |
| R-F | 0.880 ± 0.074 | 0.116 ± 0.074 | 0.000 ± 0.000 | 0.580 ± 0.054 | 0.000 ± 0.000 | 0.770 ± 0.028 | 0.876 ± 0.075 | 0.048 ± 0.004 | 0.046 (524/11373) |
| R-A | 0.905 ± 0.095 | 0.023 ± 0.017 | 0.000 ± 0.000 | 0.655 ± 0.036 | 0.424 ± 0.085 | 0.794 ± 0.005 | 0.872 ± 0.092 | 0.048 ± 0.005 | 0.045 (575/12697) |
| R-AD | 0.942 ± 0.070 | 0.017 ± 0.007 | 0.000 ± 0.001 | 0.657 ± 0.021 | 0.068 ± 0.089 | 0.794 ± 0.005 | 0.908 ± 0.059 | 0.051 ± 0.004 | 0.044 (563/12912) |
| R-ADT | 0.943 ± 0.077 | 0.021 ± 0.009 | 0.000 ± 0.001 | 0.680 ± 0.050 | 0.100 ± 0.051 | 0.791 ± 0.009 | 0.907 ± 0.073 | 0.047 ± 0.001 | 0.044 (586/13226) |
| R-CTV | 0.945 ± 0.076 | 0.029 ± 0.025 | 0.000 ± 0.000 | 0.670 ± 0.040 | 0.064 ± 0.023 | 0.789 ± 0.004 | 0.910 ± 0.069 | 0.047 ± 0.001 | 0.044 (575/13033) |
| R-CTV-abs | 0.866 ± 0.107 | 0.022 ± 0.014 | 0.002 ± 0.003 | 0.673 ± 0.066 | 0.532 ± 0.027 | 0.785 ± 0.016 | 0.830 ± 0.104 | 0.051 ± 0.004 | 0.044 (517/11634) |

### dev split (greedy policy; mean ± sd over seeds)

| Arm | Method validity | Invalid-method rate | Broken-plan rate | Twin consistency | Abstention correct | Held-out edits | Strict (single ref.) | Task-level FPR | Agent FPR on null sims |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 0.758 ± 0.073 | 0.123 ± 0.030 | 0.079 ± 0.066 | 0.800 ± 0.100 | 0.500 ± 0.316 | n/a | 0.743 ± 0.077 | 0.048 ± 0.026 | - |
| R-F | 0.893 ± 0.050 | 0.105 ± 0.050 | 0.000 ± 0.000 | 0.900 ± 0.094 | 0.000 ± 0.000 | n/a | 0.879 ± 0.053 | 0.044 ± 0.022 | - |
| R-A | 0.889 ± 0.059 | 0.036 ± 0.008 | 0.000 ± 0.000 | 0.975 ± 0.050 | 0.900 ± 0.200 | n/a | 0.865 ± 0.051 | 0.048 ± 0.000 | - |
| R-AD | 0.937 ± 0.047 | 0.036 ± 0.019 | 0.000 ± 0.000 | 0.975 ± 0.050 | 0.100 ± 0.200 | n/a | 0.913 ± 0.040 | 0.049 ± 0.002 | - |
| R-ADT | 0.939 ± 0.031 | 0.044 ± 0.008 | 0.000 ± 0.000 | 0.975 ± 0.050 | 0.000 ± 0.000 | n/a | 0.911 ± 0.032 | 0.048 ± 0.000 | - |
| R-CTV | 0.937 ± 0.030 | 0.048 ± 0.012 | 0.000 ± 0.000 | 0.975 ± 0.050 | 0.000 ± 0.000 | n/a | 0.905 ± 0.031 | 0.048 ± 0.000 | - |
| R-CTV-abs | 0.820 ± 0.087 | 0.036 ± 0.014 | 0.004 ± 0.008 | 0.975 ± 0.050 | 1.000 ± 0.000 | n/a | 0.800 ± 0.083 | 0.040 ± 0.020 | - |

### Compute parity (per arm, mean over seeds)

| Arm | Rollouts generated | Rollouts contributing a gradient | Optimiser updates | Task instances |
|---|---|---|---|---|
| S0 | 0 | 0 | 0 | 0 |
| R-F | 19,200 | 4,050 | 133 | 2,119 |
| R-A | 19,200 | 4,217 | 133 | 2,122 |
| R-AD | 19,200 | 3,732 | 129 | 2,053 |
| R-ADT | 19,201 | 5,000 | 139 | 2,237 |
| R-CTV | 19,200 | 4,986 | 139 | 2,239 |
| R-CTV-abs | 19,200 | 4,942 | 139 | 2,239 |

### Primary comparisons: R-CTV minus R-F (locked test split, paired bootstrap 2,000, Holm)

| Comparison | Difference | 95% CI | p | p (Holm) | n units |
|---|---|---|---|---|---|
| method_validity | +0.064 | [+0.050, +0.079] | 0.000 | 0.000 | 455 |
| twin_consistency | +0.091 | [+0.050, +0.141] | 0.000 | 0.000 | 88 |
| agent_fpr_reduction | +0.000 | [-0.003, +0.004] | 0.841 | 0.841 | 2619 |

Metric notes: *method validity* = executed method ∈ A(x, e) on answerable tasks; *twin consistency* = both members correct; *broken-plan rate* = the chosen plan could not be verified or executed (answerable tasks); *Strict (single ref.)* = decision matches the reference and |Δz| < 0.5 versus the registry-preferred method; *agent FPR on null sims* = rejections among fresh null simulations where the executed test's null hypothesis is true, pooled over seeds.

## Findings (compute-matched, v0.4.1 reward)
1. **The admissible-set reward (A) carries the effect.** At equal rollouts, R-F still chooses an invalid method in ~12% of answerable test tasks. The CTV-family arms are at 1.7–2.9% invalid and 0.0–0.2% broken. Validity: R-F 0.880 → R-A 0.905.
2. **R-CTV vs R-F (locked test, Holm):** method validity **+0.064 [+0.050, +0.079]**, twin consistency **+0.091 [+0.050, +0.141]**, agent FPR no detectable difference (+0.000 [−0.003, +0.004]). R-CTV is also higher on the single-reference Strict score (0.910 vs 0.876).
3. **D, T and V add no measurable benefit over A in this proxy setting.** R-A, R-AD, R-ADT and R-CTV are within seed noise of each other on validity (0.905–0.945, sd 0.07–0.10). The paper must not claim that each component helps. The defensible claim is "the admissible-set reward helps; D, T and V were not separable from A at this scale".
4. **Negative: discordance sampling starves abstention.** Abstention correctness falls from 0.42 (R-A) to 0.07 (R-AD). The post-hoc D = 1 fix (R-CTV-abs) restores 0.53, at a validity cost (0.866).
5. **The validity constraint has nothing to correct here:** all RL arms are at ≈ 4.4% agent FPR.
6. **Re-sampling consumes budget:** arms reach 129–139 optimiser updates within the same rollout budget; only 19–26% of generated rollouts contribute a gradient (zero-variance groups are frequent with group size 4). The twin-joint advantage *reduces* zero-variance waste (R-ADT/R-CTV ≈ 5,000 vs R-AD 3,732 gradient rollouts). That is a real efficiency property, but it did not translate into accuracy here.
7. **The reward-ordering fix is not uniformly positive.** On seed 2, the fix raised R-CTV validity (0.767 → 0.793) but lowered R-A validity (0.787 → 0.721): plans that had hidden as broken became visible invalid or not-admissible choices. Mean R-A validity fell 0.922 → 0.905, and every CTV arm's invalid-method rate rose. These are the honest numbers; the earlier ones were partly an artefact of the ordering.

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
