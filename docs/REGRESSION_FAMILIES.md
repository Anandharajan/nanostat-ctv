# Regression and survival task families (spec-level keys)

Seven model-based families extend the synthetic factory (`benchmark/factory/regression.py`) so that training and dev-set prompt work can cover P-Bench's Hard categories (OLS, logistic, IV, Cox, F-test) **without touching P-Bench**.

| Family | Perturbations | Admissible (canonical spec) | Invalid specs (reason) |
|---|---|---|---|
| reg_linear | none, heteroskedastic, clustered, confounded | ols_coef (HC SE if heteroskedastic; cluster-robust at the randomisation level if clustered; required covariates); mixed_lm if clustered | classical SE under heteroskedasticity; non-cluster SE under clustering; omitted confounder |
| reg_binary | none, clustered, confounded | logistic, probit, LPM (ols_coef with robust/cluster SE) | non-cluster SE under clustering; omitted confounder; Tobit (outcome type) |
| reg_iv | none | iv_2sls with the randomised offer as instrument | naive OLS on the endogenous take-up |
| reg_multiarm | none, clustered | ols_wald with the stated restriction (joint zero or equality) | non-cluster SE under clustering |
| reg_survgroup | none | log_rank; cox_ph with the group dummy | OLS on follow-up time (ignores censoring) |
| reg_cox | none, confounded | cox_ph with required covariates | omitted confounder; OLS on time |
| reg_panel | none | mixed_lm (random intercept per person); OLS with SE clustered by person | classical OLS ignoring repeated measures |

**Spec-level admissibility.** For these designs the method name is not enough. Each admissible key entry carries requirements (exposure, required covariates, allowed SE types, cluster or group variable, instruments, restriction). Each invalid entry carries a bad pattern. `benchmark/spec.py::spec_status` judges a plan's full `ModelSpec`, and the CTV reward, evaluation metrics and SFT gold plans all use it. Questions follow the P-Bench layout (research question, H0/H1, study design, variable glossary), so the agent has to read the specification from the text. No method or SE type is named.

**Spec-level twins** (wording-changed): `add_clustering`, `add_confounder`, `make_endogenous`. Each changes one design fact, so the base task's gold specification is no longer admissible on the twin (tested).

## Are the labels statistically right? Null calibration of admissible vs invalid specs
500 true-null simulations per row, small sample regime, alpha = 0.05 (`results/spec_label_nullcal.json`):

| Family / perturbation | Admissible spec | Null rejection [95% CI] | Invalid spec (reason) | Null rejection [95% CI] |
|---|---|---|---|---|
| reg_linear/clustered | ols_coef | 0.044 [0.028, 0.066] (22/500) | ols_coef (ignores_clustering) | 0.180 [0.147, 0.217] (90/500) |
| reg_linear/heteroskedastic | ols_coef | 0.056 [0.038, 0.080] (28/500) | ols_coef (ignores_heteroskedasticity) | 0.208 [0.173, 0.246] (104/500) |
| reg_linear/confounded | ols_coef | 0.040 [0.025, 0.061] (20/500) | ols_coef (omitted_confounder) | 0.600 [0.556, 0.643] (300/500) |
| reg_binary/clustered | logistic | 0.028 [0.015, 0.047] (14/500) | logistic (ignores_clustering) | 0.074 [0.053, 0.101] (37/500) |
| reg_binary/confounded | logistic | 0.044 [0.028, 0.066] (22/500) | logistic (omitted_confounder) | 0.240 [0.203, 0.280] (120/500) |
| reg_iv/none | iv_2sls | 0.044 [0.028, 0.066] (22/500) | ols_coef (endogenous_regressor) | 0.988 [0.974, 0.996] (494/500) |
| reg_multiarm/clustered | ols_wald | 0.072 [0.051, 0.098] (36/500) | ols_wald (ignores_clustering) | 0.216 [0.181, 0.255] (108/500) |
| reg_cox/confounded | cox_ph | 0.062 [0.043, 0.087] (31/500) | cox_ph (omitted_confounder) | 0.416 [0.372, 0.461] (208/500) |
| reg_panel/none | mixed_lm | 0.044 [0.028, 0.066] (22/500) | ols_coef (ignores_repeated_measures) | 0.004 [0.000, 0.014] (2/500) |

Reading:
- The labels behave as intended where an invalid spec is *liberal*: omitted confounders, naive OLS under endogeneity, ignored clustering, and classical SEs with unbalanced heteroskedastic arms all inflate the false-positive rate, while the admissible specs sit near 5%.
- **The first run caught a wrong label.** With balanced arms, classical SEs are robust to heteroskedasticity (5.4% vs 5.4%). The DGP now uses unbalanced arms with the larger variance in the smaller arm, which is where classical SEs genuinely fail.
- **Known small-sample effect:** cluster-robust (CR1) Wald/F with 30-60 clusters is slightly liberal for the multi-arm joint test. CR2 or wild-cluster bootstrap would fix this; that is future work, not claimed.
- **Panel:** ignoring repeated measures is *conservative* here because the tested effect is within-person. It is still a wrong standard error (an invalid specification), but it does not inflate Type I error. Do not describe it as such.
- These are synthetic families built from the registry's own judgements (`reviewed_by: PENDING`). A statistician should review them before any confirmatory use.

## Notes
- The CPU proxy-policy ablation skips model-based tasks (the proxy has no specification head), so its published numbers are unchanged. These families are for LLM SFT/GRPO and dev-set evaluation.
- The rule planner (B0) cannot plan these tasks unless the user declares `study.model`; on factory tasks (P-Bench style, no declared roles) it asks for clarification, as on P-Bench.
- Pilot profile (seed 20261126): 1,400 tasks, of which 627 are model-based (reg_linear 174, reg_iv 95, reg_binary 82, reg_cox 82, reg_panel 78, reg_survgroup 66, reg_multiarm 50); 66 spec twin pairs; factory time 76 s.
