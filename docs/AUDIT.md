# Repository audit against SPEC v3.0 → v3.1 (27 Nov 2026)

Classification: IMPLEMENTED+TESTED · IMPLEMENTED-UNTESTED · PARTIAL · MISSING · BLOCKED · PLANNED. The row-level evidence (code path, test, artifact) is in [REQUIREMENTS_TRACEABILITY.md](../REQUIREMENTS_TRACEABILITY.md).

## The five P0 correctness defects from the gap report: all fixed and tested

| # | Defect in v0.1.0 | Fix in v0.2.0 | Evidence |
|---|---|---|---|
| 1 | Invalid methods could execute and give a decision (status was only reported) | Confirmatory mode rejects `invalid` and `unknown`; diagnostic mode never returns a decision | `nanostat/verifier.py::_gate2`; `tests/unit/test_verifier.py::test_invalid_method_*` |
| 2 | Model chose alpha / tail / null value | `Protocol` locked from study + question; plan may only restate it; reward penalises changes | `nanostat/protocol.py`; `test_verifier.py::test_protocol_fields_are_locked`; `test_reward_constraint.py::test_changing_locked_protocol_never_increases_reward` |
| 3 | Admissible sets mixed estimands (e.g. Welch and Mann–Whitney interchangeable) | Estimand-keyed A(x,e)/B(x,e); `estimand_mismatch`; Rp over A(x,e) only | `configs/policies/m0_m1.yaml` v0.2.0; `test_verifier.py::test_estimand_lock_mean_vs_location` |
| 4 | No re-verification after preprocessing | Gate 1 validates ops vs protocol; gate 2 re-profiles and re-verifies the analysed rows, which are hashed and are exactly the rows tested | `test_verifier.py::test_gate2_*` |
| 5 | Assumption reward was a checkbox (listing check names earned credit) | Credit only when the typed verdict matches the CPU's reading of the executed diagnostic; irrelevant claims penalised | `test_reward_constraint.py::test_assumption_credit_is_evidence_grounded`, `::test_listing_every_check_earns_nothing_extra` |

## Other changes in this pass
- **Hypothesis-specific nulls** (§2.6): e.g. under unequal variances "equal means" is true but "equal distributions" is false; under symmetric leverage contamination ρ = 0 is true but independence is false. Null calibration now judges each rejection against the null of the test actually run, and runs each fixed method on the same simulations.
- **Scenario-specific constraint** (§2.7): per-scenario λ with binomial tolerances and per-scenario power references.
- **Group-exclusive splits** (§2.8): whole twin-edit types, a template domain and a scenario family are held out for the locked test split.
- **New tool**: permutation test (two-group means, paired sign-flip, Pearson r), exact enumeration when small, seeded Monte Carlo otherwise; checked against independent exact enumerations.
- **New twin edit**: estimand change (mean ↔ typical).
- **Clean-room P-Bench scorer** with a fixture sheet for the authors.
- **CPU ablation study** of the six training arms at matched budget (proxy policy; see `docs/ABLATION_RESULTS.md`).

## What remains (honest status)
| Area | Status | Why |
|---|---|---|
| P-Bench loader / preflight / coverage ceiling | BLOCKED | requires approval to download `May2222/P-Bench` |
| M4 methods (OLS HC/cluster, logistic, probit, log-rank, Cox, IV, mixed, Tobit) and clustering twin | MISSING | largest coverage lever for P-Bench Hard; next engineering block |
| Typed inspect/diagnose action loop, schema-constrained decoding | PLANNED | v3.1 Step 5 |
| LLM SFT / CTV-GRPO / all LLM arms | NOT RUN | needs a T4 (Colab/Kaggle) |
| Statistician review of policy cells | BLOCKED | all cells `reviewed_by: PENDING` |
| Independent (non-circular) test set | PLANNED | must be authored separately from the training registry |
| Second-machine bundle reproduction, R parity | NOT RUN | no second machine / no `Rscript` here |
