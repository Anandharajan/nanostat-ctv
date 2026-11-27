# Validation plan (SPEC v3.1)

Rules: dev split for every choice; locked test split, P-Bench and StatQA are touched only by frozen configurations. Unrun = NOT RUN. Negative results are reported.

| Metric | Data | Unit of analysis | Procedure | Threshold / role |
|---|---|---|---|---|
| Method validity | NanoStat locked test (answerable) | task | m̂ ∈ A(x, e) with protocol tail; seed-averaged; paired cluster bootstrap 2,000 | PRIMARY (R-CTV vs R-F), Holm |
| Agent-level FPR | ≥ 1,000 fresh null simulations per scenario (300 for the CPU ablation) | simulation | reject rate among sims whose *tested* null is true; Clopper–Pearson CI | PRIMARY, Holm; target ≤ α + ε_s |
| Twin consistency | twin pairs in locked test | pair | both members correct (valid method or correct clarify/abstain) | PRIMARY, Holm |
| Invalid-method rate | locked test | task | m̂ ∈ B(x, e) | secondary |
| Abstention correctness / false abstention | non-answerable / answerable tasks | task | reason code must match | secondary; target ≥ 90% |
| Held-out-edit accuracy | twins with `outcome_type`, `small_counts` edits | task | never seen in training | generalisation |
| Strict (single reference) | locked test | task | decision == reference ∧ \|Δz\| < 0.5 vs the preferred method | P-Bench proxy; non-inferiority margin −3 pp |
| P-Bench Raw/Strict/Relaxed pass@1/3 | P-Bench 425 × 3 | task | `benchmark/pbench/score.py`; unsupported = wrong | external; BLOCKED until download approved |
| Operational failure | all evaluations | trial | bad JSON, unknown column, execution error | target ≤ 5% |
| Efficiency | GPU runs | run | GPU-h, tokens/s, peak reserved VRAM, correct per GPU-h | ≤ 14 GiB reserved |

Statistics: paired bootstrap over tasks / pairs / simulations (2,000 resamples), McNemar only for independent pairs, Holm across the three primary comparisons, seed variance reported. No significance claim from smoke or pilot runs of the LLM.

Splits: train / dev / locked test by lineage hash; held out for test only: twin edits `outcome_type`, `small_counts`; template domain 4; scenario family (correlation, skewed). Twins share a split (tested).
