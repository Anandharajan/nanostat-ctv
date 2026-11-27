# Fisher-R1 baseline record

**Provenance.** Every value below is a *published* Fisher-R1 value (arXiv:2608.07437v2), **not re-run** by this project. They were transcribed from the maintainer's master plan (27 Nov 2026), which cites the paper's Tables 1–3 and §§4.3, 5.4. They have **not** been re-checked against the PDF in this repository's sessions. Verify each one against the primary source before any manuscript or leaderboard use.

## Protocol definitions used by our clean-room scorer (`benchmark/pbench/score.py`)

| Item | Definition |
|---|---|
| Benchmark | P-Bench (`May2222/P-Bench`, CC BY 4.0): 425 tasks; Easy 203 / Hard 222; 17 method categories |
| Trials | 3 per task; pass@1 = mean over the 3 trials; pass@3 = any trial correct |
| Raw | correct decision at α = 0.05 |
| Strict | correct decision AND \|Δz\| < 0.5, z(p) = Φ⁻¹(1 − p/2) |
| Failure handling | unsupported / failed / abstained / timed-out = incorrect |
| Fisher-R1 reward | 0.9 × z-closeness to the single canonical reference (z clipped at 5) + 0.1 × conclusion, with a validity gate (the code must run) |
| Optimiser | GRPO with DAPO-style settings (see paper) |

**Open point:** how p = 0 / p = 1 are mapped to z. Our scorer clips p to [1e-16, 1]. Confirm with the P-Bench authors (fixture sheet: `benchmark/pbench/score.py::FIXTURE_SHEET`).

## Hard Strict pass@1 (the BenchmarkList composite), published

| Model | Hard Strict pass@1 |
|---|---:|
| Fisher-R1-14B | 33.0 |
| Fisher-R1-7B | 30.6 |
| GPT-5.4 | 30.5 |
| Qwen3-Coder-30B | 27.8 |
| Qwen3-32B | 27.2 |
| DeepSeek V4 Pro | 26.3 |
| Qwen2.5-Coder-14B | 23.7 |
| DataMind-14B | 22.1 |
| Qwen2.5-Coder-7B + StatQA-FT | 21.2 |
| DataMind-7B | 19.1 |
| Qwen2.5-Coder-7B + DARE | 18.5 |
| GPT-OSS-120B | 18.2 |
| Qwen2.5-Coder-7B (base) | 13.2 |

## Failure analysis and per-method gains, published

- Residual failures: 78/213 specification errors, 65/213 perturbation-handling errors.
- Per-method gains (base → Fisher-R1, correct/total): Spearman 7 → 42 of 55; log-rank 18 → 30 of 35; Cox 5 → 12 of 20.
- Operational failures of the 7B backbone before training: 43.5% of tasks (SPEC §1).

## How NanoStat will compare

1. **Internal (primary):** R-CTV vs R-F (our clean-room re-implementation of the Fisher-R1 reward) at matched compute, same backbone. The CPU ablation in `docs/ABLATION_RESULTS.md` already uses this comparator with a proxy policy.
2. **External:** published values above, labelled "published, not re-run". Optionally re-run Fisher-R1-7B (open weights) on our harness, labelled "re-run on NanoStat harness".
